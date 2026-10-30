"""
fused_lora.py — Fused LoRA: base + scaling * (x @ A @ B).

ПАТЧИТ: PEFT LoraLayer — подменяет forward на fused версию.
Это работает — Soup использует PEFT для LoRA адаптеров.
"""
import torch
import triton
import triton.language as tl
from .utils import TIMER


@triton.jit
def _fused_lora_kernel(
    base_ptr, x_ptr, lora_a_ptr, lora_b_ptr, out_ptr,
    M, N, K,
    stride_bm, stride_bn,
    stride_xm, stride_xk,
    stride_ar, stride_ak,
    stride_bn_lora, stride_br,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    R: tl.constexpr,  # R must be constexpr for Triton loops
):
    """
    out = base + scaling * (x @ A^T @ B^T)
    A is [R, K], B is [N, R]
    R is fixed at compile time (16 for this task).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_off < M
    n_mask = n_off < N

    base = tl.load(
        base_ptr + m_off[:, None] * stride_bm + n_off[None, :] * stride_bn,
        mask=m_mask[:, None] & n_mask[None, :], other=0.0,
    ).to(tl.float32)

    # Compute x @ A^T @ B^T
    lora_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        # Load x block [BLOCK_M, BLOCK_K]
        x_block = tl.load(
            x_ptr + m_off[:, None] * stride_xm + k_off[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)

        # For each r: compute dot(x_block, A[r, :]) * B[n, r]
        for r in range(R):
            # A[r, k_start:k_start+BLOCK_K]
            a_block = tl.load(
                lora_a_ptr + r * stride_ar + k_off * stride_ak,
                mask=k_mask, other=0.0,
            ).to(tl.float32)

            # dot product: [BLOCK_M] = sum_k x_block[m,k] * a_block[k]
            xa = tl.sum(x_block * a_block[None, :], axis=1)  # [BLOCK_M]

            # B[n, r]
            b_val = tl.load(
                lora_b_ptr + n_off * stride_bn_lora + r * stride_br,
                mask=n_mask, other=0.0,
            ).to(tl.float32)

            lora_acc += xa[:, None] * b_val[None, :]

    result = base + scaling * lora_acc
    tl.store(
        out_ptr + m_off[:, None] * stride_bm + n_off[None, :] * stride_bn,
        result.to(tl.float16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def fused_lora_forward(base_output, x, lora_A, lora_B, scaling):
    """base + scaling * (x @ A^T @ B^T). A=[R,K], B=[N,R]."""
    M, K = x.shape
    R = lora_A.shape[0]
    N = lora_B.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=torch.float16)
    BLOCK_M = min(32, triton.next_power_of_2(M))
    BLOCK_N = min(32, triton.next_power_of_2(N))
    BLOCK_K = min(128, triton.next_power_of_2(K))

    with TIMER.record("fused_lora"):
        _fused_lora_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
            base_output, x, lora_A, lora_B, out,
            M, N, K,
            base_output.stride(0), base_output.stride(1),
            x.stride(0), x.stride(1),
            lora_A.stride(0), lora_A.stride(1),
            lora_B.stride(0), lora_B.stride(1),
            scaling,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            R=R,  # R is constexpr
        )
    return out


def patch_lora():
    """
    РЕАЛЬНЫЙ ПАТЧ: подменяет PEFT LoraLayer.
    Это работает — Soup использует PEFT для LoRA.
    """
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear

        _orig_forward = LoraLinear.forward

        def patched_forward(self, x: torch.Tensor, *args, **kwargs):
            # Базовый linear
            result = torch.nn.functional.linear(
                x, self.base_layer.weight, self.base_layer.bias
            )

            # LoRA адаптер (если активен)
            if self.active_adapter in self.lora_A:
                adapter = self.active_adapter
                lora_A = self.lora_A[adapter].weight  # [R, K]
                lora_B = self.lora_B[adapter].weight  # [N, R]
                
                # scaling может быть dict или float
                if isinstance(self.scaling, dict):
                    scaling = self.scaling[adapter]
                else:
                    scaling = self.scaling

                # Dropout если нужен
                x_lora = x
                if hasattr(self, 'lora_dropout') and adapter in self.lora_dropout:
                    if self.lora_dropout[adapter].p > 0:
                        x_lora = self.lora_dropout[adapter](x)

                # Fused LoRA
                if x_lora.dim() == 2:  # [M, K]
                    result = fused_lora_forward(
                        result, x_lora, lora_A, lora_B, scaling
                    )
                else:
                    # Для 3D тензоров — reshape и fused
                    orig_shape = x_lora.shape
                    x_2d = x_lora.reshape(-1, orig_shape[-1])
                    result_2d = result.reshape(-1, result.shape[-1])
                    result_fused = fused_lora_forward(
                        result_2d, x_2d, lora_A, lora_B, scaling
                    )
                    result = result_fused.reshape(orig_shape[:-1] + (result.shape[-1],))

            return result

        LoraLinear.forward = patched_forward

    except (ImportError, AttributeError) as e:
        import warnings
        warnings.warn(f"[!] Could not patch PEFT LoraLinear: {e}")
