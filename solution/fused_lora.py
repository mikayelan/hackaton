"""
fused_lora.py — Fused LoRA: base + scaling * (x @ A @ B).
ПАТЧИТ: PEFT LoraLinear.forward

ИСПРАВЛЕНИЯ v2:
  - Правильный вызов base_layer (работает с 4-bit квантованными весами)
  - Safe проверка атрибутов PEFT
  - Правильная обработка scaling (dict vs float)
  - Правильная обработка dropout
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
    R: tl.constexpr,
):
    """
    out = base + scaling * (x @ A^T @ B^T)
    A is [R, K], B is [N, R]
    R is constexpr (16 for this task).
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

    lora_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        x_block = tl.load(
            x_ptr + m_off[:, None] * stride_xm + k_off[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)

        for r in range(R):
            a_block = tl.load(
                lora_a_ptr + r * stride_ar + k_off * stride_ak,
                mask=k_mask, other=0.0,
            ).to(tl.float32)

            xa = tl.sum(x_block * a_block[None, :], axis=1)

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
            R=R,
        )
    return out


def patch_lora():
    """
    РЕАЛЬНЫЙ ПАТЧ: PEFT LoraLinear.forward

    ИСПРАВЛЕНИЕ v2:
      - Используем self.base_layer(x) вместо F.linear — работает с 4-bit
      - Safe проверка всех атрибутов
      - Правильная обработка scaling (может быть dict или float)
    """
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear

        _orig_forward = LoraLinear.forward

        def patched_forward(self, x: torch.Tensor, *args, **kwargs):
            # ИСПРАВЛЕНИЕ: вызываем base_layer(x) — это работает с квантованными весами
            # F.linear(x, self.base_layer.weight) НЕ работает с 4-bit!
            # НЕ передаём *args, **kwargs в base_layer — Linear4bit их не принимает
            result = self.base_layer(x)

            # LoRA адаптер (если активен)
            active_adapter = self.active_adapter
            if isinstance(active_adapter, str):
                adapters = [active_adapter]
            elif hasattr(active_adapter, '__iter__'):
                adapters = list(active_adapter)
            else:
                adapters = [active_adapter]

            for adapter in adapters:
                if adapter not in self.lora_A:
                    continue

                lora_A = self.lora_A[adapter].weight  # [R, K]
                lora_B = self.lora_B[adapter].weight  # [N, R]

                # scaling может быть dict или float
                if isinstance(self.scaling, dict):
                    scaling = self.scaling[adapter]
                elif hasattr(self.scaling, '__getitem__'):
                    try:
                        scaling = self.scaling[adapter]
                    except (KeyError, TypeError):
                        scaling = float(self.scaling)
                else:
                    scaling = float(self.scaling)

                # Dropout если нужен
                x_lora = x
                if hasattr(self, 'lora_dropout') and adapter in self.lora_dropout:
                    dropout_layer = self.lora_dropout[adapter]
                    if dropout_layer is not None and dropout_layer.p > 0:
                        x_lora = dropout_layer(x)

                # Fused LoRA
                if x_lora.dim() == 2:
                    result = fused_lora_forward(result, x_lora, lora_A, lora_B, scaling)
                else:
                    orig_shape = x_lora.shape
                    x_2d = x_lora.reshape(-1, orig_shape[-1])
                    result_2d = result.reshape(-1, result.shape[-1])
                    result_fused = fused_lora_forward(result_2d, x_2d, lora_A, lora_B, scaling)
                    result = result_fused.reshape(orig_shape[:-1] + (result.shape[-1],))

            return result

        LoraLinear.forward = patched_forward
    except (ImportError, AttributeError) as e:
        import warnings
        warnings.warn(f"[!] Could not patch PEFT LoraLinear: {e}")
