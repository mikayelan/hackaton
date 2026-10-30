"""
fused_silu_mul.py — Fused SiLU(x) * y в одном кернеле.

Используется в FFN Qwen2:
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    result = self.down_proj(SiLU(gate) * up)

Проблема: стандартно это 3 операции:
  1. SiLU(gate) → временный тензор
  2. temp * up → временный тензор
  3. down_proj(result)

Наш кернел: SiLU(gate) * up → сразу в один тензор.
  - Убирает один промежуточный тензор
  - Меньше трафика памяти
  - Часть тех 44% element-wise операций

ПАТЧИТ: Qwen2MLP.forward
"""

import torch
import triton
import triton.language as tl
from .utils import TIMER


@triton.jit
def _fused_silu_mul_kernel(
    x_ptr,      # [N, D] — gate output (fp16)
    y_ptr,      # [N, D] — up output (fp16)
    out_ptr,    # [N, D] — result (fp16)
    stride_n,
    N,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """out = SiLU(x) * y"""
    row = tl.program_id(0)
    if row >= N:
        return

    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    x = tl.load(x_ptr + row * stride_n + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * stride_n + offsets, mask=mask, other=0.0).to(tl.float32)

    # SiLU(x) = x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_x = x * sigmoid_x

    # Multiply by y
    result = silu_x * y

    tl.store(out_ptr + row * stride_n + offsets, result.to(tl.float16), mask=mask)


def fused_silu_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute SiLU(x) * y in one Triton kernel.
    x, y: [*, D] fp16 tensors of same shape.
    Returns: [*, D] fp16
    """
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"

    orig_shape = x.shape
    D = x.shape[-1]
    x_2d = x.reshape(-1, D)
    y_2d = y.reshape(-1, D)
    N = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    BLOCK_D = triton.next_power_of_2(D)

    with TIMER.record("fused_silu_mul"):
        _fused_silu_mul_kernel[(N,)](
            x_2d, y_2d, out,
            x_2d.stride(0),
            N,
            D=D,
            BLOCK_D=BLOCK_D,
        )

    return out.reshape(orig_shape)


def patch_silu_mul():
    """
    РЕАЛЬНЫЙ ПАТЧ: подменяет Qwen2MLP.forward чтобы использовать fused SiLU*mul.

    Оригинальный Qwen2MLP.forward:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    Наш:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        fused = fused_silu_mul(gate, up)  # SiLU(gate) * up в одном кернеле
        return self.down_proj(fused)
    """
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

        def patched_forward(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            # Fused: SiLU(gate) * up в одном кернеле
            fused = fused_silu_mul(gate, up)
            return self.down_proj(fused)

        Qwen2MLP.forward = patched_forward

    except (ImportError, AttributeError) as e:
        import warnings
        warnings.warn(f"[!] Could not patch Qwen2MLP: {e}")
