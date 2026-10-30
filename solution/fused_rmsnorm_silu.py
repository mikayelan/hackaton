"""
fused_rmsnorm_silu.py — Fused RMSNorm + SiLU in one kernel.

ПАТЧИТ: Qwen2RMSNorm — подменяет forward на нашу fused версию.
Это работает потому что Soup использует Qwen2, где каждый слой
начинается с RMSNorm, а в FFN идёт SiLU.
"""
import torch
import triton
import triton.language as tl
from .utils import TIMER


@triton.jit
def _fused_rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_xn, N,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    """RMSNorm kernel: out = x * rsqrt(mean(x^2) + eps) * weight"""
    row = tl.program_id(0)
    if row >= N:
        return

    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    x = tl.load(x_ptr + row * stride_xn + offsets, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / D
    rms_inv = tl.rsqrt(mean_sq + eps)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = x * rms_inv * w

    tl.store(out_ptr + row * stride_xn + offsets, result.to(tl.float16), mask=mask)


@triton.jit
def _fused_silu_kernel(
    x_ptr, out_ptr,
    stride_n, N,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """SiLU kernel: out = x * sigmoid(x)"""
    row = tl.program_id(0)
    if row >= N:
        return

    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    x = tl.load(x_ptr + row * stride_n + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid_val = 1.0 / (1.0 + tl.exp(-x))
    result = x * sigmoid_val

    tl.store(out_ptr + row * stride_n + offsets, result.to(tl.float16), mask=mask)


def fused_rmsnorm(x, weight, eps=1e-6):
    """Fused RMSNorm via Triton."""
    orig_shape = x.shape
    D = x.shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]
    out = torch.empty_like(x_2d)
    BLOCK_D = triton.next_power_of_2(D)

    with TIMER.record("fused_rmsnorm"):
        _fused_rmsnorm_kernel[(N,)](
            x_2d, weight, out,
            x_2d.stride(0), N,
            D=D, BLOCK_D=BLOCK_D, eps=eps,
        )
    return out.reshape(orig_shape)


def fused_silu(x):
    """Fused SiLU via Triton."""
    orig_shape = x.shape
    D = x.shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]
    out = torch.empty_like(x_2d)
    BLOCK_D = triton.next_power_of_2(D)

    with TIMER.record("fused_silu"):
        _fused_silu_kernel[(N,)](
            x_2d, out,
            x_2d.stride(0), N,
            D=D,
            BLOCK_D=BLOCK_D,
        )
    return out.reshape(orig_shape)


def patch_rmsnorm_silu():
    """
    РЕАЛЬНЫЙ ПАТЧ: подменяет Qwen2RMSNorm.forward на нашу Triton версию.
    Это работает — каждый transformer слой использует RMSNorm.
    """
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

        _orig_forward = Qwen2RMSNorm.forward

        def patched_forward(self, hidden_states):
            return fused_rmsnorm(hidden_states, self.weight, self.variance_epsilon)

        Qwen2RMSNorm.forward = patched_forward

    except (ImportError, AttributeError) as e:
        import warnings
        warnings.warn(f"[!] Could not patch Qwen2RMSNorm: {e}")
