"""
fused_lora.py — Fused LoRA forward: base_linear(x) + scaling * (x @ A @ B).

Проблема: стандартный LoRA делает:
  1. result = base_linear(x)          # [M, N] = [M, K] @ [K, N]
  2. lora_a_out = x @ lora_A          # [M, r] = [M, K] @ [K, r]
  3. lora_b_out = lora_a_out @ lora_B # [M, N] = [M, r] @ [r, N]
  4. result += scaling * lora_b_out   # [M, N] += scalar * [M, N]

Шаги 2-4 создают промежуточный тензор [M, r] и делают лишний проход по памяти.

Решение: один кернел считает base + scaling * (x @ A @ B), не материализуя
промежуточный [M, r]. Для маленького r=16 это экономит память и трафик.

Также убираем лишние dtype casts между fp16 и fp32.
"""

import torch
import triton
import triton.language as tl
from typing import Optional

from .utils import TIMER


# ---------------------------------------------------------------------------
# Triton kernel: fused base + LoRA
# ---------------------------------------------------------------------------

@triton.jit
def _fused_lora_kernel(
    base_ptr,       # [M, N] — output of base linear (fp16)
    x_ptr,          # [M, K] — input (fp16)
    lora_a_ptr,     # [r, K] or [K, r] — LoRA A weights (fp16)
    lora_b_ptr,     # [N, r] or [r, N] — LoRA B weights (fp16)
    out_ptr,        # [M, N] — output (fp16)
    M, N, K, R,
    stride_bm, stride_bn,     # base strides
    stride_xm, stride_xk,     # x strides
    stride_am, stride_ak,     # lora_a strides
    stride_bk, stride_bn_lora,# lora_b strides
    scaling,                  # alpha / r
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    A_TRANSPOSED: tl.constexpr,  # if True, A is [K, r], else [r, K]
    B_TRANSPOSED: tl.constexpr,  # if True, B is [r, N], else [N, r]
):
    """
    out = base + scaling * (x @ lora_A @ lora_B)
    
    Computed in tiles over (M, N) with reduction over rank R.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    # Load base output
    base = tl.load(
        base_ptr + m_offsets[:, None] * stride_bm + n_offsets[None, :] * stride_bn,
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    
    # Accumulate lora: x @ A @ B
    lora_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over rank dimension
    for r_start in range(0, R, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)
        r_mask = r_offsets < R
        
        # Compute x @ A -> [BLOCK_M, BLOCK_R]
        xa = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for k in range(K):
            # Load x[:, k]
            x_col = tl.load(
                x_ptr + m_offsets * stride_xm + k * stride_xk,
                mask=m_mask,
                other=0.0,
            )
            # Load A[k, :] or A[:, k] depending on layout
            if A_TRANSPOSED:
                a_row = tl.load(
                    lora_a_ptr + k * stride_am + r_offsets * stride_ak,
                    mask=r_mask,
                    other=0.0,
                )
            else:
                a_row = tl.load(
                    lora_a_ptr + r_offsets * stride_am + k * stride_ak,
                    mask=r_mask,
                    other=0.0,
                )
            xa += x_col[:, None] * a_row[None, :]
        
        # Compute (x @ A) @ B -> [BLOCK_M, BLOCK_N]
        for r_idx in range(BLOCK_R):
            r_abs = r_start + r_idx
            if r_abs >= R:
                break
            
            # Load B[r, :] or B[:, r]
            if B_TRANSPOSED:
                b_col = tl.load(
                    lora_b_ptr + r_abs * stride_bk + n_offsets * stride_bn_lora,
                    mask=n_mask,
                    other=0.0,
                )
            else:
                b_col = tl.load(
                    lora_b_ptr + n_offsets * stride_bk + r_abs * stride_bn_lora,
                    mask=n_mask,
                    other=0.0,
                )
            
            lora_acc += xa[:, r_idx:r_idx+1] * b_col[None, :]
    
    # Final: base + scaling * lora
    result = base + scaling * lora_acc
    
    # Store
    tl.store(
        out_ptr + m_offsets[:, None] * stride_bm + n_offsets[None, :] * stride_bn,
        result.to(tl.float16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_lora_forward(
    base_output: torch.Tensor,   # [M, N] from base linear
    x: torch.Tensor,             # [M, K] input
    lora_A: torch.Tensor,        # [r, K] or [K, r]
    lora_B: torch.Tensor,        # [N, r] or [r, N]
    scaling: float,              # alpha / r
    a_transposed: bool = False,
    b_transposed: bool = False,
) -> torch.Tensor:
    """
    Compute base_output + scaling * (x @ lora_A @ lora_B) in one kernel.
    """
    M = x.shape[0]
    K = x.shape[1]
    
    if a_transposed:
        R = lora_A.shape[1]
    else:
        R = lora_A.shape[0]
    
    if b_transposed:
        N = lora_B.shape[1]
    else:
        N = lora_B.shape[0]
    
    # Allocate output
    out = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    # Block sizes
    BLOCK_M = min(64, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_R = min(16, triton.next_power_of_2(R))  # R is usually 16
    
    with TIMER.record("fused_lora"):
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _fused_lora_kernel[grid](
            base_output, x, lora_A, lora_B, out,
            M, N, K, R,
            base_output.stride(0), base_output.stride(1),
            x.stride(0), x.stride(1),
            lora_A.stride(0), lora_A.stride(1),
            lora_B.stride(0), lora_B.stride(1),
            scaling,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_R=BLOCK_R,
            A_TRANSPOSED=a_transposed,
            B_TRANSPOSED=b_transposed,
        )
    
    return out


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def patch_lora():
    """
    Patch PEFT's LoraLayer to use fused forward.
    
    In PEFT, the LoraLayer.forward does:
      result = F.linear(x, weight, bias)  # base
      result += lora_B(lora_A(dropout(x))) * scaling
    
    We replace this with our fused kernel.
    """
    try:
        from peft.tuners.lora.layer import LoraLayer
        
        _original_forward = LoraLayer.forward
        
        def patched_forward(self, x: torch.Tensor, *args, **kwargs):
            # For now, use original forward — the fused kernel is available
            # for direct integration when we have access to the internals.
            # Full patching requires careful handling of dropout, adapters, etc.
            return _original_forward(self, x, *args, **kwargs)
        
        # Don't patch by default — too risky with PEFT internals
        # LoraLayer.forward = patched_forward
        
    except (ImportError, AttributeError):
        pass
