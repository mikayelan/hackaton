"""
fused_rope.py — Fused Rotary Position Embedding (RoPE).

Проблема: стандартная реализация RoPE делает:
  1. Разделить Q и K на две половины по dim
  2. Применить cos и sin к каждой половине
  3. Собрать обратно
  
Каждый шаг создаёт промежуточные тензоры. Для [B, H, S, D]:
  - 4 загрузки + 4 записи вместо 2+2

Решение: один кернел применяет RoPE к Q и K одновременно.
  - Загружаем Q1, Q2, K1, K2, cos, sin
  - Считаем: out1 = x1*cos - x2*sin, out2 = x1*sin + x2*cos
  - Пишем сразу в выход
  - Экономим ~50% трафика памяти для RoPE
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional

from .utils import TIMER


# ---------------------------------------------------------------------------
# Triton kernel: fused RoPE for Q and K
# ---------------------------------------------------------------------------

@triton.jit
def _fused_rope_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    q_out_ptr, k_out_ptr,
    stride_qt, stride_qh,
    stride_kt, stride_kh,
    stride_cos_pos,
    TOTAL_TOKENS,
    H,
    D: tl.constexpr,        # full head dim
    HALF_D: tl.constexpr,   # D // 2
    BLOCK_D: tl.constexpr,  # power of 2 >= HALF_D
):
    """Apply RoPE to Q and K simultaneously."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    if pid_t >= TOTAL_TOKENS:
        return
    
    pos = pid_t
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HALF_D
    
    cos_vals = tl.load(cos_ptr + pos * stride_cos_pos + d_offsets, mask=d_mask, other=0.0)
    sin_vals = tl.load(sin_ptr + pos * stride_cos_pos + d_offsets, mask=d_mask, other=0.0)
    
    # Q rotation
    q_base = pid_t * stride_qt + pid_h * stride_qh
    q1 = tl.load(q_ptr + q_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + q_base + HALF_D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    
    tl.store(q_out_ptr + q_base + d_offsets, (q1 * cos_vals - q2 * sin_vals).to(tl.float16), mask=d_mask)
    tl.store(q_out_ptr + q_base + HALF_D + d_offsets, (q1 * sin_vals + q2 * cos_vals).to(tl.float16), mask=d_mask)
    
    # K rotation
    k_base = pid_t * stride_kt + pid_h * stride_kh
    k1 = tl.load(k_ptr + k_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_ptr + k_base + HALF_D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    
    tl.store(k_out_ptr + k_base + d_offsets, (k1 * cos_vals - k2 * sin_vals).to(tl.float16), mask=d_mask)
    tl.store(k_out_ptr + k_base + HALF_D + d_offsets, (k1 * sin_vals + k2 * cos_vals).to(tl.float16), mask=d_mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_apply_rotary_pos_emb(
    q: torch.Tensor,          # [B, H, S, D] or [B*S, H, D]
    k: torch.Tensor,          # same shape
    cos: torch.Tensor,        # [S, D/2] or [B*S, D/2]
    sin: torch.Tensor,        # [S, D/2] or [B*S, D/2]
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to Q and K using fused Triton kernel."""
    if q.dim() == 4:
        B, H, S, D = q.shape
        total_tokens = B * S
        q_flat = q.reshape(total_tokens, H, D)
        k_flat = k.reshape(total_tokens, H, D)
    elif q.dim() == 3:
        total_tokens, H, D = q.shape
        q_flat = q
        k_flat = k
    else:
        raise ValueError(f"Expected 3D or 4D input, got {q.dim()}D")
    
    HALF_D = D // 2
    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)
    
    cos_flat = cos.reshape(-1, HALF_D) if cos.dim() == 3 else cos
    sin_flat = sin.reshape(-1, HALF_D) if sin.dim() == 3 else sin
    
    BLOCK_D = triton.next_power_of_2(HALF_D)
    
    with TIMER.record("fused_rope"):
        grid = (total_tokens, H)
        _fused_rope_kernel[grid](
            q_flat, k_flat,
            cos_flat, sin_flat,
            q_out, k_out,
            q_flat.stride(0), q_flat.stride(1),
            k_flat.stride(0), k_flat.stride(1),
            cos_flat.stride(0),
            total_tokens, H,
            D=D, HALF_D=HALF_D, BLOCK_D=BLOCK_D,
        )
    
    if q.dim() == 4:
        q_out = q_out.reshape(B, H, S, D)
        k_out = k_out.reshape(B, H, S, D)
    
    return q_out, k_out


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def patch_rope():
    """
    Patch Qwen2's apply_rotary_pos_emb to use our fused version.
    """
    try:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
        
        # Save original
        _original_apply_rotary = qwen2_mod.apply_rotary_pos_emb
        
        def patched_apply_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            # Our fused version doesn't need unsqueeze_dim
            # But we need to match the interface
            return fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        qwen2_mod.apply_rotary_pos_emb = patched_apply_rotary
        
    except (ImportError, AttributeError):
        pass
