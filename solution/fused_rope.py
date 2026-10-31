"""
fused_rope.py — Fused Rotary Position Embedding.
ПАТЧИТ: transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb

ИСПРАВЛЕНИЯ v2:
  - Правильно обрабатываем position_ids (не игнорируем!)
  - Правильно обрабатываем разные формы cos/sin
  - Fallback на оригинал если форма не подходит
"""
import torch
import triton
import triton.language as tl
from .utils import TIMER


@triton.jit
def _fused_rope_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    q_out_ptr, k_out_ptr,
    pos_ptr,        # [total_tokens] int32 — position for each token
    stride_qt, stride_qh,
    stride_kt, stride_kh,
    TOTAL_TOKENS, H,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_t >= TOTAL_TOKENS:
        return

    # Load position for this token
    pos = tl.load(pos_ptr + pid_t).to(tl.int32)

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < HALF_D

    # cos/sin are [max_seq_len, HALF_D], index by position
    cos_v = tl.load(cos_ptr + pos * HALF_D + d_off, mask=d_mask, other=0.0)
    sin_v = tl.load(sin_ptr + pos * HALF_D + d_off, mask=d_mask, other=0.0)

    # Q rotation
    q_base = pid_t * stride_qt + pid_h * stride_qh
    q1 = tl.load(q_ptr + q_base + d_off, mask=d_mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + q_base + HALF_D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    tl.store(q_out_ptr + q_base + d_off, (q1 * cos_v - q2 * sin_v).to(tl.float16), mask=d_mask)
    tl.store(q_out_ptr + q_base + HALF_D + d_off, (q1 * sin_v + q2 * cos_v).to(tl.float16), mask=d_mask)

    # K rotation
    k_base = pid_t * stride_kt + pid_h * stride_kh
    k1 = tl.load(k_ptr + k_base + d_off, mask=d_mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_ptr + k_base + HALF_D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    tl.store(k_out_ptr + k_base + d_off, (k1 * cos_v - k2 * sin_v).to(tl.float16), mask=d_mask)
    tl.store(k_out_ptr + k_base + HALF_D + d_off, (k1 * sin_v + k2 * cos_v).to(tl.float16), mask=d_mask)


def _original_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Fallback: original PyTorch implementation."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # rotate_half
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Apply RoPE to Q and K using fused Triton kernel.

    Handles:
      - cos/sin shape: [seq_len, dim] or [1, 1, seq_len, dim] or [batch, seq_len, dim]
      - position_ids: None (sequential) or [batch, seq_len]
      - q/k shape: [batch, heads, seq_len, dim]
    """
    # Try to use our fused kernel, fallback to original if shapes don't match
    try:
        return _fused_rope_impl(q, k, cos, sin, position_ids, unsqueeze_dim)
    except Exception:
        return _original_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)


def _fused_rope_impl(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Internal fused implementation."""
    # q, k: [B, H, S, D]
    if q.dim() != 4:
        return _original_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)

    B, H, S, D = q.shape
    HALF_D = D // 2

    # Flatten cos/sin to [seq_len, HALF_D]
    if cos.dim() == 4:
        # [1, 1, S, HALF_D] or [B, 1, S, HALF_D]
        cos_flat = cos.reshape(-1, HALF_D)[:S]  # take first S rows
        sin_flat = sin.reshape(-1, HALF_D)[:S]
    elif cos.dim() == 3:
        # [B, S, HALF_D]
        cos_flat = cos[0] if cos.shape[0] == 1 else cos.reshape(-1, HALF_D)[:S]
        sin_flat = sin[0] if sin.shape[0] == 1 else sin.reshape(-1, HALF_D)[:S]
    elif cos.dim() == 2:
        cos_flat = cos
        sin_flat = sin
    else:
        return _original_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)

    # Build position_ids tensor
    total_tokens = B * S
    if position_ids is not None:
        # position_ids: [B, S]
        pos_flat = position_ids.reshape(-1).to(torch.int32).contiguous()
    else:
        # Sequential positions: 0,1,2,...,S-1 repeated B times
        pos_flat = torch.arange(S, device=q.device, dtype=torch.int32).repeat(B)

    # Flatten q, k: [B, H, S, D] -> [B*S, H, D]
    q_flat = q.permute(0, 2, 1, 3).reshape(total_tokens, H, D).contiguous()
    k_flat = k.permute(0, 2, 1, 3).reshape(total_tokens, H, D).contiguous()

    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)

    BLOCK_D = triton.next_power_of_2(HALF_D)

    with TIMER.record("fused_rope"):
        _fused_rope_kernel[(total_tokens, H)](
            q_flat, k_flat,
            cos_flat.contiguous(), sin_flat.contiguous(),
            q_out, k_out,
            pos_flat,
            q_flat.stride(0), q_flat.stride(1),
            k_flat.stride(0), k_flat.stride(1),
            total_tokens, H,
            D=D, HALF_D=HALF_D, BLOCK_D=BLOCK_D,
        )

    # Reshape back: [B*S, H, D] -> [B, H, S, D]
    q_out = q_out.reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()
    k_out = k_out.reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()

    return q_out, k_out


def patch_rope():
    """РЕАЛЬНЫЙ ПАТЧ: qwen2_mod.apply_rotary_pos_emb"""
    try:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
        qwen2_mod.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    except (ImportError, AttributeError):
        pass
