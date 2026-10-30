"""
fused_rope.py — Fused Rotary Position Embedding.
ПАТЧИТ: transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb
"""
import torch
import triton
import triton.language as tl
from .utils import TIMER


@triton.jit
def _fused_rope_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    q_out_ptr, k_out_ptr,
    stride_qt, stride_qh,
    stride_kt, stride_kh,
    stride_cos_pos,
    TOTAL_TOKENS, H,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_t >= TOTAL_TOKENS:
        return

    pos = pid_t
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < HALF_D

    cos_v = tl.load(cos_ptr + pos * stride_cos_pos + d_off, mask=d_mask, other=0.0)
    sin_v = tl.load(sin_ptr + pos * stride_cos_pos + d_off, mask=d_mask, other=0.0)

    q_base = pid_t * stride_qt + pid_h * stride_qh
    q1 = tl.load(q_ptr + q_base + d_off, mask=d_mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + q_base + HALF_D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    tl.store(q_out_ptr + q_base + d_off, (q1 * cos_v - q2 * sin_v).to(tl.float16), mask=d_mask)
    tl.store(q_out_ptr + q_base + HALF_D + d_off, (q1 * sin_v + q2 * cos_v).to(tl.float16), mask=d_mask)

    k_base = pid_t * stride_kt + pid_h * stride_kh
    k1 = tl.load(k_ptr + k_base + d_off, mask=d_mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_ptr + k_base + HALF_D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    tl.store(k_out_ptr + k_base + d_off, (k1 * cos_v - k2 * sin_v).to(tl.float16), mask=d_mask)
    tl.store(k_out_ptr + k_base + HALF_D + d_off, (k1 * sin_v + k2 * cos_v).to(tl.float16), mask=d_mask)


def fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if q.dim() == 4:
        B, H, S, D = q.shape
        total = B * S
        q_flat = q.reshape(total, H, D)
        k_flat = k.reshape(total, H, D)
    else:
        total, H, D = q.shape
        q_flat = q
        k_flat = k

    HALF_D = D // 2
    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)

    cos_flat = cos.reshape(-1, HALF_D) if cos.dim() == 3 else cos
    sin_flat = sin.reshape(-1, HALF_D) if sin.dim() == 3 else sin
    BLOCK_D = triton.next_power_of_2(HALF_D)

    with TIMER.record("fused_rope"):
        _fused_rope_kernel[(total, H)](
            q_flat, k_flat, cos_flat, sin_flat,
            q_out, k_out,
            q_flat.stride(0), q_flat.stride(1),
            k_flat.stride(0), k_flat.stride(1),
            cos_flat.stride(0),
            total, H,
            D=D, HALF_D=HALF_D, BLOCK_D=BLOCK_D,
        )

    if q.dim() == 4:
        q_out = q_out.reshape(B, H, S, D)
        k_out = k_out.reshape(B, H, S, D)
    return q_out, k_out


def patch_rope():
    """РЕАЛЬНЫЙ ПАТЧ: qwen2_mod.apply_rotary_pos_emb"""
    try:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
        qwen2_mod.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    except (ImportError, AttributeError):
        pass
