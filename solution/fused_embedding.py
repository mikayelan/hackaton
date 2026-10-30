"""
fused_embedding.py — Fused Embedding lookup + FP32→FP16 cast.

Проблема: Soup хранит embedding weights в fp32 (для точности), но перед
каждым transformer-слоем кастует в fp16. Это лишний проход по памяти:
  1. Embedding lookup: ids -> fp32 tensor [B, S, D]
  2. Cast: fp32 -> fp16 tensor [B, S, D]

Решение: один кернел делает lookup и сразу пишет fp16.
  - Убирает промежуточный fp32 тензор
  - Экономит ~2× трафик памяти на embedding
  - Для D=1536, B*S=2048: экономим ~6 МБ на каждый проход
"""

import torch
import triton
import triton.language as tl
from typing import Optional

from .utils import TIMER


# ---------------------------------------------------------------------------
# Triton kernel: fused embedding + cast
# ---------------------------------------------------------------------------

@triton.jit
def _fused_embedding_cast_kernel(
    weight_ptr,     # [V, D] embedding weights (fp32)
    ids_ptr,        # [N] token ids (int64 or int32)
    out_ptr,        # [N, D] output (fp16)
    N,              # total tokens
    D: tl.constexpr,         # original embedding dimension
    BLOCK_D: tl.constexpr,   # block size (power of 2)
    VOCAB_SIZE: tl.constexpr,
):
    """Lookup embedding and cast to fp16 in one pass."""
    token_idx = tl.program_id(0)
    if token_idx >= N:
        return
    
    token_id = tl.load(ids_ptr + token_idx)
    
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    
    if token_id < 0 or token_id >= VOCAB_SIZE:
        tl.store(
            out_ptr + token_idx * D + offsets,
            tl.zeros([BLOCK_D], dtype=tl.float16),
            mask=mask,
        )
        return
    
    emb = tl.load(weight_ptr + token_id * D + offsets, mask=mask, other=0.0)
    tl.store(
        out_ptr + token_idx * D + offsets,
        emb.to(tl.float16),
        mask=mask,
    )


# scaled version removed — use fused_embedding_lookup with post-multiply


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_embedding_lookup(
    weight: torch.Tensor,    # [V, D] fp32
    ids: torch.Tensor,       # [N] int64
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Fused embedding lookup + cast to fp16.
    
    Args:
        weight: embedding matrix [V, D] in fp32
        ids: token ids [N] (any shape, will be flattened)
        scale: optional scale factor (e.g., sqrt(D))
    
    Returns:
        embeddings [N, D] in fp16
    """
    orig_shape = ids.shape
    ids_flat = ids.reshape(-1)
    N = ids_flat.shape[0]
    V, D = weight.shape
    
    out = torch.empty((N, D), device=weight.device, dtype=torch.float16)
    BLOCK_D = triton.next_power_of_2(D)
    
    with TIMER.record("fused_embedding"):
        grid = (N,)
        _fused_embedding_cast_kernel[grid](
            weight, ids_flat, out,
            N,
            D=D,
            BLOCK_D=BLOCK_D,
            VOCAB_SIZE=V,
        )
    
    result = out.reshape(*orig_shape, D)
    
    # Apply scale if needed (outside kernel for simplicity)
    if scale is not None:
        result = result * scale
    
    return result


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def patch_embedding():
    """
    Patch Soup/transformers to use fused embedding.
    
    Strategy: replace the nn.Embedding forward with our fused version
    when the embedding weight is in fp32 and output is needed in fp16.
    """
    try:
        # Patch nn.Embedding to detect fp32->fp16 pattern
        _original_embedding_forward = torch.nn.Embedding.forward
        
        def patched_embedding_forward(self, input: torch.Tensor) -> torch.Tensor:
            # Only use fused version if weight is fp32
            if self.weight.dtype == torch.float32 and torch.is_autocast_enabled():
                return fused_embedding_lookup(self.weight, input)
            return _original_embedding_forward(self, input)
        
        # Don't patch globally — too risky
        # torch.nn.Embedding.forward = patched_embedding_forward
        
    except (ImportError, AttributeError):
        pass
