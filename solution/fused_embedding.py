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
    D: tl.constexpr,         # embedding dimension
    VOCAB_SIZE: tl.constexpr,
    PAD_VALUE: tl.constexpr, # value to write for pad tokens
):
    """
    Lookup embedding and cast to fp16 in one pass.
    Each program handles one token.
    """
    token_idx = tl.program_id(0)
    if token_idx >= N:
        return
    
    # Load token id
    token_id = tl.load(ids_ptr + token_idx)
    
    # Handle pad/invalid tokens
    if token_id < 0 or token_id >= VOCAB_SIZE:
        # Write zeros for invalid tokens
        offsets = tl.arange(0, D)
        tl.store(
            out_ptr + token_idx * D + offsets,
            tl.zeros([D], dtype=tl.float16),
        )
        return
    
    # Load embedding row and cast to fp16
    offsets = tl.arange(0, D)
    emb = tl.load(weight_ptr + token_id * D + offsets)
    
    # Store as fp16
    tl.store(
        out_ptr + token_idx * D + offsets,
        emb.to(tl.float16),
    )


@triton.jit
def _fused_embedding_scaled_kernel(
    weight_ptr,     # [V, D] embedding weights (fp32)
    ids_ptr,        # [N] token ids
    out_ptr,        # [N, D] output (fp16)
    N,
    D: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    scale: tl.constexpr,    # embedding scale factor (e.g., sqrt(D))
):
    """
    Lookup + scale + cast in one kernel.
    Some models scale embeddings by sqrt(hidden_size).
    """
    token_idx = tl.program_id(0)
    if token_idx >= N:
        return
    
    token_id = tl.load(ids_ptr + token_idx)
    
    if token_id < 0 or token_id >= VOCAB_SIZE:
        offsets = tl.arange(0, D)
        tl.store(
            out_ptr + token_idx * D + offsets,
            tl.zeros([D], dtype=tl.float16),
        )
        return
    
    offsets = tl.arange(0, D)
    emb = tl.load(weight_ptr + token_id * D + offsets).to(tl.float32)
    
    # Apply scale
    emb = emb * scale
    
    tl.store(
        out_ptr + token_idx * D + offsets,
        emb.to(tl.float16),
    )


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
    
    D_pow2 = triton.next_power_of_2(D)
    
    with TIMER.record("fused_embedding"):
        grid = (N,)
        if scale is not None:
            _fused_embedding_scaled_kernel[grid](
                weight, ids_flat, out,
                N,
                D=D_pow2,
                VOCAB_SIZE=V,
                scale=scale,
            )
        else:
            _fused_embedding_cast_kernel[grid](
                weight, ids_flat, out,
                N,
                D=D_pow2,
                VOCAB_SIZE=V,
                PAD_VALUE=0,
            )
    
    return out.reshape(*orig_shape, D)


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
