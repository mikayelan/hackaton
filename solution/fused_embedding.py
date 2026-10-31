"""
fused_embedding.py — Embedding lookup + cast to fp16 in one kernel.

ПАТЧИТ: nn.Embedding.forward — когда weight в fp32 и нужен fp16 output.
Это работает — Soup хранит embedding в fp32, а перед слоем кастует в fp16.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl
from .utils import TIMER


@triton.jit
def _fused_embedding_kernel(
    weight_ptr, ids_ptr, out_ptr,
    N,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
):
    """Embedding lookup + cast to fp16."""
    token_idx = tl.program_id(0)
    if token_idx >= N:
        return

    token_id = tl.load(ids_ptr + token_idx)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    if token_id < 0 or token_id >= VOCAB_SIZE:
        tl.store(out_ptr + token_idx * D + offsets, tl.zeros([BLOCK_D], dtype=tl.float16), mask=mask)
        return

    emb = tl.load(weight_ptr + token_id * D + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + token_idx * D + offsets, emb.to(tl.float16), mask=mask)


def fused_embedding_lookup(weight, ids):
    """Fused embedding lookup + cast to fp16."""
    orig_shape = ids.shape
    ids_flat = ids.reshape(-1)
    N = ids_flat.shape[0]
    V, D = weight.shape
    out = torch.empty((N, D), device=weight.device, dtype=torch.float16)
    BLOCK_D = triton.next_power_of_2(D)

    with TIMER.record("fused_embedding"):
        _fused_embedding_kernel[(N,)](
            weight, ids_flat, out,
            N, D=D, BLOCK_D=BLOCK_D, VOCAB_SIZE=V,
        )
    return out.reshape(*orig_shape, D)


def patch_embedding():
    """
    РЕАЛЬНЫЙ ПАТЧ: подменяет nn.Embedding.forward.
    Когда weight в fp32 и включён autocast — используем fused kernel.
    """
    _orig_forward = nn.Embedding.forward

    def patched_forward(self, input: torch.Tensor) -> torch.Tensor:
        # Используем fused только если weight в fp32 и нужен fp16
        if (self.weight.dtype == torch.float32 and
            torch.is_autocast_enabled() and
            torch.get_autocast_gpu_dtype() == torch.float16):
            return fused_embedding_lookup(self.weight, input)
        return _orig_forward(self, input)

    nn.Embedding.forward = patched_forward
