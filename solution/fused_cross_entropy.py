"""
fused_cross_entropy.py — Fused Cross-Entropy with online log-sum-exp.

ИСПРАВЛЕНИЕ v3:
  - Подменяем torch.nn.functional.cross_entropy
  - Оригинальный forward работает как обычно (сохраняет autograd)
  - Внутри используется наш fused кернел
  - НЕ ломаем autograd graph!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from .utils import TIMER


@triton.jit
def _fused_ce_forward_kernel(
    logits_ptr, labels_ptr, loss_ptr,
    N, V,
    stride_ln, stride_lv,
    BLOCK_V: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    label = tl.load(labels_ptr + pid)
    if label == IGNORE_INDEX:
        tl.store(loss_ptr + pid, 0.0)
        return

    # Pass 1: find max logit (numerical stability)
    max_logit = float('-inf')
    for v_start in range(0, V, BLOCK_V):
        v_off = v_start + tl.arange(0, BLOCK_V)
        mask = v_off < V
        logits = tl.load(logits_ptr + pid * stride_ln + v_off * stride_lv, mask=mask, other=float('-inf'))
        max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))

    # Pass 2: sum(exp) + target logit
    sum_exp = 0.0
    target_logit = tl.load(logits_ptr + pid * stride_ln + label * stride_lv)

    for v_start in range(0, V, BLOCK_V):
        v_off = v_start + tl.arange(0, BLOCK_V)
        mask = v_off < V
        logits = tl.load(logits_ptr + pid * stride_ln + v_off * stride_lv, mask=mask, other=float('-inf'))
        sum_exp += tl.sum(tl.exp(logits - max_logit), axis=0)

    loss = max_logit + tl.log(sum_exp) - target_logit
    tl.store(loss_ptr + pid, loss)


@triton.jit
def _fused_ce_backward_kernel(
    logits_ptr, labels_ptr, grad_ptr,
    N, V,
    stride_ln, stride_lv,
    BLOCK_V: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    label = tl.load(labels_ptr + pid)
    if label == IGNORE_INDEX:
        for v_start in range(0, V, BLOCK_V):
            v_off = v_start + tl.arange(0, BLOCK_V)
            mask = v_off < V
            tl.store(grad_ptr + pid * stride_ln + v_off * stride_lv,
                     tl.zeros([BLOCK_V], dtype=tl.float32), mask=mask)
        return

    # Recompute max
    max_logit = float('-inf')
    for v_start in range(0, V, BLOCK_V):
        v_off = v_start + tl.arange(0, BLOCK_V)
        mask = v_off < V
        logits = tl.load(logits_ptr + pid * stride_ln + v_off * stride_lv, mask=mask, other=float('-inf'))
        max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))

    # Recompute sum_exp
    sum_exp = 0.0
    for v_start in range(0, V, BLOCK_V):
        v_off = v_start + tl.arange(0, BLOCK_V)
        mask = v_off < V
        logits = tl.load(logits_ptr + pid * stride_ln + v_off * stride_lv, mask=mask, other=float('-inf'))
        sum_exp += tl.sum(tl.exp(logits - max_logit), axis=0)

    # softmax - one_hot
    for v_start in range(0, V, BLOCK_V):
        v_off = v_start + tl.arange(0, BLOCK_V)
        mask = v_off < V
        logits = tl.load(logits_ptr + pid * stride_ln + v_off * stride_lv, mask=mask, other=float('-inf'))
        sm = tl.exp(logits - max_logit) / sum_exp
        is_label = (v_off == label).to(tl.float32)
        tl.store(grad_ptr + pid * stride_ln + v_off * stride_lv, sm - is_label, mask=mask)


class FusedCEFunction(torch.autograd.Function):
    """
    Fused cross-entropy that preserves autograd.
    """
    @staticmethod
    def forward(ctx, logits, labels, ignore_index=-100):
        N, V = logits.shape
        logits_f32 = logits.float() if logits.dtype == torch.float16 else logits
        per_token_loss = torch.empty(N, device=logits.device, dtype=torch.float32)
        BLOCK_V = 4096

        with TIMER.record("fused_ce_forward"):
            _fused_ce_forward_kernel[(N,)](
                logits_f32, labels, per_token_loss,
                N, V,
                logits_f32.stride(0), logits_f32.stride(1),
                BLOCK_V=BLOCK_V, IGNORE_INDEX=ignore_index,
            )

        valid = labels != ignore_index
        n_valid = valid.sum().item()
        loss = per_token_loss.sum() / max(n_valid, 1)

        ctx.save_for_backward(logits_f32, labels)
        ctx.ignore_index = ignore_index
        ctx.n_valid = n_valid
        ctx.BLOCK_V = BLOCK_V
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits_f32, labels = ctx.saved_tensors
        N, V = logits_f32.shape
        grad_logits = torch.empty_like(logits_f32)

        with TIMER.record("fused_ce_backward"):
            _fused_ce_backward_kernel[(N,)](
                logits_f32, labels, grad_logits,
                N, V,
                logits_f32.stride(0), logits_f32.stride(1),
                BLOCK_V=ctx.BLOCK_V, IGNORE_INDEX=ctx.ignore_index,
            )

        if ctx.n_valid > 0:
            grad_logits *= grad_output / ctx.n_valid
        return grad_logits, None, None


def fused_cross_entropy(logits, labels, ignore_index=-100):
    """
    Fused cross-entropy loss.
    Drop-in replacement for F.cross_entropy.
    """
    if logits.dim() == 3:
        B, S, V = logits.shape
        logits = logits.reshape(-1, V)
        labels = labels.reshape(-1)
    return FusedCEFunction.apply(logits, labels, ignore_index)


def patch_cross_entropy():
    """
    ИСПРАВЛЕНИЕ v3: Подменяем torch.nn.functional.cross_entropy.
    
    Оригинальный forward Qwen2 работает как обычно,
    но внутри использует наш fused кернел.
    Autograd graph сохраняется!
    """
    # Сохраняем оригинал
    _original_cross_entropy = F.cross_entropy
    
    def patched_cross_entropy(input, target, weight=None, ignore_index=-100, 
                              reduction='mean', label_smoothing=0.0):
        # Если есть weight или label_smoothing — используем оригинал
        if weight is not None or label_smoothing > 0:
            return _original_cross_entropy(input, target, weight=weight,
                                          ignore_index=ignore_index,
                                          reduction=reduction,
                                          label_smoothing=label_smoothing)
        
        # Используем наш fused кернел
        loss = fused_cross_entropy(input, target, ignore_index=ignore_index)
        
        # Reduction уже сделан (mean) в fused_cross_entropy
        # Если нужен sum — умножаем на количество валидных токенов
        if reduction == 'sum':
            valid = target != ignore_index
            loss = loss * valid.sum()
        elif reduction == 'none':
            # Для none нужно вернуть per-token loss
            # Но наш кернел уже сделал mean, так что fallback
            return _original_cross_entropy(input, target, weight=weight,
                                          ignore_index=ignore_index,
                                          reduction=reduction,
                                          label_smoothing=label_smoothing)
        
        return loss
    
    # Подменяем
    F.cross_entropy = patched_cross_entropy
    
    # Также подменяем в torch.nn.functional (на случай если импортировали напрямую)
    import torch.nn.functional
    torch.nn.functional.cross_entropy = patched_cross_entropy
