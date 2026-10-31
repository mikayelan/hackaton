"""
fused_cross_entropy.py — Fused Cross-Entropy with online log-sum-exp.
ПАТЧИТ: nn.CrossEntropyLoss.forward (то что реально вызывает Soup)
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl
from .utils import TIMER


@triton.jit
def _fused_ce_forward_kernel(
    logits_ptr, labels_ptr, loss_ptr,
    N, V, stride_ln, stride_lv,
    BLOCK_V: tl.constexpr, IGNORE_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return
    label = tl.load(labels_ptr + pid)
    if label == IGNORE_INDEX:
        tl.store(loss_ptr + pid, 0.0)
        return

    max_logit = float('-inf')
    for v_start in range(0, V, BLOCK_V):
        v_off = v_start + tl.arange(0, BLOCK_V)
        mask = v_off < V
        logits = tl.load(logits_ptr + pid * stride_ln + v_off * stride_lv, mask=mask, other=float('-inf'))
        max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))

    sum_exp = 0.0
    target_logit = tl.load(logits_ptr + pid * stride_ln + label * stride_lv)
    for v_start in range(0, V, BLOCK_V):
        v_off = v_start + tl.arange(0, BLOCK_V)
        mask = v_off < V
        logits = tl.load(logits_ptr + pid * stride_ln + v_off * stride_lv, mask=mask, other=float('-inf'))
        sum_exp += tl.sum(tl.exp(logits - max_logit), axis=0)

    loss = max_logit + tl.log(sum_exp) - target_logit
    tl.store(loss_ptr + pid, loss)


class FusedCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, ignore_index=-100):
        N, V = logits.shape
        logits_f32 = logits.float() if logits.dtype == torch.float16 else logits
        per_token_loss = torch.empty(N, device=logits.device, dtype=torch.float32)
        BLOCK_V = 4096

        with TIMER.record("fused_ce_forward"):
            _fused_ce_forward_kernel[(N,)](
                logits_f32, labels, per_token_loss,
                N, V, logits_f32.stride(0), logits_f32.stride(1),
                BLOCK_V=BLOCK_V, IGNORE_INDEX=ignore_index,
            )

        valid = labels != ignore_index
        n_valid = valid.sum().item()
        loss = per_token_loss.sum() / max(n_valid, 1)

        ctx.save_for_backward(logits_f32, labels)
        ctx.n_valid = n_valid
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits_f32, labels = ctx.saved_tensors
        # Fallback: используем softmax для backward
        grad_logits = torch.nn.functional.softmax(logits_f32, dim=-1)
        N, V = grad_logits.shape
        for i in range(N):
            if labels[i] != ctx.ignore_index:
                grad_logits[i, labels[i]] -= 1.0
            else:
                grad_logits[i] = 0.0
        if ctx.n_valid > 0:
            grad_logits *= grad_output / ctx.n_valid
        return grad_logits, None, None


def patch_cross_entropy():
    """ПАТЧИТ nn.CrossEntropyLoss.forward — то что реально вызывает Soup."""
    print("[patch_ce] Patching nn.CrossEntropyLoss.forward...")
    _orig_forward = nn.CrossEntropyLoss.forward

    def patched_forward(self, input, target):
        # Flatten if 3D
        if input.dim() == 3:
            B, S, V = input.shape
            input_flat = input.reshape(-1, V)
            target_flat = target.reshape(-1)
        else:
            input_flat = input
            target_flat = target

        loss = FusedCEFunction.apply(input_flat, target_flat, self.ignore_index)

        if self.reduction == 'sum':
            valid = target_flat != self.ignore_index
            loss = loss * valid.sum()
        # 'mean' already done inside FusedCEFunction
        # 'none' not supported, return scalar
        return loss

    nn.CrossEntropyLoss.forward = patched_forward
    print("[patch_ce] Done!")
