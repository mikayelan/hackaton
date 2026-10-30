"""
apply_kernels.py — Подключает все Triton-кернелы к Soup.

Функция apply() вызывается один раз перед обучением и патчит Soup/transformers/peft,
подменяя стандартные операции на наши fused-версии.
"""

import importlib
import sys
import warnings

_PATCHED = False


def apply():
    """
    Plug all custom Triton kernels into Soup training pipeline.
    
    Must be called once before `soup train`. Idempotent — calling twice is safe.
    
    Kernels activated:
      1. Fused Cross-Entropy  — saves ~8-10 GB memory (no [B*S*152K] logits)
      2. Fused RMSNorm + SiLU — eliminates intermediate buffers in FFN
      3. Fused LoRA forward   — fuses base linear + LoRA delta in one kernel
      4. Fused Embedding+Cast — emits fp16 directly, skips fp32 roundtrip
      5. Fused Rotary (RoPE)  — applies cos/sin to Q and K in one pass
    """
    global _PATCHED
    if _PATCHED:
        print("[Soup Kernels] Already applied, skipping.")
        return
    _PATCHED = True

    print("=" * 60)
    print("  Soup Kernel Challenge — custom Triton kernels")
    print("=" * 60)

    # --- 1. Fused Cross-Entropy Loss ---
    try:
        from .fused_cross_entropy import patch_cross_entropy
        patch_cross_entropy()
        print("[✓] Fused Cross-Entropy Loss (memory + speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused CE failed: {e}")

    # --- 2. Fused RMSNorm + SiLU ---
    try:
        from .fused_rmsnorm_silu import patch_rmsnorm_silu
        patch_rmsnorm_silu()
        print("[✓] Fused RMSNorm + SiLU (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused RMSNorm+SiLU failed: {e}")

    # --- 3. Fused LoRA ---
    try:
        from .fused_lora import patch_lora
        patch_lora()
        print("[✓] Fused LoRA forward (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused LoRA failed: {e}")

    # --- 4. Fused Embedding + Cast ---
    try:
        from .fused_embedding import patch_embedding
        patch_embedding()
        print("[✓] Fused Embedding + Cast (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused Embedding failed: {e}")

    # --- 5. Fused Rotary Position Embedding ---
    try:
        from .fused_rope import patch_rope
        patch_rope()
        print("[✓] Fused Rotary Position Embedding (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused RoPE failed: {e}")

    print("=" * 60)
    print("  All kernels applied. Ready for training.")
    print("=" * 60)
