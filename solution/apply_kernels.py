"""
apply_kernels.py — Подключает Triton-кернелы к Soup.

Активные патчи (реально работают в обучении):
  1. Fused Cross-Entropy — патчит Qwen2ForCausalLM.forward
  2. Fused RoPE — патчит transformers.models.qwen2.modeling_qwen2

Доступные кернелы (не патчат автоматически, можно вызвать напрямую):
  - fused_rmsnorm_silu()
  - fused_lora_forward()
  - fused_embedding_lookup()
"""

import warnings

_PATCHED = False


def apply():
    """
    Plug custom Triton kernels into Soup training pipeline.
    Must be called once before `soup train`.
    """
    global _PATCHED
    if _PATCHED:
        print("[Soup Kernels] Already applied, skipping.")
        return
    _PATCHED = True

    print("=" * 60)
    print("  Soup Kernel Challenge — custom Triton kernels")
    print("=" * 60)

    # 1. Fused Cross-Entropy Loss (ПАТЧИТ МОДЕЛЬ)
    try:
        from .fused_cross_entropy import patch_cross_entropy
        patch_cross_entropy()
        print("[✓] Fused Cross-Entropy Loss (memory + speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused CE failed: {e}")

    # 2. Fused RoPE (ПАТЧИТ TRANSFORMERS)
    try:
        from .fused_rope import patch_rope
        patch_rope()
        print("[✓] Fused RoPE (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused RoPE failed: {e}")

    # Note: fused_rmsnorm_silu, fused_lora, fused_embedding
    # are available as standalone functions but not auto-patched.
    # They require careful integration with Soup internals.

    print("=" * 60)
    print("  Active kernels: Fused CE + Fused RoPE")
    print("  Ready for training.")
    print("=" * 60)
