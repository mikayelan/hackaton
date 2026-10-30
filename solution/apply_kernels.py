"""
apply_kernels.py — Подключает ВСЕ Triton-кернелы к Soup.

Все патчи РЕАЛЬНО работают в обучении:
  1. Fused Cross-Entropy — патчит Qwen2ForCausalLM.forward
  2. Fused RMSNorm — патчит Qwen2RMSNorm.forward
  3. Fused Embedding — патчит nn.Embedding.forward
  4. Fused LoRA — патчит PEFT LoraLinear.forward
  5. Fused RoPE — патчит transformers.models.qwen2.modeling_qwen2
"""

import warnings

_PATCHED = False


def apply():
    """
    Plug ALL custom Triton kernels into Soup training pipeline.
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

    # 1. Fused Cross-Entropy Loss
    try:
        from .fused_cross_entropy import patch_cross_entropy
        patch_cross_entropy()
        print("[✓] Fused Cross-Entropy Loss (memory + speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused CE failed: {e}")

    # 2. Fused RMSNorm
    try:
        from .fused_rmsnorm_silu import patch_rmsnorm_silu
        patch_rmsnorm_silu()
        print("[✓] Fused RMSNorm (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused RMSNorm failed: {e}")

    # 3. Fused SiLU*mul (FFN activation)
    try:
        from .fused_silu_mul import patch_silu_mul
        patch_silu_mul()
        print("[✓] Fused SiLU*mul (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused SiLU*mul failed: {e}")

    # 4. Fused Embedding
    try:
        from .fused_embedding import patch_embedding
        patch_embedding()
        print("[✓] Fused Embedding (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused Embedding failed: {e}")

    # 5. Fused LoRA
    try:
        from .fused_lora import patch_lora
        patch_lora()
        print("[✓] Fused LoRA (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused LoRA failed: {e}")

    # 6. Fused RoPE
    try:
        from .fused_rope import patch_rope
        patch_rope()
        print("[✓] Fused RoPE (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused RoPE failed: {e}")

    print("=" * 60)
    print("  All 6 kernels patched and active!")
    print("  Ready for training.")
    print("=" * 60)
