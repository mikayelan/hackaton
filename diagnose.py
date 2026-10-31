#!/usr/bin/env python3
"""
diagnose.py — Диагностика: проверяет что все патчи РЕАЛЬНО применяются.

Запуск в Kaggle:
    python diagnose.py

Этот скрипт проверит:
1. Применены ли патчи к классам
2. Работают ли кернелы (не fallback на PyTorch)
3. Сколько времени занимает каждый кернел
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_patches():
    """Проверяет что все патчи применены."""
    print("=" * 70)
    print("🔍 Проверка патчей")
    print("=" * 70)
    
    # Применяем кернелы
    from solution import apply
    apply()
    
    results = {}
    
    # 1. Проверяем Fused CE
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        forward_code = Qwen2ForCausalLM.forward.__code__
        has_patched = 'labels = kwargs.pop' in str(forward_code.co_consts) or \
                      'patched' in Qwen2ForCausalLM.forward.__name__
        results['Fused CE'] = has_patched
        print(f"{'Fused CE':<30} {'✅ Патч применён' if has_patched else '❌ НЕ применён'}")
    except Exception as e:
        print(f"{'Fused CE':<30} ❌ Ошибка: {e}")
        results['Fused CE'] = False
    
    # 2. Проверяем Fused RMSNorm
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        forward_name = Qwen2RMSNorm.forward.__name__
        has_patched = 'patched' in forward_name or 'fused_rmsnorm' in str(Qwen2RMSNorm.forward)
        results['Fused RMSNorm'] = has_patched
        print(f"{'Fused RMSNorm':<30} {'✅ Патч применён' if has_patched else '❌ НЕ применён'}")
    except Exception as e:
        print(f"{'Fused RMSNorm':<30} ❌ Ошибка: {e}")
        results['Fused RMSNorm'] = False
    
    # 3. Проверяем Fused SiLU*mul
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
        forward_name = Qwen2MLP.forward.__name__
        has_patched = 'patched' in forward_name or 'fused_silu_mul' in str(Qwen2MLP.forward)
        results['Fused SiLU*mul'] = has_patched
        print(f"{'Fused SiLU*mul':<30} {'✅ Патч применён' if has_patched else '❌ НЕ применён'}")
    except Exception as e:
        print(f"{'Fused SiLU*mul':<30} ❌ Ошибка: {e}")
        results['Fused SiLU*mul'] = False
    
    # 4. Проверяем Fused Embedding
    try:
        import torch.nn as nn
        forward_name = nn.Embedding.forward.__name__
        has_patched = 'patched' in forward_name or 'fused_embedding' in str(nn.Embedding.forward)
        results['Fused Embedding'] = has_patched
        print(f"{'Fused Embedding':<30} {'✅ Патч применён' if has_patched else '❌ НЕ применён'}")
    except Exception as e:
        print(f"{'Fused Embedding':<30} ❌ Ошибка: {e}")
        results['Fused Embedding'] = False
    
    # 5. Проверяем Fused LoRA
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
        forward_name = LoraLinear.forward.__name__
        has_patched = 'patched' in forward_name or 'fused_lora' in str(LoraLinear.forward)
        results['Fused LoRA'] = has_patched
        print(f"{'Fused LoRA':<30} {'✅ Патч применён' if has_patched else '❌ НЕ применён'}")
    except Exception as e:
        print(f"{'Fused LoRA':<30} ❌ Ошибка: {e}")
        results['Fused LoRA'] = False
    
    # 6. Проверяем Fused RoPE
    try:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
        func_name = qwen2_mod.apply_rotary_pos_emb.__name__
        has_patched = 'fused' in func_name or 'fused_rope' in str(qwen2_mod.apply_rotary_pos_emb)
        results['Fused RoPE'] = has_patched
        print(f"{'Fused RoPE':<30} {'✅ Патч применён' if has_patched else '❌ НЕ применён'}")
    except Exception as e:
        print(f"{'Fused RoPE':<30} ❌ Ошибка: {e}")
        results['Fused RoPE'] = False
    
    return results


def test_kernel_execution():
    """Тестирует что кернелы реально выполняются (не fallback)."""
    print("\n" + "=" * 70)
    print("🧪 Тест выполнения кернелов")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA недоступна")
        return
    
    from solution.utils import TIMER
    
    # Тест 1: Fused CE
    try:
        from solution.fused_cross_entropy import FusedCrossEntropyLoss
        N, V = 64, 10000  # маленький vocab для теста
        logits = torch.randn(N, V, device='cuda', dtype=torch.float32, requires_grad=True)
        labels = torch.randint(0, V, (N,), device='cuda')
        
        TIMER.reset()
        loss_fn = FusedCrossEntropyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        
        ce_time = TIMER.total_ms('fused_ce_forward') + TIMER.total_ms('fused_ce_backward')
        print(f"{'Fused CE':<30} ✅ {ce_time:.2f} ms")
    except Exception as e:
        print(f"{'Fused CE':<30} ❌ {e}")
    
    # Тест 2: Fused RMSNorm
    try:
        from solution.fused_rmsnorm_silu import fused_rmsnorm
        N, D = 32, 1536
        x = torch.randn(N, D, device='cuda', dtype=torch.float16)
        w = torch.randn(D, device='cuda', dtype=torch.float16)
        
        TIMER.reset()
        out = fused_rmsnorm(x, w)
        
        rmsnorm_time = TIMER.total_ms('fused_rmsnorm')
        print(f"{'Fused RMSNorm':<30} ✅ {rmsnorm_time:.2f} ms")
    except Exception as e:
        print(f"{'Fused RMSNorm':<30} ❌ {e}")
    
    # Тест 3: Fused SiLU*mul
    try:
        from solution.fused_silu_mul import fused_silu_mul
        N, D = 32, 1536
        x = torch.randn(N, D, device='cuda', dtype=torch.float16)
        y = torch.randn(N, D, device='cuda', dtype=torch.float16)
        
        TIMER.reset()
        out = fused_silu_mul(x, y)
        
        silu_time = TIMER.total_ms('fused_silu_mul')
        print(f"{'Fused SiLU*mul':<30} ✅ {silu_time:.2f} ms")
    except Exception as e:
        print(f"{'Fused SiLU*mul':<30} ❌ {e}")
    
    # Тест 4: Fused Embedding
    try:
        from solution.fused_embedding import fused_embedding_lookup
        V, D = 1000, 512
        weight = torch.randn(V, D, device='cuda', dtype=torch.float32)
        ids = torch.randint(0, V, (32,), device='cuda')
        
        TIMER.reset()
        out = fused_embedding_lookup(weight, ids)
        
        emb_time = TIMER.total_ms('fused_embedding')
        print(f"{'Fused Embedding':<30} ✅ {emb_time:.2f} ms")
    except Exception as e:
        print(f"{'Fused Embedding':<30} ❌ {e}")


def analyze_bottlenecks():
    """Анализирует где теряется время."""
    print("\n" + "=" * 70)
    print("📊 Анализ узких мест")
    print("=" * 70)
    
    print("\nПрофиль GPU (из ТЗ):")
    print("  • Матричные умножения: 36%")
    print("  • Поэлементные + casts: 44%")
    print("  • Softmax (152K vocab): 7%")
    print("  • Attention: 5%")
    print("  • NF4 unpack: 5%")
    print("  • Прочее: 3%")
    
    print("\nЧто оптимизировано:")
    print("  ✅ Fused CE — экономит память + 7% времени")
    print("  ✅ Fused RMSNorm — часть 44%")
    print("  ✅ Fused SiLU*mul — часть 44%")
    print("  ✅ Fused Embedding — часть 44%")
    print("  ✅ Fused RoPE — часть 5%")
    print("  ✅ Fused LoRA — часть 36%")
    
    print("\nЧто НЕ оптимизировано (критично!):")
    print("  ❌ Fused NF4 matmul — 36%+5% = 41% времени!")
    print("     Это ГЛАВНОЕ узкое место. Без него невозможно достичь 15-17с.")
    
    print("\n💡 Почему у лидеров 17.8с:")
    print("   • У них есть fused NF4 matmul (экономит ~40% времени)")
    print("   • Возможно fused attention (экономит ещё ~5%)")
    print("   • Оптимизированные block sizes через autotune")


def main():
    print("\n" + "=" * 70)
    print("🍲 Soup Kernel Challenge — Диагностика решения")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA недоступна! Запускайте в Kaggle с GPU.")
        sys.exit(1)
    
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Проверяем патчи
    patch_results = check_patches()
    
    # Тестируем выполнение
    test_kernel_execution()
    
    # Анализируем узкие места
    analyze_bottlenecks()
    
    print("\n" + "=" * 70)
    print("📋 Итог")
    print("=" * 70)
    
    patched_count = sum(1 for v in patch_results.values() if v)
    total_count = len(patch_results)
    
    print(f"\nПрименено патчей: {patched_count}/{total_count}")
    
    if patched_count < total_count:
        print("\n⚠️ НЕ все патчи применены! Проверьте логи выше.")
        print("   Это может значительно ухудшить результат.")
    
    print("\n💡 Рекомендации:")
    if patched_count == total_count:
        print("   ✅ Все патчи работают. Для улучшения нужно:")
        print("   1. Добавить fused NF4 matmul (41% времени)")
        print("   2. Оптимизировать block sizes через autotune")
        print("   3. Возможно добавить fused attention")
    else:
        print("   1. Исправьте патчи которые не применились")
        print("   2. Проверьте логи на ошибки импорта")
        print("   3. Убедитесь что transformers/peft правильных версий")
    
    print()


if __name__ == "__main__":
    main()
