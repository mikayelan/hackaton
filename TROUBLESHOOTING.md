# 🔧 Troubleshooting — что делать если что-то сломалось

## Быстрая диагностика

Если обучение упало или loss резко отличается от базы:

### Шаг 1: Проверить какие патчи применились

```python
from solution import apply
apply()
# Смотрим вывод — какие [✓] появились, какие [!]
```

### Шаг 2: Отключить подозрительный патч

В `solution/apply_kernels.py` закомментируйте подозрительный патч:

```python
# Закомментируйте строки:
# from .fused_lora import patch_lora
# patch_lora()
```

### Шаг 3: Отключить ВСЕ патчи кроме CE

Самый безопасный вариант — оставить только Fused CE (он даёт главный выигрыш по памяти):

```python
def apply():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    
    # Только Fused CE — самый безопасный и важный
    try:
        from .fused_cross_entropy import patch_cross_entropy
        patch_cross_entropy()
        print("[✓] Fused Cross-Entropy Loss")
    except Exception as e:
        print(f"[!] Fused CE failed: {e}")
```

---

## Частые проблемы

### Проблема: "CUDA out of memory" при компиляции кернелов

**Решение:** Triton кэширует скомпилированные кернелы. Первый запуск может занять ~2 ГБ.
```python
# Увеличить кэш Triton
import os
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
```

### Проблема: Loss сильно отличается от базы (>0.5%)

**Решение:** Отключайте патчи по одному, начиная с самого сложного:
1. Fused LoRA (самый сложный)
2. Fused Embedding
3. Fused RoPE
4. Fused SiLU*mul
5. Fused RMSNorm
6. Fused CE (оставить последним — самый важный)

### Проблема: "Triton compilation error"

**Решение:** Проверить версию Triton:
```python
import triton
print(triton.__version__)  # Должна быть 3.6.0
```

### Проблема: "AttributeError: 'LoraLinear' object has no attribute 'base_layer'"

**Решение:** Версия PEFT несовместима. Отключить Fused LoRA:
```python
# В apply_kernels.py закомментировать:
# from .fused_lora import patch_lora
# patch_lora()
```

### Проблема: Тесты Soup не проходят

**Решение:** Наши патчи не должны ломать тесты. Если ломают — отключить все патчи кроме CE.

---

## Безопасный минимум

Если ничего не работает, оставьте ТОЛЬКО Fused CE:

```python
def apply():
    from .fused_cross_entropy import patch_cross_entropy
    patch_cross_entropy()
    print("[✓] Fused Cross-Entropy only")
```

Это уже даст ~2-3× экономию памяти.

---

## Как проверить что кернелы работают

```python
from solution.utils import TIMER

# После обучения:
print(TIMER.summary())
# Должны быть записи типа:
# fused_ce_forward     123.45 ms  (64 calls)
# fused_rmsnorm         45.67 ms  (64 calls)
# ...
```

Если записей нет — кернелы не вызываются. Проверить что патчи применились.
