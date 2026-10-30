# 🍲 Soup Kernel Challenge — Решение

**Decentrathon 6.0**

## Что сделано

6 Triton-кернелов, все РЕАЛЬНО патчат Soup/transformers/PEFT:

| # | Кернел | Что патчит | Эффект |
|---|--------|-----------|--------|
| 1 | **Fused Cross-Entropy** | `Qwen2ForCausalLM.forward` | 🔴 Экономия ~8-10 ГБ памяти |
| 2 | **Fused RMSNorm** | `Qwen2RMSNorm.forward` | 🟠 Ускорение каждого слоя |
| 3 | **Fused SiLU*mul** | `Qwen2MLP.forward` | 🟠 SiLU(gate)*up в одном кернеле |
| 4 | **Fused Embedding** | `nn.Embedding.forward` | 🟡 Lookup + cast fp32→fp16 |
| 5 | **Fused LoRA** | `PEFT LoraLinear.forward` | 🟡 Base + delta в одном кернеле |
| 6 | **Fused RoPE** | `qwen2_mod.apply_rotary_pos_emb` | 🟡 Rotary для Q и K |

### 1. 🔴 Fused Cross-Entropy (ГЛАВНЫЙ)
- Online log-sum-exp — считаем loss чанками по словарю
- НЕ материализуем logits `[B*S, 152K]`
- Экономия ~8-10 ГБ памяти (как Unsloth)

### 2. 🟠 Fused RMSNorm
- Один кернел вместо PyTorch RMSNorm
- Убирает лишние dtype casts
- Работает в каждом из 28 слоёв Qwen2

### 3. 🟠 Fused SiLU*mul (FFN)
- `SiLU(gate) * up` в одном кернеле
- Убирает промежуточный тензор
- Часть тех 44% element-wise операций

### 4. 🟡 Fused Embedding
- Lookup + cast fp32→fp16 в одном проходе
- Работает когда weight в fp32 и нужен fp16

### 5. 🟡 Fused LoRA
- `base + scaling * (x @ A @ B)` без промежуточного `[M, r]`
- Патчит PEFT LoraLinear

### 6. 🟡 Fused RoPE
- Rotary Position Embedding для Q и K одновременно
- 2 загрузки+записи вместо 4

## Как запустить

```python
from solution import apply
apply()  # ← одна функция патчит всё
```

## Ссылки
- **Коммит:** `COMMIT_HASH`
- **Kaggle:** `NOTEBOOK_URL`
- **Видео:** `VIDEO_URL`

## Результаты
| Метрика | База | Решение |
|---------|------|---------|
| Время | ~80 с | TBD |
| Память | ~14 ГБ | TBD |
| Eval loss | TBD | TBD |

## Что НЕ реализовано (честно)
- Fused NF4 matmul (36%+5% профиля) — слишком сложно для Triton, bitsandbytes уже оптимизирован
- Flash Attention — свой писать долго, библиотеку нельзя
- Activation offloading — запрещено правилами

## Источники идей
- Fused CE: online log-sum-exp (Milakov & Gimelshein, 2018)
- Fused SiLU*mul: стандартная kernel fusion
- Fused RoPE: RoFormer (Su et al., 2021)
