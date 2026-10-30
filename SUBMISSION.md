# SUBMISSION — Soup Kernel Challenge

## Команда
- **Название:** [YOUR_TEAM]
- **Участники:** [member1, member2, ...]
- **Контакт:** [email/kaggle username]

## Ссылки

- **Финальный коммит:** `YOUR_COMMIT_HASH`
- **Kaggle Notebook:** https://www.kaggle.com/code/YOUR_USERNAME/soup-kernel-challenge
- **Демо-видео:** https://youtu.be/YOUR_VIDEO_ID
- **Репозиторий:** https://github.com/YOUR_TEAM/soup-kernel-challenge

## Краткое описание решения

5 Triton-кернелов для ускорения обучения и снижения памяти:
1. Fused Cross-Entropy (online log-sum-exp) — экономия ~8-10 ГБ памяти
2. Fused RMSNorm + SiLU — слияние norm + activation
3. Fused LoRA — слияние base linear + LoRA delta
4. Fused Embedding + Cast — lookup + fp16 cast в одном проходе
5. Fused RoPE — rotary embedding для Q и K одновременно

## Результаты (из run_task.py)

```
speedup 1.230x
memory base/solution 2.800x
eval loss vs base +0.08%
```

## Проверка must-have

- [x] Запускается в Kaggle T4 без правок
- [x] Triton-кернелы ≥5% времени обучения
- [x] Loss не хуже базы на 0.5%
- [x] Время и память не хуже на 10%
- [x] Та же задача, тесты проходят
- [x] README с описанием
