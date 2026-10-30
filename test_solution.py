#!/usr/bin/env python3
"""
test_solution.py — Тестовый скрипт для проверки решения в Kaggle.

Запуск:
    python test_solution.py

Этот скрипт:
1. Проверяет что все кернелы компилируются
2. Запускает мини-обучение (2 шага)
3. Замеряет время и память
4. Сравнивает с базой
"""

import torch
import time
import sys
import os

# Добавляем solution в path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_kernels_compile():
    """Проверяет что все Triton кернелы компилируются."""
    print("=" * 60)
    print("🔍 Проверка компиляции кернелов...")
    print("=" * 60)
    
    try:
        from solution import apply
        apply()
        print("✅ Все кернелы применены успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка применения кернелов: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training():
    """Запускает мини-обучение для замера."""
    print("\n" + "=" * 60)
    print("🚀 Запуск мини-обучения (2 шага)...")
    print("=" * 60)
    
    try:
        # Импортируем Soup
        from soup import train
        from soup.config import load_config
        
        # Загружаем конфиг (используем маленький датасет для теста)
        config = load_config("task/task.yaml")
        
        # Уменьшаем для быстрого теста
        config.training.num_epochs = 1
        config.training.train_size = 16  # всего 16 строк
        config.training.batch_size = 4
        # Это даст 4 шага
        
        # Замеряем время
        start_time = time.time()
        start_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        # Запускаем обучение
        train(config)
        
        end_time = time.time()
        end_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        elapsed = end_time - start_time
        peak_mem = end_mem - start_mem
        
        print(f"\n✅ Мини-обучение завершено")
        print(f"   Время: {elapsed:.2f} с")
        print(f"   Пик памяти: {peak_mem:.2f} ГБ")
        
        return True, elapsed, peak_mem
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0


def compare_with_baseline(our_time, our_mem):
    """Сравнивает с базовыми результатами."""
    print("\n" + "=" * 60)
    print("📊 Сравнение с базой")
    print("=" * 60)
    
    # Базовые значения (из ТЗ)
    base_time = 80.0  # секунд
    base_mem = 14.0   # ГБ
    
    speedup = base_time / our_time if our_time > 0 else 0
    memory_saving = base_mem / our_mem if our_mem > 0 else 0
    
    print(f"\n{'Метрика':<20} {'База':<15} {'Наше':<15} {'Улучшение':<15}")
    print("-" * 65)
    print(f"{'Время (с)':<20} {base_time:<15.2f} {our_time:<15.2f} {speedup:<15.2f}×")
    print(f"{'Память (ГБ)':<20} {base_mem:<15.2f} {our_mem:<15.2f} {memory_saving:<15.2f}×")
    
    print("\n📈 Оценка результата:")
    if speedup >= 4.0 and memory_saving >= 4.0:
        print("🏆 ОТЛИЧНО! Топовый результат (как у лидеров)")
    elif speedup >= 2.0 and memory_saving >= 2.0:
        print("✅ ХОРОШО! Выше среднего")
    elif speedup >= 1.2 and memory_saving >= 1.5:
        print("👍 НЕПЛОХО! Базовый выигрыш")
    else:
        print("⚠️ СЛАБО. Нужно оптимизировать")
    
    return speedup, memory_saving


def main():
    """Главная функция."""
    print("\n" + "=" * 60)
    print("🍲 Soup Kernel Challenge — Тест решения")
    print("=" * 60)
    
    # Проверяем GPU
    if not torch.cuda.is_available():
        print("❌ CUDA недоступна! Запускайте в Kaggle с GPU.")
        sys.exit(1)
    
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} ГБ")
    
    # Проверяем компиляцию
    if not check_kernels_compile():
        print("\n❌ Кернелы не скомпилировались. Исправьте ошибки.")
        sys.exit(1)
    
    # Запускаем мини-обучение
    success, elapsed, peak_mem = test_mini_training()
    
    if not success:
        print("\n❌ Обучение не удалось. Проверьте логи.")
        sys.exit(1)
    
    # Сравниваем с базой
    speedup, memory_saving = compare_with_baseline(elapsed, peak_mem)
    
    print("\n" + "=" * 60)
    print("✅ Тест завершён!")
    print("=" * 60)
    print("\n💡 Рекомендации:")
    if speedup < 2.0:
        print("   • Проверьте что Fused CE работает (главный выигрыш по памяти)")
        print("   • Проверьте логи на наличие warnings")
    if memory_saving < 2.0:
        print("   • Fused CE должен экономить ~8-10 ГБ")
        print("   • Проверьте что патч Qwen2ForCausalLM применился")
    
    print("\n🚀 Для полного теста запустите:")
    print("   python harness/run_task.py")
    print()


if __name__ == "__main__":
    main()
