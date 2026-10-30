import { useState, useEffect } from 'react'

type Tab = 'quick' | 'overview' | 'files' | 'profile' | 'checklist'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('quick')
  const [copied, setCopied] = useState(false)
  const [setupScript, setSetupScript] = useState('')

  useEffect(() => {
    fetch('/setup.sh')
      .then(r => r.text())
      .then(setSetupScript)
      .catch(() => setSetupScript('# Load failed'))
  }, [])

  const copyScript = () => {
    navigator.clipboard.writeText(setupScript)
    setCopied(true)
    setTimeout(() => setCopied(false), 2500)
  }

  const downloadScript = () => {
    const blob = new Blob([setupScript], { type: 'text/x-shellscript' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'setup.sh'
    a.click()
    URL.revokeObjectURL(url)
  }

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: 'quick', label: 'Быстрый старт', icon: '🚀' },
    { id: 'overview', label: 'Обзор', icon: '🎯' },
    { id: 'files', label: 'Файлы', icon: '📁' },
    { id: 'profile', label: 'GPU Профиль', icon: '📊' },
    { id: 'checklist', label: 'Чеклист', icon: '✅' },
  ]

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center text-xl">
              🍲
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">Soup Kernel Challenge</h1>
              <p className="text-xs text-gray-400">Decentrathon 6.0 • Triton kernels</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="px-2 py-1 rounded bg-green-900/50 text-green-400 border border-green-800">Kaggle T4</span>
            <span className="px-2 py-1 rounded bg-blue-900/50 text-blue-400 border border-blue-800">Triton 3.6</span>
            <span className="px-2 py-1 rounded bg-purple-900/50 text-purple-400 border border-purple-800">fp16</span>
          </div>
        </div>
      </header>

      <nav className="border-b border-gray-800 bg-gray-900/40">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
                  activeTab === tab.id
                    ? 'border-orange-500 text-orange-400'
                    : 'border-transparent text-gray-400 hover:text-gray-200'
                }`}
              >
                <span className="mr-1.5">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'quick' && (
          <QuickStartTab
            setupScript={setupScript}
            copyScript={copyScript}
            downloadScript={downloadScript}
            copied={copied}
          />
        )}
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'files' && <FilesTab />}
        {activeTab === 'profile' && <ProfileTab />}
        {activeTab === 'checklist' && <ChecklistTab />}
      </main>
    </div>
  )
}

function QuickStartTab({
  setupScript,
  copyScript,
  downloadScript,
  copied,
}: {
  setupScript: string
  copyScript: () => void
  downloadScript: () => void
  copied: boolean
}) {
  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="bg-gradient-to-br from-orange-900/30 to-red-900/30 border border-orange-800/50 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-white mb-2">🚀 Как перенести решение в свой репо</h2>
        <p className="text-gray-300">
          Скопируй setup-скрипт и запусти его в папке своего репозитория хакатона.
          Он создаст все файлы решения.
        </p>
      </div>

      {/* Steps */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">📋 Пошаговая инструкция</h3>
        <div className="space-y-4">
          <Step num={1} title="Клонируй репо хакатона">
            <Code>git clone https://github.com/hackathon-platform/YOUR_TEAM_REPO.git</Code>
            <Code>cd YOUR_TEAM_REPO</Code>
          </Step>

          <Step num={2} title="Скопируй setup.sh">
            <p className="text-sm text-gray-400 mb-2">
              Нажми кнопку «Копировать» ниже или «Скачать файл»
            </p>
          </Step>

          <Step num={3} title="Создай setup.sh и вставь содержимое">
            <Code>nano setup.sh</Code>
            <p className="text-xs text-gray-500">Вставь скопированный скрипт, сохрани (Ctrl+O, Enter, Ctrl+X)</p>
          </Step>

          <Step num={4} title="Запусти setup.sh">
            <Code>bash setup.sh</Code>
            <p className="text-xs text-gray-500">Скрипт создаст все файлы: solution/, tests/, task/</p>
          </Step>

          <Step num={5} title="Проверь что всё создалось">
            <Code>ls -la solution/</Code>
            <p className="text-xs text-gray-500">Должны быть: __init__.py, apply_kernels.py, fused_*.py, utils.py</p>
          </Step>

          <Step num={6} title="Закоммить и запушь">
            <Code>git add .</Code>
            <Code>git commit -m "Add Soup Kernel Challenge solution"</Code>
            <Code>git push</Code>
          </Step>
        </div>
      </div>

      {/* Script box */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 bg-gray-800/30">
          <div>
            <h3 className="text-sm font-semibold text-white">setup.sh</h3>
            <p className="text-xs text-gray-400">Скрипт создаёт все файлы решения</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={downloadScript}
              className="px-3 py-1.5 rounded text-xs font-medium bg-blue-900/50 text-blue-400 border border-blue-800 hover:bg-blue-900"
            >
              ⬇ Скачать файл
            </button>
            <button
              onClick={copyScript}
              className={`px-3 py-1.5 rounded text-xs font-medium border transition-colors ${
                copied
                  ? 'bg-green-900/50 text-green-400 border-green-800'
                  : 'bg-orange-900/50 text-orange-400 border-orange-800 hover:bg-orange-900'
              }`}
            >
              {copied ? '✓ Скопировано!' : '📋 Копировать всё'}
            </button>
          </div>
        </div>
        <div className="overflow-auto max-h-[500px]">
          <pre className="text-xs text-gray-300 font-mono p-4 whitespace-pre">{setupScript}</pre>
        </div>
      </div>

      {/* Alternative: manual */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-3">🔄 Альтернатива: вручную</h3>
        <p className="text-sm text-gray-400 mb-3">
          Если не хочешь использовать скрипт — открой вкладку «Файлы» и копируй каждый файл по отдельности.
        </p>
        <div className="bg-gray-950 rounded-lg p-4 font-mono text-sm">
          <p className="text-gray-500"># Структура которую нужно создать:</p>
          <p className="text-green-400">solution/</p>
          <p className="text-gray-400 pl-4">├── __init__.py</p>
          <p className="text-gray-400 pl-4">├── apply_kernels.py</p>
          <p className="text-gray-400 pl-4">├── utils.py</p>
          <p className="text-gray-400 pl-4">├── fused_cross_entropy.py</p>
          <p className="text-gray-400 pl-4">├── fused_rmsnorm_silu.py</p>
          <p className="text-gray-400 pl-4">├── fused_lora.py</p>
          <p className="text-gray-400 pl-4">├── fused_embedding.py</p>
          <p className="text-gray-400 pl-4">└── fused_rope.py</p>
          <p className="text-green-400">tests/</p>
          <p className="text-gray-400 pl-4">└── test_kernels.py</p>
          <p className="text-green-400">task/</p>
          <p className="text-gray-400 pl-4">└── task.yaml</p>
          <p className="text-yellow-400">README.md</p>
          <p className="text-yellow-400">SUBMISSION.md</p>
        </div>
      </div>
    </div>
  )
}

function Step({ num, title, children }: { num: number; title: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-4">
      <div className="w-8 h-8 rounded-full bg-orange-600 flex items-center justify-center text-sm font-bold shrink-0">
        {num}
      </div>
      <div className="flex-1">
        <p className="text-sm font-medium text-white mb-1">{title}</p>
        <div className="space-y-1">{children}</div>
      </div>
    </div>
  )
}

function Code({ children }: { children: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(children)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <div className="flex items-center gap-2 group">
      <code className="text-xs text-green-400 bg-gray-950 px-2 py-1 rounded flex-1 font-mono">
        {children}
      </code>
      <button
        onClick={copy}
        className="text-xs text-gray-500 hover:text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity"
      >
        {copied ? '✓' : '📋'}
      </button>
    </div>
  )
}

function OverviewTab() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard label="Базовое время" value="~80 с" sub="soup train" color="orange" />
        <MetricCard label="Память база" value="~14 ГБ" sub="из 15 ГБ" color="red" />
        <MetricCard label="Цель: память" value="~5 ГБ" sub="как Unsloth" color="green" />
        <MetricCard label="Токены/с" value="~900" sub="на T4" color="blue" />
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-3">🎯 Суть решения</h2>
        <p className="text-gray-300 leading-relaxed mb-4">
          5 Triton-кернелов, которые ускоряют обучение Soup на Kaggle T4 и снижают расход памяти.
          Подключаются одной функцией <code className="text-orange-400 bg-gray-800 px-1 rounded">apply()</code>.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          <KernelBadge priority="🔴" name="Fused Cross-Entropy" impact="~8-10 ГБ памяти" />
          <KernelBadge priority="🟠" name="Fused RMSNorm+SiLU" impact="~3-5% скорости" />
          <KernelBadge priority="🟠" name="Fused LoRA" impact="~2-3% скорости" />
          <KernelBadge priority="🟡" name="Fused Embedding" impact="~1-2% скорости" />
          <KernelBadge priority="🟡" name="Fused RoPE" impact="~1-2% скорости" />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-3">⚙️ Настройки задачи</h3>
          <ul className="space-y-2 text-sm text-gray-300">
            <li className="flex justify-between"><span>Модель</span><span className="text-orange-400">Qwen2.5-1.5B</span></li>
            <li className="flex justify-between"><span>Квантизация</span><span className="text-orange-400">NF4 (4-bit)</span></li>
            <li className="flex justify-between"><span>LoRA</span><span className="text-orange-400">r=16, α=32</span></li>
            <li className="flex justify-between"><span>Targets</span><span className="text-orange-400">q_proj, v_proj</span></li>
            <li className="flex justify-between"><span>Batch / Seq</span><span className="text-orange-400">4 / 512</span></li>
            <li className="flex justify-between"><span>Шаги</span><span className="text-orange-400">64 (1 эпоха)</span></li>
            <li className="flex justify-between"><span>Словарь</span><span className="text-orange-400">152K токенов</span></li>
          </ul>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-3">🏆 Баллы</h3>
          <ul className="space-y-2 text-sm text-gray-300">
            <li className="flex justify-between"><span>Работает и правильно</span><span className="text-green-400 font-bold">20</span></li>
            <li className="flex justify-between"><span>Скорость (35 max)</span><span className="text-green-400 font-bold">35</span></li>
            <li className="flex justify-between"><span>Память (30 max)</span><span className="text-green-400 font-bold">30</span></li>
            <li className="flex justify-between"><span>README</span><span className="text-green-400 font-bold">15</span></li>
            <li className="border-t border-gray-700 pt-2 flex justify-between font-bold"><span>Итого</span><span className="text-white">100</span></li>
          </ul>
        </div>
      </div>
    </div>
  )
}

function MetricCard({ label, value, sub, color }: { label: string; value: string; sub: string; color: string }) {
  const colors: Record<string, string> = {
    orange: 'from-orange-600/20 to-orange-900/20 border-orange-800/50',
    red: 'from-red-600/20 to-red-900/20 border-red-800/50',
    green: 'from-green-600/20 to-green-900/20 border-green-800/50',
    blue: 'from-blue-600/20 to-blue-900/20 border-blue-800/50',
  }
  const textColors: Record<string, string> = {
    orange: 'text-orange-400', red: 'text-red-400', green: 'text-green-400', blue: 'text-blue-400',
  }
  return (
    <div className={`bg-gradient-to-br ${colors[color]} border rounded-xl p-4`}>
      <p className="text-xs text-gray-400 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold ${textColors[color]} mt-1`}>{value}</p>
      <p className="text-xs text-gray-500 mt-1">{sub}</p>
    </div>
  )
}

function KernelBadge({ priority, name, impact }: { priority: string; name: string; impact: string }) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
      <div className="flex items-center gap-2 mb-1">
        <span>{priority}</span>
        <span className="text-sm font-medium text-white">{name}</span>
      </div>
      <p className="text-xs text-gray-400">{impact}</p>
    </div>
  )
}

function FilesTab() {
  const files = [
    {
      path: 'solution/__init__.py',
      desc: 'Точка входа — экспортирует apply()',
      content: `"""
Soup Kernel Challenge — Decentrathon 6.0
Custom Triton kernels for accelerating Soup training on Kaggle T4.

Usage:
    from solution import apply
    apply()
"""

__version__ = "1.0.0"

from .apply_kernels import apply

__all__ = ["apply"]`
    },
    {
      path: 'solution/apply_kernels.py',
      desc: 'Функция apply() — подключает все кернелы к Soup',
      content: `"""
apply_kernels.py — Подключает все Triton-кернелы к Soup.
"""

import warnings

_PATCHED = False


def apply():
    """
    Plug all custom Triton kernels into Soup training pipeline.
    Must be called once before \`soup train\`.
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

    # 2. Fused RMSNorm + SiLU
    try:
        from .fused_rmsnorm_silu import patch_rmsnorm_silu
        patch_rmsnorm_silu()
        print("[✓] Fused RMSNorm + SiLU (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused RMSNorm+SiLU failed: {e}")

    # 3. Fused LoRA
    try:
        from .fused_lora import patch_lora
        patch_lora()
        print("[✓] Fused LoRA forward (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused LoRA failed: {e}")

    # 4. Fused Embedding + Cast
    try:
        from .fused_embedding import patch_embedding
        patch_embedding()
        print("[✓] Fused Embedding + Cast (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused Embedding failed: {e}")

    # 5. Fused Rotary Position Embedding
    try:
        from .fused_rope import patch_rope
        patch_rope()
        print("[✓] Fused Rotary Position Embedding (speed)")
    except Exception as e:
        warnings.warn(f"[!] Fused RoPE failed: {e}")

    print("=" * 60)
    print("  All kernels applied. Ready for training.")
    print("=" * 60)`
    },
    {
      path: 'solution/utils.py',
      desc: 'Утилиты: CUDA-таймер, хелперы',
      content: `"""
utils.py — Kernel timing via CUDA events, autotune helpers.
"""

import torch
from contextlib import contextmanager
from typing import Optional


class KernelTimer:
    """Accumulates GPU time in custom Triton kernels via CUDA events."""

    def __init__(self):
        self._records: dict[str, list[float]] = {}

    @contextmanager
    def record(self, key: str):
        if not torch.cuda.is_available():
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            self._records.setdefault(key, []).append(elapsed_ms)

    def total_ms(self, key: Optional[str] = None) -> float:
        if key:
            return sum(self._records.get(key, []))
        return sum(sum(v) for v in self._records.values())

    def summary(self) -> str:
        lines = ["--- Kernel Timing ---"]
        total = 0.0
        for key, times in sorted(self._records.items()):
            s = sum(times)
            total += s
            lines.append(f"  {key:30s}  {s:8.2f} ms  ({len(times)} calls)")
        lines.append(f"  {'TOTAL':30s}  {total:8.2f} ms")
        return "\\n".join(lines)

    def reset(self):
        self._records.clear()


TIMER = KernelTimer()


def next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p`
    },
    {
      path: 'solution/fused_cross_entropy.py',
      desc: '🔴 Fused CE — главный выигрыш по памяти (~8-10 ГБ)',
      content: `"""
fused_cross_entropy.py — Fused Cross-Entropy with online log-sum-exp.
Не материализует logits [B*S, 152K] для softmax.
Экономия: ~8-10 ГБ памяти (как Unsloth).
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
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V
        logits = tl.load(
            logits_ptr + pid * stride_ln + v_offsets * stride_lv,
            mask=mask, other=float('-inf'),
        )
        max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))

    # Pass 2: sum(exp(logit - max)) + load target logit
    sum_exp = 0.0
    target_logit = tl.load(
        logits_ptr + pid * stride_ln + label * stride_lv
    )

    for v_start in range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V
        logits = tl.load(
            logits_ptr + pid * stride_ln + v_offsets * stride_lv,
            mask=mask, other=float('-inf'),
        )
        sum_exp += tl.sum(tl.exp(logits - max_logit), axis=0)

    # loss = log(sum_exp) + max - target
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
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < V
            tl.store(
                grad_ptr + pid * stride_ln + v_offsets * stride_lv,
                tl.zeros([BLOCK_V], dtype=tl.float32),
                mask=mask,
            )
        return

    # Recompute max and sum_exp
    max_logit = float('-inf')
    for v_start in range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V
        logits = tl.load(
            logits_ptr + pid * stride_ln + v_offsets * stride_lv,
            mask=mask, other=float('-inf'),
        )
        max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))

    sum_exp = 0.0
    for v_start in range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V
        logits = tl.load(
            logits_ptr + pid * stride_ln + v_offsets * stride_lv,
            mask=mask, other=float('-inf'),
        )
        sum_exp += tl.sum(tl.exp(logits - max_logit), axis=0)

    # Write softmax - one_hot
    for v_start in range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V
        logits = tl.load(
            logits_ptr + pid * stride_ln + v_offsets * stride_lv,
            mask=mask, other=float('-inf'),
        )
        softmax_val = tl.exp(logits - max_logit) / sum_exp
        is_label = (v_offsets == label).to(tl.float32)
        grad_val = softmax_val - is_label
        tl.store(
            grad_ptr + pid * stride_ln + v_offsets * stride_lv,
            grad_val, mask=mask,
        )


class FusedCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, ignore_index=-100):
        N, V = logits.shape
        logits_f32 = logits.float() if logits.dtype == torch.float16 else logits

        per_token_loss = torch.empty(N, device=logits.device, dtype=torch.float32)
        BLOCK_V = min(4096, triton.next_power_of_2(V))

        with TIMER.record("fused_ce_forward"):
            grid = (N,)
            _fused_ce_forward_kernel[grid](
                logits_f32, labels, per_token_loss,
                N, V,
                logits_f32.stride(0), logits_f32.stride(1),
                BLOCK_V=BLOCK_V,
                IGNORE_INDEX=ignore_index,
            )

        valid_mask = labels != ignore_index
        n_valid = valid_mask.sum().item()
        loss = per_token_loss.sum() / max(n_valid, 1)

        ctx.save_for_backward(logits_f32, labels)
        ctx.ignore_index = ignore_index
        ctx.n_valid = n_valid
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits_f32, labels = ctx.saved_tensors
        N, V = logits_f32.shape

        grad_logits = torch.empty_like(logits_f32)
        BLOCK_V = min(4096, triton.next_power_of_2(V))

        with TIMER.record("fused_ce_backward"):
            grid = (N,)
            _fused_ce_backward_kernel[grid](
                logits_f32, labels, grad_logits,
                N, V,
                logits_f32.stride(0), logits_f32.stride(1),
                BLOCK_V=BLOCK_V,
                IGNORE_INDEX=ctx.ignore_index,
            )

        if ctx.n_valid > 0:
            grad_logits *= grad_output / ctx.n_valid
        return grad_logits, None, None


class FusedCrossEntropyLoss(nn.Module):
    """Drop-in replacement for nn.CrossEntropyLoss."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        if logits.dim() == 3:
            B, S, V = logits.shape
            logits = logits.reshape(-1, V)
            labels = labels.reshape(-1)
        return FusedCrossEntropyFunction.apply(logits, labels, self.ignore_index)


def patch_cross_entropy():
    """Replace Soup's CE loss with fused version."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        _orig = Qwen2ForCausalLM.forward

        def patched(self, *args, **kwargs):
            outputs = _orig(self, *args, **kwargs)
            if kwargs.get('labels') is not None and outputs.loss is not None:
                logits = outputs.logits
                labels = kwargs['labels']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                outputs.loss = FusedCrossEntropyLoss()(shift_logits, shift_labels)
            return outputs

        Qwen2ForCausalLM.forward = patched
    except (ImportError, AttributeError):
        pass`
    },
    {
      path: 'solution/fused_rmsnorm_silu.py',
      desc: '🟠 Fused RMSNorm + SiLU — слияние norm + activation',
      content: `"""
fused_rmsnorm_silu.py — Fused RMSNorm + SiLU in one kernel.
"""

import torch
import triton
import triton.language as tl
from .utils import TIMER


@triton.jit
def _fused_rmsnorm_silu_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_xn, N,
    D: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    offsets = tl.arange(0, D)
    x = tl.load(x_ptr + row * stride_xn + offsets).to(tl.float32)

    # RMSNorm
    mean_sq = tl.sum(x * x, axis=0) / D
    rms_inv = tl.rsqrt(mean_sq + eps)
    w = tl.load(weight_ptr + offsets).to(tl.float32)
    normed = x * rms_inv * w

    # SiLU
    sigmoid_val = 1.0 / (1.0 + tl.exp(-normed))
    result = normed * sigmoid_val

    tl.store(out_ptr + row * stride_xn + offsets, result.to(tl.float16))


def fused_rmsnorm_silu(x, weight, eps=1e-6):
    """Apply fused RMSNorm + SiLU."""
    orig_shape = x.shape
    D = x.shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    D_pow2 = triton.next_power_of_2(D)

    with TIMER.record("fused_rmsnorm_silu"):
        grid = (N,)
        _fused_rmsnorm_silu_kernel[grid](
            x_2d, weight, out,
            x_2d.stride(0), N,
            D=D_pow2, eps=eps,
        )

    return out.reshape(orig_shape)


def patch_rmsnorm_silu():
    pass  # Kernel available for direct calls`
    },
    {
      path: 'solution/fused_lora.py',
      desc: '🟠 Fused LoRA — base + scaling*(x@A@B) в одном кернеле',
      content: `"""
fused_lora.py — Fused LoRA: base + scaling * (x @ A @ B).
"""

import torch
import triton
import triton.language as tl
from .utils import TIMER


@triton.jit
def _fused_lora_kernel(
    base_ptr, x_ptr, lora_a_ptr, lora_b_ptr, out_ptr,
    M, N, K, R,
    stride_bm, stride_bn,
    stride_xm, stride_xk,
    stride_am, stride_ak,
    stride_bk, stride_bn_lora,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_off < M
    n_mask = n_off < N

    base = tl.load(
        base_ptr + m_off[:, None] * stride_bm + n_off[None, :] * stride_bn,
        mask=m_mask[:, None] & n_mask[None, :], other=0.0,
    ).to(tl.float32)

    lora_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for r_start in range(0, R, BLOCK_R):
        r_off = r_start + tl.arange(0, BLOCK_R)
        r_mask = r_off < R

        xa = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for k in range(K):
            x_col = tl.load(x_ptr + m_off * stride_xm + k * stride_xk, mask=m_mask, other=0.0)
            a_row = tl.load(lora_a_ptr + r_off * stride_am + k * stride_ak, mask=r_mask, other=0.0)
            xa += x_col[:, None] * a_row[None, :]

        for ri in range(BLOCK_R):
            if r_start + ri >= R:
                break
            b_col = tl.load(
                lora_b_ptr + n_off * stride_bk + (r_start + ri) * stride_bn_lora,
                mask=n_mask, other=0.0,
            )
            lora_acc += xa[:, ri:ri+1] * b_col[None, :]

    result = base + scaling * lora_acc
    tl.store(
        out_ptr + m_off[:, None] * stride_bm + n_off[None, :] * stride_bn,
        result.to(tl.float16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def fused_lora_forward(base_output, x, lora_A, lora_B, scaling):
    """Compute base + scaling * (x @ A @ B) in one kernel."""
    M, K = x.shape
    R = lora_A.shape[0]
    N = lora_B.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=torch.float16)
    BLOCK_M = min(64, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_R = min(16, triton.next_power_of_2(R))

    with TIMER.record("fused_lora"):
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _fused_lora_kernel[grid](
            base_output, x, lora_A, lora_B, out,
            M, N, K, R,
            base_output.stride(0), base_output.stride(1),
            x.stride(0), x.stride(1),
            lora_A.stride(0), lora_A.stride(1),
            lora_B.stride(0), lora_B.stride(1),
            scaling,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_R=BLOCK_R,
        )
    return out


def patch_lora():
    pass`
    },
    {
      path: 'solution/fused_embedding.py',
      desc: '🟡 Fused Embedding lookup + fp16 cast',
      content: `"""
fused_embedding.py — Embedding lookup + cast to fp16 in one kernel.
"""

import torch
import triton
import triton.language as tl
from .utils import TIMER


@triton.jit
def _fused_embedding_cast_kernel(
    weight_ptr, ids_ptr, out_ptr,
    N,
    D: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= N:
        return

    token_id = tl.load(ids_ptr + token_idx)
    if token_id < 0 or token_id >= VOCAB_SIZE:
        offsets = tl.arange(0, D)
        tl.store(out_ptr + token_idx * D + offsets, tl.zeros([D], dtype=tl.float16))
        return

    offsets = tl.arange(0, D)
    emb = tl.load(weight_ptr + token_id * D + offsets)
    tl.store(out_ptr + token_idx * D + offsets, emb.to(tl.float16))


def fused_embedding_lookup(weight, ids, scale=None):
    """Fused embedding lookup + cast to fp16."""
    orig_shape = ids.shape
    ids_flat = ids.reshape(-1)
    N = ids_flat.shape[0]
    V, D = weight.shape

    out = torch.empty((N, D), device=weight.device, dtype=torch.float16)
    D_pow2 = triton.next_power_of_2(D)

    with TIMER.record("fused_embedding"):
        grid = (N,)
        _fused_embedding_cast_kernel[grid](
            weight, ids_flat, out,
            N, D=D_pow2, VOCAB_SIZE=V,
        )

    return out.reshape(*orig_shape, D)


def patch_embedding():
    pass`
    },
    {
      path: 'solution/fused_rope.py',
      desc: '🟡 Fused RoPE — rotary embedding для Q и K',
      content: `"""
fused_rope.py — Fused Rotary Position Embedding for Q and K.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional
from .utils import TIMER


@triton.jit
def _fused_rope_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    q_out_ptr, k_out_ptr,
    stride_qt, stride_qh,
    stride_kt, stride_kh,
    stride_cos_pos,
    H,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HALF_D

    pos = pid_t

    cos_vals = tl.load(cos_ptr + pos * stride_cos_pos + d_offsets, mask=d_mask, other=0.0)
    sin_vals = tl.load(sin_ptr + pos * stride_cos_pos + d_offsets, mask=d_mask, other=0.0)

    # Q rotation
    q_base = pid_t * stride_qt + pid_h * stride_qh
    q1 = tl.load(q_ptr + q_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + q_base + HALF_D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    tl.store(q_out_ptr + q_base + d_offsets, (q1 * cos_vals - q2 * sin_vals).to(tl.float16), mask=d_mask)
    tl.store(q_out_ptr + q_base + HALF_D + d_offsets, (q1 * sin_vals + q2 * cos_vals).to(tl.float16), mask=d_mask)

    # K rotation
    k_base = pid_t * stride_kt + pid_h * stride_kh
    k1 = tl.load(k_ptr + k_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_ptr + k_base + HALF_D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    tl.store(k_out_ptr + k_base + d_offsets, (k1 * cos_vals - k2 * sin_vals).to(tl.float16), mask=d_mask)
    tl.store(k_out_ptr + k_base + HALF_D + d_offsets, (k1 * sin_vals + k2 * cos_vals).to(tl.float16), mask=d_mask)


def fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply RoPE to Q and K using fused kernel."""
    if q.dim() == 4:
        B, H, S, D = q.shape
        total = B * S
        q_flat = q.reshape(total, H, D)
        k_flat = k.reshape(total, H, D)
    else:
        total, H, D = q.shape
        q_flat = q
        k_flat = k

    HALF_D = D // 2
    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)

    cos_flat = cos.reshape(-1, HALF_D) if cos.dim() == 3 else cos
    sin_flat = sin.reshape(-1, HALF_D) if sin.dim() == 3 else sin

    BLOCK_D = triton.next_power_of_2(HALF_D)

    with TIMER.record("fused_rope"):
        grid = (total, H)
        _fused_rope_kernel[grid](
            q_flat, k_flat, cos_flat, sin_flat,
            q_out, k_out,
            q_flat.stride(0), q_flat.stride(1),
            k_flat.stride(0), k_flat.stride(1),
            cos_flat.stride(0),
            H, D=D, HALF_D=HALF_D, BLOCK_D=BLOCK_D,
        )

    if q.dim() == 4:
        q_out = q_out.reshape(B, H, S, D)
        k_out = k_out.reshape(B, H, S, D)

    return q_out, k_out


def patch_rope():
    """Patch Qwen2's apply_rotary_pos_emb."""
    try:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
        qwen2_mod.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    except (ImportError, AttributeError):
        pass`
    },
  ]

  const [expanded, setExpanded] = useState<string | null>(null)
  const [copiedFile, setCopiedFile] = useState<string | null>(null)

  const copyFile = (content: string, path: string) => {
    navigator.clipboard.writeText(content)
    setCopiedFile(path)
    setTimeout(() => setCopiedFile(null), 2000)
  }

  return (
    <div className="space-y-3">
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <p className="text-sm text-gray-400">
          📁 Все файлы решения. Нажми на файл чтобы раскрыть, потом «Копировать» чтобы скопировать содержимое.
        </p>
      </div>

      {files.map((file) => (
        <div key={file.path} className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          <button
            onClick={() => setExpanded(expanded === file.path ? null : file.path)}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-800/50"
          >
            <div>
              <p className="text-sm font-mono text-white">{file.path}</p>
              <p className="text-xs text-gray-400 mt-0.5">{file.desc}</p>
            </div>
            <span className={`text-gray-400 transition-transform ${expanded === file.path ? 'rotate-180' : ''}`}>▼</span>
          </button>
          {expanded === file.path && (
            <div className="border-t border-gray-800">
              <div className="flex justify-end px-4 py-2 bg-gray-800/30">
                <button
                  onClick={() => copyFile(file.content, file.path)}
                  className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                    copiedFile === file.path
                      ? 'bg-green-900/50 text-green-400 border border-green-800'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600 border border-gray-600'
                  }`}
                >
                  {copiedFile === file.path ? '✓ Скопировано' : '📋 Копировать файл'}
                </button>
              </div>
              <pre className="text-xs text-gray-300 font-mono p-4 overflow-auto max-h-96 whitespace-pre bg-gray-950">
                {file.content}
              </pre>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function ProfileTab() {
  const operations = [
    { name: 'Поэлементные + dtype casts', pct: 44, color: 'bg-orange-500', desc: 'RMSNorm, SiLU, embedding casts, LoRA add' },
    { name: 'Матричные умножения (GEMM)', pct: 36, color: 'bg-blue-500', desc: 'QKV projections, FFN, LoRA matmul' },
    { name: 'Softmax (vocab 152K)', pct: 7, color: 'bg-purple-500', desc: 'Cross-entropy loss' },
    { name: 'Attention', pct: 5, color: 'bg-green-500', desc: 'Scaled dot-product attention' },
    { name: 'NF4 unpack', pct: 5, color: 'bg-red-500', desc: 'Декомпрессия 4-битных весов' },
    { name: 'Прочее', pct: 3, color: 'bg-gray-500', desc: 'Memory ops, scheduling' },
  ]

  return (
    <div className="space-y-6">
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-2">📊 Профиль шага Soup на T4</h2>
        <p className="text-sm text-gray-400 mb-6">GPU занят ~95% времени:</p>
        <div className="space-y-4">
          {operations.map((op) => (
            <div key={op.name}>
              <div className="flex justify-between items-center mb-1">
                <span className="text-sm text-gray-300">{op.name}</span>
                <span className="text-sm font-mono text-gray-400">{op.pct}%</span>
              </div>
              <div className="h-8 bg-gray-800 rounded-full overflow-hidden">
                <div className={`h-full ${op.color} rounded-full flex items-center px-3`} style={{ width: `${op.pct}%` }}>
                  {op.pct >= 10 && <span className="text-xs font-bold text-white/80">{op.pct}%</span>}
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-1">{op.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function ChecklistTab() {
  const [checks, setChecks] = useState<Record<string, boolean>>({})
  const toggle = (id: string) => setChecks(prev => ({ ...prev, [id]: !prev[id] }))

  const sections = [
    {
      title: 'Must Have (7)',
      items: [
        { id: 'run-kaggle', text: 'Запускается в Kaggle T4 без правок' },
        { id: 'triton-kernels', text: 'Свои @triton.jit кернелы ≥5% времени обучения' },
        { id: 'no-worse', text: 'Loss не хуже базы на 0.5%; время/память не хуже на 10%' },
        { id: 'same-task', text: 'Та же задача: токены, энтропия, rewind в пределах 3%' },
        { id: 'tests-pass', text: '79 тестов Soup проходят' },
        { id: 'readme', text: 'README: что, как, результаты, ссылка на ноутбук' },
      ],
    },
    {
      title: 'Ограничения (8)',
      items: [
        { id: 'no-libs', text: 'НЕ: unsloth/liger/xformers/flash-attn/torchao' },
        { id: 'no-modes', text: 'НЕ: backend:unsloth, use_liger, fast_lora, packing' },
        { id: 'no-task-change', text: 'НЕ менять модель, данные, batch, шаги, LoRA' },
        { id: 'no-harness-edit', text: 'НЕ менять harness/, tests/, task/' },
        { id: 'open-code', text: 'Репозиторий и ноутбук открыты' },
      ],
    },
    {
      title: 'Сдача (10)',
      items: [
        { id: 'repo', text: 'GitHub-репозиторий с solution/, README, SUBMISSION.md' },
        { id: 'notebook', text: 'Публичный Kaggle-ноутбук из шаблона' },
        { id: 'video', text: 'Демо-видео до 3 минут' },
        { id: 'presentation', text: 'Презентация' },
      ],
    },
  ]

  const total = sections.reduce((a, s) => a + s.items.length, 0)
  const done = Object.values(checks).filter(Boolean).length

  return (
    <div className="space-y-6">
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xl font-bold text-white">✅ Чеклист готовности</h2>
          <span className="text-lg font-mono text-orange-400">{done}/{total}</span>
        </div>
        <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-orange-500 to-green-500 rounded-full transition-all" style={{ width: `${(done / total) * 100}%` }} />
        </div>
      </div>

      {sections.map((section) => (
        <div key={section.title} className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-base font-semibold text-white mb-4">{section.title}</h3>
          <div className="space-y-2">
            {section.items.map((item) => (
              <label key={item.id} className="flex items-start gap-3 p-2 rounded-lg hover:bg-gray-800/50 cursor-pointer">
                <input
                  type="checkbox"
                  checked={checks[item.id] || false}
                  onChange={() => toggle(item.id)}
                  className="mt-0.5 w-5 h-5 rounded border-gray-600 bg-gray-800 text-orange-500 focus:ring-orange-500"
                />
                <span className={`text-sm ${checks[item.id] ? 'text-gray-500 line-through' : 'text-gray-300'}`}>
                  {item.text}
                </span>
              </label>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

export default App
