import { useState } from 'react'

type Tab = 'overview' | 'profile' | 'kernels' | 'structure' | 'checklist'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('overview')
  const [expandedKernel, setExpandedKernel] = useState<string | null>(null)

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: 'overview', label: 'Обзор', icon: '🎯' },
    { id: 'profile', label: 'GPU Профиль', icon: '📊' },
    { id: 'kernels', label: 'Кернелы', icon: '⚡' },
    { id: 'structure', label: 'Структура', icon: '📁' },
    { id: 'checklist', label: 'Чеклист', icon: '✅' },
  ]

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center text-xl">
              🍲
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">Soup Kernel Challenge</h1>
              <p className="text-xs text-gray-400">Decentrathon 6.0 • 36 часов</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="px-2 py-1 rounded bg-green-900/50 text-green-400 border border-green-800">
              Kaggle T4
            </span>
            <span className="px-2 py-1 rounded bg-blue-900/50 text-blue-400 border border-blue-800">
              Triton 3.6.0
            </span>
            <span className="px-2 py-1 rounded bg-purple-900/50 text-purple-400 border border-purple-800">
              fp16
            </span>
          </div>
        </div>
      </header>

      {/* Navigation */}
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

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'profile' && <ProfileTab />}
        {activeTab === 'kernels' && (
          <KernelsTab expandedKernel={expandedKernel} setExpandedKernel={setExpandedKernel} />
        )}
        {activeTab === 'structure' && <StructureTab />}
        {activeTab === 'checklist' && <ChecklistTab />}
      </main>
    </div>
  )
}

function OverviewTab() {
  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard label="Базовое время" value="~80 с" sub="soup train" color="orange" />
        <MetricCard label="Память база" value="~14 ГБ" sub="из 15 ГБ" color="red" />
        <MetricCard label="Unsloth память" value="~2.3 ГБ" sub="в 6× меньше" color="green" />
        <MetricCard label="Токены/с" value="~900" sub="на T4" color="blue" />
      </div>

      {/* Problem Statement */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-3">🎯 Задача</h2>
        <p className="text-gray-300 leading-relaxed">
          Написать свои Triton-кернелы, которые ускоряют фиксированный запуск обучения Soup на Kaggle T4
          или снижают расход памяти при том же качестве. В решении должен быть хотя бы один собственный
          Triton-кернел, который реально работает в обучении.
        </p>
      </div>

      {/* Setup */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-3">⚙️ Настройки задачи</h3>
          <ul className="space-y-2 text-sm text-gray-300">
            <li className="flex justify-between"><span>Модель</span><span className="text-orange-400">Qwen2.5-1.5B-Instruct</span></li>
            <li className="flex justify-between"><span>Квантизация</span><span className="text-orange-400">NF4 (4-bit)</span></li>
            <li className="flex justify-between"><span>LoRA</span><span className="text-orange-400">r=16, alpha=32</span></li>
            <li className="flex justify-between"><span>LoRA targets</span><span className="text-orange-400">q_proj, v_proj</span></li>
            <li className="flex justify-between"><span>Batch</span><span className="text-orange-400">4</span></li>
            <li className="flex justify-between"><span>Seq length</span><span className="text-orange-400">до 512</span></li>
            <li className="flex justify-between"><span>Данные</span><span className="text-orange-400">256 строк</span></li>
            <li className="flex justify-between"><span>Шаги</span><span className="text-orange-400">64 (1 эпоха)</span></li>
          </ul>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-3">🏆 Оценивание</h3>
          <ul className="space-y-2 text-sm text-gray-300">
            <li className="flex justify-between"><span>Работает и правильно</span><span className="text-green-400 font-bold">20 баллов</span></li>
            <li className="flex justify-between"><span>Скорость (s = база/вы)</span><span className="text-green-400 font-bold">35 баллов</span></li>
            <li className="flex justify-between"><span>Память (m = база/вы)</span><span className="text-green-400 font-bold">30 баллов</span></li>
            <li className="flex justify-between"><span>README и воспроизводимость</span><span className="text-green-400 font-bold">15 баллов</span></li>
            <li className="border-t border-gray-700 pt-2 flex justify-between"><span className="font-semibold">Итого</span><span className="text-white font-bold">100 баллов</span></li>
          </ul>
        </div>
      </div>

      {/* Formula */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-3">📐 Формула баллов</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <p className="text-sm text-gray-400 mb-1">Скорость:</p>
            <code className="text-orange-400 text-sm">35 × ln(s) / ln(s_лучш)</code>
            <p className="text-xs text-gray-500 mt-2">s = время_базы / ваше_время</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4">
            <p className="text-sm text-gray-400 mb-1">Память:</p>
            <code className="text-orange-400 text-sm">30 × ln(m) / ln(m_лучш)</code>
            <p className="text-xs text-gray-500 mt-2">m = пик_базы / ваш_пик</p>
          </div>
        </div>
      </div>

      {/* Constraints */}
      <div className="bg-red-950/30 border border-red-900/50 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-red-400 mb-3">🚫 Нельзя</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-gray-300">
          <div>• unsloth, liger-kernel, xformers, flash-attn, torchao</div>
          <div>• backend: unsloth, use_liger</div>
          <div>• Менять модель, данные, batch, шаги, LoRA</div>
          <div>• gradient_checkpointing, activation_offloading</div>
          <div>• fast_lora, use_cut_ce, packing</div>
          <div>• Добавлять обучаемые параметры</div>
          <div>• Менять скрипты harness/, tests/, task/</div>
          <div>• Подменять адаптер, читать скрытые данные</div>
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
    orange: 'text-orange-400',
    red: 'text-red-400',
    green: 'text-green-400',
    blue: 'text-blue-400',
  }
  return (
    <div className={`bg-gradient-to-br ${colors[color]} border rounded-xl p-4`}>
      <p className="text-xs text-gray-400 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold ${textColors[color]} mt-1`}>{value}</p>
      <p className="text-xs text-gray-500 mt-1">{sub}</p>
    </div>
  )
}

function ProfileTab() {
  const operations = [
    { name: 'Матричные умножения', pct: 36, color: 'bg-blue-500', desc: 'GEMM: QKV projections, FFN, LoRA' },
    { name: 'Поэлементные + dtype casts', pct: 44, color: 'bg-orange-500', desc: 'RMSNorm, SiLU, embedding casts, LoRA add' },
    { name: 'Softmax (vocab 152K)', pct: 7, color: 'bg-purple-500', desc: 'Cross-entropy loss по огромному словарю' },
    { name: 'Attention', pct: 5, color: 'bg-green-500', desc: 'Scaled dot-product attention' },
    { name: 'NF4 unpack', pct: 5, color: 'bg-red-500', desc: 'Декомпрессия 4-битных весов' },
    { name: 'Прочее', pct: 3, color: 'bg-gray-500', desc: 'Memory ops, scheduling' },
  ]

  return (
    <div className="space-y-6">
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-2">📊 Профиль шага Soup на T4</h2>
        <p className="text-sm text-gray-400 mb-6">GPU занят ~95% времени. Распределение по операциям:</p>
        
        {/* Bar chart */}
        <div className="space-y-3">
          {operations.map((op) => (
            <div key={op.name}>
              <div className="flex justify-between items-center mb-1">
                <span className="text-sm text-gray-300">{op.name}</span>
                <span className="text-sm font-mono text-gray-400">{op.pct}%</span>
              </div>
              <div className="h-6 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className={`h-full ${op.color} rounded-full transition-all duration-1000 flex items-center justify-end pr-2`}
                  style={{ width: `${op.pct}%` }}
                >
                  {op.pct >= 10 && <span className="text-xs font-bold text-white/80">{op.pct}%</span>}
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-0.5">{op.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Opportunities */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">💡 Где искать оптимизации</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <OpportunityCard
            title="Слияние мелких операций (44%)"
            impact="Высокий"
            impactColor="text-green-400"
            items={[
              'Fused RMSNorm + SiLU',
              'Fused LoRA add + cast',
              'Fused embedding + cast to fp16',
              'Fused rotary embeddings',
            ]}
          />
          <OpportunityCard
            title="Fused Cross-Entropy (7% + память)"
            impact="Высокий (память)"
            impactColor="text-green-400"
            items={[
              'Не материализовать logits [B×S×152K]',
              'Считать loss чанками по словарю',
              'Entropy и accuracy без полного softmax',
              'Экономия ~8-10 ГБ памяти',
            ]}
          />
          <OpportunityCard
            title="4-bit matmul fusion (5% + 36%)"
            impact="Средний"
            impactColor="text-yellow-400"
            items={[
              'Fused dequant + GEMM',
              'Перепаковка NF4 в удобный формат',
              'Sub-byte addressing',
              'Векторизация по 8 весов за раз',
            ]}
          />
          <OpportunityCard
            title="Attention (5%)"
            impact="Средний"
            impactColor="text-yellow-400"
            items={[
              'Flash-attention стиль (без своих библиотек!)',
              'Tiled attention с online softmax',
              'FP16 accumulation',
            ]}
          />
        </div>
      </div>
    </div>
  )
}

function OpportunityCard({ title, impact, impactColor, items }: { title: string; impact: string; impactColor: string; items: string[] }) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
      <div className="flex justify-between items-start mb-2">
        <h4 className="text-sm font-semibold text-white">{title}</h4>
        <span className={`text-xs font-medium ${impactColor}`}>{impact}</span>
      </div>
      <ul className="space-y-1">
        {items.map((item, i) => (
          <li key={i} className="text-xs text-gray-400 flex items-start gap-1.5">
            <span className="text-orange-500 mt-0.5">›</span>
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}

function KernelsTab({ expandedKernel, setExpandedKernel }: { expandedKernel: string | null; setExpandedKernel: (k: string | null) => void }) {
  const kernels = [
    {
      id: 'fused-cross-entropy',
      name: 'Fused Cross-Entropy Loss',
      priority: '🔴 Критический',
      time: '~7% времени + огромная экономия памяти',
      description: 'Считает loss, entropy и accuracy не материализуя logits [B×S×152K]. Это главный выигрыш по памяти — Unsloth так делает и занимает в 6 раз меньше.',
      code: `import torch
import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_kernel(
    logits_ptr,      # [total_tokens, vocab_size] — но считаем чанками
    labels_ptr,      # [total_tokens]
    loss_ptr,        # [total_tokens] — выход
    entropy_ptr,     # [total_tokens] — выход (опц.)
    n_tokens,
    vocab_size,
    BLOCK_VOCAB: tl.constexpr,   # размер чанка по словарю
    BLOCK_TOKENS: tl.constexpr,  # кол-во токенов за раз
    NEED_ENTROPY: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offset = pid * BLOCK_TOKENS
    
    # Для каждого токена считаем loss чанками по словарю
    for t in range(BLOCK_TOKENS):
        token_idx = token_offset + t
        if token_idx >= n_tokens:
            continue
        
        label = tl.load(labels_ptr + token_idx)
        
        # Online log-sum-exp по чанкам словаря
        max_logit = float('-inf')
        # Pass 1: найти max
        for v_start in range(0, vocab_size, BLOCK_VOCAB):
            v_offsets = v_start + tl.arange(0, BLOCK_VOCAB)
            mask = v_offsets < vocab_size
            logits = tl.load(
                logits_ptr + token_idx * vocab_size + v_offsets,
                mask=mask, other=float('-inf')
            )
            max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))
        
        # Pass 2: sum exp и target logit
        sum_exp = 0.0
        target_logit = 0.0
        for v_start in range(0, vocab_size, BLOCK_VOCAB):
            v_offsets = v_start + tl.arange(0, BLOCK_VOCAB)
            mask = v_offsets < vocab_size
            logits = tl.load(
                logits_ptr + token_idx * vocab_size + v_offsets,
                mask=mask, other=float('-inf')
            )
            exp_logits = tl.exp(logits - max_logit)
            sum_exp += tl.sum(exp_logits, axis=0)
            
            # Загрузить target logit
            if label >= v_start and label < v_start + BLOCK_VOCAB:
                target_logit = tl.load(
                    logits_ptr + token_idx * vocab_size + label
                )
        
        log_sum_exp = max_logit + tl.log(sum_exp)
        loss = log_sum_exp - target_logit
        tl.store(loss_ptr + token_idx, loss)`,
    },
    {
      id: 'fused-rmsnorm-silu',
      name: 'Fused RMSNorm + SiLU',
      priority: '🟠 Высокий',
      time: '~15-20% времени (часть из 44% element-wise)',
      description: 'Объединяет RMSNorm (в начале FFN) и SiLU активацию в один кернел. Убирает промежуточные записи в память и лишние dtype casts.',
      code: `import torch
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_silu_kernel(
    x_ptr,          # [N, D] input
    weight_ptr,     # [D] RMSNorm weight
    out_ptr,        # [N, D] output
    stride_n,
    D: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    
    # Загрузить строку
    offsets = tl.arange(0, D)
    x = tl.load(x_ptr + row * stride_n + offsets).to(tl.float32)
    
    # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    mean_sq = tl.sum(x * x, axis=0) / D
    rms = tl.rsqrt(mean_sq + eps)
    normed = x * rms
    
    w = tl.load(weight_ptr + offsets).to(tl.float32)
    normed = normed * w
    
    # SiLU: x * sigmoid(x)
    sigmoid_val = tl.sigmoid(normed)
    result = normed * sigmoid_val
    
    # Сохранить в fp16
    tl.store(out_ptr + row * stride_n + offsets, result.to(tl.float16))`,
    },
    {
      id: 'fused-lora-add',
      name: 'Fused LoRA Update + Cast',
      priority: '🟠 Высокий',
      time: '~10% времени (часть из 44% element-wise)',
      description: 'Объединяет: output = base_output + (alpha/r) * (x @ lora_A @ lora_B). Убирает промежуточные тензоры и лишние casts между fp32/fp16.',
      code: `import torch
import triton
import triton.language as tl

@triton.jit
def fused_lora_add_kernel(
    base_ptr,       # [M, N] — основной выход linear
    lora_a_ptr,     # [r, K] — LoRA A (K = hidden, r = rank)
    lora_b_ptr,     # [N, r] — LoRA B  
    x_ptr,          # [M, K] — вход
    out_ptr,        # [M, N] — результат
    M, N, K,
    stride_bm, stride_bn,
    stride_xm, stride_xk,
    stride_ar, stride_ak,
    stride_bn_lora, stride_br,
    alpha_over_r: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    # Загрузить base output
    base = tl.load(
        base_ptr + m_offsets[:, None] * stride_bm + n_offsets[None, :] * stride_bn,
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0
    ).to(tl.float32)
    
    # Считаем lora: x @ A @ B * (alpha/r)
    lora_out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for r_start in range(0, BLOCK_R, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)
        r_mask = r_offsets < BLOCK_R
        
        # x @ lora_A -> [M, r]
        xa = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for k in range(K):
            x_col = tl.load(
                x_ptr + m_offsets[:, None] * stride_xm + k * stride_xk,
                mask=m_mask[:, None], other=0.0
            )
            a_col = tl.load(
                lora_a_ptr + r_offsets[None, :] * stride_ar + k * stride_ak,
                mask=r_mask[None, :], other=0.0
            )
            xa += x_col * a_col
        
        # (x @ A) @ lora_B -> [M, N]
        for r_idx in range(BLOCK_R):
            b_col = tl.load(
                lora_b_ptr + n_offsets[None, :] * stride_bn_lora + (r_start + r_idx) * stride_br,
                mask=n_mask[None, :], other=0.0
            )
            lora_out += xa[:, r_idx:r_idx+1] * b_col
    
    result = base + lora_out * alpha_over_r
    tl.store(
        out_ptr + m_offsets[:, None] * stride_bm + n_offsets[None, :] * stride_bn,
        result.to(tl.float16),
        mask=m_mask[:, None] & n_mask[None, :]
    )`,
    },
    {
      id: 'fused-embedding-cast',
      name: 'Fused Embedding + FP32→FP16 Cast',
      priority: '🟡 Средний',
      time: '~5% времени',
      description: 'Soup хранит embedding в fp32 и кастует перед каждым слоем. Этот кернел сразу выдаёт fp16, убирая лишний проход по памяти.',
      code: `import torch
import triton
import triton.language as tl

@triton.jit
def fused_embedding_cast_kernel(
    weight_ptr,     # [vocab_size, hidden_dim] в fp32
    ids_ptr,        # [batch, seq_len] int64
    out_ptr,        # [batch, seq_len, hidden_dim] в fp16
    n_tokens,
    hidden_dim: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    if token_id >= n_tokens:
        return
    
    idx = tl.load(ids_ptr + token_id)
    
    # Загрузить строку embedding и сразу кастовать в fp16
    offsets = tl.arange(0, hidden_dim)
    emb = tl.load(weight_ptr + idx * hidden_dim + offsets).to(tl.float16)
    
    tl.store(out_ptr + token_id * hidden_dim + offsets, emb)`,
    },
    {
      id: 'fused-rotary',
      name: 'Fused Rotary Position Embedding',
      priority: '🟡 Средний',
      time: '~3-5% времени',
      description: 'Объединяет применение RoPE к Q и K в один кернел с учётом cos/sin таблиц. Убирает промежуточные тензоры.',
      code: `import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_rope_kernel(
    q_ptr,          # [B, H, S, D]
    k_ptr,          # [B, H, S, D]
    cos_ptr,        # [S, D]
    sin_ptr,        # [S, D]
    q_out_ptr,
    k_out_ptr,
    B, H, S,
    D: tl.constexpr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    if pid_b >= B or pid_h >= H or pid_s >= S:
        return
    
    half_d = D // 2
    offsets = tl.arange(0, half_d)
    
    # Загрузить q и k
    q_base = pid_b * stride_qb + pid_h * stride_qh + pid_s * stride_qs
    k_base = pid_b * stride_kb + pid_h * stride_kh + pid_s * stride_ks
    
    q1 = tl.load(q_ptr + q_base + offsets)
    q2 = tl.load(q_ptr + q_base + half_d + offsets)
    k1 = tl.load(k_ptr + k_base + offsets)
    k2 = tl.load(k_ptr + k_base + half_d + offsets)
    
    # Загрузить cos, sin
    cos = tl.load(cos_ptr + pid_s * D + offsets)
    sin = tl.load(sin_ptr + pid_s * D + offsets)
    
    # Применить RoPE
    q_out1 = q1 * cos - q2 * sin
    q_out2 = q1 * sin + q2 * cos
    k_out1 = k1 * cos - k2 * sin
    k_out2 = k1 * sin + k2 * cos
    
    # Сохранить
    tl.store(q_out_ptr + q_base + offsets, q_out1)
    tl.store(q_out_ptr + q_base + half_d + offsets, q_out2)
    tl.store(k_out_ptr + k_base + offsets, k_out1)
    tl.store(k_out_ptr + k_base + half_d + offsets, k_out2)`,
    },
  ]

  return (
    <div className="space-y-4">
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-2">⚡ Triton-кернелы для решения</h2>
        <p className="text-sm text-gray-400 mb-4">
          Приоритет по потенциальному выигрышу. Каждый кернел должен занимать ≥5% времени обучения суммарно.
        </p>
      </div>

      {kernels.map((kernel) => (
        <div key={kernel.id} className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          <button
            onClick={() => setExpandedKernel(expandedKernel === kernel.id ? null : kernel.id)}
            className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-gray-800/50 transition-colors"
          >
            <div>
              <div className="flex items-center gap-3">
                <h3 className="text-base font-semibold text-white">{kernel.name}</h3>
                <span className="text-xs">{kernel.priority}</span>
              </div>
              <p className="text-sm text-gray-400 mt-1">{kernel.time}</p>
            </div>
            <span className={`text-gray-400 transition-transform ${expandedKernel === kernel.id ? 'rotate-180' : ''}`}>
              ▼
            </span>
          </button>
          {expandedKernel === kernel.id && (
            <div className="px-6 pb-6 border-t border-gray-800">
              <p className="text-sm text-gray-300 mt-4 mb-4">{kernel.description}</p>
              <div className="bg-gray-950 rounded-lg p-4 overflow-x-auto">
                <pre className="text-xs text-gray-300 font-mono whitespace-pre">{kernel.code}</pre>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function StructureTab() {
  return (
    <div className="space-y-6">
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-4">📁 Структура решения</h2>
        <div className="bg-gray-950 rounded-lg p-4 font-mono text-sm">
          <div className="text-gray-500"># Репозиторий команды</div>
          <div className="text-green-400">├── solution/</div>
          <div className="text-green-400">│   ├── __init__.py          <span className="text-gray-500"># apply() функция</span></div>
          <div className="text-green-400">│   ├── fused_cross_entropy.py <span className="text-gray-500"># Fused CE loss</span></div>
          <div className="text-green-400">│   ├── fused_rmsnorm_silu.py  <span className="text-gray-500"># Fused norm + activation</span></div>
          <div className="text-green-400">│   ├── fused_lora.py          <span className="text-gray-500"># Fused LoRA update</span></div>
          <div className="text-green-400">│   ├── fused_embedding.py     <span className="text-gray-500"># Embedding + cast</span></div>
          <div className="text-green-400">│   └── fused_rope.py          <span className="text-gray-500"># Rotary embeddings</span></div>
          <div className="text-blue-400">├── task/</div>
          <div className="text-blue-400">│   └── task.yaml              <span className="text-gray-500"># Конфиг (не менять!)</span></div>
          <div className="text-blue-400">├── harness/                   <span className="text-gray-500"># Скрипты замера (не менять!)</span></div>
          <div className="text-blue-400">├── tests/                     <span className="text-gray-500"># Тесты Soup (не менять!)</span></div>
          <div className="text-blue-400">├── templates/</div>
          <div className="text-blue-400">│   └── kaggle_notebook.ipynb</div>
          <div className="text-yellow-400">├── README.md                <span className="text-gray-500"># Обязательно!</span></div>
          <div className="text-yellow-400">└── SUBMISSION.md            <span className="text-gray-500"># Ссылки на коммит и ноутбук</span></div>
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">🔌 solution/__init__.py — apply()</h3>
        <div className="bg-gray-950 rounded-lg p-4 overflow-x-auto">
          <pre className="text-xs text-gray-300 font-mono whitespace-pre">{`"""
Soup Kernel Challenge solution.
Plugs custom Triton kernels into Soup training.
"""

from .fused_cross_entropy import FusedCrossEntropyLoss
from .fused_rmsnorm_silu import fused_rmsnorm_silu
from .fused_lora import fused_lora_forward
from .fused_embedding import FusedEmbeddingCast
from .fused_rope import fused_apply_rotary


def apply():
    """
    Plug all custom kernels into Soup.
    Called once before training starts.
    """
    import soup
    
    # 1. Replace cross-entropy loss with fused version
    #    This saves massive memory (no [B*S*152K] logits)
    soup.training.loss.replace_loss(FusedCrossEntropyLoss())
    
    # 2. Monkey-patch RMSNorm + SiLU fusion in FFN
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    _original_forward = Qwen2MLP.forward
    def patched_mlp_forward(self, x):
        # Fuse down_proj's RMSNorm + SiLU
        return fused_rmsnorm_silu(x, self, _original_forward)
    Qwen2MLP.forward = patched_mlp_forward
    
    # 3. Patch LoRA forward to use fused kernel
    from peft.tuners.lora import LoraLayer
    _original_lora = LoraLayer.forward
    def patched_lora_forward(self, x, *args, **kwargs):
        return fused_lora_forward(self, x, _original_lora, *args, **kwargs)
    LoraLayer.forward = patched_lora_forward
    
    # 4. Replace embedding with fused cast version
    soup.training.embeddings.replace(FusedEmbeddingCast)
    
    # 5. Patch rotary embeddings
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
    qwen2_mod.apply_rotary_pos_emb = fused_apply_rotary
    
    print("[Soup Kernels] All kernels applied successfully")
    print("[Soup Kernels] - Fused Cross-Entropy (memory + speed)")
    print("[Soup Kernels] - Fused RMSNorm+SiLU (speed)")
    print("[Soup Kernels] - Fused LoRA (speed)")
    print("[Soup Kernels] - Fused Embedding cast (speed)")
    print("[Soup Kernels] - Fused RoPE (speed)")`}</pre>
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">🐳 Kaggle Notebook порядок</h3>
        <div className="space-y-3">
          {[
            { step: 1, title: 'Install dependencies', code: '!pip install soup-cli[train]==0.77.0' },
            { step: 2, title: 'Clone solution repo', code: '!git clone https://github.com/YOUR_TEAM/repo.git && cd repo && git checkout COMMIT' },
            { step: 3, title: 'Import and apply kernels', code: 'import sys; sys.path.insert(0, "repo")\nfrom solution import apply\napply()' },
            { step: 4, title: 'Run training + measurement', code: '!python repo/harness/run_task.py' },
          ].map((item) => (
            <div key={item.step} className="flex gap-3 items-start">
              <div className="w-8 h-8 rounded-full bg-orange-600 flex items-center justify-center text-sm font-bold shrink-0">
                {item.step}
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-white">{item.title}</p>
                <code className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded mt-1 inline-block whitespace-pre-wrap">
                  {item.code}
                </code>
              </div>
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
      title: 'Must Have (7. Must have)',
      items: [
        { id: 'run-kaggle', text: 'Решение запускается в чистом Kaggle-ноутбуке на T4 без правок' },
        { id: 'triton-kernels', text: 'Свои @triton.jit кернелы из solution/ работают и ≥5% времени обучения' },
        { id: 'no-worse', text: 'Loss не хуже базы на 0.5%; время и память не хуже на 10%' },
        { id: 'same-task', text: 'Та же задача: токены, энтропия, точность, rewind в пределах 3%' },
        { id: 'tests-pass', text: '79 тестов Soup проходят с патчами' },
        { id: 'readme', text: 'README: что сделано, как запустить, результаты, ссылка на ноутбук' },
      ],
    },
    {
      title: 'Жёсткие ограничения (8)',
      items: [
        { id: 'no-libs', text: 'НЕ используются unsloth/liger/xformers/flash-attn/torchao' },
        { id: 'no-modes', text: 'НЕ включены backend:unsloth, use_liger, fast_lora, packing и т.д.' },
        { id: 'no-task-change', text: 'НЕ изменены модель, данные, batch, шаги, LoRA, optimizer' },
        { id: 'no-harness-edit', text: 'НЕ изменены harness/, tests/, task/' },
        { id: 'no-cheat', text: 'НЕ обходим замер: одна GPU, нет фоновых процессов' },
        { id: 'open-code', text: 'Репозиторий и ноутбук открыты для жюри' },
      ],
    },
    {
      title: 'Сдача (10)',
      items: [
        { id: 'repo', text: 'GitHub-репозиторий команды с solution/, README, SUBMISSION.md' },
        { id: 'notebook', text: 'Публичный Kaggle-ноутбук из шаблона с результатами' },
        { id: 'video', text: 'Демо-видео до 3 минут' },
        { id: 'presentation', text: 'Презентация: что ускоряли, как устроен кернел, результаты' },
        { id: 'submission-md', text: 'SUBMISSION.md: ссылка на коммит + ссылка на ноутбук' },
      ],
    },
    {
      title: 'README (11.1)',
      items: [
        { id: 'readme-what', text: 'Что сделано: какие операции, как устроены кернелы, источники идей' },
        { id: 'readme-how', text: 'Как запустить: шаги, ссылки на Kaggle-ноутбук и финальный коммит' },
        { id: 'readme-results', text: 'Результаты: таблица vs база (время, память, loss) + ограничения' },
      ],
    },
  ]

  const totalItems = sections.reduce((acc, s) => acc + s.items.length, 0)
  const checkedItems = Object.values(checks).filter(Boolean).length

  return (
    <div className="space-y-6">
      {/* Progress */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xl font-bold text-white">✅ Чеклист готовности</h2>
          <span className="text-lg font-mono text-orange-400">{checkedItems}/{totalItems}</span>
        </div>
        <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-orange-500 to-green-500 rounded-full transition-all duration-300"
            style={{ width: `${(checkedItems / totalItems) * 100}%` }}
          />
        </div>
      </div>

      {sections.map((section) => (
        <div key={section.title} className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-base font-semibold text-white mb-4">{section.title}</h3>
          <div className="space-y-2">
            {section.items.map((item) => (
              <label
                key={item.id}
                className="flex items-start gap-3 p-2 rounded-lg hover:bg-gray-800/50 cursor-pointer transition-colors"
              >
                <input
                  type="checkbox"
                  checked={checks[item.id] || false}
                  onChange={() => toggle(item.id)}
                  className="mt-0.5 w-5 h-5 rounded border-gray-600 bg-gray-800 text-orange-500 focus:ring-orange-500 focus:ring-offset-0"
                />
                <span className={`text-sm ${checks[item.id] ? 'text-gray-500 line-through' : 'text-gray-300'}`}>
                  {item.text}
                </span>
              </label>
            ))}
          </div>
        </div>
      ))}

      {/* Timeline */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-base font-semibold text-white mb-4">⏰ Таймлайн (36 часов)</h3>
        <div className="space-y-4">
          {[
            { time: '0-4ч', task: 'Разобраться со стартовым репозиторием, запустить базу, изучить профиль', color: 'bg-blue-500' },
            { time: '4-12ч', task: 'Написать Fused Cross-Entropy (главный выигрыш по памяти)', color: 'bg-orange-500' },
            { time: '12-20ч', task: 'Fused RMSNorm+SiLU, Fused LoRA, Fused Embedding', color: 'bg-orange-500' },
            { time: '20-28ч', task: 'Тестирование, отладка, прогон всех 79 тестов', color: 'bg-yellow-500' },
            { time: '28-32ч', task: 'Замеры на Kaggle, оптимизация BLOCK_SIZE, tuning', color: 'bg-green-500' },
            { time: '32-36ч', task: 'README, видео, презентация, SUBMISSION.md, финальный пуш', color: 'bg-purple-500' },
          ].map((item) => (
            <div key={item.time} className="flex gap-4 items-start">
              <div className="w-16 text-xs font-mono text-gray-400 shrink-0 pt-0.5">{item.time}</div>
              <div className={`w-3 h-3 rounded-full ${item.color} shrink-0 mt-1`} />
              <p className="text-sm text-gray-300">{item.task}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default App
