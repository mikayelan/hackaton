"""
utils.py — Общие утилиты: замер времени кернелов через CUDA events,
autotune-конфиги, проверка доступности GPU.
"""

import torch
import time
from contextlib import contextmanager
from typing import Optional


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------

def check_gpu() -> dict:
    """Return basic GPU info for logging."""
    if not torch.cuda.is_available():
        return {"available": False}
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": round(props.total_mem / 1e9, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "device": dev,
    }


# ---------------------------------------------------------------------------
# Kernel timing via CUDA events
# ---------------------------------------------------------------------------

class KernelTimer:
    """
    Accumulates GPU time spent inside custom Triton kernels.
    Uses CUDA events for accurate measurement.
    
    Usage:
        timer = KernelTimer()
        with timer.record("fused_ce"):
            output = fused_ce_forward(...)
        print(timer.summary())
    """
    
    def __init__(self):
        self._records: dict[str, list[float]] = {}
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
        self._current_key: Optional[str] = None
    
    @contextmanager
    def record(self, key: str):
        """Time a block of GPU work."""
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
            
            if key not in self._records:
                self._records[key] = []
            self._records[key].append(elapsed_ms)
    
    def total_ms(self, key: Optional[str] = None) -> float:
        """Total milliseconds for a kernel (or all kernels)."""
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
        return "\n".join(lines)
    
    def reset(self):
        self._records.clear()


# Global timer instance
TIMER = KernelTimer()


# ---------------------------------------------------------------------------
# Autotune helpers
# ---------------------------------------------------------------------------

def get_block_sizes(total: int, max_block: int = 4096) -> int:
    """Pick a reasonable block size that divides evenly or is close."""
    for bs in [max_block, 2048, 1024, 512, 256, 128]:
        if total >= bs:
            return bs
    return min(total, 64)


def next_power_of_2(n: int) -> int:
    """Round up to next power of 2."""
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def safe_fp16(x: torch.Tensor) -> torch.Tensor:
    """Cast to fp16 if not already."""
    if x.dtype != torch.float16:
        return x.to(torch.float16)
    return x


def safe_fp32(x: torch.Tensor) -> torch.Tensor:
    """Cast to fp32 if not already."""
    if x.dtype != torch.float32:
        return x.to(torch.float32)
    return x
