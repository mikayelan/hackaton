"""
Soup Kernel Challenge — Decentrathon 6.0
Custom Triton kernels for accelerating Soup training on Kaggle T4.

Usage:
    from solution import apply
    apply()
"""

__version__ = "1.0.0"

from .apply_kernels import apply

__all__ = ["apply"]
