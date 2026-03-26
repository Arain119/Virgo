"""Training workflows for Virgo Trader.

This package contains the training pipeline used to build PPO-based trading agents.
"""

from __future__ import annotations

from typing import Any

__all__ = ["main", "train"]


def __getattr__(name: str) -> Any:
    """Lazy attribute loading to avoid importing heavy ML dependencies at package import time."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .pipeline import main, train

    return main if name == "main" else train
