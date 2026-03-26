"""Training entrypoint module.

The core training implementation lives under `virgo_trader.training.pipeline`.
This wrapper preserves a stable module path for subprocess invocations
(e.g. `python -m virgo_trader.train`).
"""

from __future__ import annotations

from typing import Any, Dict


def train(
    train_params: Dict[str, Any], dashboard=None, progress_callback=None, stop_flag_func=None
):
    """Train a PPO agent using the configured trading environment."""
    from .training.pipeline import train as _train

    return _train(
        train_params,
        dashboard=dashboard,
        progress_callback=progress_callback,
        stop_flag_func=stop_flag_func,
    )


def main() -> None:
    """Console script entrypoint for training (used by `virgo-trader-train`)."""
    from .training.pipeline import main as _main

    _main()


__all__ = ["main", "train"]


if __name__ == "__main__":
    main()
