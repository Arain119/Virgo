"""Backtest orchestration helpers.

The lower-level simulation engine lives in `virgo_trader.simulation`. This
package provides a thin application-facing API for running backtests and
producing JSON-ready outputs (metrics, trades, equity curve).
"""

from .runner import run_backtest

__all__ = ["run_backtest"]
