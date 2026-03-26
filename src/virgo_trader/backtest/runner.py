"""Backtest runner utilities.

This module provides a stable, UI/CLI-friendly backtest entrypoint that builds
simulation parameters, runs `run_simulation_for_worker`, computes performance
metrics, and returns JSON-serializable outputs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..data.stock_pools import get_stock_pool_codes
from ..simulation.sim_trader import run_simulation_for_worker
from ..utils.database_manager import get_sessions_for_model
from ..utils.paths import REPORTS_DIR, ensure_dir, migrate_legacy_files, resolve_model_zip_path
from ..utils.performance_metrics import calculate_performance_metrics
from ..utils.sb3_compat import infer_ppo_model_spaces

SSE50_BASELINE_SYMBOL = "510050.SH"


def _infer_agent_type(model_name: str, train_params: Dict[str, Any]) -> str:
    configured = str(train_params.get("agent_type") or "").strip().lower()
    if configured:
        return configured

    upper = model_name.upper()
    if "TRANSFORMER" in upper:
        return "transformer"
    if "CROSS_ATTENTION" in upper:
        return "cross_attention"
    if "MULTISCALE_CNN" in upper:
        return "multiscale_cnn"
    return "multiscale_cnn"


def _resolve_stock_pool(model_name: str, train_params: Dict[str, Any]) -> list[str]:
    stock_pool_source = train_params.get("resolved_stock_pool") or train_params.get("stock_pool")
    pool = [str(code).strip() for code in (stock_pool_source or []) if str(code).strip()]
    if pool:
        return pool

    upper = model_name.upper()
    if "SSE50_POOL" in upper:
        # The SSE50 pool registry includes 510050.SH as an extra keyword. Models trained on
        # constituents expect only the 50 stock codes.
        return [
            code
            for code in get_stock_pool_codes("sse50", use_live=False)
            if code != SSE50_BASELINE_SYMBOL
        ]

    fallback = str(train_params.get("stock_code") or SSE50_BASELINE_SYMBOL).strip()
    return [fallback] if fallback else [SSE50_BASELINE_SYMBOL]


def _serialize_trades(trades_df: pd.DataFrame) -> list[dict[str, Any]]:
    if trades_df is None or getattr(trades_df, "empty", True):
        return []
    out = trades_df.reset_index()
    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].astype(str)
    return out.to_dict(orient="records")


def _serialize_portfolio(series: pd.Series) -> dict[str, float]:
    if series is None or getattr(series, "empty", True):
        return {}
    raw = series.to_dict()
    return {str(k): float(v) for k, v in raw.items()}


def _load_latest_train_params(model_name: str) -> Dict[str, Any]:
    sessions = get_sessions_for_model(model_name)
    if not sessions:
        return {}
    sessions_sorted = sorted(sessions, key=lambda s: s.get("start_time", ""), reverse=True)
    return json.loads(sessions_sorted[0].get("train_parameters") or "{}")


def build_sim_params(
    *,
    model_name: str,
    model_path: Path,
    start_date: str,
    end_date: str,
    train_params: Optional[Dict[str, Any]] = None,
    sentiment_dataset: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build `sim_params` accepted by `run_simulation_for_worker`."""
    params = dict(train_params or {})
    stock_pool = _resolve_stock_pool(model_name, params)

    window_size = int(params.get("window_size") or 0)
    if window_size <= 0:
        window_size = int(infer_ppo_model_spaces(model_path).window_size)

    agent_type = _infer_agent_type(model_name, params)
    commission = float(params.get("commission_rate", 0.0003))
    slippage = float(params.get("slippage", 0.0001))
    base_symbol = SSE50_BASELINE_SYMBOL if len(stock_pool) > 1 else stock_pool[0]

    sim_params: Dict[str, Any] = {
        "stock_code": stock_pool[0],
        "stock_pool": stock_pool,
        "base_symbol": base_symbol,
        "start_date": start_date,
        "end_date": end_date,
        "commission_rate": commission,
        "slippage": slippage,
        "window_size": window_size,
        "model_name": model_name,
        "model_path": str(model_path),
        "agent_type": agent_type,
    }
    dataset = (sentiment_dataset or "").strip()
    if dataset:
        sim_params["sentiment_dataset"] = dataset
    if seed is not None:
        sim_params["seed"] = int(seed)
    return sim_params


def run_backtest(
    *,
    model_name: str,
    model_path: Optional[Path] = None,
    start_date: str,
    end_date: str,
    output_label: str = "backtest",
    sentiment_dataset: Optional[str] = None,
    seed: Optional[int] = None,
    write_report: bool = True,
) -> Dict[str, Any]:
    """Run a backtest and return a JSON-ready payload."""
    migrate_legacy_files()

    resolved_path = model_path or resolve_model_zip_path(model_name)
    if not resolved_path.exists():
        raise FileNotFoundError(resolved_path)
    model_name = model_name or resolved_path.stem

    train_params = _load_latest_train_params(model_name)
    if not sentiment_dataset:
        sentiment_dataset = (
            str(
                train_params.get("sentiment_dataset")
                or train_params.get("sentiment_dataset_path")
                or train_params.get("sentiment_data_path")
                or ""
            ).strip()
            or None
        )

    sim_params = build_sim_params(
        model_name=model_name,
        model_path=resolved_path,
        start_date=start_date,
        end_date=end_date,
        train_params=train_params,
        sentiment_dataset=sentiment_dataset,
        seed=seed,
    )

    results = run_simulation_for_worker(sim_params)
    portfolio_series = results["portfolio_history"]
    trades_df = results["trades"]
    benchmark_df = results.get("base_kline")
    if benchmark_df is None or (hasattr(benchmark_df, "empty") and benchmark_df.empty):
        benchmark_df = results.get("kline_data")

    metrics = calculate_performance_metrics(
        portfolio_history=portfolio_series,
        trades=trades_df,
        kline_data=benchmark_df,
    )

    payload: Dict[str, Any] = {
        "model_name": model_name,
        "model_path": str(resolved_path),
        "backtest_window": {"start": start_date, "end": end_date},
        "sentiment_dataset": sentiment_dataset,
        "stock_pool": results.get("stock_pool") or sim_params.get("stock_pool", []),
        "metrics": metrics,
        "trades": _serialize_trades(trades_df),
        "portfolio_history": _serialize_portfolio(portfolio_series),
    }

    if write_report:
        ensure_dir(REPORTS_DIR)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"backtest_{output_label}_{model_name}_{timestamp}.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["report_path"] = str(report_path)

    return payload
