"""Train -> backtest -> metrics pipeline (CLI).

Provides a packaged replacement for legacy pipeline scripts. Intended for
smoke/regression runs and headless experiments.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _load_reward_config(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("reward_config must be a JSON object")
    return data


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train -> backtest -> metrics pipeline.")

    parser.add_argument("--model_name", type=str, default="", help="Optional model name override.")
    parser.add_argument("--stock_code", type=str, default="510050.SH", help="Single asset code.")
    parser.add_argument(
        "--stock_pool", type=str, default="", help="Optional stock pool key (e.g. sse50)."
    )
    parser.add_argument(
        "--start_date", type=str, required=True, help="Training start date (YYYYMMDD)."
    )
    parser.add_argument("--end_date", type=str, required=True, help="Training end date (YYYYMMDD).")
    parser.add_argument(
        "--episodes", type=int, default=2, help="Training episodes (keep small for smoke)."
    )
    parser.add_argument("--window_size", type=int, default=30, help="Override window size.")
    parser.add_argument(
        "--learning_frequency", type=int, default=1024, help="PPO n_steps / update frequency."
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        choices=["multiscale_cnn", "transformer", "cross_attention"],
        default="multiscale_cnn",
        help="Policy backbone.",
    )
    parser.add_argument("--commission_rate", type=float, default=0.0003)
    parser.add_argument("--slippage", type=float, default=0.0001)
    parser.add_argument(
        "--sentiment_dataset", type=str, default="", help="Optional sentiment JSONL path."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--reward_config", type=Path, default=None, help="Optional reward config JSON file."
    )

    parser.add_argument(
        "--backtest_start_date", type=str, required=True, help="Backtest start date (YYYYMMDD)."
    )
    parser.add_argument(
        "--backtest_end_date", type=str, required=True, help="Backtest end date (YYYYMMDD)."
    )
    parser.add_argument(
        "--output_label",
        type=str,
        default="pipeline",
        help="Label used in report filename (default: pipeline).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the JSON report (defaults to reports directory).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Delay heavy imports so `--help` works in minimal environments.
    from virgo_trader.backtest.runner import build_sim_params
    from virgo_trader.data.stock_pools import get_stock_pool_codes, get_stock_pool_label
    from virgo_trader.simulation.sim_trader import run_simulation_for_worker
    from virgo_trader.train import train
    from virgo_trader.utils.paths import (
        REPORTS_DIR,
        ensure_dir,
        migrate_legacy_files,
        resolve_model_zip_path,
    )
    from virgo_trader.utils.performance_metrics import calculate_performance_metrics

    migrate_legacy_files()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name.strip() or f"pipeline_{args.agent_type}_{timestamp}"

    stock_pool_key = args.stock_pool.strip() or None
    if stock_pool_key:
        codes = list(get_stock_pool_codes(stock_pool_key))
        stock_pool_label = get_stock_pool_label(stock_pool_key)
    else:
        codes = [args.stock_code.strip()]
        stock_pool_label = None

    train_params: Dict[str, Any] = {
        "model_name": model_name,
        "stock_pool": codes,
        "stock_pool_key": stock_pool_key,
        "stock_pool_label": stock_pool_label,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "commission_rate": float(args.commission_rate),
        "slippage": float(args.slippage),
        "episodes": int(args.episodes),
        "learning_frequency": max(0, int(args.learning_frequency)),
        "window_size": max(0, int(args.window_size)),
        "agent_type": str(args.agent_type).lower(),
    }
    if args.seed is not None:
        train_params["seed"] = int(args.seed)
    if args.sentiment_dataset.strip():
        train_params["sentiment_dataset"] = args.sentiment_dataset.strip()
    if args.reward_config is not None:
        train_params["reward_config"] = _load_reward_config(args.reward_config)

    print(f"[1/2] Training: {model_name}")
    train(train_params)

    model_path = resolve_model_zip_path(model_name)
    print(f"[2/2] Backtesting: {model_path}")

    sim_params = build_sim_params(
        model_name=model_name,
        model_path=model_path,
        start_date=args.backtest_start_date,
        end_date=args.backtest_end_date,
        train_params=train_params,
        sentiment_dataset=args.sentiment_dataset.strip() or None,
        seed=args.seed,
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

    payload = {
        "model_name": model_name,
        "model_path": str(model_path),
        "train_window": {"start": args.start_date, "end": args.end_date},
        "backtest_window": {"start": args.backtest_start_date, "end": args.backtest_end_date},
        "stock_pool": codes,
        "stock_pool_key": stock_pool_key,
        "sentiment_dataset": args.sentiment_dataset.strip() or None,
        "seed": args.seed,
        "metrics": metrics,
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(args.out))
        return 0

    ensure_dir(REPORTS_DIR)
    report_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"pipeline_{args.output_label}_{model_name}_{report_ts}.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
