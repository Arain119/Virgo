"""Simulation / backtest runner for trained trading agents.

Prepares single-asset and multi-asset environments from market datasets, loads
trained PPO models, and runs rollouts to produce performance summaries.
"""

import logging
from pathlib import Path

import pandas as pd

from ..data.data_fetcher import get_stock_data
from ..data.dataset_builder import attach_sentiment_features, load_sentiment_loader
from ..data.feature_engineer import FeatureEngineer
from ..environment.multi_asset_env import MultiAssetTradingEnv
from ..environment.trading_env import TradingEnv
from ..utils.paths import resolve_model_zip_path
from ..utils.sb3_compat import infer_ppo_model_spaces, load_ppo_model


def _pad_feature_frame(processed_df: pd.DataFrame, *, target_dim: int) -> pd.DataFrame:
    """
    Pad a feature-engineered dataframe with zero-valued placeholder columns.

    This is used during backtesting when the model expects sentiment features but
    the sentiment dataset is unavailable. We preserve observation dimensionality
    by appending `sentiment_pad_*` columns (so envs keep sentiment columns at the
    end of the feature vector).
    """
    if processed_df is None or processed_df.empty:
        return processed_df
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")

    current_dim = int(processed_df.shape[1])
    if current_dim == target_dim:
        return processed_df
    if current_dim > target_dim:
        raise ValueError(f"Feature dim mismatch: current={current_dim} expected={target_dim}")

    missing = int(target_dim - current_dim)
    out = processed_df.copy()
    for idx in range(missing):
        col = f"sentiment_pad_{idx:03d}"
        if col in out.columns:
            continue
        out[col] = 0.0
    return out


def _prepare_single_asset_env(
    stock_code,
    start_date,
    end_date,
    window_size,
    commission_rate,
    slippage,
    use_calendar_features=True,
    sentiment_loader=None,
    sentiment_columns=None,
    expected_feature_dim: int | None = None,
):
    start_buffer = pd.to_datetime(start_date) - pd.DateOffset(days=window_size + 50)
    raw_data = get_stock_data(stock_code, start_buffer.strftime("%Y%m%d"), end_date)
    if raw_data.empty:
        raise ValueError(f"无法获取模拟数据: {stock_code}")
    feature_engineer = FeatureEngineer(use_calendar_features=use_calendar_features)
    processed_data = feature_engineer.process(raw_data.copy())
    processed_data = attach_sentiment_features(
        processed_df=processed_data,
        symbol=stock_code,
        sentiment_loader=sentiment_loader,
        sentiment_columns=sentiment_columns or [],
    )
    if expected_feature_dim is not None:
        processed_data = _pad_feature_frame(processed_data, target_dim=int(expected_feature_dim))
    sim_start_date_dt = pd.to_datetime(start_date)
    sim_raw_data = raw_data[raw_data.index >= sim_start_date_dt]
    sim_processed_data = processed_data.loc[sim_raw_data.index]
    env = TradingEnv(
        df=sim_processed_data,
        raw_df=sim_raw_data,
        window_size=window_size,
        commission_rate=commission_rate,
        slippage=slippage,
    )
    return env, sim_raw_data


def _prepare_multi_asset_env(
    stock_codes,
    start_date,
    end_date,
    window_size,
    commission_rate,
    slippage,
    use_calendar_features=True,
    sentiment_loader=None,
    sentiment_columns=None,
    expected_num_assets: int | None = None,
    expected_per_symbol_feature_dim: int | None = None,
):
    start_buffer = pd.to_datetime(start_date) - pd.DateOffset(days=window_size + 50)
    feature_engineer = FeatureEngineer(use_calendar_features=use_calendar_features)
    datasets = []
    skipped: list[tuple[str, str]] = []
    for code in stock_codes:
        try:
            raw = get_stock_data(code, start_buffer.strftime("%Y%m%d"), end_date)
        except Exception as exc:
            skipped.append((code, str(exc)))
            continue
        if raw.empty:
            skipped.append((code, "empty"))
            continue
        processed = feature_engineer.process(raw.copy())
        processed = attach_sentiment_features(
            processed_df=processed,
            symbol=code,
            sentiment_loader=sentiment_loader,
            sentiment_columns=sentiment_columns or [],
        )
        if expected_per_symbol_feature_dim is not None:
            processed = _pad_feature_frame(
                processed, target_dim=int(expected_per_symbol_feature_dim)
            )
        sim_start_date_dt = pd.to_datetime(start_date)
        sim_raw = raw[raw.index >= sim_start_date_dt]
        sim_processed = processed.loc[sim_raw.index]
        datasets.append({"symbol": code, "raw_df": sim_raw, "df": sim_processed})
        if expected_num_assets is not None and len(datasets) >= expected_num_assets:
            break

    if expected_num_assets is not None and len(datasets) != expected_num_assets:
        raise ValueError(
            f"股票池数据不足：期望 {expected_num_assets} 支标的，实际拿到 {len(datasets)} 支。"
            f"（可能是成分股变更/新股在回测区间无数据） skipped={skipped[:10]}{'...' if len(skipped) > 10 else ''}"
        )
    if not datasets:
        raise ValueError(f"无法获取模拟数据（全部标的失败）。skipped={skipped[:10]}")

    env = MultiAssetTradingEnv(
        datasets=datasets,
        window_size=window_size,
        commission_rate=commission_rate,
        slippage=slippage,
    )
    symbol_kline_map = {item["symbol"]: item["raw_df"] for item in datasets}
    used_symbols = [item["symbol"] for item in datasets]
    return env, symbol_kline_map, used_symbols


def run_simulation_for_worker(sim_params, status_callback=None):
    stock_code = sim_params["stock_code"]
    stock_pool = sim_params.get("stock_pool") or [stock_code]
    base_symbol = sim_params.get("base_symbol") or stock_code
    start_date = sim_params["start_date"]
    end_date = sim_params["end_date"]
    commission_rate = sim_params["commission_rate"]
    slippage = sim_params["slippage"]
    window_size_raw = sim_params.get("window_size")
    model_name = sim_params["model_name"]
    model_path_override = sim_params.get("model_path")
    sentiment_dataset_path = sim_params.get("sentiment_dataset")
    seed_raw = sim_params.get("seed")
    model_path = (
        Path(model_path_override) if model_path_override else resolve_model_zip_path(model_name)
    )
    agent_type = sim_params.get("agent_type", "multiscale_cnn").lower()
    use_calendar_features = agent_type != "multiscale_cnn"

    seed_value = None
    try:
        if seed_raw is not None and str(seed_raw).strip() != "":
            seed_value = int(seed_raw)
    except (TypeError, ValueError):
        seed_value = None

    sentiment_loader, sentiment_columns = load_sentiment_loader(sentiment_dataset_path)
    spaces_info = infer_ppo_model_spaces(model_path)
    inferred_window_size = int(spaces_info.window_size)
    try:
        window_size = int(window_size_raw) if window_size_raw else inferred_window_size
    except (TypeError, ValueError):
        window_size = inferred_window_size
    if window_size != inferred_window_size:
        logging.warning(
            "window_size mismatch for %s: provided=%s inferred=%s. Using inferred.",
            model_path.name,
            window_size,
            inferred_window_size,
        )
        window_size = inferred_window_size
    if sentiment_loader and sentiment_columns and status_callback:
        status_callback(
            f"已加载情绪特征: {sentiment_dataset_path} (features={len(sentiment_columns)})"
        )

    try:
        model = load_ppo_model(model_path, device="cpu")
        if status_callback:
            status_callback("已加载训练好的模型。")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"模型文件未找到: {model_path}。请先训练模型。") from exc

    expected_assets = int(spaces_info.action_dim or 1)

    if status_callback:
        status_callback(f"开始为 {model_name} 回测（标的: {', '.join(stock_pool)}）...")

    benchmark_series = None
    actual_pool = list(stock_pool)
    if expected_assets > 1:
        expected_aux_dim = 2 * expected_assets + 1
        expected_market_dim = int(spaces_info.obs_dim - expected_aux_dim)
        if expected_market_dim <= 0:
            raise ValueError(
                f"Invalid inferred obs_dim={spaces_info.obs_dim} for assets={expected_assets}."
            )
        if expected_market_dim % expected_assets != 0:
            raise ValueError(
                f"Expected market_dim {expected_market_dim} not divisible by assets={expected_assets}."
            )
        expected_per_symbol_dim = int(expected_market_dim // expected_assets)

        env, symbol_kline_map, actual_pool = _prepare_multi_asset_env(
            stock_pool,
            start_date,
            end_date,
            window_size,
            commission_rate,
            slippage,
            use_calendar_features=use_calendar_features,
            sentiment_loader=sentiment_loader,
            sentiment_columns=sentiment_columns,
            expected_num_assets=expected_assets,
            expected_per_symbol_feature_dim=expected_per_symbol_dim,
        )
        kline_data = None
        normalized = env.price_df / env.price_df.iloc[0]
        benchmark_series = normalized.mean(axis=1)
    else:
        expected_market_dim = int(spaces_info.obs_dim - 3)
        if expected_market_dim <= 0:
            raise ValueError(
                f"Invalid inferred obs_dim={spaces_info.obs_dim} for single-asset env."
            )
        env, kline_data = _prepare_single_asset_env(
            stock_code,
            start_date,
            end_date,
            window_size,
            commission_rate,
            slippage,
            use_calendar_features=use_calendar_features,
            sentiment_loader=sentiment_loader,
            sentiment_columns=sentiment_columns,
            expected_feature_dim=expected_market_dim,
        )
        symbol_kline_map = {stock_code: kline_data}
        actual_pool = [stock_code]

    if base_symbol not in actual_pool and actual_pool:
        logging.warning(
            "base_symbol '%s' not available after data resolution; falling back to '%s'.",
            base_symbol,
            actual_pool[0],
        )
        base_symbol = actual_pool[0]

    base_kline = get_stock_data(base_symbol, start_date, end_date)
    if base_kline.empty:
        fallback = symbol_kline_map.get(base_symbol)
        if fallback is not None:
            base_kline = fallback
    if base_kline.empty and kline_data is not None and base_symbol == stock_code:
        base_kline = kline_data

    if status_callback:
        status_callback("创建环境并进入模拟交易循环...")

    if seed_value is not None:
        obs, _ = env.reset(seed=seed_value)
    else:
        obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)

    if status_callback:
        status_callback("模拟结束，正在处理结果...")

    if len(stock_pool) > 1:
        index = env.price_df.index[: len(env.portfolio_history)]
    else:
        index = kline_data.index[: len(env.portfolio_history)]

    trades_df = pd.DataFrame(env.trades)
    if not trades_df.empty:
        if "symbol" not in trades_df.columns:
            trades_df["symbol"] = stock_code
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
        trades_df.set_index("timestamp", inplace=True)
    symbol_trades_map = {}
    if not trades_df.empty:
        for symbol, group in trades_df.groupby("symbol"):
            symbol_trades_map[symbol] = group.copy()

    results = {
        "stock_code": stock_code,
        "kline_data": kline_data,
        "trades": trades_df,
        "portfolio_history": pd.Series(env.portfolio_history, index=index),
        "benchmark_series": benchmark_series,
        "base_symbol": base_symbol,
        "base_kline": base_kline,
        "symbol_kline_map": symbol_kline_map,
        "symbol_trades_map": symbol_trades_map,
        "start_date": start_date,
        "end_date": end_date,
        "stock_pool": actual_pool,
    }
    return results
