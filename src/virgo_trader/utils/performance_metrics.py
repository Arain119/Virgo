"""Performance metric utilities for backtests and simulations.

Computes return/risk statistics, benchmark comparisons, and trade-level metrics
used by CLI commands and the dashboard.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _collect_realized_pnls(trades: pd.DataFrame) -> List[float]:
    """Compute realized PnL values from a trade blotter (supports multi-asset via `symbol`)."""
    if trades is None or trades.empty:
        return []

    has_symbol = "symbol" in trades.columns
    positions: Dict[str, Tuple[float, float]] = {}  # symbol -> (quantity, avg_cost)
    realized: List[float] = []

    for _, trade in trades.iterrows():
        side = trade.get("side")
        if side not in {"BUY", "SELL"}:
            continue

        symbol = str(trade.get("symbol", "__default__")) if has_symbol else "__default__"
        try:
            trade_quantity = float(trade["quantity"])
            trade_price = float(trade["price"])
        except (KeyError, TypeError, ValueError):
            continue

        position_quantity, position_avg_cost = positions.get(symbol, (0.0, 0.0))

        if side == "BUY":
            current_total_cost = position_avg_cost * position_quantity
            new_total_cost = current_total_cost + (trade_price * trade_quantity)
            position_quantity += trade_quantity
            if position_quantity > 0:
                position_avg_cost = new_total_cost / position_quantity
            positions[symbol] = (position_quantity, position_avg_cost)
            continue

        if position_quantity <= 0:
            continue

        sell_quantity = min(trade_quantity, position_quantity)
        realized.append((trade_price - position_avg_cost) * sell_quantity)

        position_quantity -= sell_quantity
        if position_quantity <= 0:
            position_quantity = 0.0
            position_avg_cost = 0.0
        positions[symbol] = (position_quantity, position_avg_cost)

    return [float(pnl) for pnl in realized]


def _calculate_realized_pnl(trades: pd.DataFrame) -> Tuple[float, int]:
    """
    通过模拟持仓变化，计算胜率与交易次数（已实现 SELL 笔数）。
    """
    if trades.empty:
        return 0.0, 0

    realized_pnls = _collect_realized_pnls(trades)
    if not realized_pnls:
        return 0.0, 0

    winning_trades = sum(1 for pnl in realized_pnls if pnl > 0)
    win_rate = winning_trades / len(realized_pnls)

    return win_rate, len(realized_pnls)


def _calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    计算各种技术指标
    """
    indicators = {}

    # 移动平均线
    indicators["MA5"] = df["close"].rolling(window=5).mean()
    indicators["MA20"] = df["close"].rolling(window=20).mean()

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    indicators["MACD"] = ema12 - ema26
    indicators["MACD_signal"] = indicators["MACD"].ewm(span=9).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    indicators["RSI"] = 100 - (100 / (1 + rs))

    # 布林带
    indicators["BB_middle"] = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    indicators["BB_upper"] = indicators["BB_middle"] + (bb_std * 2)
    indicators["BB_lower"] = indicators["BB_middle"] - (bb_std * 2)

    return indicators


def _simulate_strategy_returns(
    df: pd.DataFrame, strategy_type: str, initial_capital: float = 10000
) -> float:
    """
    模拟各种策略的收益率
    """
    if df.empty or len(df) < 30:
        return 0.0

    df = df.copy()
    indicators = _calculate_technical_indicators(df)

    position = 0  # 0: 空仓, 1: 持仓
    capital = initial_capital
    shares = 0

    for i in range(30, len(df)):  # 从第30天开始，确保技术指标有效
        price = df["close"].iloc[i]

        if strategy_type == "MA_CROSS":
            # 双均线策略
            if indicators["MA5"].iloc[i] > indicators["MA20"].iloc[i] and position == 0:
                # 金叉买入
                shares = capital / price
                capital = 0
                position = 1
            elif indicators["MA5"].iloc[i] < indicators["MA20"].iloc[i] and position == 1:
                # 死叉卖出
                capital = shares * price
                shares = 0
                position = 0

        elif strategy_type == "MACD":
            # MACD策略
            if (
                indicators["MACD"].iloc[i] > indicators["MACD_signal"].iloc[i]
                and indicators["MACD"].iloc[i - 1] <= indicators["MACD_signal"].iloc[i - 1]
                and position == 0
            ):
                # MACD金叉买入
                shares = capital / price
                capital = 0
                position = 1
            elif (
                indicators["MACD"].iloc[i] < indicators["MACD_signal"].iloc[i]
                and indicators["MACD"].iloc[i - 1] >= indicators["MACD_signal"].iloc[i - 1]
                and position == 1
            ):
                # MACD死叉卖出
                capital = shares * price
                shares = 0
                position = 0

        elif strategy_type == "RSI":
            # RSI策略
            if indicators["RSI"].iloc[i] < 30 and position == 0:
                # RSI超卖买入
                shares = capital / price
                capital = 0
                position = 1
            elif indicators["RSI"].iloc[i] > 70 and position == 1:
                # RSI超买卖出
                capital = shares * price
                shares = 0
                position = 0

        elif strategy_type == "BOLLINGER":
            # 布林带策略
            if price < indicators["BB_lower"].iloc[i] and position == 0:
                # 触及下轨买入
                shares = capital / price
                capital = 0
                position = 1
            elif price > indicators["BB_upper"].iloc[i] and position == 1:
                # 触及上轨卖出
                capital = shares * price
                shares = 0
                position = 0

        elif strategy_type == "MOMENTUM":
            # 动量策略 - 价格突破20日高点买入，跌破20日低点卖出
            high_20 = df["high"].iloc[i - 20 : i].max()
            low_20 = df["low"].iloc[i - 20 : i].min()

            if price > high_20 and position == 0:
                # 突破买入
                shares = capital / price
                capital = 0
                position = 1
            elif price < low_20 and position == 1:
                # 跌破卖出
                capital = shares * price
                shares = 0
                position = 0

    # 计算最终价值
    final_price = df["close"].iloc[-1]
    final_value = capital + shares * final_price

    return (final_value - initial_capital) / initial_capital


def _calculate_advanced_metrics(
    portfolio_history: pd.Series, kline_data: pd.DataFrame, trades: pd.DataFrame, hold_return: float
) -> Dict[str, str]:
    """
    计算高级性能指标
    """
    metrics = {}

    if portfolio_history.empty or len(portfolio_history) < 2:
        return {
            key: "0.00"
            for key in [
                "信息比率",
                "卡尔马比率",
                "索提诺比率",
                "最大连续亏损天数",
                "收益波动率",
                "Beta值",
                "Alpha值",
                "VaR (95%)",
                "平均持仓时间",
                "盈亏比",
            ]
        }

    daily_returns = portfolio_history.pct_change().dropna()

    # 1. 信息比率 (Information Ratio)
    if len(daily_returns) > 1:
        excess_return = daily_returns.mean() - hold_return / len(daily_returns)
        tracking_error = daily_returns.std()
        info_ratio = excess_return / tracking_error if tracking_error > 0 else 0
    else:
        info_ratio = 0
    metrics["信息比率"] = f"{info_ratio:.2f}"

    # 2. 卡尔马比率 (Calmar Ratio)
    total_return = (
        portfolio_history.iloc[-1] - portfolio_history.iloc[0]
    ) / portfolio_history.iloc[0]
    days = (portfolio_history.index[-1] - portfolio_history.index[0]).days
    annualized_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else 0

    cumulative_max = portfolio_history.cummax()
    drawdown = (portfolio_history - cumulative_max) / cumulative_max
    max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.01

    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
    metrics["卡尔马比率"] = f"{calmar_ratio:.2f}"

    # 3. 索提诺比率 (Sortino Ratio)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.01
    sortino_ratio = (
        (daily_returns.mean() * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
    )
    metrics["索提诺比率"] = f"{sortino_ratio:.2f}"

    # 4. 最大连续亏损天数
    losses = daily_returns < 0
    max_consecutive_losses = 0
    current_consecutive = 0
    for loss in losses:
        if loss:
            current_consecutive += 1
            max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
        else:
            current_consecutive = 0
    metrics["最大连续亏损天数"] = f"{max_consecutive_losses}天"

    # 5. 收益波动率
    volatility = daily_returns.std() * np.sqrt(252)
    metrics["收益波动率"] = f"{volatility:.2%}"

    # 6. Beta值和Alpha值（相对于持有收益）
    if kline_data is not None and not kline_data.empty and len(daily_returns) > 1:
        try:
            benchmark_returns = kline_data["close"].pct_change().dropna()
            # 对齐时间序列
            min_len = min(len(daily_returns), len(benchmark_returns))
            if min_len > 1:
                port_ret = daily_returns.iloc[-min_len:]
                bench_ret = benchmark_returns.iloc[-min_len:]

                covariance = np.cov(port_ret, bench_ret)[0][1]
                benchmark_variance = np.var(bench_ret)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                alpha = port_ret.mean() - beta * bench_ret.mean()
            else:
                beta, alpha = 0, 0
        except (KeyError, TypeError, ValueError):
            beta, alpha = 0, 0
    else:
        beta, alpha = 0, 0

    metrics["Beta值"] = f"{beta:.2f}"
    metrics["Alpha值"] = f"{alpha:.3f}"

    # 7. VaR (Value at Risk) 95%置信度
    if len(daily_returns) > 0:
        var_95 = np.percentile(daily_returns, 5)
    else:
        var_95 = 0
    metrics["VaR (95%)"] = f"{var_95:.2%}"

    # 8. 平均持仓时间
    if not trades.empty:
        trades_local = trades.copy()
        if not isinstance(trades_local.index, pd.DatetimeIndex):
            try:
                trades_local.index = pd.to_datetime(trades_local.index, errors="coerce")
                trades_local = trades_local.loc[~trades_local.index.isna()]
            except (TypeError, ValueError):
                trades_local = trades.copy()

        hold_periods: List[int] = []
        if "side" in trades_local.columns and not trades_local.empty:
            if "symbol" in trades_local.columns:
                groups = trades_local.groupby("symbol", sort=False)
            else:
                groups = [("__default__", trades_local)]

            for _, group in groups:
                group_sorted = group.sort_index()
                buy_queue: List[pd.Timestamp] = []
                for ts, row in group_sorted.iterrows():
                    side = row.get("side")
                    if side == "BUY":
                        buy_queue.append(ts)
                    elif side == "SELL" and buy_queue:
                        buy_ts = buy_queue.pop(0)
                        days = int((ts - buy_ts).days)
                        if days > 0:
                            hold_periods.append(days)

        avg_hold_time = float(np.mean(hold_periods)) if hold_periods else 0.0
    else:
        avg_hold_time = 0.0

    metrics["平均持仓时间"] = f"{avg_hold_time:.1f}天"

    # 9. 盈亏比
    if not trades.empty:
        realized_pnls = _collect_realized_pnls(trades)
        profits = [pnl for pnl in realized_pnls if pnl > 0]
        losses = [-pnl for pnl in realized_pnls if pnl < 0]
        avg_profit = float(np.mean(profits)) if profits else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        profit_loss_ratio = (avg_profit / avg_loss) if avg_loss > 0 else 0.0
    else:
        profit_loss_ratio = 0.0

    metrics["盈亏比"] = f"{profit_loss_ratio:.2f}"

    return metrics


def calculate_performance_metrics(
    portfolio_history: pd.Series,
    trades: pd.DataFrame,
    kline_data: pd.DataFrame = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, str]:
    """
    计算并返回一套完整的量化交易性能指标。

    Args:
        portfolio_history: 资产组合历史价值序列
        trades: 交易记录DataFrame
        kline_data: K线数据DataFrame，用于计算持有收益率和基准策略
        risk_free_rate: 无风险利率
    """
    if portfolio_history.empty:
        return {
            "总收益率": "0.00%",
            "年化收益率": "0.00%",
            "夏普比率": "0.00",
            "最大回撤": "0.00%",
            "胜率": "0.00%",
            "总交易次数": 0,
            "持有收益率": "0.00%",
            "信息比率": "0.00",
            "卡尔马比率": "0.00",
            "索提诺比率": "0.00",
            "最大连续亏损天数": "0天",
            "收益波动率": "0.00%",
            "Beta值": "0.00",
            "Alpha值": "0.000",
            "VaR (95%)": "0.00%",
            "平均持仓时间": "0.0天",
            "盈亏比": "0.00",
            "双均线策略收益率": "0.00%",
            "MACD策略收益率": "0.00%",
            "RSI策略收益率": "0.00%",
            "布林带策略收益率": "0.00%",
            "动量策略收益率": "0.00%",
        }

    # ===== 现有核心指标 =====

    # 1. 总收益率
    initial_value = portfolio_history.iloc[0]
    final_value = portfolio_history.iloc[-1]
    total_return = (final_value - initial_value) / initial_value

    # 2. 胜率和交易次数
    win_rate, total_trades = _calculate_realized_pnl(trades.copy())

    # 3. 最大回撤
    cumulative_max = portfolio_history.cummax()
    drawdown = (portfolio_history - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() if not drawdown.empty else 0

    # 4. 年化收益率
    days = (portfolio_history.index[-1] - portfolio_history.index[0]).days
    if days > 0:
        annualized_return = (1 + total_return) ** (365.0 / days) - 1
    else:
        annualized_return = 0.0

    # 5. 夏普比率
    daily_returns = portfolio_history.pct_change().dropna()
    if not daily_returns.empty and daily_returns.std() > 0 and days > 0:
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # 6. 持有收益率
    hold_return = 0.0
    excess_return_pct = 0.0
    if kline_data is not None and not kline_data.empty:
        try:
            initial_price = kline_data.iloc[0]["close"]
            final_price = kline_data.iloc[-1]["close"]
            hold_return = (final_price - initial_price) / initial_price
        except (KeyError, IndexError):
            hold_return = 0.0

    excess_return_pct = total_return - hold_return
    tracking_error_metric = 0.0
    correlation_metric = 0.0
    if kline_data is not None and not kline_data.empty:
        try:
            benchmark_daily = kline_data["close"].pct_change().dropna()
            aligned = pd.concat([daily_returns, benchmark_daily], axis=1, join="inner").dropna()
            if not aligned.empty:
                aligned.columns = ["strategy", "benchmark"]
                tracking_error_metric = (
                    aligned["strategy"] - aligned["benchmark"]
                ).std() * np.sqrt(252)
                if aligned["strategy"].std() > 0 and aligned["benchmark"].std() > 0:
                    correlation_metric = aligned["strategy"].corr(aligned["benchmark"])
        except Exception:
            tracking_error_metric = 0.0
            correlation_metric = 0.0

    # ===== 新增性能评估指标 =====
    advanced_metrics = _calculate_advanced_metrics(
        portfolio_history, kline_data, trades, hold_return
    )

    # ===== 基准策略收益率 =====
    benchmark_returns = {}
    if kline_data is not None and not kline_data.empty:
        strategies = [
            ("双均线策略收益率", "MA_CROSS"),
            ("MACD策略收益率", "MACD"),
            ("RSI策略收益率", "RSI"),
            ("布林带策略收益率", "BOLLINGER"),
            ("动量策略收益率", "MOMENTUM"),
        ]

        for strategy_name, strategy_type in strategies:
            try:
                strategy_return = _simulate_strategy_returns(kline_data, strategy_type)
                benchmark_returns[strategy_name] = f"{strategy_return:.2%}"
            except Exception:
                benchmark_returns[strategy_name] = "0.00%"
    else:
        for strategy_name, _ in [
            ("双均线策略收益率", "MA_CROSS"),
            ("MACD策略收益率", "MACD"),
            ("RSI策略收益率", "RSI"),
            ("布林带策略收益率", "BOLLINGER"),
            ("动量策略收益率", "MOMENTUM"),
        ]:
            benchmark_returns[strategy_name] = "0.00%"

    # ===== 组合所有指标（按指定顺序） =====
    result = {
        # 现有核心指标
        "总收益率": f"{total_return:.2%}",
        "年化收益率": f"{annualized_return:.2%}",
        "夏普比率": f"{sharpe_ratio:.2f}",
        "最大回撤": f"{max_drawdown:.2%}",
        "胜率": f"{win_rate:.2%}",
        "总交易次数": total_trades,
        "持有收益率": f"{hold_return:.2%}",
        "超额收益率": f"{excess_return_pct:.2%}",
        "跟踪误差": f"{tracking_error_metric:.2%}",
        "指数相关性": f"{correlation_metric:.2f}",
        # 新增性能评估指标
        **advanced_metrics,
        # 基准策略收益率
        **benchmark_returns,
    }

    return result
