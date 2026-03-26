"""Gymnasium-compatible single-asset trading environment.

The environment consumes feature-engineered market data and models transaction
costs/slippage. It exposes continuous target-position actions for RL training.
"""

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..utils.normalizer import OnlineNormalizer
from .reward import RewardConfig, compute_composite_reward


class TradingEnv(gym.Env):
    """
    一个完全端到端的、符合Gym接口的股票交易环境。
    - 状态是包含历史窗口的二维矩阵。
    - 动作是直接的目标仓位。
    - 包含交易成本和差分夏普比率奖励。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        raw_df: pd.DataFrame,
        window_size: int,
        initial_balance: float = 100000.0,
        commission_rate: float = 0.0003,
        slippage: float = 0.0001,
        dsr_lookback: int = 252,
        start_step: int = 0,
        symbol: Optional[str] = None,
        reward_config: Optional[RewardConfig] = None,
        random_start: bool = False,
        episode_length: int = 0,
    ):
        super(TradingEnv, self).__init__()

        # Note: df is now expected to be the feature-engineered but NOT normalized data
        common_index = df.index.intersection(raw_df.index)
        self.df = df.loc[common_index]
        sentiment_cols = [col for col in self.df.columns if "sentiment_" in col]
        if sentiment_cols:
            base_cols = [col for col in self.df.columns if col not in sentiment_cols]
            self.df = self.df[base_cols + sentiment_cols]
        self.market_feature_dim = self.df.shape[1]
        self.sentiment_feature_dim = len(sentiment_cols)
        self.raw_df = raw_df.loc[common_index]
        self._feature_array = self.df.to_numpy(dtype=np.float32, copy=True)
        self._close_array = self.raw_df["close"].to_numpy(dtype=np.float64, copy=True)
        self._timestamps = self.raw_df.index.to_numpy()

        self.window_size = window_size
        self.initial_balance = initial_balance
        self.start_step = max(
            start_step, self.window_size
        )  # Ensure we have enough data for the first observation
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.dsr_lookback = dsr_lookback
        self.symbol = symbol
        self.reward_config = reward_config or RewardConfig()
        self.last_reward_components = {}
        self.random_start = bool(random_start)
        self.episode_length = max(0, int(episode_length or 0))
        self.episode_start_step = self.start_step
        self._steps_in_episode = 0

        # 动作空间：连续，-1 (全仓卖空) to 1 (全仓买入)
        # 为简化，我们先只做多 (0 to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # 状态空间：(窗口大小, 特征数量 + 3个账户/动作特征)
        # 账户特征：仓位比例, 余额比例
        # 动作特征: 历史动作
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.df.shape[1] + 3),
            dtype=np.float32,
        )

        # Initialize the online normalizer for market features
        self.normalizer = OnlineNormalizer(shape=(self.df.shape[1],))

        # Use a deque for a bounded history window
        self.action_history = deque([0] * self.window_size, maxlen=self.window_size)
        self.idle_steps = 0

        # Benchmark & diversification trackers
        self.benchmark_worth = self.initial_balance
        self.benchmark_history = []
        self.asset_return_history = []
        self.strategy_return_history = []
        self.corr_window = max(60, self.window_size)

    def _sample_start_step(self) -> int:
        if not self.random_start:
            return int(self.start_step)

        data_len = int(len(self.df))
        min_start = int(self.start_step)
        max_start = max(min_start, data_len - 2)
        if self.episode_length > 0:
            max_start = min(max_start, data_len - int(self.episode_length) - 2)
        max_start = max(min_start, max_start)
        if max_start <= min_start:
            return min_start
        return int(self.np_random.integers(min_start, max_start + 1))

    def reset(self, *, seed=None, options=None):
        # `options` is part of the Gymnasium API; currently unused but kept for signature compatibility.
        _ = options
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_worth = self.initial_balance
        self.trades = []  # 记录交易历史

        self.episode_start_step = self._sample_start_step()
        self.current_step = int(self.episode_start_step)
        self._steps_in_episode = 0

        # 移除sharpe_ratio和max_drawdown的初始化，因为它们将由一个专门的函数在评估结束时计算并返回。

        # Reset and pre-fill the normalizer with data prior to the start_step
        self.normalizer = OnlineNormalizer(shape=(self.df.shape[1],))
        # Use a warm-up period of at least window_size
        warmup_data = self._feature_array[self.current_step - self.window_size : self.current_step]
        for row in warmup_data:
            self.normalizer.update(row)

        self.portfolio_history = [self.initial_balance] * self.window_size
        self.portfolio_peak = self.initial_balance
        self.benchmark_worth = self.initial_balance
        self.benchmark_history = [self.initial_balance] * self.window_size
        self.asset_return_history = [0.0] * self.window_size
        self.strategy_return_history = [0.0] * self.window_size
        self.action_history.clear()
        self.action_history.extend([0] * self.window_size)
        self.idle_steps = 0

        self.dsr_a = 0
        self.dsr_b = 0
        self.dsr_eta = 1.0 / self.dsr_lookback
        self.last_reward_components = {}

        return self._next_observation(), {}

    def calculate_performance_metrics(self):
        """在回测结束时计算并返回最终的性能指标"""
        portfolio_values = pd.Series(self.portfolio_history)
        benchmark_values = pd.Series(self.benchmark_history[: len(self.portfolio_history)])

        strategy_returns = portfolio_values.pct_change().dropna()
        benchmark_returns = benchmark_values.pct_change().dropna()

        sharpe_ratio = 0.0
        if not strategy_returns.empty and strategy_returns.std() > 0:
            sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)

        max_drawdown = 0.0
        if not portfolio_values.empty:
            cumulative_max = portfolio_values.cummax()
            drawdown = (portfolio_values - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min()

        excess_return = 0.0
        tracking_error = 0.0
        beta = 0.0
        correlation = 0.0

        if not strategy_returns.empty and not benchmark_returns.empty:
            aligned = pd.concat(
                [strategy_returns, benchmark_returns], axis=1, join="inner"
            ).dropna()
            if not aligned.empty:
                aligned.columns = ["strategy", "benchmark"]
                strat = aligned["strategy"]
                bench = aligned["benchmark"]
                excess_series = strat - bench

                cumulative_strategy = (1 + strat).prod() - 1
                cumulative_benchmark = (1 + bench).prod() - 1
                excess_return = cumulative_strategy - cumulative_benchmark

                tracking_error = excess_series.std() * np.sqrt(252)

                bench_var = bench.var()
                if bench_var > 0:
                    beta = np.cov(strat, bench)[0][1] / bench_var

                if strat.std() > 0 and bench.std() > 0:
                    correlation = strat.corr(bench)

        return {
            "sharpe_ratio": np.nan_to_num(sharpe_ratio),
            "max_drawdown": np.nan_to_num(max_drawdown),
            "excess_return": np.nan_to_num(excess_return),
            "tracking_error": np.nan_to_num(tracking_error),
            "beta": np.nan_to_num(beta),
            "correlation": np.nan_to_num(correlation),
        }

    def _next_observation(self):
        # Get the historical window of raw market features
        market_features_raw = self._feature_array[
            self.current_step - self.window_size : self.current_step
        ]

        # Normalize the features using the online normalizer (vectorized for performance)
        market_features = self.normalizer.normalize_batch(market_features_raw)

        # Create historical window for account data
        last_price = float(self._close_array[self.current_step - 1])
        position_ratio = (
            self.shares_held * last_price / self.total_worth if self.total_worth != 0 else 0
        )
        balance_ratio = self.balance / self.total_worth if self.total_worth != 0 else 1

        account_features = np.full((self.window_size, 2), [position_ratio, balance_ratio])

        # 获取历史窗口的动作数据
        action_features = np.array(self.action_history).reshape(-1, 1)

        # 拼接成最终的状态
        obs = np.concatenate([market_features, account_features, action_features], axis=1)
        return obs.astype(np.float32)

    def step(self, action):
        safe_action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
        target_position = float(np.clip(safe_action[0], 0.0, 1.0))  # 动作是目标仓位比例
        self.action_history.append(target_position)

        prev_total_worth = self.total_worth
        current_price = float(self._close_array[self.current_step])

        # 使用当前价格估算的组合价值计算仓位，避免使用上一步 total_worth 导致的偏差
        portfolio_value = self.balance + self.shares_held * current_price
        if portfolio_value > 0:
            current_position = (self.shares_held * current_price) / portfolio_value
        else:
            current_position = 0.0
        position_diff = target_position - current_position

        # 统计连续低仓位时长，用于惩罚长期锁仓
        if abs(target_position) < 0.05 and abs(position_diff) < 0.01:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        # 根据差异执行交易（用当前组合价值计算目标仓位）
        target_value = target_position * portfolio_value
        current_value = self.shares_held * current_price
        value_diff = target_value - current_value

        if value_diff > 0:  # 买入
            buy_value = value_diff
            buy_price = current_price * (1 + self.slippage)
            shares_to_buy = buy_value / buy_price
            cost = shares_to_buy * buy_price
            commission = cost * self.commission_rate
            if self.balance >= cost + commission:
                self.balance -= cost + commission
                self.shares_held += shares_to_buy
                self.trades.append(
                    {
                        "timestamp": self._timestamps[self.current_step],
                        "side": "BUY",
                        "price": buy_price,
                        "quantity": shares_to_buy,
                        "symbol": self.symbol,
                    }
                )
        elif value_diff < 0:  # 卖出
            sell_value = abs(value_diff)
            sell_price = current_price * (1 - self.slippage)
            desired_shares = sell_value / sell_price if sell_price > 0 else 0.0
            shares_to_sell = min(self.shares_held, desired_shares)
            if shares_to_sell > 0:
                revenue = shares_to_sell * sell_price
                commission = revenue * self.commission_rate
                self.balance += revenue - commission
                self.shares_held -= shares_to_sell
                self.trades.append(
                    {
                        "timestamp": self._timestamps[self.current_step],
                        "side": "SELL",
                        "price": sell_price,
                        "quantity": shares_to_sell,
                        "symbol": self.symbol,
                    }
                )

        # 更新资产与基准
        self.total_worth = self.balance + self.shares_held * current_price
        self.portfolio_history.append(self.total_worth)
        if self.total_worth > self.portfolio_peak:
            self.portfolio_peak = self.total_worth

        prev_price = float(self._close_array[self.current_step - 1])
        benchmark_return = (current_price - prev_price) / prev_price if prev_price != 0 else 0.0
        self.benchmark_worth *= 1 + benchmark_return
        self.benchmark_history.append(self.benchmark_worth)

        # 计算增强的多重奖励函数
        log_return = (
            np.log(self.total_worth) - np.log(prev_total_worth) if prev_total_worth != 0 else 0
        )
        strategy_simple_return = (
            (self.total_worth - prev_total_worth) / prev_total_worth
            if prev_total_worth != 0
            else 0.0
        )
        self.strategy_return_history.append(strategy_simple_return)
        self.asset_return_history.append(benchmark_return)

        # 趋势一致性奖励（鼓励顺势交易）
        if self.current_step > 1:
            price_change = (current_price - prev_price) / prev_price if prev_price != 0 else 0.0
            position_change = target_position - (
                self.action_history[-2] if len(self.action_history) > 1 else 0
            )
            # 当价格变化和仓位变化方向一致时给予正奖励
            trend_consistency = np.tanh(price_change * position_change * 10)  # 使用tanh限制奖励范围
        else:
            trend_consistency = 0

        reward, self.dsr_a, self.dsr_b, reward_components = compute_composite_reward(
            log_return=float(log_return),
            simple_return=float(strategy_simple_return),
            benchmark_return=float(benchmark_return),
            portfolio_history=self.portfolio_history,
            portfolio_peak=float(self.portfolio_peak),
            dsr_a=float(self.dsr_a),
            dsr_b=float(self.dsr_b),
            dsr_eta=float(self.dsr_eta),
            strategy_return_history=self.strategy_return_history,
            asset_return_history=self.asset_return_history,
            corr_window=int(self.corr_window),
            idle_steps=int(self.idle_steps),
            trend_consistency=float(trend_consistency),
            config=self.reward_config,
        )
        self.last_reward_components = reward_components

        # Update the normalizer with the new data point from the current step
        new_data_point = self._feature_array[self.current_step]
        self.normalizer.update(new_data_point)

        self.current_step += 1
        self._steps_in_episode += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        if (
            self.episode_length > 0
            and self._steps_in_episode >= self.episode_length
            and not terminated
        ):
            truncated = True

        # 移除这里的调用，指标计算将由外部在评估结束后显式调用

        obs = self._next_observation()

        # info 字典可以传递额外信息，但我们已将trades作为公共属性
        info = {}

        # 在新版Gymnasium中，step返回5个值
        return obs, reward, terminated, truncated, info
