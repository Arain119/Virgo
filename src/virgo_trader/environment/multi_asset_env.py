"""Gymnasium environment for multi-asset portfolio trading.

The environment exposes observations built from multiple symbols and accepts
continuous portfolio weights as actions. Rewards are computed via composite
metrics (return, drawdown, etc.) with online normalization.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..utils.normalizer import OnlineNormalizer
from .reward import RewardConfig, compute_composite_reward


class MultiAssetTradingEnv(gym.Env):
    """
    同步在多支标的上进行训练的投资组合环境。
    动作向量表示目标仓位（0~1），智能体可在上证50股票池等多标的上同时交易。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        datasets: Sequence[Dict[str, object]],
        window_size: int,
        initial_balance: float = 100000.0,
        commission_rate: float = 0.0003,
        slippage: float = 0.0001,
        dsr_lookback: int = 252,
        reward_config: Optional[RewardConfig] = None,
        start_step: int = 0,
        random_start: bool = False,
        episode_length: int = 0,
    ):
        super().__init__()
        if not datasets:
            raise ValueError("datasets 不能为空。")

        self.datasets = list(datasets)
        self.symbols = [item["symbol"] for item in self.datasets]
        self.num_assets = len(self.symbols)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.dsr_lookback = dsr_lookback
        self.reward_config = reward_config or RewardConfig()
        self.last_reward_components = {}
        self.start_step = max(int(start_step), int(self.window_size))
        self.random_start = bool(random_start)
        self.episode_length = max(0, int(episode_length or 0))
        self.episode_start_step = int(self.start_step)
        self._steps_in_episode = 0

        feature_frames: List[pd.DataFrame] = []
        price_frames: List[pd.Series] = []
        common_index = None
        for item in self.datasets:
            df = item["df"]
            raw_df = item["raw_df"]
            symbol = item["symbol"]
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
            common_index = common_index.intersection(raw_df.index)

        if common_index is None or len(common_index) <= self.window_size:
            raise ValueError("股票池共有的数据区间不足以支持训练，请调整时间范围。")

        for item in self.datasets:
            symbol = item["symbol"]
            df = item["df"].loc[common_index].add_prefix(f"{symbol}_")
            feature_frames.append(df)
            price_frames.append(item["raw_df"].loc[common_index, "close"].rename(symbol))

        self.df = pd.concat(feature_frames, axis=1).sort_index()
        sentiment_cols = [col for col in self.df.columns if "sentiment_" in col]
        if sentiment_cols:
            base_cols = [col for col in self.df.columns if col not in sentiment_cols]
            self.df = self.df[base_cols + sentiment_cols]
        self.sentiment_feature_dim = len(sentiment_cols)
        self.price_df = pd.concat(price_frames, axis=1).sort_index()
        self.sequence_length = len(self.df)
        self._feature_array = self.df.to_numpy(dtype=np.float32, copy=True)
        self._price_array = self.price_df.to_numpy(dtype=np.float64, copy=True)
        self._timestamps = self.price_df.index.to_numpy()

        self.observation_dim = self.df.shape[1] + self.num_assets + 1 + self.num_assets
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.observation_dim),
            dtype=np.float32,
        )

        self.normalizer = OnlineNormalizer(shape=(self.df.shape[1],))
        self.action_history = deque(
            [np.zeros(self.num_assets, dtype=np.float32) for _ in range(self.window_size)],
            maxlen=self.window_size,
        )
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
        self.shares_held = np.zeros(self.num_assets, dtype=np.float32)
        self.total_worth = self.initial_balance
        self.trades = []
        self.episode_start_step = self._sample_start_step()
        self.current_step = int(self.episode_start_step)
        self._steps_in_episode = 0

        self.normalizer = OnlineNormalizer(shape=(self.df.shape[1],))
        warmup_start = max(0, self.current_step - self.window_size)
        warmup_slice = self._feature_array[warmup_start : self.current_step]
        for row in warmup_slice:
            self.normalizer.update(row)

        self.portfolio_history = [self.initial_balance] * self.window_size
        self.portfolio_peak = self.initial_balance
        self.benchmark_worth = self.initial_balance
        self.benchmark_history = [self.initial_balance] * self.window_size
        self.asset_return_history = [0.0] * self.window_size
        self.strategy_return_history = [0.0] * self.window_size
        self.action_history.clear()
        self.action_history.extend(
            [np.zeros(self.num_assets, dtype=np.float32) for _ in range(self.window_size)]
        )
        self.idle_steps = 0

        self.dsr_a = 0.0
        self.dsr_b = 0.0
        self.dsr_eta = 1.0 / self.dsr_lookback
        self.last_reward_components = {}

        return self._next_observation(), {}

    def _next_observation(self):
        start = self.current_step - self.window_size
        market_features_raw = self._feature_array[start : self.current_step]
        market_features = self.normalizer.normalize_batch(market_features_raw)

        current_prices = self._price_array[self.current_step - 1]
        holdings_value = self.shares_held * current_prices
        if self.total_worth > 0:
            position_ratios = holdings_value / self.total_worth
            balance_ratio = self.balance / self.total_worth
        else:
            position_ratios = np.zeros(self.num_assets)
            balance_ratio = 1.0
        account_vec = np.concatenate([position_ratios, [balance_ratio]])
        account_features = np.tile(account_vec, (self.window_size, 1))

        action_features = np.array(self.action_history)

        obs = np.concatenate([market_features, account_features, action_features], axis=1)
        return obs.astype(np.float32)

    def step(self, action):
        safe_action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
        target_weights = np.clip(safe_action, 0.0, 1.0).astype(np.float64)
        weight_sum = float(target_weights.sum())
        if weight_sum > 1.0:
            target_weights = target_weights / weight_sum
        self.action_history.append(target_weights.astype(np.float32))

        prev_total_worth = self.total_worth
        current_prices = self._price_array[self.current_step]
        prev_prices = self._price_array[self.current_step - 1]

        current_values = self.shares_held * current_prices
        portfolio_value = self.balance + current_values.sum()
        target_values = target_weights * portfolio_value
        value_diff = target_values - current_values

        # Idle penalty tracking：若目标仓位基本为0则视作闲置
        max_weight = float(target_weights.max()) if target_weights.size else 0.0
        if max_weight < 0.05:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        timestamp = self._timestamps[self.current_step]

        # 先执行卖出以释放资金（向量化计算 + 稀疏记录交易明细）
        sell_mask = (value_diff < 0) & (current_prices > 0) & (self.shares_held > 0)
        if np.any(sell_mask):
            sell_prices = current_prices * (1 - self.slippage)
            desired_sell_value = np.minimum(-value_diff, self.shares_held * current_prices)

            shares_to_sell = np.zeros_like(self.shares_held, dtype=np.float64)
            safe_sell_prices = np.where(sell_prices > 0, sell_prices, 1.0)
            shares_to_sell[sell_mask] = np.minimum(
                self.shares_held[sell_mask],
                desired_sell_value[sell_mask] / safe_sell_prices[sell_mask],
            )

            revenue = shares_to_sell * sell_prices
            commission = revenue * self.commission_rate
            net_revenue = np.maximum(0.0, revenue - commission)

            self.balance += float(net_revenue.sum())
            self.shares_held = self.shares_held - shares_to_sell

            for idx in np.flatnonzero(shares_to_sell > 0):
                self.trades.append(
                    {
                        "timestamp": timestamp,
                        "side": "SELL",
                        "price": float(sell_prices[idx]),
                        "quantity": float(shares_to_sell[idx]),
                        "symbol": self.symbols[idx],
                    }
                )

        current_values = self.shares_held * current_prices
        portfolio_value = self.balance + current_values.sum()
        target_values = target_weights * portfolio_value
        value_diff = target_values - current_values

        # 再执行买入：按差额比例分配现金，避免逐个循环带来的顺序偏差
        buy_mask = (value_diff > 0) & (current_prices > 0) & (self.balance > 0)
        if np.any(buy_mask) and self.balance > 0:
            buy_prices = current_prices * (1 + self.slippage)
            desired_cash = np.where(buy_mask, value_diff, 0.0)

            required_total = float((desired_cash * (1 + self.commission_rate)).sum())
            if required_total > 0:
                scale = min(1.0, float(self.balance) / required_total)
                cash_spend = desired_cash * scale
                shares_to_buy = np.divide(
                    cash_spend,
                    np.where(buy_prices > 0, buy_prices, 1.0),
                    out=np.zeros_like(cash_spend),
                    where=buy_prices > 0,
                )

                cost = cash_spend
                commission = cost * self.commission_rate
                total_cost = cost + commission
                total_cost_sum = float(total_cost.sum())

                # 数值稳定性兜底：避免浮点误差导致余额为负
                if total_cost_sum > self.balance and total_cost_sum > 0:
                    adjust = float(self.balance) / total_cost_sum
                    shares_to_buy *= adjust
                    cost *= adjust
                    commission *= adjust
                    total_cost *= adjust
                    total_cost_sum = float(total_cost.sum())

                if total_cost_sum > 0:
                    self.balance -= total_cost_sum
                    self.shares_held = self.shares_held + shares_to_buy

                    for idx in np.flatnonzero(shares_to_buy > 0):
                        self.trades.append(
                            {
                                "timestamp": timestamp,
                                "side": "BUY",
                                "price": float(buy_prices[idx]),
                                "quantity": float(shares_to_buy[idx]),
                                "symbol": self.symbols[idx],
                            }
                        )

        self.total_worth = self.balance + np.dot(self.shares_held, current_prices)
        self.portfolio_history.append(self.total_worth)
        if self.total_worth > self.portfolio_peak:
            self.portfolio_peak = self.total_worth

        price_returns = np.divide(
            current_prices - prev_prices,
            np.where(prev_prices == 0, 1, prev_prices),
            out=np.zeros_like(current_prices),
            where=prev_prices != 0,
        )
        benchmark_return = float(np.nanmean(price_returns))
        self.benchmark_worth *= 1 + benchmark_return
        self.benchmark_history.append(self.benchmark_worth)

        log_return = (
            np.log(self.total_worth) - np.log(prev_total_worth) if prev_total_worth > 0 else 0.0
        )
        strategy_simple_return = (
            (self.total_worth - prev_total_worth) / prev_total_worth
            if prev_total_worth > 0
            else 0.0
        )
        self.strategy_return_history.append(strategy_simple_return)
        self.asset_return_history.append(benchmark_return)

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
            trend_consistency=0.0,
            config=self.reward_config,
        )
        self.last_reward_components = reward_components

        new_row = self._feature_array[self.current_step]
        self.normalizer.update(new_row)

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

        obs = self._next_observation()
        info = {"symbols": self.symbols}
        return obs, reward, terminated, truncated, info
