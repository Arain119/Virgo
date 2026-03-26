"""Feature engineering for market OHLCV data.

Computes technical indicators via `pandas_ta` and adds optional calendar/time
features. Normalization is intentionally handled inside environments to avoid
lookahead bias.
"""

import logging

import numpy as np
import pandas as pd

# pandas-ta (<=0.3.x) expects legacy NumPy aliases like `numpy.NaN`, which were
# removed in NumPy 2.x. Provide a compatibility alias before importing pandas_ta.
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]

import pandas_ta as ta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Calculates technical indicators using pandas-ta and adds time-based features.
    Normalization is handled within the environment to prevent lookahead bias.
    """

    def __init__(self, use_calendar_features: bool = True):
        self.use_calendar_features = use_calendar_features

    def process(self, df: pd.DataFrame):
        """
        Processes the entire DataFrame, adding indicators and time features.

        Args:
            df (pd.DataFrame): The raw OHLCV data.
        """
        df_processed = df.copy()

        # 确保有足够的数据进行技术指标计算
        if len(df) < 50:
            logger.warning("Data length (%d) is too small for full feature engineering.", len(df))
            # 对于小数据集，只添加基本特征
            df_processed["SMA_20"] = df["close"].rolling(20, min_periods=1).mean()
            df_processed["returns"] = df["close"].pct_change().fillna(0)
            df_processed["volatility_10"] = (
                df_processed["returns"].rolling(10, min_periods=1).std().fillna(0)
            )

            # 添加时间特征
            df_processed["day_of_week_sin"] = np.sin(2 * np.pi * df_processed.index.dayofweek / 7)
            df_processed["day_of_week_cos"] = np.cos(2 * np.pi * df_processed.index.dayofweek / 7)
            df_processed["month_sin"] = np.sin(2 * np.pi * df_processed.index.month / 12)
            df_processed["month_cos"] = np.cos(2 * np.pi * df_processed.index.month / 12)

            if self.use_calendar_features:
                df_processed = self._add_calendar_features(df_processed)

            # 填充缺失值
            df_processed.ffill(inplace=True)
            df_processed.fillna(0, inplace=True)

            return df_processed

        # 1. 使用稳定的技术指标策略（对于充足的数据）
        try:
            # RSI指标
            rsi_14 = ta.rsi(df["close"], length=14)
            if rsi_14 is not None:
                df_processed["RSI_14"] = rsi_14

            rsi_7 = ta.rsi(df["close"], length=7)
            if rsi_7 is not None:
                df_processed["RSI_7"] = rsi_7

            # MACD指标
            macd_data = ta.macd(df["close"])
            if macd_data is not None and not macd_data.empty:
                for col in macd_data.columns:
                    df_processed[col] = macd_data[col]

            # 布林带
            bbands_data = ta.bbands(df["close"], length=20)
            if bbands_data is not None and not bbands_data.empty:
                for col in bbands_data.columns:
                    df_processed[col] = bbands_data[col]

            # 移动平均线
            sma_10 = ta.sma(df["close"], length=10)
            if sma_10 is not None:
                df_processed["SMA_10"] = sma_10

            sma_20 = ta.sma(df["close"], length=20)
            if sma_20 is not None:
                df_processed["SMA_20"] = sma_20

            sma_50 = ta.sma(df["close"], length=50)
            if sma_50 is not None:
                df_processed["SMA_50"] = sma_50

            ema_12 = ta.ema(df["close"], length=12)
            if ema_12 is not None:
                df_processed["EMA_12"] = ema_12

            ema_26 = ta.ema(df["close"], length=26)
            if ema_26 is not None:
                df_processed["EMA_26"] = ema_26

            # ADX指标（趋势强度）
            try:
                adx_data = ta.adx(df["high"], df["low"], df["close"], length=14)
                if adx_data is not None and "ADX_14" in adx_data.columns:
                    df_processed["ADX_14"] = adx_data["ADX_14"]
            except Exception as exc:
                logger.debug("ADX calculation failed: %s", exc)

        except Exception as e:
            logger.warning("Some technical indicators failed to calculate: %s", e)

        # 确保基本移动平均线存在
        if "SMA_20" not in df_processed.columns or df_processed["SMA_20"].isna().all():
            df_processed["SMA_20"] = df["close"].rolling(20, min_periods=1).mean()
        if "SMA_50" not in df_processed.columns or df_processed["SMA_50"].isna().all():
            df_processed["SMA_50"] = df["close"].rolling(50, min_periods=1).mean()

        # 2. 添加高效的衍生特征
        # 价格位置指标 - 反映价格在近期区间的相对位置
        high_10_max = df["high"].rolling(10, min_periods=1).max()
        low_10_min = df["low"].rolling(10, min_periods=1).min()
        df_processed["price_position_10"] = (df["close"] - low_10_min) / (
            high_10_max - low_10_min + 1e-8
        )

        high_20_max = df["high"].rolling(20, min_periods=1).max()
        low_20_min = df["low"].rolling(20, min_periods=1).min()
        df_processed["price_position_20"] = (df["close"] - low_20_min) / (
            high_20_max - low_20_min + 1e-8
        )

        # 成交量特征 - 量价关系分析
        volume_sma = df["volume"].rolling(20, min_periods=1).mean()
        df_processed["volume_sma_ratio"] = df["volume"] / volume_sma
        df_processed["volume_price_trend"] = df["volume"] * np.sign(
            df["close"].diff()
        )  # 量价配合度

        # 波动率和动量特征
        df_processed["returns"] = df["close"].pct_change()
        df_processed["volatility_10"] = df_processed["returns"].rolling(10, min_periods=1).std()
        df_processed["volatility_20"] = df_processed["returns"].rolling(20, min_periods=1).std()
        df_processed["momentum_3"] = df["close"] / df["close"].shift(3) - 1
        df_processed["momentum_10"] = df["close"] / df["close"].shift(10) - 1

        # 市场结构特征
        df_processed["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]  # 日内波幅比
        df_processed["close_position"] = (df["close"] - df["low"]) / (
            df["high"] - df["low"] + 1e-8
        )  # 收盘价在日内的位置

        # 趋势和反转信号（添加安全检查）
        if "SMA_20" in df_processed.columns and df_processed["SMA_20"].notna().any():
            df_processed["price_vs_sma20"] = (
                df["close"] / df_processed["SMA_20"] - 1
            )  # 价格相对于20日均线的偏离度
            df_processed["sma_slope"] = df_processed["SMA_20"].diff(5)  # 20日均线的斜率（趋势强度）
        else:
            df_processed["price_vs_sma20"] = 0
            df_processed["sma_slope"] = 0

        # 3. 添加周期性时间特征
        df_processed["day_of_week_sin"] = np.sin(2 * np.pi * df_processed.index.dayofweek / 7)
        df_processed["day_of_week_cos"] = np.cos(2 * np.pi * df_processed.index.dayofweek / 7)
        df_processed["month_sin"] = np.sin(2 * np.pi * df_processed.index.month / 12)
        df_processed["month_cos"] = np.cos(2 * np.pi * df_processed.index.month / 12)

        if self.use_calendar_features:
            df_processed = self._add_calendar_features(df_processed)

        # 4. 处理缺失值
        # 首先用前向填充
        df_processed.ffill(inplace=True)
        # 然后用0填充剩余的缺失值（主要是开头的数据）
        df_processed.fillna(0, inplace=True)

        # 5. 处理无穷值
        df_processed.replace([np.inf, -np.inf], 0, inplace=True)

        return df_processed

    def _add_calendar_features(self, df_processed: pd.DataFrame) -> pd.DataFrame:
        """添加节假日、季度末等交易日历相关特征。"""
        if df_processed.empty or not isinstance(df_processed.index, pd.DatetimeIndex):
            return df_processed

        date_series = df_processed.index.to_series()

        df_processed["is_month_start"] = date_series.dt.is_month_start.astype(int)
        df_processed["is_month_end"] = date_series.dt.is_month_end.astype(int)
        df_processed["is_quarter_start"] = date_series.dt.is_quarter_start.astype(int)
        df_processed["is_quarter_end"] = date_series.dt.is_quarter_end.astype(int)
        df_processed["is_year_start"] = date_series.dt.is_year_start.astype(int)
        df_processed["is_year_end"] = date_series.dt.is_year_end.astype(int)

        days_in_month = date_series.dt.days_in_month
        df_processed["days_from_month_start"] = (date_series.dt.day - 1).astype(int)
        df_processed["days_to_month_end"] = (days_in_month - date_series.dt.day).astype(int)

        quarter_end_month = date_series.dt.quarter * 3
        quarter_start_month = quarter_end_month - 2
        quarter_start = pd.to_datetime(
            {"year": date_series.dt.year, "month": quarter_start_month, "day": 1}
        )
        quarter_end = quarter_start + pd.offsets.MonthEnd(2)

        df_processed["days_from_quarter_start"] = (date_series - quarter_start).dt.days.astype(int)
        df_processed["days_to_quarter_end"] = (quarter_end - date_series).dt.days.astype(int)

        year_start = pd.to_datetime({"year": date_series.dt.year, "month": 1, "day": 1})
        year_end = pd.to_datetime({"year": date_series.dt.year, "month": 12, "day": 31})

        df_processed["days_from_year_start"] = (date_series - year_start).dt.days.astype(int)
        df_processed["days_to_year_end"] = (year_end - date_series).dt.days.astype(int)

        df_processed["is_week_of_quarter_end"] = (
            df_processed["days_to_quarter_end"].abs() <= 5
        ).astype(int)
        df_processed["is_week_of_month_end"] = (df_processed["days_to_month_end"] <= 5).astype(int)

        prev_gap = date_series.diff().dt.days.fillna(1).astype(int) - 1
        next_gap = (-date_series.diff(-1).dt.days.fillna(1).astype(int)) - 1
        prev_gap = prev_gap.clip(lower=0)
        next_gap = next_gap.clip(lower=0)

        df_processed["prev_gap_days"] = prev_gap.values
        df_processed["next_gap_days"] = next_gap.values
        df_processed["is_post_holiday"] = (df_processed["prev_gap_days"] >= 2).astype(int)
        df_processed["is_pre_holiday"] = (df_processed["next_gap_days"] >= 2).astype(int)
        df_processed["long_break_span"] = df_processed[["prev_gap_days", "next_gap_days"]].max(
            axis=1
        )
        df_processed["is_quarter_turn"] = (
            (df_processed["is_quarter_start"] + df_processed["is_quarter_end"]) > 0
        ).astype(int)

        return df_processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from virgo_trader.data.data_fetcher import get_stock_data

    raw_data = get_stock_data("600519", "20220101", "20230101")
    if not raw_data.empty:
        engineer = FeatureEngineer()

        # 模拟训练/测试分割
        train_size = int(len(raw_data) * 0.8)
        train_df = raw_data.iloc[:train_size]
        test_df = raw_data.iloc[train_size:]

        # Process both datasets
        processed_train_data = engineer.process(train_df)
        processed_test_data = engineer.process(test_df)

        logger.info("--- Processed Train Data ---\n%s", processed_train_data.head())
        logger.info("--- Processed Test Data ---\n%s", processed_test_data.head())
        logger.info("Train Data Description:\n%s", processed_train_data.describe())
        logger.info("Test Data Description:\n%s", processed_test_data.describe())
