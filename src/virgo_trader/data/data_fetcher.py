"""Market data fetching utilities (OHLCV) with local caching.

Sources include AkShare, Sina, and Tencent endpoints. Pickle caches are used
only for locally trusted data to speed up repeated experiments.
"""

import json
import logging
import random
import re

import akshare as ak
import pandas as pd
import requests

from ..utils.paths import MARKET_DATA_CACHE_DIR, ensure_dir

logger = logging.getLogger(__name__)


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has date index and standard OHLCV columns."""
    if df.empty:
        return pd.DataFrame()
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.columns = df.columns.astype(str)
    return df.sort_index()


def _is_end_date_covered(
    last_date: pd.Timestamp, end_date: pd.Timestamp, tolerance_days: int = 7
) -> bool:
    """
    Whether time series coverage is considered sufficient up to `end_date`.

    We allow a small `tolerance_days` window to accommodate cases where `end_date` is a non-trading day
    (e.g. backtests often pass `YYYY0101` as the "end of previous year").
    """
    if last_date is None or pd.isna(last_date):
        return False
    if last_date >= end_date:
        return True
    try:
        return (end_date - last_date).days <= tolerance_days
    except Exception:
        return False


def _fetch_from_akshare_etf(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    etf_hist_df = ak.fund_etf_hist_em(
        symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq"
    )
    if etf_hist_df.empty:
        return pd.DataFrame()
    etf_hist_df.rename(
        columns={
            "\u65e5\u671f": "date",
            "\u5f00\u76d8": "open",
            "\u6536\u76d8": "close",
            "\u6700\u9ad8": "high",
            "\u6700\u4f4e": "low",
            "\u6210\u4ea4\u91cf": "volume",
        },
        inplace=True,
    )
    return _standardize_dataframe(etf_hist_df)


def _fetch_from_sina_etf(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Sina 接口需要 "sh"/"sz" 前缀
    normalized = stock_code.lower()
    if normalized.endswith(".sh"):
        prefix = "sh"
        core = normalized.replace(".sh", "")
    elif normalized.endswith(".sz"):
        prefix = "sz"
        core = normalized.replace(".sz", "")
    else:
        # 默认按照长度判断，A股代码通常为 6 位
        prefix = "sh" if normalized.startswith(("5", "6")) else "sz"
        core = normalized

    sina_symbol = f"{prefix}{core}"
    try:
        df = ak.fund_etf_hist_sina(symbol=sina_symbol)
    except Exception as exc:
        logger.warning("Sina ETF fetch failed for %s: %s", stock_code, exc)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df.rename(
        columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
        inplace=True,
    )

    standardized = _standardize_dataframe(df)
    if standardized.empty:
        return standardized

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    return standardized.loc[(standardized.index >= start_ts) & (standardized.index <= end_ts)]


def _fetch_from_tx_stock(
    core_code: str, market: str, start_date: str, end_date: str
) -> pd.DataFrame:
    prefix = "sh" if market.upper() == "SH" else "sz"
    symbol = f"{prefix}{core_code}"
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    start_year = start_ts.year
    end_year = end_ts.year
    frames = []
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"

    for year in range(start_year, end_year + 1):
        params = {
            "_var": f"kline_dayqfq{year}",
            "param": f"{symbol},day,{year}-01-01,{year + 1}-12-31,640,qfq",
            # Non-cryptographic random used only as a cache-busting query parameter.
            "r": f"0.{str(random.random()).split('.')[-1]}",  # noqa: S311
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36",
            "Referer": "https://gu.qq.com",
        }
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Tencent stock fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()

        payload_start = response.text.find("=")
        if payload_start == -1:
            continue
        try:
            data_json = json.loads(response.text[payload_start + 1 :])
        except json.JSONDecodeError:
            continue
        symbol_data = data_json.get("data", {}).get(symbol)
        if not symbol_data:
            continue
        day_list = symbol_data.get("qfqday") or symbol_data.get("day") or []
        if not day_list:
            continue
        temp_df = pd.DataFrame(day_list)
        if temp_df.empty:
            continue
        temp_df = temp_df.iloc[:, :6]
        temp_df.columns = ["date", "open", "close", "high", "low", "volume"]
        frames.append(temp_df)

    if not frames:
        return pd.DataFrame()

    big_df = pd.concat(frames, ignore_index=True)
    big_df["date"] = pd.to_datetime(big_df["date"], errors="coerce")
    big_df.dropna(subset=["date"], inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        big_df[col] = pd.to_numeric(big_df[col], errors="coerce")
    big_df.dropna(subset=numeric_cols, inplace=True)
    big_df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    big_df.sort_values("date", inplace=True)
    big_df = big_df[(big_df["date"] >= start_ts) & (big_df["date"] <= end_ts)]
    if big_df.empty:
        return pd.DataFrame()

    big_df = big_df[["date", "open", "high", "low", "close", "volume"]]
    return _standardize_dataframe(big_df)


def _fetch_from_sina_stock(
    core_code: str, market: str, start_date: str, end_date: str
) -> pd.DataFrame:
    prefix = "sh" if market.upper() == "SH" else "sz"
    symbol = f"{prefix}{core_code}"
    url = (
        "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_kline=/CN_MarketDataService.getKLineData"
    )
    params = {
        "symbol": symbol,
        "scale": "240",
        "ma": "5",
        "datalen": "1500",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36",
        "Referer": "https://finance.sina.com.cn",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Sina stock fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()

    match = re.search(r"var\s+[^\n=]+=\s*(\(.+\))\s*;", resp.text, flags=re.S)
    if not match:
        return pd.DataFrame()
    payload = match.group(1).strip()
    if payload.startswith("(") and payload.endswith(")"):
        payload = payload[1:-1]
    payload = payload.strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    required_cols = {"day", "open", "high", "low", "close", "volume"}
    if not required_cols <= set(df.columns):
        return pd.DataFrame()

    df = df[["day", "open", "high", "low", "close", "volume"]].copy()
    df.rename(columns={"day": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    df.sort_values("date", inplace=True)

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    if df.empty:
        return pd.DataFrame()
    df.rename(columns={"volume": "volume"}, inplace=True)
    return _standardize_dataframe(df)


def _fetch_from_akshare_stock_daily(
    core_code: str, market: str, start_date: str, end_date: str, adjust: str = "qfq"
) -> pd.DataFrame:
    """Fetch A-share daily data from AkShare (qfq/hfq supported)."""
    prefix = "sh" if market.upper() == "SH" else "sz"
    symbol = f"{prefix}{core_code}"
    try:
        df = ak.stock_zh_a_daily(symbol=symbol, adjust=adjust)
    except Exception as exc:
        logger.warning("AkShare stock daily fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    standardized = _standardize_dataframe(df)
    if standardized.empty:
        return standardized

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    return standardized.loc[(standardized.index >= start_ts) & (standardized.index <= end_ts)]


def _normalize_security_code(stock_code: str) -> tuple[str, str]:
    code = (stock_code or "").strip().upper()
    if not code:
        return "", ""
    if "." in code:
        core, market = code.split(".", 1)
    else:
        if code.startswith(("5", "6", "9")):
            market = "SH"
        else:
            market = "SZ"
        core = code
    return core, market


def _is_probable_etf(code: str) -> bool:
    return code.startswith(("5", "15", "16", "51", "56", "58", "159"))


def _cache_key(stock_code: str, prefer_qfq: bool) -> str:
    normalized = (stock_code or "").strip().upper()
    normalized = normalized.replace(".", "_")
    normalized = normalized.replace("/", "_")
    tag = "qfq" if prefer_qfq else "raw"
    return f"{normalized}__{tag}"


def _try_load_cache(
    stock_code: str,
    start_date: str,
    end_date: str,
    *,
    prefer_qfq: bool,
) -> tuple[pd.DataFrame, bool]:
    cache_path = ensure_dir(MARKET_DATA_CACHE_DIR) / f"{_cache_key(stock_code, prefer_qfq)}.pkl"
    if not cache_path.exists():
        return pd.DataFrame(), False
    try:
        # Local trusted cache only; never load untrusted pickle files.
        cached = pd.read_pickle(cache_path)  # noqa: S301
    except Exception as exc:
        logger.warning("Failed to read market cache %s: %s", cache_path, exc)
        return pd.DataFrame(), False

    if cached is None or getattr(cached, "empty", True):
        return pd.DataFrame(), False

    if not isinstance(cached.index, pd.DatetimeIndex):
        try:
            cached = cached.copy()
            cached.index = pd.to_datetime(cached.index, errors="coerce")
        except Exception:
            return pd.DataFrame(), False

    cached = cached.sort_index()
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    if cached.index.min() <= start_ts and _is_end_date_covered(cached.index.max(), end_ts):
        return cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)], True

    sliced = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)]
    if not sliced.empty:
        logger.info(
            "Market cache hit (partial) for %s: cached=[%s,%s] requested=[%s,%s]",
            stock_code,
            cached.index.min().date(),
            cached.index.max().date(),
            start_ts.date(),
            end_ts.date(),
        )
    return sliced, False


def _persist_cache(stock_code: str, df: pd.DataFrame, *, prefer_qfq: bool) -> None:
    if df is None or df.empty:
        return

    cache_path = ensure_dir(MARKET_DATA_CACHE_DIR) / f"{_cache_key(stock_code, prefer_qfq)}.pkl"
    try:
        existing = pd.DataFrame()
        if cache_path.exists():
            try:
                # Local trusted cache only; never load untrusted pickle files.
                existing = pd.read_pickle(cache_path)  # noqa: S301
            except Exception:
                existing = pd.DataFrame()

        combined = df if existing is None or existing.empty else pd.concat([existing, df], axis=0)
        if not isinstance(combined.index, pd.DatetimeIndex):
            combined = combined.copy()
            combined.index = pd.to_datetime(combined.index, errors="coerce")
        combined = combined[~combined.index.isna()]
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_pickle(cache_path)
    except Exception as exc:
        logger.warning("Failed to persist market cache %s: %s", cache_path, exc)


def get_stock_data(
    stock_code: str, start_date: str, end_date: str, *, prefer_qfq: bool = True
) -> pd.DataFrame:
    """
    获取股票/ETF 日线行情数据（OHLCV）。

    Args:
        stock_code: 证券代码，例如 '600519.SH'/'600519' 或 '510050.SH'.
        start_date: 起始日期 'YYYYMMDD'.
        end_date: 结束日期 'YYYYMMDD'.
        prefer_qfq: 优先使用前复权（qfq）数据源。

    Returns:
        标准化后的行情数据 DataFrame，index 为 datetime，包含 open/high/low/close/volume。
    """
    core_code, market = _normalize_security_code(stock_code)
    if not core_code:
        return pd.DataFrame()

    normalized_code = f"{core_code}.{market}"
    cached, cache_complete = _try_load_cache(
        normalized_code, start_date, end_date, prefer_qfq=prefer_qfq
    )
    if cache_complete and not cached.empty:
        return cached
    is_etf = _is_probable_etf(core_code)
    end_ts = pd.to_datetime(end_date)

    if is_etf:
        if prefer_qfq:
            ak_df = _fetch_from_akshare_etf(core_code, start_date, end_date)
            if not ak_df.empty:
                _persist_cache(normalized_code, ak_df, prefer_qfq=prefer_qfq)
                if _is_end_date_covered(ak_df.index.max(), end_ts):
                    return ak_df
                logger.warning(
                    "AkShare ETF data incomplete for %s (max=%s, end=%s); trying fallback sources.",
                    normalized_code,
                    ak_df.index.max().date(),
                    end_ts.date(),
                )

            sina_df = _fetch_from_sina_etf(normalized_code, start_date, end_date)
            if not sina_df.empty:
                logger.warning("Using non-qfq ETF data for %s (Sina fallback).", normalized_code)
                _persist_cache(normalized_code, sina_df, prefer_qfq=prefer_qfq)
                if not _is_end_date_covered(sina_df.index.max(), end_ts):
                    logger.warning(
                        "Sina ETF data incomplete for %s (max=%s, end=%s).",
                        normalized_code,
                        sina_df.index.max().date(),
                        end_ts.date(),
                    )
                return sina_df
        else:
            sina_df = _fetch_from_sina_etf(normalized_code, start_date, end_date)
            if not sina_df.empty:
                _persist_cache(normalized_code, sina_df, prefer_qfq=prefer_qfq)
                if _is_end_date_covered(sina_df.index.max(), end_ts):
                    return sina_df
                logger.warning(
                    "Sina ETF data incomplete for %s (max=%s, end=%s); trying fallback sources.",
                    normalized_code,
                    sina_df.index.max().date(),
                    end_ts.date(),
                )

            ak_df = _fetch_from_akshare_etf(core_code, start_date, end_date)
            if not ak_df.empty:
                _persist_cache(normalized_code, ak_df, prefer_qfq=prefer_qfq)
                if not _is_end_date_covered(ak_df.index.max(), end_ts):
                    logger.warning(
                        "AkShare ETF data incomplete for %s (max=%s, end=%s).",
                        normalized_code,
                        ak_df.index.max().date(),
                        end_ts.date(),
                    )
                return ak_df

        logger.warning("Unable to fetch ETF data for %s.", normalized_code)
        if not cached.empty:
            logger.warning(
                "Falling back to partial cached ETF data for %s (requested=[%s,%s]).",
                normalized_code,
                start_date,
                end_date,
            )
            return cached
        return pd.DataFrame()

    if prefer_qfq:
        tx_df = _fetch_from_tx_stock(core_code, market, start_date, end_date)
        if not tx_df.empty:
            _persist_cache(normalized_code, tx_df, prefer_qfq=prefer_qfq)
            if _is_end_date_covered(tx_df.index.max(), end_ts):
                return tx_df
            logger.warning(
                "Tencent stock data incomplete for %s (max=%s, end=%s); trying fallback sources.",
                normalized_code,
                tx_df.index.max().date(),
                end_ts.date(),
            )

        ak_df = _fetch_from_akshare_stock_daily(
            core_code, market, start_date, end_date, adjust="qfq"
        )
        if not ak_df.empty:
            _persist_cache(normalized_code, ak_df, prefer_qfq=prefer_qfq)
            if _is_end_date_covered(ak_df.index.max(), end_ts):
                return ak_df
            logger.warning(
                "AkShare stock data incomplete for %s (max=%s, end=%s); trying fallback sources.",
                normalized_code,
                ak_df.index.max().date(),
                end_ts.date(),
            )

        sina_df = _fetch_from_sina_stock(core_code, market, start_date, end_date)
        if not sina_df.empty:
            logger.warning("Using non-qfq stock data for %s (Sina fallback).", normalized_code)
            _persist_cache(normalized_code, sina_df, prefer_qfq=prefer_qfq)
            if not _is_end_date_covered(sina_df.index.max(), end_ts):
                logger.warning(
                    "Sina stock data incomplete for %s (max=%s, end=%s).",
                    normalized_code,
                    sina_df.index.max().date(),
                    end_ts.date(),
                )
            return sina_df
    else:
        sina_df = _fetch_from_sina_stock(core_code, market, start_date, end_date)
        if not sina_df.empty:
            _persist_cache(normalized_code, sina_df, prefer_qfq=prefer_qfq)
            if _is_end_date_covered(sina_df.index.max(), end_ts):
                return sina_df
            logger.warning(
                "Sina stock data incomplete for %s (max=%s, end=%s); trying fallback sources.",
                normalized_code,
                sina_df.index.max().date(),
                end_ts.date(),
            )

        tx_df = _fetch_from_tx_stock(core_code, market, start_date, end_date)
        if not tx_df.empty:
            logger.info("Falling back to Tencent source for %s.", normalized_code)
            _persist_cache(normalized_code, tx_df, prefer_qfq=prefer_qfq)
            if _is_end_date_covered(tx_df.index.max(), end_ts):
                return tx_df
            logger.warning(
                "Tencent stock data incomplete for %s (max=%s, end=%s); trying fallback sources.",
                normalized_code,
                tx_df.index.max().date(),
                end_ts.date(),
            )

        ak_df = _fetch_from_akshare_stock_daily(core_code, market, start_date, end_date, adjust="")
        if not ak_df.empty:
            logger.info("Falling back to AkShare daily source for %s.", normalized_code)
            _persist_cache(normalized_code, ak_df, prefer_qfq=prefer_qfq)
            if not _is_end_date_covered(ak_df.index.max(), end_ts):
                logger.warning(
                    "AkShare stock data incomplete for %s (max=%s, end=%s).",
                    normalized_code,
                    ak_df.index.max().date(),
                    end_ts.date(),
                )
            return ak_df

    if not cached.empty:
        logger.warning(
            "Falling back to partial cached stock data for %s (requested=[%s,%s]).",
            normalized_code,
            start_date,
            end_date,
        )
        return cached

    raise ValueError(f"无法获取行情数据: {normalized_code}")


if __name__ == "__main__":
    # 这是一个简单的使用示例
    # 获取贵州茅台从2021年1月1日到2023年1月1日的数据
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    maotai_data = get_stock_data(stock_code="600519", start_date="20210101", end_date="20230101")
    if not maotai_data.empty:
        logger.info(
            "Successfully fetched data for 600519 (Kweichow Moutai). Rows=%d", len(maotai_data)
        )
