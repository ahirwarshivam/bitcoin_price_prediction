"""
Data Ingestion and Synchronization Module.
Fetches the latest Bitcoin and macro-economic data (S&P 500, DXY, Gold, VIX),
handles market holidays/weekends, combines live feeds with historical baselines,
and provides robust error recovery and fallback mechanisms.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    RAW_HISTORICAL_BTC_CSV,
    DXY_CSV,
    VIX_CSV,
    SP500_CSV,
    GOLD_CSV,
    DATA_DIR,
    BINANCE_KLINES_URL,
    BINANCE_SYMBOL,
    YFINANCE_TICKERS
)

CACHE_MERGED_CSV = DATA_DIR / "merged_market_data.csv"


def fetch_binance_btc(start_date: str = "2026-08-20") -> pd.DataFrame:
    """
    Fetch daily BTCUSDT klines from Binance API starting from a given date.
    Returns DataFrame with Open time, Open, High, Low, Close, Volume.
    """
    try:
        start_dt = pd.to_datetime(start_date, utc=True)
        start_ts = int(start_dt.timestamp() * 1000)

        params = {
            "symbol": BINANCE_SYMBOL,
            "interval": "1d",
            "startTime": start_ts,
            "limit": 1000
        }
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Binance API returned status code {response.status_code}: {response.text[:200]}")

        data = response.json()
        if not data:
            return pd.DataFrame()

        columns = [
            "Open time", "Open", "High", "Low", "Close", "Volume",
            "Close time", "Quote asset volume", "Number of trades",
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
        ]
        df = pd.DataFrame(data, columns=columns)
        df["Date"] = pd.to_datetime(df["Open time"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"[Warning] Binance API fetch failed: {e}. Attempting fallback...")
        return pd.DataFrame()


def fetch_yfinance_btc(start_date: str = "2026-08-20") -> pd.DataFrame:
    """Fallback Bitcoin data fetcher via Yahoo Finance."""
    try:
        ticker = YFINANCE_TICKERS["BTC_FALLBACK"]
        start_dt = (pd.to_datetime(start_date) - timedelta(days=5)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start_dt, interval="1d", progress=False)

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Normalize date
        dates = pd.to_datetime(df["Date"])

        if dates.dt.tz is not None:
            df["Date"] = dates.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        else:
            df["Date"] = dates.dt.normalize()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Yahoo Finance BTC-USD volume is quote-currency volume.
        # Convert it to BTC base-asset volume to match Binance/training data.
        df["Volume"] = df["Volume"] / df["Close"]

        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        print(f"[Warning] Yahoo Finance BTC fallback failed: {e}")
        return pd.DataFrame()


def fetch_macro_series(ticker: str, start_date: str = "2026-08-20") -> pd.DataFrame:
    """Fetch daily closing series for a macro instrument from Yahoo Finance."""
    try:
        start_dt = (pd.to_datetime(start_date) - timedelta(days=10)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start_dt, interval="1d", progress=False)
        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            # Yahoo Finance returns multi-index
            if 'Close' in df.columns.levels[0]:
                close_series = df['Close'].iloc[:, 0]
            else:
                close_series = df.iloc[:, 0]
        else:
            close_series = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]

        # Handle index conversion properly
        idx = pd.to_datetime(df.index)
        if getattr(idx, 'tz', None) is not None:
            norm_dates = idx.tz_convert('UTC').tz_localize(None).normalize()
        else:
            norm_dates = idx.normalize()

        res = pd.DataFrame({
            "Date": norm_dates,
            "Close": pd.to_numeric(close_series.values, errors="coerce")
        }).dropna()
        return res
    except Exception as e:
        print(f"[Warning] Yahoo Finance fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def load_historical_baseline() -> pd.DataFrame:
    """Loads and formats the historical baseline dataset up to August 2026."""
    maindf = pd.read_csv(RAW_HISTORICAL_BTC_CSV)
    df = maindf.drop(columns=['Ignore', 'Close time'], errors='ignore')

    dxy = pd.read_csv(DXY_CSV)
    vix = pd.read_csv(VIX_CSV)
    sp500 = pd.read_csv(SP500_CSV)
    gold = pd.read_csv(GOLD_CSV)

    df['Date'] = pd.to_datetime(df['Open time'], utc=True).dt.tz_localize(None).dt.normalize()
    for x in [dxy, vix, sp500, gold]:
        x['Date'] = pd.to_datetime(x['Date'], utc=True).dt.tz_localize(None).dt.normalize()

    dxy = dxy.drop(columns=['Unnamed: 0'], errors='ignore')

    df = df.merge(dxy, on='Date', how='left')
    df = df.merge(vix, on='Date', how='left')
    df = df.merge(sp500, on='Date', how='left')
    df = df.merge(gold, on='Date', how='left')

    ext = ['DXY', 'VIX', 'SP500', 'Gold']
    df[ext] = df[ext].ffill()
    df = df.dropna(subset=ext).reset_index(drop=True)
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'DXY', 'VIX', 'SP500', 'Gold']]


def fetch_all_market_data(force_refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Combines historical dataset with the newest live market data.
    Forward-fills external market indicators on weekends and holidays.
    Returns (cleaned_merged_df, status_info_dict).
    """
    status_info = {
        "status": "OK",
        "btc_source": "Historical Baseline",
        "macro_source": "Historical Baseline",
        "latest_date": None,
        "latest_btc_price": None,
        "is_live_updated": False,
        "error_message": None,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    try:
        # Load baseline
        base_df = load_historical_baseline()
        last_base_date = base_df['Date'].max()
        status_info["latest_date"] = last_base_date.strftime("%Y-%m-%d")
        status_info["latest_btc_price"] = float(base_df.iloc[-1]['Close'])

        # Attempt to fetch live incremental data
        btc_new = fetch_binance_btc(start_date=last_base_date.strftime("%Y-%m-%d"))
        if not btc_new.empty:
            status_info["btc_source"] = "Binance API (Live)"
        else:
            btc_new = fetch_yfinance_btc(start_date=last_base_date.strftime("%Y-%m-%d"))
            if not btc_new.empty:
                status_info["btc_source"] = "Yahoo Finance BTC (Live Fallback)"

        if not btc_new.empty:
            # Fetch updated macro series
            sp500_new = fetch_macro_series(YFINANCE_TICKERS["SP500"], start_date=last_base_date.strftime("%Y-%m-%d"))
            dxy_new = fetch_macro_series(YFINANCE_TICKERS["DXY"], start_date=last_base_date.strftime("%Y-%m-%d"))
            gold_new = fetch_macro_series(YFINANCE_TICKERS["GOLD"], start_date=last_base_date.strftime("%Y-%m-%d"))
            vix_new = fetch_macro_series(YFINANCE_TICKERS["VIX"], start_date=last_base_date.strftime("%Y-%m-%d"))

            # Merge incremental new records
            if not sp500_new.empty:
                btc_new = btc_new.merge(sp500_new.rename(columns={"Close": "SP500"}), on="Date", how="left")
                status_info["macro_source"] = "Yahoo Finance (Live)"
            if not dxy_new.empty:
                btc_new = btc_new.merge(dxy_new.rename(columns={"Close": "DXY"}), on="Date", how="left")
            if not gold_new.empty:
                btc_new = btc_new.merge(gold_new.rename(columns={"Close": "Gold"}), on="Date", how="left")
            if not vix_new.empty:
                btc_new = btc_new.merge(vix_new.rename(columns={"Close": "VIX"}), on="Date", how="left")

            # Concatenate baseline with new incremental observations
            combined = pd.concat([base_df, btc_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)

            # Forward-fill external indicators
            ext_cols = ['SP500', 'DXY', 'Gold', 'VIX']
            for col in ext_cols:
                if col in combined.columns:
                    combined[col] = combined[col].ffill()

            combined = combined.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume']).reset_index(drop=True)

            status_info["is_live_updated"] = True
            status_info["latest_date"] = combined['Date'].max().strftime("%Y-%m-%d")
            status_info["latest_btc_price"] = float(combined.iloc[-1]['Close'])

            # Cache the combined data
            combined.to_csv(CACHE_MERGED_CSV, index=False)
            return combined, status_info

        # If live fetch was empty/not needed, return baseline
        base_df.to_csv(CACHE_MERGED_CSV, index=False)
        return base_df, status_info

    except Exception as e:
        status_info["status"] = "WARNING_USING_CACHED"
        status_info["error_message"] = str(e)
        print(f"[Error in fetch_all_market_data]: {e}")

        # Check if cache exists
        if CACHE_MERGED_CSV.exists():
            cached_df = pd.read_csv(CACHE_MERGED_CSV)
            cached_df['Date'] = pd.to_datetime(cached_df['Date'])
            status_info["latest_date"] = cached_df['Date'].max().strftime("%Y-%m-%d")
            status_info["latest_btc_price"] = float(cached_df.iloc[-1]['Close'])
            return cached_df, status_info

        # Fall back to baseline
        base_df = load_historical_baseline()
        status_info["latest_date"] = base_df['Date'].max().strftime("%Y-%m-%d")
        status_info["latest_btc_price"] = float(base_df.iloc[-1]['Close'])
        return base_df, status_info
