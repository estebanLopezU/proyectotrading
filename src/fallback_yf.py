"""Respaldo Yahoo Finance cuando Binance/ccxt no disponible.
Convierte AAPL/MSFT/BTC-USD a DataFrame OHLCV estandar UTC.
"""
from __future__ import annotations

import pandas as pd


def yf_to_ohlcv(symbol: str = "BTC-USD", period: str = "7d", interval: str = "15m") -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"Yahoo sin datos para {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df.reset_index()
    dtcol = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
    df["timestamp"] = pd.to_datetime(df[dtcol], utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
    df["is_closed"] = True
    df.loc[df.index[-1], "is_closed"] = False
    df["stale"] = False
    return df.sort_values("timestamp").reset_index(drop=True)
