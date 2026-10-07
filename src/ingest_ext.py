"""Fetch paginado: mas historial que el limite de 1000 de Binance."""
from __future__ import annotations

import time

import pandas as pd
from loguru import logger


def fetch_ohlcv_paginated(symbol: str, timeframe: str, total: int = 5000) -> pd.DataFrame:
    import ccxt

    ex = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    tf_ms = ex.parse_timeframe(timeframe) * 1000  # parse_timeframe devuelve SEGUNDOS
    # Empezamos desde el pasado para no pedir velas futuras (Binance devuelve vacio)
    since = ex.milliseconds() - total * tf_ms
    all_rows: list = []
    remaining = total
    while remaining > 0:
        limit = min(1000, remaining)
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            logger.warning(f"paginacion fallo tras {len(all_rows)} velas: {e}")
            break
        if not batch:
            break
        all_rows.extend(batch)
        since = batch[-1][0] + tf_ms
        remaining -= len(batch)
        if len(batch) < limit:
            break
        time.sleep(0.2)

    df = pd.DataFrame(all_rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.info(f"PAGINADO {symbol} {timeframe}: {len(df)} velas")
    return df