"""Ingesta OHLCV via ccxt (multi-exchange) con cache y reintentos.

Contrato: devuelve DataFrame con columnas
timestamp[UTC tz-aware], open, high, low, close, volume + stale, is_closed.
NUNCA dibuja. Solo datos limpios.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import ccxt
import pandas as pd
from loguru import logger

from config.settings import SETTINGS, cache_path_for


@dataclass
class FetchResult:
    df: pd.DataFrame
    stale: bool
    source: str  # 'live' | 'cache'
    latency_ms: float = 0.0


def _build_exchange(exchange_id: str = "binance"):
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Exchange '{exchange_id}' no existe en ccxt")
    klass = getattr(ccxt, exchange_id)
    return klass({"enableRateLimit": True, "timeout": 20000})


def _normalize_ohlcv(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def get_ohlcv(symbol: str | None = None, timeframe: str | None = None, limit: int | None = None) -> FetchResult:
    symbol = symbol or SETTINGS.symbol
    timeframe = timeframe or SETTINGS.timeframe
    limit = limit or SETTINGS.limit
    cache_path = cache_path_for(symbol, timeframe)

    t0 = time.perf_counter()
    # 1) Intento live con reintentos
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            ex = _build_exchange(SETTINGS.exchange_id)
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = _normalize_ohlcv(raw)
            if len(df) < 30:
                raise RuntimeError(f"Respuesta muy corta: {len(df)} velas")
            # Marca vela actual como no cerrada
            df["is_closed"] = True
            df.loc[df.index[-1], "is_closed"] = False
            df["stale"] = False
            # Guarda cache del ultimo fetch bueno
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            df.to_csv(cache_path, index=False)
            latency = (time.perf_counter() - t0) * 1000
            logger.info(f"LIVE {symbol} {timeframe} n={len(df)} lat={latency:.0f}ms intento={attempt}")
            return FetchResult(df=df, stale=False, source="live", latency_ms=latency)
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning(f"fetch intento {attempt} fallo: {e}")
            time.sleep(2 * attempt)

    # 2) Fallback a cache
    latency = (time.perf_counter() - t0) * 1000
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        # asegura tz-aware UTC
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        if "is_closed" not in df.columns:
            df["is_closed"] = True
        df["stale"] = True
        logger.warning(f"CACHE {symbol} {timeframe} n={len(df)} causa={last_err}")
        return FetchResult(df=df, stale=True, source="cache", latency_ms=latency)

    raise RuntimeError(f"Sin conexion y sin cache ({cache_path}): {last_err}")


def fetch_ticker_24h(symbol: str | None = None) -> dict:
    """Precio actual, cambio 24h, alto/bajo. Si falla, {}."""
    symbol = symbol or SETTINGS.symbol
    try:
        ex = _build_exchange(SETTINGS.exchange_id)
        t = ex.fetch_ticker(symbol)
        return {
            "last": _safe_float(t.get("last"), 0.0),
            "pct_24h": _safe_float(t.get("percentage"), 0.0),
            "high_24h": _safe_float(t.get("high"), 0.0),
            "low_24h": _safe_float(t.get("low"), 0.0),
            "base_vol": _safe_float(t.get("baseVolume"), 0.0),
            "quote_vol": _safe_float(t.get("quoteVolume"), 0.0),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"ticker fallo: {e}")
        return {}
