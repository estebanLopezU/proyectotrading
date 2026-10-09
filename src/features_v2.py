"""Features V2: analitica de datos avanzada para alta precision.
Todo sin look-ahead: solo usa informacion hasta la vela actual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import _ema, _rsi, _sma


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, w: int = 14) -> pd.Series:
    ll = low.rolling(w, min_periods=w).min()
    hh = high.rolling(w, min_periods=w).max()
    return ((close - ll) / (hh - ll).replace(0, np.nan) * 100).fillna(50.0)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, w: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = pd.Series(tr, index=close.index).ewm(alpha=1 / w, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1 / w, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1 / w, adjust=False).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1 / w, adjust=False).mean()


def add_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """18+ features analiticas sin look-ahead."""
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    c, h, low, v = d["close"], d["high"], d["low"], d["volume"]

    # --- Tendencia ---
    d["sma20"] = _sma(c, 20)
    d["sma50"] = _sma(c, 50)
    d["ema12"] = _ema(c, 12)
    d["ema26"] = _ema(c, 26)
    d["trend_strength"] = (d["sma20"] - d["sma50"]) / d["sma50"].replace(0, np.nan)
    d["price_vs_ema12"] = (c - d["ema12"]) / d["ema12"].replace(0, np.nan)

    # --- Momentum ---
    d["rsi14"] = _rsi(c, 14)
    d["stoch_k"] = _stoch_k(h, low, c, 14)
    e12, e26 = d["ema12"], d["ema26"]
    d["macd"] = e12 - e26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    d["roc_5"] = c.pct_change(5) * 100
    d["roc_10"] = c.pct_change(10) * 100
    d["roc_20"] = c.pct_change(20) * 100

    # --- Volatilidad ---
    mid = _sma(c, 20)
    sd = c.rolling(20, min_periods=20).std()
    d["bb_high"] = mid + 2 * sd
    d["bb_mid"] = mid
    d["bb_low"] = mid - 2 * sd
    rng = (d["bb_high"] - d["bb_low"]).replace(0, np.nan)
    d["bb_pct"] = (c - d["bb_low"]) / rng
    d["bb_width"] = rng / mid.replace(0, np.nan)
    pc = c.shift(1)
    tr = pd.concat([(h - low), (h - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14, min_periods=14).mean()
    d["atr_pct"] = d["atr14"] / c * 100.0
    d["vol_20"] = c.pct_change().rolling(20).std() * np.sqrt(20)

    # --- Fuerza / tendencia ADX ---
    d["adx14"] = _adx(h, low, c, 14)

    # --- Volumen ---
    d["vol_sma20"] = _sma(v, 20)
    d["vol_ratio"] = v / d["vol_sma20"].replace(0, np.nan)
    d["obv_slope"] = (np.sign(c.diff()).fillna(0) * v).cumsum().pct_change(10)

    # --- Estructura de precio ---
    d["ret_1"] = c.pct_change()
    d["high_low_range"] = (h - low) / c * 100
    d["close_pos_range"] = (c - low) / (h - low).replace(0, np.nan)  # 1=cierra en max
    d["gap_up"] = (c > c.shift(1)).astype(int)

    # --- Ratios de probabilidad (condicionales históricas) ---
    # Probabilidad empírica de subida en ventana móvil (sin look-ahead)
    fut_up = (c.shift(-1) > c).astype(float)
    d["emp_p_up_50"] = fut_up.shift(1).rolling(50, min_periods=20).mean().fillna(0.5)
    # Nivel de RSI en percentil de las ultimas 100 velas
    d["rsi_percentile"] = d["rsi14"].rolling(100, min_periods=30).rank(pct=True).fillna(0.5)

    return d
