"""Indicadores quant sin look-ahead bias (pandas/numpy puro).
Sin dependencia 'ta' para maxima compatibilidad Python 3.12.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()


def _ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=w, adjust=False, min_periods=w).mean()


def _rsi(close: pd.Series, w: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / w, adjust=False, min_periods=w).mean()
    al = loss.ewm(alpha=1 / w, adjust=False, min_periods=w).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    close, high, low = out["close"], out["high"], out["low"]

    out["sma20"] = _sma(close, 20)
    out["sma50"] = _sma(close, 50)
    out["ema12"] = _ema(close, 12)
    out["ema26"] = _ema(close, 26)
    out["rsi14"] = _rsi(close, 14)

    e12 = _ema(close, 12)
    e26 = _ema(close, 26)
    out["macd"] = e12 - e26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    mid = _sma(close, 20)
    sd = close.rolling(20, min_periods=20).std()
    out["bb_mid"] = mid
    out["bb_high"] = mid + 2 * sd
    out["bb_low"] = mid - 2 * sd
    rng = (out["bb_high"] - out["bb_low"]).replace(0, np.nan)
    out["bb_pct"] = (close - out["bb_low"]) / rng

    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=14).mean()
    out["atr_pct"] = out["atr14"] / close * 100.0

    out["ret_1"] = close.pct_change()
    out["vol_20"] = out["ret_1"].rolling(20).std() * np.sqrt(20)

    out["cross_golden"] = (out["sma20"] > out["sma50"]) & (out["sma20"].shift(1) <= out["sma50"].shift(1))
    out["cross_death"] = (out["sma20"] < out["sma50"]) & (out["sma20"].shift(1) >= out["sma50"].shift(1))
    out[["cross_golden", "cross_death"]] = out[["cross_golden", "cross_death"]].fillna(False)
    return out


def latest_signal(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"label": "SIN DATOS", "score": 0, "reasons": []}
    r = df.iloc[-1]
    reasons: list[str] = []
    score = 0
    rsi = float(r.get("rsi14", 50) or 50)
    if rsi < 30:
        score += 2
        reasons.append(f"RSI {rsi:.1f} sobreventa")
    elif rsi > 70:
        score -= 2
        reasons.append(f"RSI {rsi:.1f} sobrecompra")
    else:
        reasons.append(f"RSI {rsi:.1f} neutral")
    if bool(r.get("cross_golden", False)):
        score += 2
        reasons.append("Cruce dorado SMA20>SMA50")
    if bool(r.get("cross_death", False)):
        score -= 2
        reasons.append("Cruce muerte SMA20<SMA50")
    mh = float(r.get("macd_hist", 0) or 0)
    if mh > 0:
        score += 1
        reasons.append("MACD positivo")
    else:
        score -= 1
        reasons.append("MACD negativo")
    try:
        bbp = float(r.get("bb_pct", 0.5))
        if bbp < 0.05:
            score += 1
            reasons.append("Banda inferior Bollinger")
        elif bbp > 0.95:
            score -= 1
            reasons.append("Banda superior Bollinger")
    except Exception:
        pass
    label = "COMPRAR (regla)" if score >= 3 else ("VENDER (regla)" if score <= -3 else "MANTENER (regla)")
    return {"label": label, "score": score, "reasons": reasons}

