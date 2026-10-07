"""Features ML sin look-ahead: solo pasado -> target futuro.
Target: 1 si close en +horizon velas > close actual, 0 si no.
"""
from __future__ import annotations

import pandas as pd

FEATURE_COLS = [
    "rsi14", "macd", "macd_hist", "bb_pct", "atr_pct",
    "ret_1", "vol_20", "sma20", "sma50", "ema12", "ema26", "close",
]


def make_dataset(df: pd.DataFrame, horizon: int = 5):
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    d["sma_ratio"] = d["sma20"] / d["sma50"].replace(0, float("nan"))
    d["ema_diff"] = (d["ema12"] - d["ema26"]) / d["close"]
    d["close_sma20"] = d["close"] / d["sma20"]
    cols = ["rsi14", "macd", "macd_hist", "bb_pct", "atr_pct",
            "ret_1", "vol_20", "sma_ratio", "ema_diff", "close_sma20"]
    d["target"] = (d["close"].shift(-horizon) > d["close"]).astype(int)
    d = d.dropna(subset=cols + ["target"]).reset_index(drop=True)
    return d[cols], d["target"].astype(int), cols, d


def latest_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    d["sma_ratio"] = d["sma20"] / d["sma50"].replace(0, float("nan"))
    d["ema_diff"] = (d["ema12"] - d["ema26"]) / d["close"]
    d["close_sma20"] = d["close"] / d["sma20"]
    return d[cols].tail(1)
