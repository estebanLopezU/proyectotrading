"""Features ML sin look-ahead: solo pasado -> target futuro.
Target: 1 si close en +horizon velas > close actual, 0 si no.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "rsi14", "macd", "macd_hist", "macd_abs", "bb_pct", "bb_width", "atr_pct",
    "ret_1", "vol_20", "sma20", "sma50", "ema12", "ema26", "close",
]


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features adicionales para el modelo sin usar datos futuros."""
    d = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Features básicas que ya existían
    d["sma_ratio"] = d["sma20"] / d["sma50"].replace(0, float("nan"))
    d["ema_diff"] = (d["ema12"] - d["ema26"]) / d["close"]
    d["close_sma20"] = d["close"] / d["sma20"]

    # Nuevas features
    # Absoluto del MACD (intensidad del movimiento)
    d["macd_abs"] = d["macd"].abs()

    # Ancho de las bandas de Bollinger relativo
    d["bb_width"] = (d["bb_high"] - d["bb_low"]) / d["bb_mid"].replace(0, np.nan)

    # Relación precio/SMA50 (cómo se sitúa respecto a la media a largo plazo)
    d["close_vs_sma50"] = d["close"] / d["sma50"].replace(0, np.nan)

    # Momentum de 5 velas (cambio acumulado)
    d["momentum_5"] = d["close"].pct_change(5)

    # Volatilidad 5 velas (desviación estándar de retornos)
    d["vol_5"] = d["ret_1"].rolling(5).std() * np.sqrt(5) * 100.0

    return d


def make_dataset(df: pd.DataFrame, horizon: int = 5):
    d = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Añadimos todas las features (básicas + extras)
    d = add_ml_features(d)

    # Columnas que se usarán como entrada al modelo
    cols = [
        "rsi14", "macd", "macd_hist", "macd_abs", "bb_pct", "bb_width",
        "atr_pct", "ret_1", "vol_20", "sma_ratio", "ema_diff",
        "close_sma20", "close_vs_sma50", "momentum_5", "vol_5"
    ]

    # Asegurar que todas las columnas existan (por si hay pocos datos)
    for col in cols:
        if col not in d.columns:
            d[col] = np.nan

    # Target: 1 si el precio sube +horizon velas, 0 si baja
    d["target"] = (d["close"].shift(-horizon) > d["close"]).astype(int)

    # Eliminar filas con datos incompletos
    d = d.dropna(subset=cols + ["target"]).reset_index(drop=True)

    X = d[cols]
    y = d["target"].astype(int)
    return X, y, cols, d


def latest_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Obtiene las features del último tick para clasificar."""
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    d = add_ml_features(d)

    # Asegurar que todas las columnas esperadas estén presentes
    for col in cols:
        if col not in d.columns:
            d[col] = np.nan

    return d[cols].tail(1)
