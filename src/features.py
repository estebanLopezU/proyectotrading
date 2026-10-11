"""Features ML sin look-ahead: solo pasado -> target futuro.
Target: 1 si close en +horizon velas > close actual, 0 si no.

NOTA: Este modulo es un wrapper retrocompatible de features_v2.
Toda la logica real vive en features_v2.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features_v2 import add_features_v2

# Columnas usadas por el modelo (definicion unificada)
FEATURE_COLS = [
    "trend_strength", "rsi14", "stoch_k", "macd", "macd_hist",
    "bb_pct", "bb_width", "atr_pct", "roc_5", "roc_10",
    "adx14", "vol_ratio", "ret_1", "vol_20", "close_pos_range",
]


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper retrocompatible: delega en add_features_v2."""
    return add_features_v2(df)


def make_dataset(df: pd.DataFrame, horizon: int = 5):
    """Crea dataset de entrenamiento sin look-ahead.

    Args:
        df: DataFrame con columnas OHLCV (timestamp, open, high, low, close, volume)
        horizon: numero de velas futuras para definir el target

    Returns:
        X: DataFrame de features
        y: Series de targets (1 = sube, 0 = baja)
        cols: lista de columnas usadas
        d: DataFrame completo con todas las columnas
    """
    d = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Anade todas las features v2
    d = add_features_v2(d)

    # Asegurar que todas las columnas existan
    for col in FEATURE_COLS:
        if col not in d.columns:
            d[col] = np.nan

    # Target: 1 si el precio sube +horizon velas, 0 si baja
    d["target"] = (d["close"].shift(-horizon) > d["close"]).astype(int)

    # Eliminar filas con datos incompletos
    d = d.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

    X = d[FEATURE_COLS]
    y = d["target"].astype(int)
    return X, y, FEATURE_COLS, d


def latest_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Obtiene las features del ultimo tick para clasificar.

    Args:
        df: DataFrame con columnas OHLCV
        cols: lista de columnas a retornar

    Returns:
        DataFrame con 1 fila (la ultima vela) y las columnas solicitadas
    """
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    d = add_features_v2(d)

    # Asegurar que todas las columnas esperadas esten presentes
    for col in cols:
        if col not in d.columns:
            d[col] = np.nan

    return d[cols].tail(1)
