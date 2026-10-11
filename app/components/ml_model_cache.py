"""Cache del modelo ML para no recargar en cada rerun."""
from __future__ import annotations

import glob

import joblib
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_ml_model_cached(symbol: str, timeframe: str):
    """Carga el modelo ML una sola vez y lo cachea.

    Args:
        symbol: Símbolo (ej: "BTC/USDT")
        timeframe: Temporalidad (ej: "1h")

    Returns:
        Tupla (modelo, path) o (None, None) si no hay modelo.
    """
    safe = symbol.replace("/", "_").replace(":", "_")

    # Prioriza modelos de señal validados
    hits = sorted(glob.glob(f"models/signal_{safe}_{timeframe}_h*.joblib"))
    if hits:
        return joblib.load(hits[-1]), hits[-1]

    # Luego cualquier modelo de señal para el símbolo
    any_sig = sorted(glob.glob(f"models/signal_{safe}_*.joblib"))
    if any_sig:
        return joblib.load(any_sig[-1]), any_sig[-1]

    # Finalmente cualquier modelo genérico
    for pattern in ["models/cal_*.joblib", "models/rf_*.joblib"]:
        for p in sorted(glob.glob(pattern)):
            try:
                return joblib.load(p), p
            except Exception:  # noqa: BLE001
                continue

    return None, None
