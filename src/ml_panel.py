"""src/ml_panel.py
Panel ML V2 integrado al dashboard.
Uso honesto: solo senala si la precision >=80% (validada out-of-sample) + confluencia tecnica.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scorer import score_signal  # noqa: E402


def has_trained_model(symbol: str) -> bool:
    """True si existe algun bundle entrenado para el simbolo (sin cargarlo)."""
    import glob
    safe = symbol.replace("/", "_").replace(":", "_")
    hits = (
        sorted(glob.glob(f"models/signal_{safe}_*.joblib"))
        + sorted(glob.glob(f"models/cal_{safe}_*.joblib"))
        + sorted(glob.glob(f"models/rf_{safe}_*.joblib"))
    )
    return bool(hits)


def ml_signal(df, symbol: str, timeframe: str = "15m", horizon: int = 6,
              source: str = "local", api_base: str = "http://127.0.0.1:8000") -> dict:
    """
    Devuelve un diccionario con:
      - label: COMPRAR / VENDER / SIN SENAL / SIN MODELO
      - proba: probabilidad calibrada (solo si senal)
      - confidence: 0..1 (confluencia de condiciones + calibracion)
      - precision: precision historica validada (0..1)
      - n_test: cantidad de muestras validating
      - calibrated: bool
      - conditions: {adx, rsi, bb_pct, vol_ratio, trend_strength, roc_5}
      - note: mensaje explicativo

    source: "local" (modelo joblib) o "api" (endpoint /predict de FastAPI).
    """
    if source == "api":
        try:
            import httpx
            r = httpx.get(
                f"{api_base.rstrip('/')}/predict",
                params={"symbol": symbol, "timeframe": timeframe},
                timeout=8,
            )
            r.raise_for_status()
            ml = r.json().get("ml") or {}
            ml.setdefault("ok", True)
            return ml
        except Exception as e:  # noqa: BLE001
            return {"label": "ERROR API", "ok": False, "note": str(e)[:120],
                    "prob_up": None, "prob_down": None, "fire": False,
                    "precision": 0.0, "wilson_lb": 0.0, "n_test": 0}
    return score_signal(df, symbol, timeframe, horizon=horizon)


if __name__ == "__main__":
    import json
    from src.ingest_ext import fetch_ohlcv_paginated
    from src.indicators import add_indicators
    from src.features_v2 import add_features_v2

    raw = fetch_ohlcv_paginated("BTC/USDT", "15m", total=1000)
    raw["is_closed"] = True
    df = add_features_v2(add_indicators(raw))
    res = score_signal(df, "BTC/USDT", "15m", horizon=6)
    print(json.dumps(res, default=str, indent=2))