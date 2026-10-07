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


def ml_signal(df, symbol: str, timeframe: str = "15m", horizon: int = 6) -> dict:
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
    """
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