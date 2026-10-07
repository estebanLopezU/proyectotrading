"""Prediccion con modelo entrenado. Si no hay modelo, usa regla Fase 1.
Uso: python scripts/predict_live.py --symbol BTC/USDT --timeframe 15m
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import joblib
from src.features import latest_features
from src.indicators import add_indicators, latest_signal
from src.ingest import get_ohlcv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="15m")
    a = ap.parse_args()
    res = get_ohlcv(a.symbol, a.timeframe, 300)
    df = add_indicators(res.df)
    safe = a.symbol.replace("/", "_")
    path = f"models/rf_{safe}_{a.timeframe}.joblib"
    try:
        bundle = joblib.load(path)
        X = latest_features(df, bundle["cols"])
        proba = float(bundle["model"].predict_proba(X)[0][1])
        label = "COMPRAR (ML)" if proba >= 0.6 else ("VENDER (ML)" if proba <= 0.4 else "MANTENER (ML)")
        print(f"ML {label} p_subida={proba:.2f} acc_test={bundle.get('acc', 0):.3f} fuente={res.source}")
    except Exception as e:
        sig = latest_signal(df)
        print(f"REGLA {sig['label']} score={sig['score']} ({e})")


if __name__ == "__main__":
    main()
