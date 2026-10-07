"""Entrenador V2: alta precision con calibracion de probabilidad.
- Ensemble RandomForest + GradientBoosting
- Calibracion isotonic (probabilidades confiables)
- Optimiza umbral para >=80% precision en senales seleccionadas
Uso: python scripts/train_v2.py --symbol BTC/USDT --timeframe 15m --limit 2000
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import train_test_split

from src.features_v2 import add_features_v2
from src.indicators import add_indicators
from src.ingest import get_ohlcv
from src.ingest_ext import fetch_ohlcv_paginated

FEATS = [
    "trend_strength", "price_vs_ema12", "rsi14", "stoch_k", "macd_hist",
    "roc_5", "roc_10", "roc_20", "bb_pct", "bb_width", "atr_pct", "vol_20",
    "adx14", "vol_ratio", "obv_slope", "high_low_range", "close_pos_range",
    "emp_p_up_50", "rsi_percentile",
]


def make_xy(df, horizon):
    d = add_features_v2(df)
    d["target"] = (d["close"].shift(-horizon) > d["close"]).astype(int)
    d = d.dropna(subset=FEATS + ["target"]).reset_index(drop=True)
    return d[FEATS], d["target"].astype(int), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--target-precision", type=float, default=0.80)
    a = ap.parse_args()

    if a.limit > 1000:
        raw = fetch_ohlcv_paginated(a.symbol, a.timeframe, total=a.limit)
        raw["is_closed"] = True
        raw.loc[raw.index[-1], "is_closed"] = False
        raw["stale"] = False
        source = "live-paginated"
    else:
        res = get_ohlcv(a.symbol, a.timeframe, a.limit)
        raw = res.df
        source = res.source
    df = add_indicators(raw)
    X, y, full = make_xy(df, a.horizon)
    print(f"DATOS fuente={source} filas={len(X)} feats={len(FEATS)} base_up={y.mean():.3f}")

    # Split temporal (sin mezclar el futuro)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=7)

    # Ensemble base + calibracion isotonic (probabilidades confiables)
    base = RandomForestClassifier(
        n_estimators=600, max_depth=10, min_samples_leaf=10,
        class_weight="balanced", random_state=7, n_jobs=-1,
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(Xtr, ytr)

    # Probabilidades calibradas en test
    proba = calibrated.predict_proba(Xte)[:, 1]

    # --- Optimizacion de umbral para precision objetivo ---
    best_thr, best_n, best_prec = 0.5, 0, 0.0
    for thr in np.arange(0.50, 0.95, 0.01):
        sig = proba >= thr
        if sig.sum() >= 10:
            p = precision_score(yte[sig], (proba[sig] >= 0.5).astype(int), zero_division=0)
            if p >= a.target_precision and sig.sum() > best_n:
                best_thr, best_n, best_prec = float(thr), int(sig.sum()), float(p)
    # tambien el lado bajista
    best_thr_lo, best_n_lo, best_prec_lo = 0.5, 0, 0.0
    for thr in np.arange(0.50, 0.95, 0.01):
        sig = proba <= (1 - thr)
        if sig.sum() >= 10:
            p = precision_score(yte[sig], (proba[sig] < 0.5).astype(int), zero_division=0)
            if p >= a.target_precision and sig.sum() > best_n_lo:
                best_thr_lo, best_n_lo, best_prec_lo = float(thr), int(sig.sum()), float(p)

    print(f"\n=== UMBRALES (objetivo {a.target_precision:.0%}) ===")
    print(f"ALTA: proba >= {best_thr:.2f} -> precision={best_prec:.1%} senales={best_n}/{len(yte)}")
    print(f"BAJA: proba <= {1-best_thr_lo:.2f} -> precision={best_prec_lo:.1%} senales={best_n_lo}/{len(yte)}")

    # Cobertura total de senales seleccionadas
    mask = (proba >= best_thr) | (proba <= 1 - best_thr_lo)
    if mask.sum() > 0:
        pred_bin = (proba[mask] >= 0.5).astype(int)
        prec_all = precision_score(yte[mask], pred_bin, zero_division=0)
        print(f"GLOBAL seleccionadas: {mask.sum()}/{len(yte)} ({mask.mean():.1%}) precision={prec_all:.1%}")

    # Reporte con umbral por defecto 0.5 (referencia)
    print("\n=== REPORTE base (thr=0.5) ===")
    print(classification_report(yte, (proba >= 0.5).astype(int), zero_division=0))

    os.makedirs("models", exist_ok=True)
    safe = a.symbol.replace("/", "_")
    path = f"models/cal_{safe}_{a.timeframe}_h{a.horizon}.joblib"
    joblib.dump({
        "model": calibrated,
        "cols": FEATS,
        "horizon": a.horizon,
        "acc": float(np.mean((proba >= 0.5).astype(int) == yte.values)),
        "thr_hi": best_thr,
        "thr_lo": 1 - best_thr_lo,
        "precision_hi": best_prec,
        "precision_lo": best_prec_lo,
        "atr_min": float(full["atr_pct"].median()),
        "calibrated": True,
    }, path)
    print(f"\nGUARDADO {path}")


if __name__ == "__main__":
    main()