"""Validacion honesta: busca la mejor precision alcanzable con confluencia.
Combina proba calibrada del ML + condiciones de mercado.
Reporta precision, Wilson LB y tamano de muestra.
Uso: python scripts/validate_signals.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.features_v2 import add_features_v2
from src.indicators import add_indicators
from src.ingest_ext import fetch_ohlcv_paginated

FEATS = [
    "trend_strength", "price_vs_ema12", "rsi14", "stoch_k", "macd_hist",
    "roc_5", "roc_10", "roc_20", "bb_pct", "bb_width", "atr_pct", "vol_20",
    "adx14", "vol_ratio", "obv_slope", "high_low_range", "close_pos_range",
    "emp_p_up_50", "rsi_percentile",
]
HORIZON = 3


def wilson_lb(p, n, z=1.96):
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (center - margin) / denom


def main():
    print("Descargando 5000 velas 15m...")
    raw = fetch_ohlcv_paginated("BTC/USDT", "15m", total=5000)
    raw["is_closed"] = True
    df = add_features_v2(add_indicators(raw))
    df["target"] = (df["close"].shift(-HORIZON) > df["close"]).astype(int)
    df = df.dropna(subset=FEATS + ["target"]).reset_index(drop=True)
    X, y = df[FEATS], df["target"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=7)
    base = RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=15,
        class_weight="balanced", random_state=7, n_jobs=-1,
    )
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(Xtr, ytr)
    proba = cal.predict_proba(Xte)[:, 1]
    te = Xte.copy()
    te["proba"] = proba
    te["y"] = yte.values

    print(f"\nn_test={len(te)} precision_base(>=0.5)={(proba>=0.5).mean():.1%} "
          f"P(y=1|proba>=0.5)={te.loc[te['proba']>=0.5,'y'].mean() if (proba>=0.5).any() else 0:.1%}")

    # --- Barra 1: solo umbrales de proba ---
    print("\n=== SOLO PROBA (umbral creciente) ===")
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        for side, mask in [("ALTA", te["proba"] >= thr), ("BAJA", te["proba"] <= 1 - thr)]:
            n = int(mask.sum())
            if n < 10:
                continue
            correct = (te.loc[mask, "y"] == (1 if side == "ALTA" else 0)).mean()
            lb = wilson_lb(correct, n)
            tag = ">=80%!" if correct >= 0.80 else ""
            print(f"{side} proba{'>' if side=='ALTA' else '<'}{thr if side=='ALTA' else round(1-thr,2)}: "
                  f"n={n:4d} precision={correct:.1%} LB={lb:.1%} {tag}")

    # --- Barra 2: confluencia ML + condicion ---
    print("\n=== CONFLUENCIA ML + CONDICION ===")
    conds = {
        "adx>25": te["adx14"] > 25,
        "adx<20": te["adx14"] < 20,
        "rsi<40": te["rsi14"] < 40,
        "rsi>60": te["rsi14"] > 60,
        "bb_lo": te["bb_pct"] < 0.25,
        "bb_hi": te["bb_pct"] > 0.75,
        "roc5<0": te["roc_5"] < 0,
        "roc5>0": te["roc_5"] > 0,
        "vol_hi": te["vol_ratio"] > 1.3,
        "trend_up": te["trend_strength"] > 0,
        "trend_dn": te["trend_strength"] < 0,
        "atr_hi": te["atr_pct"] > te["atr_pct"].median(),
    }
    best = []
    for thr in [0.60, 0.65, 0.70, 0.75]:
        for cname, cmask in conds.items():
            for side in ["ALTA", "BAJA"]:
                if side == "ALTA":
                    mask = (te["proba"] >= thr) & cmask
                    yhat = 1
                else:
                    mask = (te["proba"] <= 1 - thr) & cmask
                    yhat = 0
                n = int(mask.sum())
                if n < 15:
                    continue
                correct = (te.loc[mask, "y"] == yhat).mean()
                lb = wilson_lb(correct, n)
                if correct >= 0.75:
                    best.append((correct, lb, n, thr, cname, side))
    best.sort(reverse=True, key=lambda t: (t[1], t[0]))
    for correct, lb, n, thr, cname, side in best[:20]:
        tag = ">=80%!" if correct >= 0.80 else ""
        print(f"{side} proba{'>=' if side=='ALTA' else '<='}{thr if side=='ALTA' else round(1-thr,2)} + {cname}: "
              f"n={n:4d} precision={correct:.1%} LB={lb:.1%} {tag}")

    # --- Mejor doble condicion con ML ---
    print("\n=== ML + DOBLE CONDICION ===")
    best2 = []
    cn = list(conds)
    for thr in [0.60, 0.65, 0.70]:
        for i, c1 in enumerate(cn):
            for c2 in cn[i + 1:]:
                cmask = conds[c1] & conds[c2]
                for side in ["ALTA", "BAJA"]:
                    if side == "ALTA":
                        mask = (te["proba"] >= thr) & cmask
                        yhat = 1
                    else:
                        mask = (te["proba"] <= 1 - thr) & cmask
                        yhat = 0
                    n = int(mask.sum())
                    if n < 12:
                        continue
                    correct = (te.loc[mask, "y"] == yhat).mean()
                    lb = wilson_lb(correct, n)
                    if correct >= 0.75:
                        best2.append((correct, lb, n, thr, f"{c1}+{c2}", side))
    best2.sort(reverse=True, key=lambda t: (t[1], t[0]))
    for correct, lb, n, thr, cname, side in best2[:15]:
        tag = ">=80%!" if correct >= 0.80 else ""
        print(f"{side} p{'>=' if side=='ALTA' else '<='}{thr if side=='ALTA' else round(1-thr,2)} + {cname}: "
              f"n={n:4d} precision={correct:.1%} LB={lb:.1%} {tag}")


if __name__ == "__main__":
    main()