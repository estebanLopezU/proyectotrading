"""Evaluacion HONESTA del edge (walk-forward + Wilson LB).
Objetivo: saber que precision direccional es REALMENTE alcanzable out-of-sample
antes de emitir senales para apostar. NO guarda modelos; solo mide.

Uso: python scripts/evaluate_edge.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from src.features_v2 import add_features_v2
from src.indicators import add_indicators
from src.ingest_ext import fetch_ohlcv_paginated

FEATS = [
    "trend_strength", "price_vs_ema12", "rsi14", "stoch_k", "macd_hist",
    "roc_5", "roc_10", "roc_20", "bb_pct", "bb_width", "atr_pct", "vol_20",
    "adx14", "vol_ratio", "obv_slope", "high_low_range", "close_pos_range",
    "emp_p_up_50", "rsi_percentile",
]


def wilson_lb(p, n, z=1.96):
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (center - margin) / denom


def best_threshold(proba, yte, side, min_n=15):
    """Mejor umbral con precision mas alta (y mas senales a igualdad)."""
    best = (0.0, 0.0, 0, None)  # prec, lb, n, thr
    for thr in np.arange(0.50, 0.96, 0.01):
        if side == "ALTA":
            m = proba >= thr
        else:
            m = proba <= (1 - thr)
        n = int(m.sum())
        if n < min_n:
            continue
        yhat = 1 if side == "ALTA" else 0
        correct = (yte[m] == yhat).mean()
        lb = wilson_lb(correct, n)
        # prioriza LB (limite inferior honesto), luego precision, luego n
        if (lb, correct, n) > (best[1], best[0], best[2]):
            best = (float(correct), float(lb), n, float(thr))
    return best


def walk_forward_proba(X, y, n_splits=5):
    """Predicciones out-of-sample agregadas via walk-forward temporal."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oos = np.full(len(y), np.nan)
    for tr, te in tscv.split(X):
        base = RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=15,
            class_weight="balanced", random_state=7, n_jobs=-1,
        )
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(X.iloc[tr], y.iloc[tr])
        oos[te] = cal.predict_proba(X.iloc[te])[:, 1]
    return oos


def eval_config(symbol, tf, horizon, total=6000):
    print(f"\n{'='*66}\n {symbol} {tf} horizon={horizon}  ({total} velas)\n{'='*66}")
    raw = fetch_ohlcv_paginated(symbol, tf, total=total)
    if len(raw) < 500:
        print("  datos insuficientes")
        return None
    raw["is_closed"] = True
    df = add_features_v2(add_indicators(raw))
    df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)
    df = df.dropna(subset=FEATS + ["target"]).reset_index(drop=True)
    X, y = df[FEATS], df["target"].astype(int)
    if len(X) < 400:
        print("  muestras insuficientes tras dropna")
        return None
    print(f"  muestras={len(X)}  base_up={y.mean():.1%}")

    proba = walk_forward_proba(X, y)
    valid = ~np.isnan(proba)
    proba_v, yv = proba[valid], y.values[valid]
    print(f"  OOS evaluadas={len(proba_v)}")

    base_acc = ((proba_v >= 0.5).astype(int) == yv).mean()
    print(f"  accuracy base(thr=0.5)={base_acc:.1%}")

    for side in ("ALTA", "BAJA"):
        prec, lb, n, thr = best_threshold(proba_v, yv, side)
        if thr is not None:
            tag = "  <-- cumple >=80%!" if prec >= 0.80 else ""
            print(f"  {side}: mejor thr={thr:.2f} precision={prec:.1%} "
                  f"WilsonLB={lb:.1%} n={n}{tag}")
        else:
            print(f"  {side}: sin umbral con n>=15")

    # Confluencia: proba alta + tendencia/volatilidad
    print("  --- confluencia ML + condicion ---")
    conds = {
        "trend_up": df.loc[valid, "trend_strength"].values > 0,
        "trend_dn": df.loc[valid, "trend_strength"].values < 0,
        "adx>25": df.loc[valid, "adx14"].values > 25,
        "atr_hi": df.loc[valid, "atr_pct"].values > df["atr_pct"].median(),
        "vol_hi": df.loc[valid, "vol_ratio"].values > 1.2,
    }
    rows = []
    for thr in (0.60, 0.65, 0.70):
        for cn, cm in conds.items():
            for side in ("ALTA", "BAJA"):
                if side == "ALTA":
                    m = (proba_v >= thr) & cm
                    yhat = 1
                else:
                    m = (proba_v <= 1 - thr) & cm
                    yhat = 0
                n = int(m.sum())
                if n < 15:
                    continue
                correct = (yv[m] == yhat).mean()
                lb = wilson_lb(correct, n)
                rows.append((correct, lb, n, thr, cn, side))
    rows.sort(reverse=True, key=lambda t: (t[1], t[0]))
    for correct, lb, n, thr, cn, side in rows[:5]:
        tag = "  >=80%!" if correct >= 0.80 else ""
        s = ">=" if side == "ALTA" else "<="
        v = thr if side == "ALTA" else round(1 - thr, 2)
        print(f"    {side} p{s}{v}+{cn}: n={n} prec={correct:.1%} LB={lb:.1%}{tag}")
    return df


if __name__ == "__main__":
    # Configuraciones a evaluar (de corto a mas largo)
    configs = [
        ("BTC/USDT", "15m", 3),
        ("BTC/USDT", "15m", 6),
        ("BTC/USDT", "1h", 3),
        ("BTC/USDT", "1h", 6),
    ]
    for sym, tf, hz in configs:
        try:
            eval_config(sym, tf, hz)
        except Exception as e:
            import traceback
            print(f"  FALLO {tf} h={hz}: {e}")
            traceback.print_exc()
