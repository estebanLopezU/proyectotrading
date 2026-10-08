"""Entrenador de SENAL honesta para apostar (1h / horizonte 3).
- Walk-forward out-of-sample para precision HONESTA (sin mezclar futuro).
- Busca la mejor regla (direccion x umbral proba x condicion) con
  Wilson LB >= 80% y n >= 30 (suficientes muestras para confiar).
- Entrena el modelo final calibrado y guarda modelo + regla + metricas.

Solo emite senal si el edge esta validado. Uso:
    python scripts/train_signal.py --symbol BTC/USDT --timeframe 1h --horizon 3
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import joblib
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

# Condiciones candidatas (reversion a la media / regimen). Pocas, para
# limitar sobreajuste por busqueda.
CONDS = {
    "trend_dn": lambda d: d["trend_strength"] < 0,
    "trend_up": lambda d: d["trend_strength"] > 0,
    "adx_hi": lambda d: d["adx14"] > 25,
    "atr_hi": lambda d: d["atr_pct"] > d["atr_pct"].median(),
    "vol_hi": lambda d: d["vol_ratio"] > 1.2,
    "none": lambda d: np.ones(len(d), dtype=bool),
}


def wilson_lb(p, n, z=1.96):
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return float((center - margin) / denom)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--total", type=int, default=8000)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--min-lb", type=float, default=0.80)
    ap.add_argument("--min-ret", type=float, default=0.0,
                    help="umbral de retorno (ej 0.003 = 0.3%) para el objetivo ALTA")
    a = ap.parse_args()

    print(f"Descargando {a.total} velas {a.timeframe}...")
    raw = fetch_ohlcv_paginated(a.symbol, a.timeframe, total=a.total)
    raw["is_closed"] = True
    df = add_features_v2(add_indicators(raw))
    # Objetivo: 1 si el precio sube mas de min_ret en +horizon velas (filtra micro-ruido)
    fwd = df["close"].shift(-a.horizon) / df["close"] - 1.0
    df["target"] = (fwd > a.min_ret).astype(int)
    df = df.dropna(subset=FEATS + ["target"]).reset_index(drop=True)
    X, y = df[FEATS], df["target"].astype(int)
    print(f"muestras={len(X)} tasa_positiva={y.mean():.1%} min_ret={a.min_ret:.3%}")

    # --- Walk-forward OOS ---
    tscv = TimeSeriesSplit(n_splits=6)
    oos = np.full(len(y), np.nan)
    for tr, te in tscv.split(X):
        base = RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=15,
            class_weight="balanced", random_state=7, n_jobs=-1,
        )
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(X.iloc[tr], y.iloc[tr])
        oos[te] = cal.predict_proba(X.iloc[te])[:, 1]

    valid = ~np.isnan(oos)
    p, yv, dv = oos[valid], y.values[valid], df.loc[valid].reset_index(drop=True)
    print(f"OOS evaluadas={len(p)}")

    # --- Busqueda de la mejor regla honesta ---
    best = None  # (lb, prec, n, direction, thr, cond)
    all_rows = []
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        for cname, cfn in CONDS.items():
            cmask = np.asarray(cfn(dv), dtype=bool)
            for direction in ("ALTA", "BAJA"):
                if direction == "ALTA":
                    m = (p >= thr) & cmask
                    yhat = 1
                else:
                    m = (p <= 1 - thr) & cmask
                    yhat = 0
                n = int(m.sum())
                if n < a.min_n:
                    continue
                prec = float((yv[m] == yhat).mean())
                lb = wilson_lb(prec, n)
                all_rows.append((lb, prec, n, direction, float(thr), cname))
                if lb >= a.min_lb:
                    key = (lb, prec, n)
                    if best is None or key > (best[0], best[1], best[2]):
                        best = (lb, prec, n, direction, float(thr), cname)

    # Diagnostico honesto: top-10 por Wilson LB (aunque no lleguen a la barra)
    all_rows.sort(reverse=True, key=lambda t: (t[0], t[1], t[2]))
    print("\n=== TOP 10 reglas por Wilson LB (diagnostico) ===")
    for lb, prec, n, direction, thr, cname in all_rows[:10]:
        v = thr if direction == "ALTA" else round(1 - thr, 2)
        s = ">=" if direction == "ALTA" else "<="
        print(f"  {direction} p{s}{v}+{cname}: n={n} prec={prec:.1%} LB={lb:.1%}")

    if best is None:
        print("\nNO se encontro regla con Wilson LB >= %.0f%% y n >= %d."
              % (a.min_lb * 100, a.min_n))
        print("Edge insuficiente para apostar de forma honesta. NO se guarda modelo.")
        return

    lb, prec, n, direction, thr, cname = best
    print("\n=== MEJOR REGLA HONESTA ===")
    print(f"{direction} | proba {'>=' if direction=='ALTA' else '<='} "
          f"{thr if direction=='ALTA' else round(1-thr,2)} + {cname}")
    print(f"precision={prec:.1%}  WilsonLB={lb:.1%}  n={n}")

    # --- Modelo final calibrado en TODOS los datos ---
    base = RandomForestClassifier(
        n_estimators=600, max_depth=10, min_samples_leaf=10,
        class_weight="balanced", random_state=7, n_jobs=-1,
    )
    final = CalibratedClassifierCV(base, method="isotonic", cv=5)
    final.fit(X, y)

    os.makedirs("models", exist_ok=True)
    safe = a.symbol.replace("/", "_")
    path = f"models/signal_{safe}_{a.timeframe}_h{a.horizon}.joblib"
    joblib.dump({
        "model": final,
        "cols": FEATS,
        "horizon": a.horizon,
        "timeframe": a.timeframe,
        "atr_min": float(df["atr_pct"].median()),
        "calibrated": True,
        "min_ret": float(a.min_ret),
        "target_desc": f"P(subida > {a.min_ret:.2%} en {a.horizon} velas de {a.timeframe})",
        "rule": {
            "direction": direction,
            "proba_thr": thr,
            "cond": cname,
            "precision": prec,
            "wilson_lb": lb,
            "n": n,
        },
        "thr_hi": thr if direction == "ALTA" else 1 - thr,
        "thr_lo": 1 - thr if direction == "BAJA" else thr,
        "precision_hi": prec,
        "precision_lo": prec,
    }, path)
    print(f"\nGUARDADO {path}")


if __name__ == "__main__":
    main()

