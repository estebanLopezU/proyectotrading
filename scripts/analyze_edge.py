"""Analitica de datos: busca condiciones con precision >=80% historica.
Metodologia: regla de condicion -> probabilidad empirica de subida con IC.
Uso: python scripts/analyze_edge.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.features_v2 import add_features_v2
from src.indicators import add_indicators
from src.ingest_ext import fetch_ohlcv_paginated

HORIZON = 3
MIN_N = 60  # minimo de observaciones para confiar


def wilson_lb(p, n, z=1.96):
    """Limite inferior de Wilson: confianza conservadora de la precision."""
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
    df["fut_up"] = (df["close"].shift(-HORIZON) > df["close"]).astype(int)
    df = df.dropna(subset=["adx14", "rsi14", "bb_pct", "atr_pct"]).reset_index(drop=True)
    df = df.iloc[:-HORIZON]
    base = df["fut_up"].mean()
    print(f"Muestra n={len(df)} prob_base_subida={base:.3f} horizonte={HORIZON} velas\n")

    # Condiciones candidatas (regimen + momentum + volatilidad)
    conds = {
        "ADX>25 (tendencia)": df["adx14"] > 25,
        "ADX>30 (tendencia fuerte)": df["adx14"] > 30,
        "ADX<20 (rango)": df["adx14"] < 20,
        "RSI<30 (sobreventa)": df["rsi14"] < 30,
        "RSI>70 (sobrecompra)": df["rsi14"] > 70,
        "RSI 40-60 (neutral)": df["rsi14"].between(40, 60),
        "bb_pct<0.1 (banda inf)": df["bb_pct"] < 0.1,
        "bb_pct>0.9 (banda sup)": df["bb_pct"] > 0.9,
        "roc_5>0 (momentum+)": df["roc_5"] > 0,
        "roc_5<0 (momentum-)": df["roc_5"] < 0,
        "vol_ratio>1.5 (vol alta)": df["vol_ratio"] > 1.5,
        "atr_pct>mediana": df["atr_pct"] > df["atr_pct"].median(),
        "trend_strength>0 (alcista)": df["trend_strength"] > 0,
        "trend_strength<0 (bajista)": df["trend_strength"] < 0,
        "close_pos_range>0.7 (cierra alto)": df["close_pos_range"] > 0.7,
        "stoch_k<20": df["stoch_k"] < 20,
        "stoch_k>80": df["stoch_k"] > 80,
    }

    # Combinaciones de 2 condiciones (confluencia)
    names = list(conds)
    results = []
    print("=== CONDICIONES SIMPLES ===")
    for name, mask in conds.items():
        n = int(mask.sum())
        if n < MIN_N:
            continue
        p = df.loc[mask, "fut_up"].mean()
        lb = wilson_lb(p, n)
        flag = "***" if lb >= 0.60 and p >= 0.80 else ("**" if p >= 0.65 else "")
        print(f"{name:35s} n={n:5d} P(subida)={p:.1%} WilsonLB={lb:.1%} {flag}")
        results.append((name, n, p, lb))

    print("\n=== COMBINACIONES (confluencia) con P>=70% ===")
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            mask = conds[n1] & conds[n2]
            n = int(mask.sum())
            if n < MIN_N:
                continue
            p = df.loc[mask, "fut_up"].mean()
            lb = wilson_lb(p, n)
            if p >= 0.70:
                flag = ">=80%!" if p >= 0.80 else ""
                print(f"{n1} & {n2}: n={n} P={p:.1%} LB={lb:.1%} {flag}")

    # Triple confluencia de las mejores
    print("\n=== TRIPLES con P>=75% ===")
    key = ["adx14", "trend_strength", "roc_5", "bb_pct", "rsi14", "vol_ratio"]
    cands = {k: conds[n] for k, n in [
        ("adx25", "ADX>25 (tendencia)"), ("trend_up", "trend_strength>0 (alcista)"),
        ("trend_dn", "trend_strength<0 (bajista)"), ("roc_p", "roc_5>0 (momentum+)"),
        ("roc_n", "roc_5<0 (momentum-)"), ("vol_h", "vol_ratio>1.5 (vol alta)"),
        ("bb_lo", "bb_pct<0.1 (banda inf)"), ("bb_hi", "bb_pct>0.9 (banda sup)"),
    ] if n in conds}
    kn = list(cands)
    for i, a in enumerate(kn):
        for j, b in enumerate(kn):
            for c in kn:
                if not (i < j < c):
                    continue
                mask = cands[a] & cands[b] & cands[c]
                n = int(mask.sum())
                if n < 40:
                    continue
                p = df.loc[mask, "fut_up"].mean()
                lb = wilson_lb(p, n)
                if p >= 0.75:
                    print(f"{a}&{b}&{c}: n={n} P={p:.1%} LB={lb:.1%} {'>=80%!' if p>=0.80 else ''}")


if __name__ == "__main__":
    main()