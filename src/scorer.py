"""src/scorer.py
Scorer de senales con metrica HONESTA (validacion walk-forward + Wilson LB).

Prioriza los bundles "signal_*" (regla validada). Solo emite COMPRAR/VENDER
cuando la regla quedo validada con Wilson LB >= 80% y las condiciones actuales
la cumplen. Siempre devuelve la probabilidad direccional.
"""
import math
import glob
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _safe_float(x, default: float = 0.0) -> float:
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return default
    return fx if not math.isnan(fx) else default

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest import get_ohlcv
from src.indicators import add_indicators
from src.features_v2 import add_features_v2
from src.features import latest_features


def wilson_lb(p: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return float((center - margin) / denom)


def load_bundle(symbol: str, timeframe: str):
    """Prioriza modelos de senal validados (signal_*)."""
    safe = symbol.replace("/", "_")
    hits = sorted(glob.glob(f"models/signal_{safe}_{timeframe}_h*.joblib"))
    if hits:
        return joblib.load(hits[-1]), hits[-1]
    any_sig = sorted(glob.glob(f"models/signal_{safe}_*.joblib"))
    if any_sig:
        return joblib.load(any_sig[-1]), any_sig[-1]
    for p in sorted(glob.glob("models/cal_*.joblib")) + sorted(glob.glob("models/rf_*.joblib")):
        try:
            return joblib.load(p), p
        except Exception:
            continue
    return None, None


def _cond_holds(cond: str, d2) -> bool:
    """Evalua la condicion de la regla sobre la ultima vela de d2 (features_v2)."""
    if d2 is None or len(d2) == 0:
        return True
    try:
        if cond == "trend_dn":
            return float(d2["trend_strength"].iloc[-1]) < 0
        if cond == "trend_up":
            return float(d2["trend_strength"].iloc[-1]) > 0
        if cond == "adx_hi":
            return float(d2["adx14"].iloc[-1]) > 25
        if cond == "atr_hi":
            return float(d2["atr_pct"].iloc[-1]) > float(d2["atr_pct"].median())
        if cond == "vol_hi":
            return float(d2["vol_ratio"].iloc[-1]) > 1.2
        return True  # "none"
    except Exception:
        return True

def score_signal(df, symbol: str, timeframe: str = "1h", horizon: int = 3):
    """Predice direccion y, si la regla validada lo permite, emite COMPRAR/VENDER."""
    b, path = load_bundle(symbol, timeframe)
    meta = {"label": "SIN MODELO", "proba": None, "confidence": 0.0, "precision": 0.0,
            "n_test": 0, "calibration": False, "note": "sin modelo entrenado",
            "ok": False, "prob_up": None, "prob_down": None, "horizon_velas": None,
            "wilson_lb": 0.0, "rule": None, "model_timeframe": None, "fire": False}
    if b is None:
        return meta

    try:
        cols = b["cols"]
        model_tf = b.get("timeframe", timeframe)
        # Datos frescos en el timeframe del modelo (correcto aunque el grafico sea otro tf)
        try:
            res = get_ohlcv(symbol, model_tf, 300)
            src_df = res.df
            data_note = f"fuente={res.source}"
        except Exception:
            src_df = df
            data_note = "df_recibido"

        is_v2 = bool(cols) and cols[0] in ("trend_strength", "rsi14") and "stoch_k" in cols
        if is_v2:
            d2 = add_features_v2(src_df)
            X = d2[cols].tail(1)
        else:
            d2 = None
            X = latest_features(src_df, cols)

        proba = float(b["model"].predict_proba(X)[0][1])
        prob_up, prob_down = proba, 1.0 - proba
        horizon_velas = int(b.get("horizon", 0) or 0)

        # ---- Camino NUEVO: regla validada (signal_*) ----
        rule = b.get("rule")
        if rule and _safe_float(rule.get("wilson_lb", 0), 0.0) >= 0.80:
            direction = rule["direction"]
            thr = float(rule["proba_thr"])
            cond = rule["cond"]
            prec = float(rule["precision"])
            lb = float(rule["wilson_lb"])
            n = int(rule["n"])
            holds = _cond_holds(cond, d2)
            if direction == "ALTA":
                fire = bool(proba >= thr and holds)
                thr_txt = f"proba>={thr:.2f}"
            else:
                fire = bool(proba <= (1 - thr) and holds)
                thr_txt = f"proba<={1 - thr:.2f}"
            label = ("COMPRAR" if direction == "ALTA" else "VENDER") if fire else "SIN SENAL"
            return {
                "label": label, "proba": proba, "confidence": lb, "precision": prec,
                "n_test": n, "calibration": bool(b.get("calibrated")),
                "thr_hi": thr if direction == "ALTA" else 1 - thr,
                "thr_lo": 1 - thr if direction == "BAJA" else thr,
                "conditions": {"activas": [cond] if holds else [], "n_cond": 1 if holds else 0,
                               "cond_requerida": cond, "cond_cumplida": holds},
                "final": proba if fire else None, "label_final": label,
                "note": (f"Regla: {direction} {thr_txt} + {cond} | precision={prec:.1%} "
                         f"WilsonLB={lb:.1%} (n={n}) | {data_note}"),
                "calibrated": True, "path": path, "prec_validada": prec, "ic_wilson": lb,
                "ok": True, "prob_up": prob_up, "prob_down": prob_down,
                "horizon_velas": horizon_velas, "direction": "SUBE" if prob_up >= 0.5 else "BAJA",
                "wilson_lb": lb, "rule": rule, "model_timeframe": model_tf, "fire": fire,
                "min_ret": _safe_float(b.get("min_ret", 0.0), 0.0),
                "target_desc": b.get("target_desc", ""),
            }

        # ---- Camino ANTIGUO (bundles cal_/rf_ sin regla): solo probabilidad ----
        prec = _safe_float(b.get("precision_hi", 0), 0.0)
        return {
            "label": "SIN SENAL (sin regla validada)", "proba": proba, "confidence": 0.0,
            "precision": prec, "n_test": int(b.get("n_test", 0) or 0),
            "calibration": bool(b.get("calibrated")),
            "thr_hi": float(b.get("thr_hi", 0.6)), "thr_lo": float(b.get("thr_lo", 0.4)),
            "conditions": {"activas": [], "n_cond": 0},
            "final": None, "label_final": "SIN SENAL",
            "note": (f"Modelo sin regla validada (precision_hi={prec:.1%}). "
                     f"Reentrena con scripts/train_signal.py | {data_note}"),
            "calibrated": bool(b.get("calibrated")), "path": path,
            "prec_validada": prec, "ic_wilson": 0.0, "ok": True,
            "prob_up": prob_up, "prob_down": prob_down, "horizon_velas": horizon_velas,
            "direction": "SUBE" if prob_up >= 0.5 else "BAJA",
            "wilson_lb": 0.0, "rule": None, "model_timeframe": model_tf, "fire": False,
        }
    except Exception as e:
        return {"label": "ERROR", "proba": None, "confidence": 0.0, "precision": 0.0,
                "n_test": 0, "calibration": False, "note": str(e)[:120], "ok": False,
                "prob_up": None, "prob_down": None, "horizon_velas": None,
                "wilson_lb": 0.0, "rule": None, "model_timeframe": None, "fire": False}

