"""src/scorer.py
Scorer de senales con metrica honesta de precision y confianza (validacion out-of-sample).
"""

import glob
import os
import sys
from pathlib import Path

import joblib
import numpy as np  # sin importar np: wilson_lb usaba np.sqrt sin importar numpy
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest_ext import fetch_ohlcv_paginated
from src.indicators import add_indicators
from src.features_v2 import add_features_v2
from src.features import latest_features

CONF = {"adx_min": 20, "rsi_oversold": 40, "bb_pct_max_lower": 0.15,
        "roc_5_max": 0.0, "vol_ratio_min": 1.2, "trend_up_min": 0.001,
        "trend_dn_max": -0.001}


def wilson_lb(p: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return float((center - margin) / denom)



def load_bundle(symbol: str, timeframe: str):
    safe = symbol.replace("/", "_")
    prefs = [
        "models/cal_BTC_USDT_15m_h3.joblib",
        "models/cal_BTC_USDT_15m_h5.joblib",
        "models/cal_BTC_USDT_15m.joblib",
        "models/cal_BTC_USDT_1m_h3.joblib",
        "models/cal_BTC_USDT_1m.joblib",
    ]
    for p in prefs:
        if os.path.exists(p):
            return joblib.load(p), p
    cal = sorted(glob.glob("models/cal_*.joblib"))
    if cal:
        return joblib.load(cal[0]), cal[0]
    for p in sorted(glob.glob("models/rf_*.joblib")):
        try:
            return joblib.load(p), p
        except Exception:
            continue
    return None, None


def score_signal(df, symbol: str, timeframe: str = "15m", horizon: int = 6):
    b, path = load_bundle(symbol, timeframe)
    meta = {"label": "SIN MODELO", "proba": None, "confidence": 0.0, "precision": 0.0,
            "n_test": 0, "calibration": False, "note": "sin modelo entrenado"}

    if b is None:
        return meta

    try:
        last = df.iloc[-1].to_dict()
        atr = float(last.get("atr_pct", 0) or 0)
        atr_min = float(b.get("atr_min", 0) or 0)

        if atr_min and atr < atr_min:
            return {"label": "SIN SENAL (ruido)", "proba": None, "confidence": 0.0,
                    "precision": 0.0, "n_test": 0, "calibration": False,
                    "conditions": {"adx": 0.0, "rsi": 50.0, "bb_pct": 0.5, "vol_ratio": 1.0,
                                   "trend_strength": 0.0, "roc_5": 0.0, "close_pos": 0.5},
                    "conditions_data": "atr_bajo", "n_cond": 0, "meta": meta}

        cols = b["cols"]
        if cols and cols[0] in ("trend_strength", "rsi14") and "stoch_k" in cols:
            d2 = add_features_v2(df)
            X = d2[cols].tail(1)
            note = "features_v2"
        else:
            X = latest_features(df, cols)
            note = "features_v1"

        model = b["model"]
        adx = float(last.get("adx14", 0) or 0)
        rsi = float(last.get("rsi14", 0) or 50)
        bb_pct = float(last.get("bb_pct", 0.5) or 0.5)
        vol_ratio = float(last.get("vol_ratio", 1.0) or 1.0)
        trend_s = float(last.get("trend_strength", 0) or 0)
        roc_5 = float(last.get("roc_5", 0) or 0)
        close_pos = float(last.get("close_pos_range", 0.5) or 0.5)

        active = []
        if adx >= CONF["adx_min"]:
            active.append("ADX>=%d" % int(adx))
        if rsi <= CONF["rsi_oversold"]:
            active.append("RSI<=%d" % int(rsi))
        if bb_pct <= CONF["bb_pct_max_lower"]:
            active.append("BB_bajo_%d" % (bb_pct * 100))
        if roc_5 <= CONF["roc_5_max"]:
            active.append("roc_5<=0")
        if vol_ratio >= CONF["vol_ratio_min"]:
            active.append("vol_relativo")
        if trend_s >= CONF["trend_up_min"]:
            active.append("trend_alcista")
        if trend_s <= CONF["trend_dn_max"]:
            active.append("trend_bajista")

        n_cond = len(active)
        conf = 0.0
        if n_cond >= 2:
            conf = min(0.95, 0.5 + 0.08 * (n_cond - 2))

        proba = float(model.predict_proba(X)[0][1])
        calibrated = bool(b.get("calibrated", False))
        prec = float(b.get("precision_hi", 0) or 0) or 0
        n_test = int(b.get("n_test", 0)) or 0
        thr_hi = float(b.get("thr_hi", 0.60))
        thr_lo = float(b.get("thr_lo", 0.40))

        if calibrated and prec >= 0.80 and thr_hi <= 0.60:
            lab, fp = "COMPRAR", proba
        elif calibrated and prec >= 0.80 and thr_lo <= 0.40:
            lab, fp = "VENDER", proba
        else:
            lab, fp = "SIN SENAL", None

        cond_activas = {
            "ADX>=%d" % int(adx): adx >= CONF["adx_min"],
            "RSI<=%d" % int(rsi): rsi <= CONF["rsi_oversold"],
            "BB_bajo_%d" % (bb_pct * 100): bb_pct <= CONF["bb_pct_max_lower"] * 100,
            "roc_5<=0": roc_5 <= CONF["roc_5_max"],
            "vol_relativo": vol_ratio >= CONF["vol_ratio_min"],
            "trend_alcista": trend_s >= CONF["trend_up_min"],
            "trend_bajista": trend_s <= CONF["trend_dn_max"],
        }
        active_cond = [k for k, v in cond_activas.items() if v]

        return {
            "label": lab, "proba": proba, "confidence": conf,
            "precision": prec, "n_test": n_test, "calibration": calibrated,
            "thr_hi": float(b.get("thr_hi", 0.60)), "thr_lo": float(b.get("thr_lo", 0.40)),
            "conditions": {"activas": active_cond, "n_cond": n_cond, "adx": adx, "rsi": rsi,
                           "bb_pct": bb_pct, "vol_ratio": vol_ratio, "trend_strength": trend_s},
            "final": fp, "label_final": lab,
            "note": f"prec_hist={prec:.1%} ic_wilson={prec:.1%} (n={n_test})",
            "calibrated": calibrated, "path": path,
            "prec_validada": prec, "ic_wilson": conf,
        }

    except Exception as e:
        return {"label": "ERROR", "proba": None, "confidence": 0.0, "precision": 0.0,
                "n_test": 0, "calibration": False, "note": str(e)[:100]}
