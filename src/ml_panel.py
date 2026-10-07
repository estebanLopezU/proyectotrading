"""Panel ML integrado al dashboard Fase 2+3 (con filtro ATR calibrado).
Busca modelo por timeframe exacto; si no existe, usa el disponible del simbolo.
"""
import glob
import os

import joblib
from src.features import latest_features


def _load_bundle(symbol: str, timeframe: str):
    safe = symbol.replace("/", "_")
    # 1) Intentos por timeframe exacto
    paths = [
        f"models/rf_{safe}_{timeframe}_h5.joblib",
        f"models/rf_{safe}_{timeframe}_h3.joblib",
        f"models/rf_{safe}_{timeframe}.joblib",
    ]
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p), p
    # 2) Respaldo: cualquier modelo del simbolo (ej: entrenado en 15m)
    for p in sorted(glob.glob(f"models/rf_{safe}_*.joblib")):
        try:
            return joblib.load(p), p
        except Exception:
            continue
    return None, None


def ml_signal(df, symbol: str, timeframe: str) -> dict:
    b, path = _load_bundle(symbol, timeframe)
    if b is None:
        return {"ok": False, "label": "Modelo no entrenado"}
    try:
        atr_min = float(b.get("atr_min", 0) or 0)
        cur_atr = float(df["atr_pct"].iloc[-1] or 0)
        note_tf = ""
        if f"_{timeframe}_" not in os.path.basename(path):
            note_tf = f" (modelo de {os.path.basename(path).split('_')[-1].replace('.joblib','')})"
        if atr_min and cur_atr < atr_min:
            return {"ok": True, "label": "MANTENER (ML ruido)", "proba": 0.5,
                    "acc": float(b.get("acc", 0)), "path": path,
                    "note": f"ATR {cur_atr:.3f} < min {atr_min:.3f}: mercado sin fuerza"}
        X = latest_features(df, b["cols"])
        p = float(b["model"].predict_proba(X)[0][1])
        lbl = "COMPRAR (ML)" if p >= 0.6 else ("VENDER (ML)" if p <= 0.4 else "MANTENER (ML)")
        return {"ok": True, "label": lbl, "proba": p, "acc": float(b.get("acc", 0)),
                "path": path, "note": note_tf}
    except Exception as e:
        return {"ok": False, "label": f"Error modelo: {str(e)[:80]}"}

