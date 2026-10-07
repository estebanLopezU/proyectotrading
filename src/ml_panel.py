"""Panel ML integrado al dashboard Fase 2+3 (con filtro ATR calibrado)."""
import joblib
from src.features import latest_features


def ml_signal(df, symbol: str, timeframe: str) -> dict:
    safe = symbol.replace("/", "_")
    paths = [
        f"models/rf_{safe}_{timeframe}_h5.joblib",
        f"models/rf_{safe}_{timeframe}_h3.joblib",
        f"models/rf_{safe}_{timeframe}.joblib",
    ]
    for path in paths:
        try:
            b = joblib.load(path)
            atr_min = float(b.get("atr_min", 0) or 0)
            cur_atr = float(df["atr_pct"].iloc[-1] or 0)
            if atr_min and cur_atr < atr_min:
                return {"ok": True, "label": "MANTENER (ML ruido)", "proba": 0.5,
                        "acc": float(b.get("acc", 0)), "path": path,
                        "note": f"ATR {cur_atr:.3f} < min {atr_min:.3f}: mercado sin fuerza"}
            X = latest_features(df, b["cols"])
            p = float(b["model"].predict_proba(X)[0][1])
            lbl = "COMPRAR (ML)" if p >= 0.6 else ("VENDER (ML)" if p <= 0.4 else "MANTENER (ML)")
            return {"ok": True, "label": lbl, "proba": p, "acc": float(b.get("acc", 0)), "path": path}
        except Exception:
            continue
    return {"ok": False, "label": "Modelo no entrenado"}

