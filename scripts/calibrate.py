"""Calibracion Fase 3: elige mejor timeframe/horizon por ACC + filtros ATR.
Uso: python scripts/calibrate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.features import make_dataset
from src.indicators import add_indicators
from src.ingest import get_ohlcv

CANDS = [("15m", 5, 600), ("1h", 3, 800), ("1h", 5, 800), ("4h", 3, 500)]


def main():
    best = None
    for tf, hz, lim in CANDS:
        try:
            res = get_ohlcv("BTC/USDT", tf, lim)
            df = add_indicators(res.df)
            X, y, cols, full = make_dataset(df, hz)
            # Filtro ruido: solo velas con ATR% > mediana
            med = full["atr_pct"].median()
            mask = (full["atr_pct"] >= med).to_numpy()
            Xf, yf = X[mask], y[mask]
            Xtr, Xte, ytr, yte = train_test_split(Xf, yf, test_size=0.25, shuffle=False)
            clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=8, random_state=7, n_jobs=-1)
            clf.fit(Xtr, ytr)
            acc = accuracy_score(yte, clf.predict(Xte))
            print(f"{tf} h={hz} n={len(Xf)} ACC={acc:.3f} fuente={res.source}")
            if best is None or acc > best[0]:
                safe = "BTC_USDT"
                path = f"models/rf_{safe}_{tf}_h{hz}.joblib"
                joblib.dump({"model": clf, "cols": cols, "horizon": hz, "acc": acc, "atr_min": float(med)}, path)
                best = (acc, path, tf, hz, float(med))
        except Exception as e:
            print(f"{tf} h={hz} FALLO: {e}")
    if best:
        print(f"MEJOR ACC={best[0]:.3f} -> {best[1]} (tf={best[2]} h={best[3]} atr_min={best[4]:.3f})")
    else:
        print("Sin candidatos validos.")


if __name__ == "__main__":
    main()
