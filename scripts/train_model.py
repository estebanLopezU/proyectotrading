"""Entrena RandomForest para predecir sube(1)/baja(0) en +horizon velas.
Uso:
    python scripts/train_model.py --symbol BTC/USDT --timeframe 15m --limit 1000 --horizon 5
Guarda: models/rf_{symbol}_{timeframe}_h{horizon}.joblib
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.features import make_dataset
from src.indicators import add_indicators
from src.ingest import get_ohlcv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--min_samples_leaf", type=int, default=4)
    a = ap.parse_args()

    res = get_ohlcv(a.symbol, a.timeframe, a.limit)
    df = add_indicators(res.df)
    X, y, cols, full = make_dataset(df, a.horizon)
    print(f"DATOS fuente={res.source} filas={len(full)} feats={len(cols)}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, shuffle=False, random_state=7
    )
    clf = RandomForestClassifier(
        n_estimators=a.n_estimators,
        max_depth=a.max_depth,
        min_samples_leaf=a.min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        random_state=7,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"ACC test={acc:.3f}  (n_train={len(Xtr)}, n_test={len(Xte)})")
    print(classification_report(yte, pred, zero_division=0))

    os.makedirs("models", exist_ok=True)
    safe = a.symbol.replace("/", "_")
    path = f"models/rf_{safe}_{a.timeframe}_h{a.horizon}.joblib"
    joblib.dump(
        {
            "model": clf,
            "cols": cols,
            "horizon": a.horizon,
            "acc": acc,
            "atr_min": float(full["atr_pct"].median()),
        },
        path,
    )
    print(f"GUARDADO {path}")


if __name__ == "__main__":
    main()
