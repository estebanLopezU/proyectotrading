"""Backtest walk-forward simple: entrena en pasado, opera en futuro.
Regla: proba>=0.6 COMPRA, proba<=0.4 VENDE. Solo LONG.
Uso: python scripts/backtest.py --symbol BTC/USDT --timeframe 1h --limit 1500
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.features import make_dataset
from src.indicators import add_indicators
from src.ingest import get_ohlcv
from src.paper import PaperAccount


def max_drawdown(equity: list[float]) -> float:
    peak, mdd = equity[0], 0.0
    for e in equity:
        peak = max(peak, e)
        mdd = min(mdd, (e / peak - 1))
    return mdd * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=5)
    a = ap.parse_args()

    res = get_ohlcv(a.symbol, a.timeframe, a.limit)
    df = add_indicators(res.df)
    X, y, cols, full = make_dataset(df, a.horizon)
    prices = full["close"].to_numpy()
    ts = full["timestamp"].astype(str).to_numpy()

    split = int(len(X) * 0.6)
    clf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=7, n_jobs=-1)
    clf.fit(X.iloc[:split], y.iloc[:split])

    acct = PaperAccount()
    curve = []
    for i in range(split, len(X)):
        p = float(prices[i])
        proba = float(clf.predict_proba(X.iloc[[i]])[0][1])
        if proba >= 0.6:
            acct.buy(p, ts[i])
        elif proba <= 0.4:
            acct.sell(p, ts[i])
        curve.append(acct.equity(p))
    if acct.qty > 0:
        acct.sell(float(prices[-1]), ts[-1])
        curve[-1] = acct.equity(float(prices[-1]))

    eq0, eq1 = 10000.0, curve[-1] if curve else 10000.0
    ret = (eq1 / eq0 - 1) * 100
    rets = np.diff(curve) / np.array(curve[:-1]) if len(curve) > 2 else np.array([0])
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(24 * 365)) if len(rets) > 2 else 0
    wins = sum(1 for j in range(1, len(curve)) if curve[j] > curve[j - 1])
    print(f"BACKTEST {a.symbol} {a.timeframe} fuente={res.source}")
    print(f"Equity {eq0:.0f} -> {eq1:.0f} | Ret {ret:.2f}% | MDD {max_drawdown(curve):.2f}% | Sharpe~{sharpe:.2f}")
    print(f"Trades={len(acct.trades)} WinSteps={wins}/{len(curve)}")
    print("AVISO: historico no garantiza futuro. Solo educativo.")


if __name__ == "__main__":
    main()
