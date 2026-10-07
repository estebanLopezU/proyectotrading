"""API Fase 4 - Trading Prototype (FastAPI).
Endpoints: /health /price /predict
Correr: uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Query
from src.indicators import add_indicators
from src.ingest import fetch_ticker_24h, get_ohlcv
from src.ml_panel import ml_signal

app = FastAPI(title="Trading Prototype API", version="1.0-fase4")


@app.get("/health")
def health():
    return {"ok": True, "service": "trading-prototype", "phase": 4}


@app.get("/price")
def price(symbol: str = Query(default="BTC/USDT"), timeframe: str = Query(default="1m")):
    res = get_ohlcv(symbol, timeframe, 5)
    tick = fetch_ticker_24h(symbol)
    last = float(res.df["close"].iloc[-1])
    return {"symbol": symbol, "timeframe": timeframe, "last": tick.get("last") or last,
            "ticker": tick, "source": res.source, "stale": res.stale, "latency_ms": round(res.latency_ms)}


@app.get("/predict")
def predict(symbol: str = Query(default="BTC/USDT"), timeframe: str = Query(default="15m")):
    res = get_ohlcv(symbol, timeframe, 300)
    df = add_indicators(res.df)
    ml = ml_signal(df, symbol, timeframe)
    price_now = float(df["close"].iloc[-1])
    rsi = float(df["rsi14"].iloc[-1] or 50)
    return {"symbol": symbol, "timeframe": timeframe, "price": price_now, "rsi14": round(rsi, 2),
            "source": res.source, "stale": res.stale, "ml": ml,
            "signal": ml.get("label"), "precision": ml.get("precision", 0),
            "thresholds": {"lo": ml.get("thr_lo"), "hi": ml.get("thr_hi")},
            "disclaimer": "Educativo. Precision historica validada en test; no garantiza futuro."}
