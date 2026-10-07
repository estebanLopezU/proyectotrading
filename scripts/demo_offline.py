"""Demo offline Fase 1: prueba dashboard sin internet ni ccxt.
Genera velas sinteticas + indicadores y guarda PNG para validar.
Uso: python scripts/demo_offline.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.indicators import add_indicators, latest_signal

rng = np.random.default_rng(42)
n = 200
price = 67000 + np.cumsum(rng.normal(0, 35, n))
df = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"),
        "open": price,
        "high": price + 20,
        "low": price - 20,
        "close": price + rng.normal(0, 5, n),
        "volume": np.abs(rng.normal(12, 3, n)),
    }
)
out = add_indicators(df)
sig = latest_signal(out)
print(f"VELAS={len(out)} PRECIO={out['close'].iloc[-1]:.2f} RSI={out['rsi14'].iloc[-1]:.1f}")
print(f"SENAL={sig['label']} score={sig['score']} | {' | '.join(sig['reasons'])}")
print("OK demo_offline: nucleo trader funciona sin internet.")
