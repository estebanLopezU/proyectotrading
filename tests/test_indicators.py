import numpy as np
import pandas as pd

from src.indicators import add_indicators


def _sample(n=120):
    rng = np.random.default_rng(7)
    price = 67000 + np.cumsum(rng.normal(0, 40, n))
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"),
            "open": price,
            "high": price + 15,
            "low": price - 15,
            "close": price + rng.normal(0, 5, n),
            "volume": np.abs(rng.normal(10, 2, n)),
        }
    )
    return df


def test_add_indicators_columns():
    out = add_indicators(_sample())
    for c in ["sma20", "sma50", "ema12", "ema26", "rsi14", "macd", "macd_signal", "bb_high", "bb_low", "atr14"]:
        assert c in out.columns, f"falta {c}"


def test_rsi_range():
    out = add_indicators(_sample()).dropna(subset=["rsi14"])
    assert out["rsi14"].between(0, 100).all()


def test_no_infinite():
    out = add_indicators(_sample())
    assert np.isfinite(out[["close", "sma20", "ema12"]].dropna().to_numpy()).all()
