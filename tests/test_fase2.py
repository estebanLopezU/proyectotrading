import numpy as np
import pandas as pd

from src.features import latest_features, make_dataset
from src.indicators import add_indicators
from src.paper import PaperAccount


def _df(n=300):
    rng = np.random.default_rng(3)
    p = 60000 + np.cumsum(rng.normal(0, 30, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": p, "high": p + 12, "low": p - 12,
            "close": p, "volume": 10,
        }
    )


def test_features_no_leak():
    out = add_indicators(_df())
    X, y, cols, full = make_dataset(out, 5)
    assert len(X) > 100 and set(cols) <= set(full.columns)
    assert set(y.unique().tolist()) <= {0, 1}


def test_latest_features_row():
    out = add_indicators(_df())
    _, _, cols, _ = make_dataset(out, 5)
    assert len(latest_features(out, cols)) == 1


def test_paper_account():
    a = PaperAccount()
    assert a.buy(100, "t1")["ok"] is True
    assert a.buy(100, "t2")["ok"] is False
    eq = a.equity(110)
    assert eq > 10000
    assert a.sell(110, "t3")["ok"] is True
    assert a.qty == 0 and a.cash > 0
