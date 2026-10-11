"""Tests para scorer.py - nucleo del ML."""
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.scorer import load_bundle, score_signal, wilson_lb


def _df(n=300):
    """Genera DataFrame de prueba con OHLCV."""
    rng = np.random.default_rng(42)
    p = 60000 + np.cumsum(rng.normal(0, 30, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": p,
            "high": p + 12,
            "low": p - 12,
            "close": p,
            "volume": np.abs(rng.normal(10, 2, n)),
        }
    )


def test_wilson_lb_perfect():
    """Wilson LB nunca debe dar 1.0 con n finito."""
    lb = wilson_lb(1.0, 100)
    assert lb < 1.0
    assert lb > 0.0


def test_wilson_lb_zero():
    """Wilson LB con n=0 debe dar 0."""
    assert wilson_lb(0.5, 0) == 0.0


def test_wilson_lb_symmetry():
    """Wilson LB(p) ~= 1 - Wilson LB(1-p) con tolerancia razonable.

    Nota: El intervalo de Wilson no es perfectamente simetrico,
    pero la asimetria es pequena para n grande.
    """
    lb_up = wilson_lb(0.7, 200)
    lb_down = wilson_lb(0.3, 200)
    assert abs(lb_up - (1 - lb_down)) < 0.15


def test_score_signal_no_model():
    """Sin modelo debe devolver SIN MODELO.

    Mockea load_bundle para que devuelva None.
    """
    df = _df()
    with patch("src.scorer.load_bundle", return_value=(None, None)):
        result = score_signal(df, "FAKE/COIN", "1h")
    assert result["label"] == "SIN MODELO"
    assert result["ok"] is False


def test_load_bundle_no_model():
    """Sin archivos de modelo debe devolver None.

    Mockea glob.glob para simular que no hay modelos.
    """
    with patch("src.scorer.glob.glob", return_value=[]):
        bundle, path = load_bundle("FAKE/COIN", "1h")
    assert bundle is None
    assert path is None


def test_score_signal_returns_required_keys():
    """score_signal debe devolver todas las claves requeridas."""
    df = _df()
    with patch("src.scorer.load_bundle", return_value=(None, None)):
        result = score_signal(df, "FAKE/COIN", "1h")
    required_keys = [
        "label", "proba", "confidence", "precision", "n_test",
        "calibration", "note", "ok", "prob_up", "prob_down",
        "horizon_velas", "wilson_lb", "rule", "model_timeframe", "fire",
    ]
    for key in required_keys:
        assert key in result, f"falta clave: {key}"
