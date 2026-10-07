"""Trade statistics: calcula estadisticas de trading por periodo.
Incluye equity curve, monthly stats, drawdown, reward:risk metrics.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def trades_to_df(trades: list) -> pd.DataFrame:
    """Convierte lista de trades cerrados a DataFrame."""
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("side") == "SELL"]
    if not closed:
        return pd.DataFrame()
    df = pd.DataFrame(closed)
    df["ts"] = pd.to_datetime(df["ts"])
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    return df


def get_period_mask(df: pd.DataFrame, period: str) -> pd.Series:
    """Devuelve mask booleana para trades dentro del periodo."""
    now = pd.Timestamp.now(tz="UTC")
    if period == "week":
        start = now - pd.Timedelta(days=7)
    elif period == "month":
        start = now - pd.Timedelta(days=30)
    elif period == "year":
        start = now - pd.Timedelta(days=365)
    else:  # all
        return pd.Series([True] * len(df), index=df.index) if not df.empty else pd.Series(dtype=bool)
    if df.empty:
        return pd.Series(dtype=bool)
    return df["ts"] >= start


def period_stats(df: pd.DataFrame, period: str) -> dict:
    """Calcula estadisticas RR, win rate, PnL para un periodo."""
    if df.empty:
        return {"rr": 0.0, "win_rate": 0.0, "pnl": 0.0, "n": 0, "wins": 0, "losses": 0, "pf": 0.0}
    # Asegurar que ts es datetime
    if "ts" in df.columns:
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"])
    mask = get_period_mask(df, period)
    sub = df[mask]
    if sub.empty:
        return {
            "rr": 0.0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "n": 0,
            "wins": 0,
            "losses": 0,
            "pf": 0.0,
        }
    wins = sub[sub["pnl"] > 0]
    losses = sub[sub["pnl"] <= 0]
    n = len(sub)
    win_rate = len(wins) / n if n > 0 else 0.0
    gross_profit = wins["pnl"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_rr = sub["rr"].mean() if sub["rr"].notna().any() else 0.0
    return {
        "rr": avg_rr,
        "win_rate": win_rate,
        "pnl": sub["pnl"].sum(),
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "pf": pf,
    }


def build_equity_curve(trades: list) -> pd.DataFrame:
    """Construye curva de equity a partir de trades cerrados.
    Usa equity_curve si existe, sino reconstruye desde trades."""
    import pandas as pd
    # Si hay equity_curve registrado, usarlo
    if hasattr(trades, 'equity_curve') and trades.equity_curve:
        ec = pd.DataFrame(trades.equity_curve)
        ec["ts"] = pd.to_datetime(ec["ts"])
        return ec.sort_values("ts").reset_index(drop=True)
    # Si es una lista plana de trades
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("side") == "SELL"] if isinstance(trades, list) else []
    if not closed:
        return pd.DataFrame()
    df = pd.DataFrame(closed)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df["equity"] = 10000.0 + df["pnl"].cumsum()
    return df[["ts", "equity"]]


def monthly_stats(trades: list, years: list = None) -> pd.DataFrame:
    """Tabla de estadisticas mensuales (win rate, RR, PnL)."""
    df = trades_to_df(trades)
    if df.empty:
        return pd.DataFrame()
    if years is None:
        years = sorted(df["ts"].dt.year.unique().tolist())
    rows = []
    for y in years:
        for m in range(1, 13):
            sub = df[(df["ts"].dt.year == y) & (df["ts"].dt.month == m)]
            if sub.empty:
                continue
            wins = sub[sub["pnl"] > 0]
            losses = sub[sub["pnl"] <= 0]
            n = len(sub)
            win_rate = len(wins) / n if n > 0 else 0
            rr_mean = sub["rr"].mean() if sub["rr"].notna().any() else None
            rows.append({
                "Year": y,
                "Month": m,
                "Trades": n,
                "Win Rate": f"{win_rate*100:.1f}%",
                "RR": f"{rr_mean:.2f}" if rr_mean is not None else "-",
                "PnL": float(sub["pnl"].sum()),
                "Avg Win": float(wins["pnl"].mean()) if not wins.empty else None,
                "Avg Loss": float(losses["pnl"].mean()) if not losses.empty else None,
            })
    out = pd.DataFrame(rows)
    if not out.empty and "PnL" in out.columns:
        out["PnL"] = pd.to_numeric(out["PnL"], errors="coerce")
        out["Avg Win"] = pd.to_numeric(out["Avg Win"], errors="coerce")
        out["Avg Loss"] = pd.to_numeric(out["Avg Loss"], errors="coerce")
    return out


def drawdown_curve(equity: pd.DataFrame) -> pd.DataFrame:
    """Calcula drawdown a partir de equity curve."""
    if equity.empty or "equity" not in equity.columns:
        return pd.DataFrame()
    eq = equity.copy()
    eq["peak"] = eq["equity"].cummax()
    eq["drawdown"] = (eq["equity"] - eq["peak"]) / eq["peak"] * 100
    return eq


def risk_reward_summary(trades: list) -> dict:
    """Analisis completo de Riesgo/Beneficio."""
    df = trades_to_df(trades)
    if df.empty:
        return {"rr_list": [], "avg_rr": 0, "median_rr": 0, "rr_std": 0}
    rr_valid = df["rr"].dropna()
    rr_list = rr_valid.tolist() if not rr_valid.empty else []
    return {
        "rr_list": rr_list,
        "avg_rr": rr_valid.mean() if not rr_valid.empty else 0,
        "median_rr": rr_valid.median() if not rr_valid.empty else 0,
        "rr_std": rr_valid.std() if not rr_valid.empty else 0,
    }
