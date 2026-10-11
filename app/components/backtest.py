"""Backtest interactivo con equity curve."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.indicators import add_indicators, latest_signal
from src.ingest import get_ohlcv


def run_simple_backtest(
    df: pd.DataFrame,
    initial_cash: float = 10000.0,
    fee: float = 0.00075,
) -> dict:
    """Ejecuta un backtest simple basado en señales técnicas.

    Estrategia:
    - Compra cuando score >= 3 (señal COMPRAR)
    - Vende cuando score <= -3 (señal VENDER)
    """
    cash = initial_cash
    qty = 0.0
    equity_curve = []
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        ts = str(row["timestamp"])

        # Calcular señal
        sig = latest_signal(df.iloc[:i + 1])
        score = sig.get("score", 0)

        # Lógica de trading
        if score >= 3 and qty == 0 and cash > 0:
            # Comprar
            q = (cash / price) * (1 - fee)
            cost = q * price * (1 + fee)
            if cost <= cash:
                qty = q
                cash -= cost
                trades.append({"ts": ts, "side": "BUY", "price": price, "qty": q})

        elif score <= -3 and qty > 0:
            # Vender
            proceeds = qty * price * (1 - fee)
            pnl = proceeds - (qty * price * (1 + fee))
            trades.append({"ts": ts, "side": "SELL", "price": price, "qty": qty, "pnl": pnl})
            cash += proceeds
            qty = 0.0

        equity = cash + qty * price
        equity_curve.append({"ts": ts, "equity": equity, "price": price})

    # Calcular métricas
    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) == 0:
        return {"error": "Sin datos"}

    total_return = (eq_df["equity"].iloc[-1] / initial_cash - 1) * 100
    n_trades = len([t for t in trades if t["side"] == "SELL"])

    # Calcular max drawdown
    cummax = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - cummax) / cummax
    max_dd = drawdown.min() * 100

    # Win rate
    wins = [t for t in trades if t["side"] == "SELL" and t.get("pnl", 0) > 0]
    losses = [t for t in trades if t["side"] == "SELL" and t.get("pnl", 0) <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

    return {
        "equity_curve": eq_df,
        "trades": trades,
        "total_return": total_return,
        "n_trades": n_trades,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "wins": len(wins),
        "losses": len(losses),
    }


def render_backtest(symbol: str, timeframe: str) -> None:
    """Renderiza el backtest interactivo."""
    st.subheader("Backtest interactivo", icon=":material/history:")

    with st.expander("Configurar backtest", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            initial_cash = st.number_input("Capital inicial ($)", min_value=100, max_value=1000000, value=10000, step=1000)
        with col2:
            lookback = st.number_input("Velas a analizar", min_value=100, max_value=2000, value=500, step=100)

        if st.button("Ejecutar backtest", type="primary"):
            with st.spinner("Ejecutando backtest..."):
                try:
                    res = get_ohlcv(symbol, timeframe, lookback)
                    df = add_indicators(res.df)
                    result = run_simple_backtest(df, initial_cash=initial_cash)

                    if "error" in result:
                        st.error(result["error"])
                        return

                    # Métricas
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Retorno total", f"{result['total_return']:+.2f}%")
                    m_col2.metric("Operaciones", result['n_trades'])
                    m_col3.metric("Max drawdown", f"{result['max_drawdown']:.2f}%")
                    m_col4.metric("Win rate", f"{result['win_rate']:.1f}%")

                    # Equity curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result["equity_curve"]["ts"],
                        y=result["equity_curve"]["equity"],
                        name="Equity",
                        line=dict(color="#60A5FA", width=2),
                        fill='tozeroy',
                        fillcolor="rgba(96,165,250,0.1)",
                    ))
                    fig.add_hline(y=initial_cash, line_dash="dash", line_color="#94A3B8",
                                 annotation_text="Capital inicial")
                    fig.update_layout(
                        template="plotly_dark",
                        title="Curva de equity",
                        yaxis_title="Equity ($)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#CBD5E1"),
                        height=300,
                    )
                    st.plotly_chart(fig, width="stretch")

                    # Tabla de trades
                    if result["trades"]:
                        trades_df = pd.DataFrame(result["trades"])
                        st.dataframe(trades_df.tail(20), width="stretch", hide_index=True)

                except Exception as e:  # noqa: BLE001
                    st.error(f"Error en backtest: {e}")
