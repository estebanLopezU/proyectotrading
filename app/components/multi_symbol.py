"""Panel de comparación multi-símbolo."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.indicators import add_indicators
from src.ingest import fetch_ticker_24h, get_ohlcv


def render_multi_symbol(symbols: list[str], timeframe: str) -> None:
    """Renderiza tabla comparativa y gráfico de múltiples símbolos."""
    st.subheader("Comparación multi-símbolo", icon=":material/compare_arrows:")

    # Cargar datos para cada símbolo
    data = {}
    for sym in symbols:
        try:
            res = get_ohlcv(sym, timeframe, 100)
            df = add_indicators(res.df)
            tick = fetch_ticker_24h(sym)
            data[sym] = {"df": df, "tick": tick, "stale": res.stale, "source": res.source}
        except Exception as e:  # noqa: BLE001
            st.warning(f"No se pudo cargar {sym}: {e}")

    if not data:
        st.warning("No se pudieron cargar datos para ningún símbolo.")
        return

    # Tabla comparativa
    rows = []
    for sym, d in data.items():
        df = d["df"]
        tick = d["tick"]
        last = df.iloc[-1]
        rows.append({
            "Símbolo": sym,
            "Precio": float(tick.get("last") or last["close"]),
            "Cambio 24h %": float(tick.get("pct_24h") or 0.0),
            "RSI 14": float(last.get("rsi14", 50)),
            "ATR %": float(last.get("atr_pct", 0)),
            "Volumen": float(df["volume"].iloc[-1]),
            "Estado": "EN VIVO" if not d["stale"] else "CACHÉ",
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Gráfico comparativo de rendimiento normalizado
    fig = go.Figure()
    for sym, d in data.items():
        df = d["df"]
        closes = df["close"].tail(50)
        if len(closes) > 0:
            normalized = (closes / closes.iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=df["timestamp"].tail(50),
                y=normalized,
                name=sym,
                mode="lines",
            ))

    fig.update_layout(
        template="plotly_dark",
        title="Rendimiento normalizado (últimas 50 velas)",
        yaxis_title="Rendimiento %",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1"),
        height=300,
        legend=dict(orientation="h", y=1.1, x=0),
    )
    st.plotly_chart(fig, width="stretch")
