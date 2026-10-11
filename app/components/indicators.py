"""Panel de indicadores técnicos."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_indicators(df: pd.DataFrame) -> None:
    """Renderiza el panel de indicadores técnicos."""
    st.subheader("Indicadores técnicos", icon=":material/query_stats:")
    st.caption("Lectura de la última vela")

    last = df.iloc[-1]
    rsi_val = float(last.get("rsi14", 50) or 50)
    bb_val = float(last.get("bb_pct", 0.5) or 0.5)
    _hi24 = float(df["high"].tail(24).max())
    _lo24 = float(df["low"].tail(24).min())
    range_pos = (float(last["close"]) - _lo24) / (_hi24 - _lo24) if _hi24 > _lo24 else 0.5
    macd_hist = float(last.get("macd_hist", 0.0) or 0.0)

    ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
    with ind_col1:
        with st.container(border=True):
            st.metric("RSI 14", f"{rsi_val:.1f}")
            st.progress(min(max(rsi_val, 0), 100) / 100,
                        text="Sobreventa < 30 · Sobrecompra > 70")
            if rsi_val > 70:
                st.markdown(":red-badge[Sobrecompra]")
            elif rsi_val < 30:
                st.markdown(":green-badge[Sobreventa]")
    with ind_col2:
        with st.container(border=True):
            st.metric("Bollinger %B", f"{bb_val:.2f}")
            st.progress(min(max(bb_val, 0), 1), text="Rango habitual 0 – 1")
            if bb_val > 0.8:
                st.markdown(":red-badge[Banda superior]")
            elif bb_val < 0.2:
                st.markdown(":green-badge[Banda inferior]")
    with ind_col3:
        with st.container(border=True):
            st.metric("Posición en rango", f"{range_pos:.0%}", "Últimas 24 velas")
            st.progress(min(max(range_pos, 0), 1))
    with ind_col4:
        with st.container(border=True):
            st.metric("MACD histograma", f"{macd_hist:,.2f}")
            st.progress(min(max(50 + macd_hist / float(last["close"]) * 10000, 0), 100) / 100)
            if macd_hist > 0:
                st.markdown(":green-badge[Momentum alcista]")
            else:
                st.markdown(":red-badge[Momentum bajista]")
