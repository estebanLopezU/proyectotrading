"""Gráfico de velas + volumen."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render_chart(df: pd.DataFrame, symbol: str) -> None:
    """Renderiza el gráfico de velas con volumen."""
    st.subheader("Gráfico de velas", icon=":material/candlestick_chart:")
    plot_df = df.tail(200)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25],
    )
    fig.add_trace(go.Candlestick(
        x=plot_df["timestamp"],
        open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"],
        increasing_line_color="#34D399", decreasing_line_color="#F87171",
        name=symbol,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["sma20"],
        line=dict(color="#60A5FA", width=1.4), name="SMA 20",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["sma50"],
        line=dict(color="#FBBF24", width=1.4), name="SMA 50",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["bb_low"],
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["bb_high"],
        line=dict(width=0), fill="tonexty", fillcolor="rgba(96,165,250,.09)",
        hoverinfo="skip", name="Bollinger",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=plot_df["timestamp"], y=plot_df["volume"],
        marker_color=["#34D399" if c >= o else "#F87171"
                      for c, o in zip(plot_df["close"], plot_df["open"], strict=True)],
        name="Volumen",
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family="Inter, sans-serif"),
        margin=dict(l=16, r=16, t=10, b=16),
        height=540,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.06, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#1E293B", zeroline=False)
    st.plotly_chart(fig, width="stretch")
