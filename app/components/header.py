"""Header del dashboard con precio y estado."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.utils import fmt_money


def render_header(
    symbol: str,
    timeframe: str,
    price: float,
    change_24h: float,
    source: str,
    latency_ms: float,
    n_velas: int,
    stale: bool,
) -> None:
    """Renderiza el header con precio, cambio 24h y estado de conexión."""
    chg_color = "#34D399" if change_24h >= 0 else "#F87171"
    estado = "EN VIVO" if not stale else "DATOS EN CACHÉ"

    st.markdown(f"""
    <div class="app-header">
        <div>
            <div class="title">Mercado <span class="sep">/</span> {symbol}</div>
            <div class="meta">Terminal de análisis · {timeframe} · Señales técnicas, modelo ML y simulación</div>
            <div class="meta" style="margin-top:8px">
                <span class="status-pill {'status-live' if not stale else 'status-stale'}">
                    <span class="status-dot"></span>{estado}
                </span>
                <span style="margin-left:6px">Fuente: {source} · {latency_ms:.0f} ms · {n_velas} velas</span>
            </div>
        </div>
        <div class="price-box">
            <div class="price" style="color:{chg_color}">{fmt_money(price)}</div>
            <div style="color:{chg_color};font-weight:700">{change_24h:+.2f}% · 24 h</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
