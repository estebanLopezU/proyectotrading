"""Simulador de trading - versión mínima y garantizada."""
from __future__ import annotations

import streamlit as st


def render_trading_game(symbol: str, timeframe: str) -> None:
    """Versión mínima: siempre muestra algo."""
    # Siempre inicializar si no existe
    if "game_cash" not in st.session_state:
        st.session_state.game_cash = 10000.0
    
    # Título y capital AL INICIO - absolutamente visible
    st.success(f"🎮 Simulador Trading activo: {symbol} {timeframe}")
    st.write(f"**Capital:** ${st.session_state.game_cash:,.2f}")
    
    # Botón de test
    if st.button("Probar operación"):
        st.session_state.game_cash += 100
        st.rerun()