"""Simulador de trading - versión simplificada."""
from __future__ import annotations

import streamlit as st


def render_trading_game(symbol: str, timeframe: str) -> None:
    """Panel simulador de trading - versión mínima."""
    st.write("DEBUG: Trading Game activo")
    
    # Inicializar capital si no existe
    if "game_cash" not in st.session_state:
        st.session_state.game_cash = 10000.0
    
    # Mostrar información simple
    st.info(f"Simulador - {symbol} {timeframe}")
    st.write("Capital:", st.session_state.game_cash)
    
    # Botón simple
    if st.button("Cerrar debug"):
        st.rerun()