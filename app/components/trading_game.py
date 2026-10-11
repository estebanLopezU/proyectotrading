"""Panel de simulación de trading interactivo."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.indicators import add_indicators, latest_signal
from src.ingest import fetch_ticker_24h, get_ohlcv
from src.ml_panel import ml_signal
from src.utils import fmt_money


def render_trading_game(symbol: str, timeframe: str) -> None:
    """Panel interactivo para simular apuestas de trading."""
    st.subheader("Simulador de Trading", icon=":material/sports_esports:")

    # Inicializar estado del juego
    if "game_cash" not in st.session_state:
        st.session_state.game_cash = 10000.0
        st.session_state.game_qty = 0.0
        st.session_state.game_history = []
        st.session_state.game_active_bet = None

    # Cargar datos
    try:
        res = get_ohlcv(symbol, timeframe, 300)
        df = add_indicators(res.df)
        tick = fetch_ticker_24h(symbol)
    except Exception as e:  # noqa: BLE001
        st.error(f"Error cargando datos: {e}")
        return

    last = df.iloc[-1]
    price = float(tick.get("last") or last["close"])
    sig = latest_signal(df)
    ml_pred = ml_signal(df, symbol, timeframe)

    # Mostrar información del mercado
    with st.container(border=True):
        st.markdown("**Estado del Mercado**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precio Actual", fmt_money(price))
        col2.metric("RSI 14", f"{float(last.get('rsi14', 50)):.1f}")
        col3.metric("MACD", f"{float(last.get('macd_hist', 0)):,.2f}")
        col4.metric("ATR %", f"{float(last.get('atr_pct', 0)):.2f}%")

    # Mostrar predicción ML
    with st.container(border=True):
        st.markdown("**Predicción del Modelo ML**")
        if ml_pred.get("ok"):
            prob_up = float(ml_pred.get("prob_up") or 0.5)
            prob_down = float(ml_pred.get("prob_down") or 0.5)
            label = ml_pred.get("label", "SIN SEÑAL")
            fire = bool(ml_pred.get("fire"))
            prec = float(ml_pred.get("precision") or 0.0)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prob. Subida", f"{prob_up:.1%}")
                st.progress(prob_up)
            with col2:
                st.metric("Prob. Bajada", f"{prob_down:.1%}")
                st.progress(prob_down)

            if fire:
                if "COMPRAR" in label:
                    st.success(f"SEÑAL: {label} · Precisión: {prec:.1%}")
                elif "VENDER" in label:
                    st.warning(f"SEÑAL: {label} · Precisión: {prec:.1%}")
            else:
                st.info(f"Sin señal activa · Precisión histórica: {prec:.1%}")
        else:
            st.info("Modelo ML no disponible")

    # Panel de apuesta
    with st.container(border=True):
        st.markdown("**Hacer una Apuesta**")

        col1, col2 = st.columns(2)
        with col1:
            direction = st.radio(
                "Dirección",
                ["LARGO (sube)", "CORTO (baja)", "MANTENER"],
                horizontal=True,
                key="bet_direction"
            )
        with col2:
            bet_pct = st.slider(
                "% del capital a apostar",
                min_value=5,
                max_value=100,
                value=25,
                step=5,
                key="bet_amount"
            )

        bet_amount = st.session_state.game_cash * (bet_pct / 100)

        col1, col2, col3 = st.columns(3)
        col1.metric("Capital Disponible", f"${st.session_state.game_cash:,.2f}")
        col2.metric("Monto a Apostar", f"${bet_amount:,.2f}")
        col3.metric("Posición Actual", f"{st.session_state.game_qty:.6f}")

        # Botón para ejecutar apuesta
        if st.button("Ejecutar Apuesta", type="primary", use_container_width=True):
            if direction == "MANTENER":
                st.info("Manteniendo posición. No se ejecuta ninguna operación.")
            else:
                # Registrar la apuesta
                bet = {
                    "timestamp": str(last["timestamp"]),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "direction": direction,
                    "entry_price": price,
                    "amount": bet_amount,
                    "prob_up": float(ml_pred.get("prob_up") or 0.5) if ml_pred.get("ok") else 0.5,
                    "status": "OPEN",
                }
                st.session_state.game_active_bet = bet
                st.session_state.game_history.append(bet)
                st.success(f"Apuesta registrada: {direction} a ${price:,.2f}")

        # Mostrar apuesta activa
        if st.session_state.game_active_bet:
            bet = st.session_state.game_active_bet
            st.markdown(f"**Apuesta Activa:** {bet['direction']} a ${bet['entry_price']:,.2f}")

            if st.button("Cerrar Apuesta", type="secondary"):
                # Calcular resultado
                current_price = price
                entry_price = bet["entry_price"]
                amount = bet["amount"]

                if "LARGO" in bet["direction"]:
                    pnl_pct = (current_price / entry_price - 1)
                else:  # CORTO
                    pnl_pct = (1 - current_price / entry_price)

                pnl = amount * pnl_pct
                st.session_state.game_cash += pnl

                # Actualizar historial
                bet["status"] = "CLOSED"
                bet["exit_price"] = current_price
                bet["pnl"] = pnl
                bet["pnl_pct"] = pnl_pct * 100

                st.session_state.game_active_bet = None

                if pnl > 0:
                    st.success(f"Ganancia: ${pnl:,.2f} ({pnl_pct*100:+.2f}%)")
                else:
                    st.error(f"Pérdida: ${pnl:,.2f} ({pnl_pct*100:+.2f}%)")

                st.rerun()

    # Historial de apuestas
    if st.session_state.game_history:
        with st.expander(f"Historial de Apuestas ({len(st.session_state.game_history)})"):
            history_df = pd.DataFrame(st.session_state.game_history)
            st.dataframe(history_df.tail(10), use_container_width=True)

            # Estadísticas
            closed_bets = [b for b in st.session_state.game_history if b.get("status") == "CLOSED"]
            if closed_bets:
                wins = [b for b in closed_bets if b.get("pnl", 0) > 0]
                losses = [b for b in closed_bets if b.get("pnl", 0) <= 0]
                total_pnl = sum(b.get("pnl", 0) for b in closed_bets)

                col1, col2, col3 = st.columns(3)
                col1.metric("Ganadas", len(wins))
                col2.metric("Perdidas", len(losses))
                col3.metric("P&L Total", f"${total_pnl:,.2f}")
