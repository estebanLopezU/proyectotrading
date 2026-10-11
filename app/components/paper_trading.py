"""Panel de paper trading simulado."""
from __future__ import annotations

import streamlit as st


def render_paper_trading(px: float) -> None:
    """Renderiza el panel de paper trading con cuenta simulada $10,000."""
    st.subheader("Paper trading simulado ($10,000)", icon=":material/account_balance_wallet:")

    if "paper_cash" not in st.session_state:
        st.session_state.paper_cash = 10000.0
        st.session_state.paper_qty = 0.0
        st.session_state.paper_log = []

    eq = st.session_state.paper_cash + st.session_state.paper_qty * px
    pnl_pct = (eq / 10000 - 1) * 100

    with st.container(horizontal=True):
        st.metric("Patrimonio", f"${eq:,.2f}", f"{pnl_pct:+.2f}%", border=True)
        st.metric("Efectivo", f"${st.session_state.paper_cash:,.2f}", border=True)
        st.metric("Posición", f"{st.session_state.paper_qty:.6f}", border=True)

    b1, b2 = st.columns(2)
    if b1.button(":material/trending_up: Comprar sim (todo)", width="stretch"):
        if st.session_state.paper_qty == 0 and st.session_state.paper_cash > 0:
            q = (st.session_state.paper_cash / px) * 0.99925
            st.session_state.paper_qty = q
            st.session_state.paper_cash = 0.0
            st.session_state.paper_log.append(f"BUY ${px:.2f} × {q:.6f}")
            st.rerun()
    if b2.button(":material/trending_down: Vender sim (todo)", width="stretch"):
        if st.session_state.paper_qty > 0:
            st.session_state.paper_cash = st.session_state.paper_qty * px * 0.99925
            st.session_state.paper_log.append(f"SELL ${px:.2f} → ${st.session_state.paper_cash:,.2f}")
            st.session_state.paper_qty = 0.0
            st.rerun()

    if st.session_state.paper_log:
        with st.expander(f"Registro de operaciones ({len(st.session_state.paper_log)})", expanded=True):
            for log_entry in st.session_state.paper_log[-10:]:
                st.markdown(f'<div class="trade-log">{log_entry}</div>', unsafe_allow_html=True)
