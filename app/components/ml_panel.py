"""Panel de señales ML."""
from __future__ import annotations

import streamlit as st

from src.alerts import send_telegram


def render_ml_panel(
    ml_pred: dict,
    symbol: str,
    timeframe: str,
    df,
) -> None:
    """Renderiza el panel de señales ML validadas."""
    st.markdown("**Señal ML validada**")
    if ml_pred.get("ok"):
        prob_up = float(ml_pred.get("prob_up") or 0.5)
        fire = bool(ml_pred.get("fire"))
        label = ml_pred.get("label", "SIN SEÑAL")
        prec = float(ml_pred.get("precision") or 0.0)
        lb = float(ml_pred.get("wilson_lb") or 0.0)
        n_test = int(ml_pred.get("n_test") or 0)
        mtf = ml_pred.get("model_timeframe", timeframe)
        hz = ml_pred.get("horizon_velas", 0)
        min_ret = float(ml_pred.get("min_ret") or 0.0)

        if fire and "COMPRAR" in label:
            st.markdown('<div class="signal-banner banner-buy">SEÑAL: COMPRAR (LARGO)</div>',
                        unsafe_allow_html=True)
        elif fire and "VENDER" in label:
            st.markdown('<div class="signal-banner banner-sell">SEÑAL: VENDER (CORTO)</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="signal-banner banner-none">SIN SEÑAL · el edge validado no se cumple ahora</div>',
                        unsafe_allow_html=True)

        st.progress(min(max(prob_up, 0.0), 1.0))
        st.markdown(
            f"**P(subida > {min_ret:.1%}) = {prob_up:.1%}** · "
            f"Modelo: {mtf} · horizonte: {hz} velas"
        )
        if ml_pred.get("target_desc"):
            st.caption(f"Objetivo: {ml_pred['target_desc']}")
        st.caption(
            f"Precisión validada: {prec:.1%} · IC95% Wilson LB: {lb:.1%} · muestras: {n_test}"
        )
        cond = ml_pred.get("conditions") or {}
        if cond:
            req = cond.get("cond_requerida", "-")
            cumple = cond.get("cond_cumplida", False)
            st.caption(f"Condición requerida: {req} → "
                       f"{'cumplida' if cumple else 'NO cumplida'}")

        # Alerta Telegram una sola vez por vela (evita spam en cada rerun)
        if fire and prec >= 0.80:
            sig_key = f"{symbol}|{mtf}|{label}|{df['timestamp'].iloc[-1]}"
            if st.session_state.get("last_alert") != sig_key:
                r = send_telegram(
                    f"{symbol} {mtf} {label} p={ml_pred.get('proba', 0):.2f} "
                    f"prec={prec:.0%} LB={lb:.0%}"
                )
                if r.get("ok"):
                    st.session_state.last_alert = sig_key
                    st.toast("Alerta Telegram enviada")
    else:
        st.info(f"ML no disponible: {ml_pred.get('label', 'Modelo no entrenado')}")

    st.caption(
        "Educativo · precisión histórica validada, no garantiza el futuro. Gestiona tu riesgo."
    )
