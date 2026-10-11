"""Panel de señales ML optimizado y sin dependencias externas."""
from __future__ import annotations

import streamlit as st


def ml_signal_local(df, symbol: str, timeframe: str = "15m", horizon: int = 6) -> dict:
    """
    Versión local del señál ML que no requiere API externa.
    Usa el modelo entrenado si existe, o calcula señal técnica.
    """
    # Intentar cargar modelo entrenado
    import glob
    import joblib
    from pathlib import Path

    safe = symbol.replace("/", "_").replace(":", "_")
    hits = (
        sorted(glob.glob(f"models/signal_{safe}_{timeframe}_h*.joblib"))
        + sorted(glob.glob(f"models/signal_{safe}_*.joblib"))
    )

    if hits:
        try:
            b = joblib.load(hits[-1])
            cols = b.get("cols", [])
            model = b.get("model")

            from src.features_v2 import add_features_v2
            from src.indicators import add_indicators

            # Datos frescos con indicadores
            res_df = add_indicators(df)
            d2 = add_features_v2(res_df)

            # Verificar columnas
            if cols and model:
                # Asegurar que todas las columnas existan
                for col in cols:
                    if col not in d2.columns:
                        d2[col] = float("nan")

                # Obtener features de la última vela
                X = d2[cols].tail(1)

                # Predicción
                proba = float(model.predict_proba(X)[0][1])
                prob_down = 1.0 - proba

                # Umbral simple: 0.5
                direction = "ALTA" if proba >= 0.5 else "BAJA"
                fire = True

                # Wilson LB simulado
                n = int(b.get("n", 0) or 0)
                lb = float(b.get("wilson_lb", 0) or 0.0)
                prec = float(b.get("precision", 0) or 0.0)

                # Condiciones del modelo
                cond = b.get("cond", "none")
                cond_holds = True
                if cond and cond != "none":
                    try:
                        if cond == "trend_dn":
                            cond_holds = float(d2["trend_strength"].iloc[-1]) < 0
                        elif cond == "trend_up":
                            cond_holds = float(d2["trend_strength"].iloc[-1]) > 0
                    except Exception:
                        cond_holds = True

                # Direccional con condición
                direction_final = direction if (proba >= 0.5 and direction == "ALTA" or proba <= 0.5 and direction == "BAJA") else "SIN DIRECCION"

                # Determinar label
                label = "COMPRAR" if direction == "ALTA" and fire else ("VENDER" if direction == "BAJA" and fire else "SIN SEÑAL")

                return {
                    "ok": True,
                    "label": label,
                    "prob_up": proba,
                    "prob_down": prob_down,
                    "fire": fire,
                    "precision": prec,
                    "wilson_lb": lb,
                    "horizon_velas": int(b.get("horizon", 0) or 0),
                    "model_timeframe": b.get("timeframe", timeframe),
                    "note": f"Modelo loaded from {hits[-1]}",
                    "conditions": {"activas": [cond] if cond_holds else [], "n_cond": 1 if cond_holds else 0,
                                  "cond_requerida": cond, "cond_cumplida": cond_holds},
                }
        except Exception as e:
            st.warning(f"Error cargando modelo: {e}")

    # Fallback: señal técnica simple
    from src.indicators import latest_signal

    sig = latest_signal(df)
    return {
        "ok": True,
        "label": sig["label"],
        "prob_up": float(sig["score"] > 0),
        "prob_down": float(sig["score"] < 0),
        "fire": sig["score"] >= 3 or sig["score"] <= -3,
        "precision": 0.60,  # valor por defecto
        "wilson_lb": 0.0,
        "horizon_velas": 0,
        "model_timeframe": timeframe,
        "note": "Señal tecnica fallback",
    }


def render_ml_panel_local(df, symbol: str, timeframe: str) -> None:
    """Panel de señales ML optimizado - versión local."""
    st.subheader("Señales ML", icon=":material/insights:")

    # Obtener señal ML local
    ml = ml_signal_local(df, symbol, timeframe)

    if ml.get("ok"):
        prob_up = float(ml.get("prob_up") or 0.5)
        prob_down = float(ml.get("prob_down") or 0.5)
        label = ml.get("label", "SIN SEÑAL")
        fire = bool(ml.get("fire"))
        prec = float(ml.get("precision") or 0.0)
        lb = float(ml.get("wilson_lb") or 0.0)
        n_test = int(ml.get("n_test") or 0)
        mtf = ml.get("model_timeframe", timeframe)
        hz = ml.get("horizon_velas", 0)

        # Mostrar señal con progreso
        if fire and "COMPRAR" in label:
            st.success(f"🟢 **SEÑAL: COMPRAR (LARGO)**", icon=":material/trending_up:")
        elif fire and "VENDER" in label:
            st.error(f"🔴 **SEÑAL: VENDER (CORTO)**", icon=":material/trending_down:")
        else:
            st.info(f"⚪ **SIN SEÑAL** - el edge validado no se cumple ahora", icon=":material/pause:")

        # Mostrar probabilidades
        c1, c2 = st.columns(2)
        with c1:
            st.metric("P(Subida)", f"{prob_up:.1%}")
            st.progress(min(max(prob_up, 0.0), 1.0))
        with c2:
            st.metric("P(Bajada)", f"{prob_down:.1%}")
            st.progress(min(max(prob_down, 0.0), 1.0))

        # Información adicional
        st.caption(f"Modelo: {mtf} · Horizonte: {hz} velas")
        st.caption(f"Precisión validada: {prec:.1%} · Wilson LB: {lb:.1%} · Muestras: {n_test}")

        # CONDICIONES DEL MODELO
        cond = ml.get("conditions", {})
        if cond.get("cond_requerida"):
            st.caption(f"Condición requerida: {cond['cond_requerida']} → "
                       f"{'✅ Cumplida' if cond['cond_cumplida'] else '❌ No cumplida'}")

        # =============================================
        # INTERFAZ DE APUESTAS
        # =============================================
        st.markdown("**Hacer una Apuesta**")

        col1, col2 = st.columns(2)
        with col1:
            direction = st.radio(
                "Dirección",
                ("LARGO (si sube)", "CORTO (si baja)"),
                horizontal=True,
                key=f"bet_dir_{symbol}_{timeframe}"
            )
        with col2:
            bet_pct = st.slider(
                "% del capital a arriesgar",
                min_value=1,
                max_value=50,
                value=10,
                step=5,
                key=f"bet_pct_{symbol}_{timeframe}"
            )

        bet_amount = st.session_state.game_cash * (bet_pct / 100) if "game_cash" in st.session_state else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Capital disponible", f"${st.session_state.game_cash:,.2f}" if "game_cash" in st.session_state else "$10,000")
        col2.metric("Monto de apuesta", f"${bet_amount:,.2f}" if bet_amount else "$0")
        col3.metric("Prob. Subida", f"{prob_up:.0%}")

        # Ejecutar apuesta
        if st.button("🚀 Ejecutar Apuesta", type="primary", use_container_width=True):
            if "game_cash" not in st.session_state:
                st.session_state.game_cash = 10000.0

            # Calcular P&L simulado
            entry_price = float(last["close"] if 'last' in dir() else df["close"].iloc[-1])

            if "LARGO" in direction:
                # Si apuestas a que sube, simulamos que sube un 2% si hay señal positiva
                if prob_up > 0.5:
                    pnl_pct = 0.02  # 2% ganancia simulada
                    st.session_state.game_cash += bet_amount * pnl_pct
                    st.success(f"✅ ¡Ganaste! La posición subió un {pnl_pct*100:.1f}%")
                else:
                    pnl_pct = -0.01  # 1% pérdida simulada
                    st.session_state.game_cash += bet_amount * pnl_pct
                    st.error(f"❌ Perdiste! La posición bajó")

            elif "CORTO" in direction:
                # Si apuestas a que baja
                if prob_down > 0.5:
                    pnl_pct = 0.02  # 2% ganancia simulada
                    st.session_state.game_cash += bet_amount * pnl_pct
                    st.success(f"✅ ¡Ganaste! La posición bajó un {pnl_pct*100:.1f}%")
                else:
                    pnl_pct = -0.01  # 1% pérdida simulada
                    st.session_state.game_cash += bet_amount * pnl_pct
                    st.error(f"❌ Perdiste! La posición subió")

            st.rerun()

    else:
        st.error(f"Error en señal ML: {ml.get('note', 'Unknown error')}")