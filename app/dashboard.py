"""Dashboard Fase 1-4 - Trading Prototype en vivo.
Framework: Streamlit + Plotly. Datos: ccxt Binance.
Correr: streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import SETTINGS
from src.alerts import send_telegram
from src.indicators import add_indicators, latest_signal
from src.ingest import fetch_ticker_24h, get_ohlcv
from src.ml_panel import has_trained_model, ml_signal
from src.utils import fmt_money, setup_logger

setup_logger()
st.set_page_config(
    page_title="Trading Dashboard",
    layout="wide",
    page_icon=":material/candlestick_chart:",
)

# =============================================
# CSS - tema oscuro financiero (único bloque)
# =============================================
st.markdown("""
<style>
    /* ---- Header glassmorphism ---- */
    .app-header {
        position: relative;
        background: linear-gradient(120deg, rgba(30,41,59,.72) 0%, rgba(15,23,42,.88) 60%, rgba(30,58,95,.55) 100%);
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 18px;
        padding: 24px 30px;
        margin-bottom: 22px;
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 14px;
        box-shadow: 0 20px 50px rgba(2,6,23,.45);
        overflow: hidden;
    }
    .app-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #60A5FA, #34D399 45%, transparent 90%);
        opacity: .85;
    }
    .app-header .title { font-size: 1.55rem; font-weight: 800; color: #F8FAFC; letter-spacing: -.03em; }
    .app-header .title .sep { color: #60A5FA; }
    .app-header .meta { color: #94A3B8; font-size: .85rem; margin-top: 4px; }
    .app-header .price-box { text-align: right; }
    .app-header .price { font-size: 1.75rem; font-weight: 800; letter-spacing: -.03em;
                         font-variant-numeric: tabular-nums; }
    /* ---- Status pill ---- */
    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 3px 12px; border-radius: 999px; font-size: .72rem; font-weight: 700;
        border: 1px solid transparent;
    }
    .status-live { background: rgba(52,211,153,.12); color: #34D399; border-color: rgba(52,211,153,.35); }
    .status-stale { background: rgba(251,191,36,.12); color: #FBBF24; border-color: rgba(251,191,36,.35); }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
    .status-live .status-dot { background: #34D399; animation: blink 2s infinite; }
    .status-stale .status-dot { background: #FBBF24; }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: .35; } }
    /* ---- Signal banners ---- */
    .signal-banner {
        padding: 14px; border-radius: 14px; text-align: center;
        font-size: 1.15rem; font-weight: 800; letter-spacing: .02em;
        transition: transform .15s ease;
    }
    .banner-buy { background: linear-gradient(135deg, rgba(52,211,153,.16), rgba(16,185,129,.28));
                  color: #A7F3D0; border: 1px solid rgba(52,211,153,.4);
                  box-shadow: 0 8px 28px rgba(52,211,153,.12); }
    .banner-sell { background: linear-gradient(135deg, rgba(248,113,113,.16), rgba(239,68,68,.28));
                   color: #FECDD3; border: 1px solid rgba(248,113,113,.4);
                   box-shadow: 0 8px 28px rgba(248,113,113,.12); }
    .banner-none { background: rgba(30,41,59,.65); color: #94A3B8; border: 1px solid #334155; }
    /* ---- Tarjetas (metrics nativos con border=True) ---- */
    [data-testid="stMetric"] {
        border-radius: 14px !important;
        transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(96,165,250,.45) !important;
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(2,6,23,.4);
    }
    /* ---- Contenedor del gráfico ---- */
    [data-testid="stPlotlyChart"] {
        border: 1px solid #334155; border-radius: 16px;
        background: rgba(15,23,42,.6); padding: 8px;
        transition: border-color .2s ease;
    }
    [data-testid="stPlotlyChart"]:hover { border-color: rgba(96,165,250,.4); }
    /* ---- Chips del log de trades ---- */
    .trade-log {
        padding: 8px 12px; background: rgba(30,41,59,.6); border: 1px solid #334155;
        border-radius: 10px; margin: 4px 0; font-size: .82rem; color: #CBD5E1;
        font-variant-numeric: tabular-nums;
    }
    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0B1120; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
</style>
""", unsafe_allow_html=True)

# =============================================
# Sidebar
# =============================================
with st.sidebar:
    st.subheader(":material/tune: Controles")

    symbol = st.text_input("Símbolo (ccxt)", value=SETTINGS.symbol) or SETTINGS.symbol
    timeframe = st.selectbox("Temporalidad", list(SETTINGS.timeframes), index=3) or "1h"
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh = st.select_slider(
        "Intervalo (s)", options=[5, 15, 30, 60], value=15, disabled=not auto_refresh
    )

    ml_source = st.segmented_control(
        "Fuente de predicción ML", ["Local", "API"], default="Local"
    )
    api_base = ""
    if ml_source == "API":
        api_base = st.text_input("URL API", value="http://127.0.0.1:8000")

    if st.button(":material/refresh: Actualizar ahora", width="stretch"):
        st.cache_data.clear()
        st.toast("Caché limpiada")

    st.subheader(":material/monitor_heart: Estado del sistema")
    model_ok = has_trained_model(symbol)
    if model_ok:
        st.markdown(
            ":green-badge[Modelo ML entrenado] :green-badge[Precisión mín. 80%] "
            ":blue-badge[Binance LIVE]"
        )
    else:
        st.markdown(
            ":red-badge[Modelo ML no entrenado] :green-badge[Precisión mín. 80%] "
            ":blue-badge[Binance LIVE]"
        )
        st.caption("Ejecuta `python scripts/train_signal.py` para entrenar un modelo.")

    st.caption("Prototipo educativo · no ejecuta órdenes reales")

if auto_refresh:
    st_autorefresh(interval=refresh * 1000, key="dashboard_autorefresh")

# =============================================
# Carga de datos
# =============================================
@st.cache_data(ttl=10, show_spinner=False)
def load_data(symbol_: str, timeframe_: str):
    try:
        res = get_ohlcv(symbol_, timeframe_)
        df = add_indicators(res.df)
        tick = fetch_ticker_24h(symbol_)
        return df, res.stale, res.source, res.latency_ms, tick
    except Exception:  # noqa: BLE001
        return None, True, "ERROR", 0, None

# =============================================
# Contenido principal
# =============================================
try:
    with st.spinner("Conectando a Binance..."):
        df, stale, source, latency_ms, tick = load_data(symbol, timeframe)
        tick = tick or {}

    if df is None or df.empty:
        st.warning("Sin velas disponibles. Verifica la conexión y el símbolo.")
        st.stop()

    last = df.iloc[-1]
    price = float(tick.get("last") or last["close"])
    sig = latest_signal(df)
    ml_pred = ml_signal(df, symbol, timeframe, source="local" if ml_source != "API" else "api",
                        api_base=api_base)

    change_24h = float(tick.get("pct_24h") or 0.0)
    high_24h = float(tick.get("high_24h") or df["high"].tail(24).max())
    low_24h = float(tick.get("low_24h") or df["low"].tail(24).min())
    atr_pct = last.get("atr_pct", float("nan"))
    estado = "EN VIVO" if not stale else "DATOS EN CACHÉ"

    # =============================================
    # Header
    # =============================================
    chg_color = "#34D399" if change_24h >= 0 else "#F87171"
    st.markdown(f"""
    <div class="app-header">
        <div>
            <div class="title">Mercado <span class="sep">/</span> {symbol}</div>
            <div class="meta">Terminal de análisis · {timeframe} · Señales técnicas, modelo ML y simulación</div>
            <div class="meta" style="margin-top:8px">
                <span class="status-pill {'status-live' if not stale else 'status-stale'}">
                    <span class="status-dot"></span>{estado}
                </span>
                <span style="margin-left:6px">Fuente: {source} · {latency_ms:.0f} ms · {len(df)} velas</span>
            </div>
        </div>
        <div class="price-box">
            <div class="price" style="color:{chg_color}">{fmt_money(price)}</div>
            <div style="color:{chg_color};font-weight:700">{change_24h:+.2f}% · 24 h</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =============================================
    # KPIs con sparklines
    # =============================================
    spark = df["close"].tail(48).tolist()
    vol_series = df["volume"].tail(48).tolist()
    with st.container(horizontal=True):
        st.metric("Precio", fmt_money(price), f"{change_24h:+.2%}",
                  border=True, chart_data=spark, chart_type="line")
        st.metric("Máximo 24 h", fmt_money(high_24h), border=True,
                  chart_data=df["high"].tail(24).tolist())
        st.metric("Mínimo 24 h", fmt_money(low_24h), border=True,
                  chart_data=df["low"].tail(24).tolist())
        st.metric("Volatilidad ATR",
                  f"{float(atr_pct):.2f}%" if pd.notna(atr_pct) else "—",
                  f"Vol. {df['volume'].iloc[-1]:,.0f}", border=True,
                  chart_data=vol_series, chart_type="bar")

    # =============================================
    # Indicadores técnicos
    # =============================================
    st.subheader("Indicadores técnicos", icon=":material/query_stats:")
    st.caption("Lectura de la última vela")

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

    # =============================================
    # Señales
    # =============================================
    st.subheader("Señales técnicas y ML", icon=":material/insights:")
    sig_col1, sig_col2 = st.columns(2)

    with sig_col1:
        with st.container(border=True):
            st.markdown("**Señal por regla**")
            if "COMPRAR" in sig["label"]:
                st.success(f"{sig['label']} · score {sig['score']:+d}", icon=":material/trending_up:")
            elif "VENDER" in sig["label"]:
                st.error(f"{sig['label']} · score {sig['score']:+d}", icon=":material/trending_down:")
            else:
                st.info(f"{sig['label']} · score {sig['score']:+d}", icon=":material/pause:")
            for reason in sig["reasons"]:
                st.markdown(f"- {reason}")

    with sig_col2:
        with st.container(border=True):
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

    # =============================================
    # Gráfico de velas + volumen
    # =============================================
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

    # =============================================
    # Paper trading
    # =============================================
    st.subheader("Paper trading simulado ($10,000)", icon=":material/account_balance_wallet:")

    if "paper_cash" not in st.session_state:
        st.session_state.paper_cash = 10000.0
        st.session_state.paper_qty = 0.0
        st.session_state.paper_log = []

    px = float(last["close"])
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

    st.caption(
        "Prototipo educativo Fases 1-4 · ML con validación honesta (≥80% precisión) · "
        "Paper trading $10k simulado · Alertas Telegram · No ejecuta órdenes reales"
    )

except Exception as e:  # noqa: BLE001
    st.error(f"Error en el dashboard: {e}")
    st.exception(e)
