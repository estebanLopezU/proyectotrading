"""Dashboard Fase 1-4 - Trading Prototype en vivo.
Framework: Streamlit + Plotly. Datos: ccxt Binance.
Correr: streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.trading_game import render_trading_game
from app.components.header import render_header
from app.components.indicators import render_indicators
from app.components.ml_panel import render_ml_panel
from app.components.chart import render_chart
from app.components.backtest import render_backtest
from app.components.feature_importance import render_feature_importance
from app.components.paper_trading import render_paper_trading
from app.components.multi_symbol import render_multi_symbol
from config.settings import SETTINGS
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

    # =============================================
    # Header
    # =============================================
    chg_color = "#34D399" if float(tick.get("pct_24h") or 0.0) >= 0 else "#F87171"
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
                <span style="margin-left:6px">Fuente: {source} · {latency_ms:.0f} ms · {len(df)} velas</span>
            </div>
        </div>
        <div class="price-box">
            <div class="price" style="color:{chg_color}">{fmt_money(price)}</div>
            <div style="color:{chg_color};font-weight:700">{float(tick.get("pct_24h") or 0.0):+.2f}% · 24 h</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =============================================
    # SIMULADOR DE TRADING - POSICIÓN PRIMERO
    # =============================================
    render_trading_game(symbol, timeframe)

    # =============================================
    # KPIs con sparklines
    # =============================================
    spark = df["close"].tail(48).tolist()
    vol_series = df["volume"].tail(48).tolist()
    with st.container(horizontal=True):
        st.metric("Precio", fmt_money(price), f"{float(tick.get('pct_24h') or 0.0):+.2%}",
                  border=True, chart_data=spark, chart_type="line")
        st.metric("Máximo 24 h", fmt_money(float(tick.get("high_24h") or df["high"].tail(24).max())), border=True,
                  chart_data=df["high"].tail(24).tolist())
        st.metric("Mínimo 24 h", fmt_money(float(tick.get("low_24h") or df["low"].tail(24).min())), border=True,
                  chart_data=df["low"].tail(24).tolist())
        st.metric("Volatilidad ATR",
                  f"{float(last.get('atr_pct', 0)):.2f}%" if pd.notna(last.get("atr_pct")) else "—",
                  f"Vol. {df['volume'].iloc[-1]:,.0f}", border=True,
                  chart_data=vol_series, chart_type="bar")

    # =============================================
    # Indicadores técnicos
    # =============================================
    render_indicators(df)

    # =============================================
    # Señales
    # =============================================
    st.subheader("Señales técnicas y ML", icon=":material/insights:")
    sig_col1, sig_col2 = st.columns(2)

    with sig_col1:
        with st.container(border=True):
            st.markdown("**Señal por regla**")
            sig = latest_signal(df)
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
            render_ml_panel(ml_signal(df, symbol, timeframe, source="local" if ml_source != "API" else "api",
                                      api_base=api_base), symbol, timeframe, df)

    # =============================================
    # Gráfico de velas + volumen
    # =============================================
    render_chart(df, symbol)

    # =============================================
    # Backtest interactivo
    # =============================================
    render_backtest(symbol, timeframe)

    # =============================================
    # Feature Importance
    # =============================================
    with st.expander("Feature Importance del modelo"):
        render_feature_importance(symbol, timeframe)

    # =============================================
    # Paper trading
    # =============================================
    render_paper_trading(float(last["close"]))

    st.caption(
        "Prototipo educativo Fases 1-4 · ML con validación honesta (≥80% precisión) · "
        "Paper trading $10k simulado · Alertas Telegram · No ejecuta órdenes reales"
    )

except Exception as e:  # noqa: BLE001
    st.error(f"Error en el dashboard: {e}")
    st.exception(e)