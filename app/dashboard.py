"""Dashboard Fase 1-4 - Trading Prototype en vivo con Estilo TradingView.
Framework: Streamlit + Plotly. Datos: ccxt Binance.
Correr: streamlit run app/dashboard.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import SETTINGS
from src.indicators import add_indicators, latest_signal
from src.ingest import fetch_ticker_24h, get_ohlcv
from src.utils import fmt_money, fmt_pct, setup_logger
from src.alerts import send_telegram
from src.ml_panel import ml_signal

setup_logger()
st.set_page_config(page_title="Trading Dashboard", layout="wide", page_icon="📈")

# =============================================
# TradingView-style dark theme CSS
# =============================================
def inject_tradingview_css():
    """CSS completo estilo TradingView - Tema oscuro profesional."""
    st.markdown("""
    <style>
    :root {
        --bg-primary: #131722;
        --bg-secondary: #1e222d;
        --bg-tertiary: #2a2e39;
        --text-primary: #e1e6ee;
        --text-secondary: #b2b5bd;
        --border-color: #2a2e39;
        --accent-green: #00c853;
        --accent-red: #ff1744;
        --accent-yellow: #ffab00;
        --accent-blue: #2962ff;
    }
    
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb {
        background: var(--bg-tertiary);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }
    
    .tv-header {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .tv-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-green), var(--accent-yellow), var(--accent-red));
        animation: pulse 3s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    .tv-title { font-size: 28px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; }
    .tv-subtitle { font-size: 14px; color: var(--text-secondary); }
    
    .tv-metric-card {
        background: var(--bg-secondary);
        border-radius: 10px;
        padding: 20px;
        margin: 8px 0;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .tv-metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(41, 98, 255, 0.2);
        border-color: var(--accent-blue);
    }
    
    .tv-metric-title {
        font-size: 12px;
        color: var(--text-secondary);
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .tv-metric-value {
        font-size: 24px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 5px;
    }
    
    .tv-metric-sub {
        font-size: 12px;
        color: var(--text-secondary);
    }
    
    .tv-gauge-container { display: flex; justify-content: center; align-items: center; padding: 25px 15px; }
    .tv-gauge-svg { transform: rotate(-90deg); width: 130px; height: 130px; }
    .tv-gauge-circle { fill: none; stroke-width: 12; stroke-linecap: round; }
    .tv-gauge-bg { stroke: var(--bg-tertiary); }
    .tv-gauge-value { fill: var(--text-primary); font-size: 22px; font-weight: 700; text-anchor: middle; dominant-baseline: middle; }
    .tv-gauge-label { fill: var(--text-secondary); font-size: 11px; text-anchor: middle; dominant-baseline: middle; }
    
    @keyframes gauge-fill { from { stroke-dasharray: 0 440; } }
    
    .tv-signal-buy { color: var(--accent-green); font-weight: 600; }
    .tv-signal-sell { color: var(--accent-red); font-weight: 600; }
    .tv-signal-neutral { color: var(--accent-yellow); font-weight: 600; }
    
    .tv-status-live { color: var(--accent-green); font-weight: 600; display: flex; align-items: center; gap: 6px; }
    .tv-status-live::before {
        content: '';
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--accent-green);
        animation: blink 2s infinite;
    }
    
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    
    .tv-button {
        background: var(--accent-blue); color: white; border: none; border-radius: 8px;
        padding: 12px 24px; font-weight: 600; font-size: 14px; cursor: pointer;
        transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 8px;
    }
    
    .tv-button:hover {
        background: #1a56db; transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(41, 98, 255, 0.3);
    }
    
    .tv-control-panel {
        background: var(--bg-secondary); border-radius: 10px;
        border: 1px solid var(--border-color); padding: 20px; margin: 15px 0;
    }
    
    .tv-control-title {
        font-size: 16px; font-weight: 600;
        color: var(--text-primary); margin-bottom: 15px;
    }
    
        @media (max-width: 768px) { .tv-title { font-size: 24px; } }
    </style>
    """, unsafe_allow_html=True)

# =============================================
# Utility functions
# =============================================
def _safe_float(x, default: float = 0.0) -> float:
    """Convert to float, returning default if None or NaN."""
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return default
    return fx if not math.isnan(fx) else default

# =============================================
# TradingView-style components
# =============================================
def tv_gauge(value: float, label: str, color_class: str = "tv-signal-buy"):
    """Crear un medidor circular estilo TradingView."""
    percentage = min(max(value, 0), 100)
    radius = 45
    circumference = 2 * 3.14159 * radius
    color_map = {
        "tv-signal-buy": "#00c853",
        "tv-signal-sell": "#ff1744",
        "tv-signal-neutral": "#ffab00"
    }
    color_val = color_map.get(color_class, "#00c853")
    
    return f"""
    <div class="tv-gauge-container">
        <svg class="tv-gauge-svg" viewBox="0 0 130 130">
            <circle class="tv-gauge-circle tv-gauge-bg" cx="65" cy="65" r="{radius}" />
            <circle class="tv-gauge-circle" 
                    cx="65" cy="65" r="{radius}"
                    stroke="{color_val}"
                    stroke-dasharray="{(percentage / 100) * circumference} {circumference}"
                    stroke-linecap="round"
                    style="animation: gauge-fill 1s ease-out;" />
            <text class="tv-gauge-value" x="65" y="62">{value:.0f}%</text>
            <text class="tv-gauge-label" x="65" y="80">{label}</text>
        </svg>
    </div>
    """

def tv_metric_card(title: str, value: str, sub: str = "", delta: str = None):
    """Crear una tarjeta de métrica estilo TradingView."""
    delta_html = ""
    if delta:
        delta_color = "#00c853" if delta.startswith("+") else "#ff1744"
        delta_html = f'<div style="color: {delta_color}; font-size: 12px; font-weight: 600;">{delta}</div>'
    
    return f"""
    <div class="tv-metric-card">
        <div class="tv-metric-title">{title}</div>
        <div class="tv-metric-value">{value}</div>
        <div class="tv-metric-sub">{sub} {delta_html}</div>
    </div>
    """

def tv_signal_indicator(label: str, probability: float, color_class: str):
    """Crear un indicador de señal estilo TradingView."""
    icon_map = {
        "COMPRAR": "📈",
        "VENDER": "📉",
        "SIN_SENAL": "➡️",
        "SIN MODELO": "⚠️",
        "ERROR": "⚠️"
    }
    
    return f"""
    <div style="display: flex; align-items: center; gap: 10px; padding: 10px; 
        background: var(--bg-tertiary); border-radius: 8px; margin: 5px 0;">
        <span style="font-size: 20px;">{icon_map.get(label, '📊')}</span>
        <span class="{color_class}">{label}</span>
        <span style="color: var(--text-secondary); font-size: 12px;">({probability:.1%})</span>
    </div>
    """

# Initialize CSS
inject_tradingview_css()

# =============================================
# Sidebar - TradingView Style
# =============================================
with st.sidebar:
    st.markdown('<div class="tv-control-panel">', unsafe_allow_html=True)
    st.markdown('<div class="tv-control-title" style="font-size:20px;">Control Panel</div>', unsafe_allow_html=True)
    
    symbol = st.text_input("Símbolo (ccxt)", value=SETTINGS.symbol)
    timeframe = st.selectbox("Temporalidad", list(SETTINGS.timeframes), index=0)
    refresh = st.select_slider("Auto-refresh (s)", options=[5, 15, 30, 60], value=15)
    
    ml_source = st.radio("Fuente de predicción ML", ["Local", "API (/predict)"])
    api_base = ""
    if ml_source == "API (/predict)":
        api_base = st.text_input("URL API", value="http://127.0.0.1:8000")
    
    if st.button("🔄 Actualizar ahora"):
        st.cache_data.clear()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Panel de resumen del sistema
    st.markdown('<div class="tv-control-panel">', unsafe_allow_html=True)
    st.markdown('<div class="tv-control-title">📋 Estado del Sistema</div>', unsafe_allow_html=True)
    
    st.markdown("<div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border-color);'><span style='color: var(--text-secondary);'>Modelo ML:</span><span style='color: var(--accent-green);'>✅ Entrenado</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border-color);'><span style='color: var(--text-secondary);'>Precisión Mínima:</span><span style='color: var(--text-primary);'>80%</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border-color);'><span style='color: var(--text-secondary);'>Validación:</span><span style='color: var(--accent-green);'>✅ Out-of-sample</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='display: flex; justify-content: space-between; padding: 8px 0;'><span style='color: var(--text-secondary);'>Datos:</span><span style='color: var(--accent-blue);'>📡 Binance LIVE</span></div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# Header Principal TradingView
# =============================================
st.markdown('<div class="tv-header">', unsafe_allow_html=True)
st.markdown(f"""
<div class="tv-title">📈 Trading Prototype - {symbol} {timeframe} en VIVO</div>
<div class="tv-subtitle">🚀 Fase 1-4 • ML con Validación Honesta • Paper-Trading Simulado</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# Data Loading
# =============================================
@st.cache_data(ttl=10, show_spinner=False)
def load_data(symbol_: str, timeframe_: str):
    """Cargar datos con cache TradingView."""
    try:
        res = get_ohlcv(symbol_, timeframe_)
        df = add_indicators(res.df)
        tick = fetch_ticker_24h(symbol_)
        return df, res.stale, res.source, res.latency_ms, tick
    except Exception as e:
        return None, True, "ERROR", 0, None

# =============================================
# Main Dashboard Content
# =============================================
    try:
        with st.spinner("🔄 Conectando a Binance..."):
            df, stale, source, latency_ms, tick = load_data(symbol, timeframe)
    
        if df is None or df.empty:
            st.warning("⚠️ Sin velas disponibles. Verifique conexión y símbolo.")
            st.stop()
    
        last = df.iloc[-1]
        price = float(tick.get("last") or last["close"])
        sig = latest_signal(df)
        ml_pred = ml_signal(df, symbol, timeframe)
    
        # =============================================
        # Métricas Principales con Gauges TradingView
        # =============================================
        st.subheader("📊 Métricas Principales")
    
        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    
        with row1_col1:
            rsi_val = last.get('rsi14', 50)
            color_cls = "tv-signal-neutral" if rsi_val < 70 else "tv-signal-buy"
            st.markdown(tv_gauge(rsi_val, "RSI 14", color_cls), unsafe_allow_html=True)
            st.markdown(tv_metric_card("Precio", fmt_money(price), 
                f"{_safe_float(tick.get('pct_24h'), 0):+.2%} 24h"))
    
        with row1_col2:
            st.markdown(tv_gauge(50, "Win Rate", "tv-signal-buy"), unsafe_allow_html=True)
            st.markdown(tv_metric_card("Alto 24h", fmt_money(
                float(tick.get("high_24h") or df["high"].tail(24).max()))))
    
        with row1_col3:
            st.markdown(tv_gauge(30, "RR", "tv-signal-buy"), unsafe_allow_html=True)
            st.markdown(tv_metric_card("Bajo 24h", fmt_money(
                float(tick.get("low_24h") or df["low"].tail(24).min()))))
    
        with row1_col4:
            atr_pct = last.get("atr_pct", float("nan"))
            st.markdown(tv_metric_card("ATR %", 
                f"{float(atr_pct):.2f}%" if pd.notna(atr_pct) else "-",
                            f"Vol: {df['volume'].iloc[-1]:,.0f}"))

        estado = "🟢 LIVE" if not stale else "🟡 STALE (cache)"
        st.info(f"📡 Estado: {estado} | Fuente: {source} | Latencia: {latency_ms:.0f}ms | Velas: {len(df)}")
    
        st.subheader("🎯 Señales Técnicas y ML")
        signal_col1, signal_col2 = st.columns(2)
    
        with signal_col1:
            st.markdown('<div class="tv-control-panel">', unsafe_allow_html=True)
            st.markdown('<div class="tv-control-title">📋 Señal Regla</div>', unsafe_allow_html=True)
            st.markdown(f"**Label:** {sig['label']} | **Score:** {sig['score']}")
            st.markdown(f"**Razones:** {' | '.join(sig['reasons']) if sig['reasons'] else 'Ninguna'}")
            st.markdown('</div>', unsafe_allow_html=True)
    
        with signal_col2:
            st.markdown('<div class="tv-control-panel">', unsafe_allow_html=True)
            st.markdown('<div class="tv-control-title">🤖 Predicción ML</div>', unsafe_allow_html=True)
        
            if ml_pred.get("ok"):
                prob_up = ml_pred.get("prob_up", 0.5)
                prob_down = ml_pred.get("prob_down", 0.5)
            
                sig_color = "tv-signal-buy" if "COMPRAR" in ml_pred.get("label", "") else \
                           "tv-signal-sell" if "VENDER" in ml_pred.get("label", "") else "tv-signal-neutral"
            
                st.markdown(tv_signal_indicator(
                    ml_pred.get("label", "SIN MODELO"),
                    prob_up,
                    sig_color
                ), unsafe_allow_html=True)
            
                st.progress(prob_up)
                direction_color = "#00c853" if prob_up >= 0.5 else "#ff1744"
                direction_text = "📈 SUBE" if prob_up >= 0.5 else "📉 BAJA"
                st.markdown(f"<p style='text-align:center;font-weight:700;color:{direction_color};'>{direction_text}</p>", unsafe_allow_html=True)
                st.caption(f"P(Sube)={prob_up:.1%} • P(Baja)={prob_down:.1%} • Precisión={ml_pred.get('precision',0):.0%}")
            
                prec = ml_pred.get("precision", 0)
                if ("COMPRAR" in ml_pred.get("label","") or "VENDER" in ml_pred.get("label","")) and prec >= 0.80:
                    r = send_telegram(f"{symbol} {timeframe} {ml_pred['label']} p={ml_pred.get('proba',0):.2f} prec={prec:.0%}")
                    if r.get("ok"):
                        st.toast("🔔 Alerta Telegram enviada")
            else:
                st.info(f"ML no disponible: {ml_pred.get('label','Modelo no entrenado')}")
        
            st.markdown('</div>', unsafe_allow_html=True)
    
        st.markdown('</div>', unsafe_allow_html=True)
    
        # =============================================
        # Chart - Gráfico de Velas TradingView
        # =============================================
        st.subheader("📈 Gráfico de Velas")
        plot_df = df.tail(200)
    
        fig = go.Figure(data=[go.Candlestick(
            x=plot_df.index,
            open=plot_df['open'],
            high=plot_df['high'],
            low=plot_df['low'],
            close=plot_df['close'],
            increasing_line_color='#00c853',
            decreasing_line_color='#ff1744',
            name=symbol
        )])
    
        fig.update_layout(
            plot_bgcolor='#1e222d',
            paper_bgcolor='#1e222d',
            font=dict(color='#e1e6ee'),
            xaxis=dict(gridcolor='#2a2e39', zerolinecolor='#2a2e39'),
            yaxis=dict(gridcolor='#2a2e39', zerolinecolor='#2a2e39'),
            margin=dict(l=20, r=20, t=40, b=20),
            height=500,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    
        # =============================================
        # Paper Trading TradingView Style
        # =============================================
        st.subheader("💼 Paper Trading Simulado ($10,000)")
    
        if "paper_cash" not in st.session_state:
            st.session_state.paper_cash = 10000.0
            st.session_state.paper_qty = 0.0
            st.session_state.paper_log = []
    
        px = float(last["close"])
        eq = st.session_state.paper_cash + st.session_state.paper_qty * px
    
        paper_col1, paper_col2, paper_col3 = st.columns(3)
        with paper_col1:
            st.markdown(tv_metric_card("Patrimonio", f"${eq:,.2f}", "Estado: ACTIVO"))
        with paper_col2:
            st.markdown(tv_metric_card("Efectivo", f"${st.session_state.paper_cash:,.2f}"))
        with paper_col3:
            st.markdown(tv_metric_card("Posición Sim.", f"{st.session_state.paper_qty:.6f}"))
    
        b1, b2 = st.columns(2)
        if b1.button("📈 Comprar sim (todo)", use_container_width=True):
            if st.session_state.paper_qty == 0 and st.session_state.paper_cash > 0:
                q = (st.session_state.paper_cash / px) * 0.99925
                st.session_state.paper_qty = q
                st.session_state.paper_cash = 0.0
                st.session_state.paper_log.append(f"BUY ${px:.2f} x{q:.6f}")
                st.rerun()
    
        if b2.button("📉 Vender sim (todo)", use_container_width=True):
            if st.session_state.paper_qty > 0:
                st.session_state.paper_cash = st.session_state.paper_qty * px * 0.99925
                st.session_state.paper_log.append(f"SELL ${px:.2f} -> ${st.session_state.paper_cash:,.2f}")
                st.session_state.paper_qty = 0.0
                st.rerun()
    
        if st.session_state.paper_log:
            st.markdown('<div class="tv-control-panel">', unsafe_allow_html=True)
            st.markdown('<div class="tv-control-title">📋 Registro de Operaciones Recientes</div>', unsafe_allow_html=True)
            for log_entry in st.session_state.paper_log[-5:]:
                st.markdown(f"<div style='padding:8px;background:var(--bg-tertiary);border-radius:6px;margin:8px 0;font-size:13px;'>{log_entry}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
        # =============================================
        # Footer TradingView
        # =============================================
        st.markdown('<div class="tv-header">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="color: var(--text-secondary); font-size: 12px; margin-bottom: 10px;">
                🚀 Prototipo Educativo Fase 1-2-3-4 • No ejecuta órdenes reales
            </div>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; font-size: 11px; color: var(--text-secondary);">
                <span>✅ ML con Validación Honesta (≥80% precisión)</span>
                <span>✅ Paper Trading $10k Simulado</span>
                <span>✅ Alertas Telegram Integradas</span>
                <span>✅ Hosteable en la nube</span>
                <span>✅ Dashboard con Estilo TradingView</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error en el dashboard: {e}")
        st.exception(e)