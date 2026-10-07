"""Trading Dashboard Pro - Fase 4.
Panel profesional con métricas avanzadas, equity curve, drawdown,
stats detalladas de trades, risk metrics y control de paper-trading.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import SETTINGS
from src.indicators import add_indicators, latest_signal
from src.ingest import fetch_ticker_24h, get_ohlcv
from src.paper import PaperAccount
from src.trade_stats import (
    period_stats,
    build_equity_curve,
    monthly_stats,
    drawdown_curve,
    risk_reward_summary,
    trades_to_df,
)
from src.ml_panel import ml_signal
from src.utils import setup_logger, utc_now


def _safe_float(x, default: float = 0.0) -> float:
    """Convert to float, returning default if None or NaN."""
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return default
    return fx if not math.isnan(fx) else default

setup_logger()

st.set_page_config(
    page_title="TradingPro - Dashboard",
    layout="wide",
    page_icon="✅",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════
# CSS - Estilo Profesional Dark
# ═══════════════════════════════════════
st.markdown("""
<style>
    :root {
        --bg: #0a0a0f;
        --bg-card: #111118;
        --bg-card-hover: #1a1a28;
        --border: #2a2a3a;
        --border-light: #3a3a4a;
        --text: #e0e0e8;
        --text-dim: #88889a;
        --text-bright: #ffffff;
        --green: #50fa7b;
        --green-bg: #2a4a2a;
        --red: #ff5555;
        --red-bg: #4a2a2a;
        --cyan: #8be9fd;
        --yellow: #f1fa8c;
        --orange: #ffb86c;
        --radius: 8px;
        --radius-sm: 4px;
        --font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: var(--font-family);
    }
    [data-testid="stSidebar"] {
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius) !important;
        padding: 12px 16px !important;
        transition: background 0.2s;
    }
    [data-testid="stMetric"]:hover { background: var(--bg-card-hover); }
    [data-testid="stMetricLabel"] {
        color: var(--text-dim) !important;
        font-size: 0.7rem !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-bright) !important;
        font-size: 1.3rem !important;
        font-weight: 700;
        font-family: var(--font-family);
    }
    [data-testid="stMetricDelta"] { font-size: 0.85rem !important; font-weight: 600; }
    .delta-positive { color: var(--green) !important; }
    .delta-negative { color: var(--red) !important; }
    .delta-neutral { color: var(--yellow) !important; }
    .main-header {
        background: linear-gradient(135deg, #111118 0%, #1a1a2a 50%, #111118 100%);
        border-bottom: 2px solid var(--border);
        padding: 16px 24px;
        margin-bottom: 20px;
        border-radius: var(--radius) !important;
    }
    .main-header h1 {
        color: var(--text-bright);
        font-weight: 800;
        margin: 0 0 4px 0;
        font-size: 1.6rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header h1 .icon { font-size: 1.8rem; }
    .main-header .subtitle { color: var(--text-dim); font-size: 0.85rem; }
    .section-title {
        color: var(--text-bright) !important;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--border);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-title .badge {
        font-size: 0.65rem;
        background: var(--border);
        color: var(--text-dim);
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 500;
    }
    .grid-2 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
        gap: 16px;
    }
    .grid-3 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 16px;
    }
    .grid-4 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 16px;
    }
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 700;
        font-family: var(--font-family);
        letter-spacing: 0.3px;
    }
    .pill-buy { background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }
    .pill-sell { background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }
    .pill-neutral { background: #2a2a3a; color: var(--text-dim); border: 1px solid var(--border); }
    .status-live { color: var(--green); font-weight: 700; }
    .status-stale { color: var(--orange); font-weight: 700; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-card); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
    .divider { border-top: 1px solid var(--border); margin: 20px 0; }
    .metric-row { display: flex; gap: 16px; flex-wrap: wrap; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Control")
    symbol = st.text_input("Símbolo", value=SETTINGS.symbol, label_visibility="collapsed")
    timeframe = st.selectbox("Temporalidad", list(SETTINGS.timeframes), index=0, label_visibility="collapsed")
    refresh_ms = st.slider("Auto-refresh (s)", 5, 60, 15, label_visibility="collapsed")
    ml_source = st.radio("ML", ["Local", "API"], horizontal=True, label_visibility="collapsed")
    api_url = st.text_input("API URL", "http://127.0.0.1:8000", label_visibility="collapsed")
    if st.button("🔄 Recargar", type="secondary", label_visibility="collapsed"):
        st.cache_data.clear()
    st.divider()
    st.markdown("### 📊 Paper Trading")
    paper = st.session_state.get("paper", PaperAccount(cash=10000.0))
    st.session_state.paper = paper
    px = float(st.session_state.get("last_price", 0))
    eq = paper.equity(px)
    st.metric("Equity", f"${eq:,.2f}", f"${(eq-10000):+,.2f}", label_visibility="collapsed")
    st.metric("Cash", f"${paper.cash:,.2f}", label_visibility="collapsed")
    st.metric("Qty", f"{paper.qty:.6f}", label_visibility="collapsed")
    col_buy, col_sell, col_reset = st.columns(3)
    with col_buy:
        if st.button("🟢 BUY", type="primary", label_visibility="collapsed"):
            r = paper.buy(px, ts=utc_now().isoformat(), strategy="rule", rr=2.0)
            st.session_state["last_trade"] = ("BUY", r)
            st.rerun()
    with col_sell:
        if st.button("🔴 SELL", type="secondary", label_visibility="collapsed"):
            if paper.qty > 0:
                r = paper.sell(px, ts=utc_now().isoformat(), strategy="rule", exit_reason="manual")
                st.session_state["last_trade"] = ("SELL", r)
                st.rerun()
            else:
                st.info("Sin posición", icon="ℹ️")
    with col_reset:
        if st.button("🗑️ Reset", type="secondary", label_visibility="collapsed"):
            st.session_state.paper = PaperAccount(cash=10000.0)
            st.rerun()
    st.divider()
    st.markdown("### ℹ️ Acerca de")
    st.caption("Prototipo educativo Fase 1-4")
    st.caption("Datos: Binance vía ccxt")
    st.caption("No ejecuta órdenes reales")

# ═══════════════════════════════════════
# Carga de datos
# ═══════════════════════════════════════
@st.cache_data(ttl=10, show_spinner=False)
def load_data(symbol_: str, timeframe_: str):
    res = get_ohlcv(symbol_, timeframe_)
    df = add_indicators(res.df)
    tick = fetch_ticker_24h(symbol_)
    return df, res, tick

try:
    with st.spinner("Conectando a Binance..."):
        df, res, tick = load_data(symbol, timeframe)
except Exception as e:
    st.error(f"Sin datos: {e}")
    st.stop()

if df.empty:
    st.warning("Sin velas")
    st.stop()

last = df.iloc[-1]
price = float(tick.get("last") or last["close"])
sig = latest_signal(df)
source = "LIVE" if not res.stale else "STALE"
latency = res.latency_ms

st.session_state.last_price = price

# ═══════════════════════════════════════
# Header
# ═══════════════════════════════════════
st.markdown(f"""
<div class="main-header">
    <h1>
        <span class="icon">📈</span>
        TradingPro Dashboard
        <span class="badge">Fase 4</span>
    </h1>
    <div style="display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
        <span style="color: var(--text-dim); font-size: 0.85rem;">
            {symbol} · {timeframe}
        </span>
        <span class="pill {'status-live' if source == 'LIVE' else 'status-stale'}">
            {source}
        </span>
        <span style="color: var(--text-dim); font-size: 0.85rem; font-family: var(--font-family);">
            {latency:.0f}ms
        </span>
    </div>
</div>
""", unsafe_allow_html=True)



# ═══════════════════════════════════════
# 1. KPIs - Rendimiento por periodo
# ═══════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Rendimiento por Periodo</div>', unsafe_allow_html=True)

trades_df = trades_to_df(paper.trades)

col_w, col_m, col_y, col_a = st.columns(4, gap="large")

for period_name, period_key, col in [
    ("This Week", "week", col_w),
    ("This Month", "month", col_m),
    ("This Year", "year", col_y),
    ("All Time", "all", col_a),
]:
    with col:
        s = period_stats(trades_df, period_key)
        rr = s["rr"]
        wr = s["win_rate"] * 100
        pnl = s["pnl"]
        color = "#50fa7b" if pnl >= 0 else "#ff5555"
        st.metric(
            label=period_name,
            value=f"{rr:.2f} RR",
            delta=f"{wr:.1f}% WR | ${abs(pnl):,.0f} PnL",
            delta_color="off",
        )
        st.markdown(f'<div style="color:{color}; font-size:0.75rem; margin-top:4px;">PnL: ${pnl:,.2f}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# 2. Estado del mercado + ML
# ═══════════════════════════════════════
st.markdown('<div class="section-title">📈 Mercado + Señal</div>', unsafe_allow_html=True)
ml = ml_signal(df, symbol, timeframe)

cm1, cm2, cm3, cm4 = st.columns(4, gap="large")
pct = _safe_float(tick.get("pct_24h"), 0)
c1 = "delta-positive" if pct >= 0 else "delta-negative"
cm1.metric("Precio", f"${price:,.2f} {symbol[:3]}", f"{pct:+.2f}%", delta_color=c1)
cm2.metric("RSI 14", f"{_safe_float(last.get('rsi14'), 0):.1f}", delta_color="delta-neutral")
cm2.metric("ATR 14", f"{_safe_float(last.get('atr_pct'), 0)*100:.2f}%", delta_color="delta-neutral")
cm3.metric("24h High", f"${_safe_float(tick.get('high_24h'), 0):,.2f}", delta_color="delta-neutral")
cm4.metric("24h Low", f"${_safe_float(tick.get('low_24h'), 0):,.2f}", delta_color="delta-neutral")

st.divider()
st.markdown('<div class="grid-2" style="grid-template-columns: 1fr 1fr; gap:16px;">')

# Columna izquierda: equity curve mini
st.markdown("### Balance de Cuenta")
eq = paper.equity(price)
st.metric("Equity", f"${eq:,.2f}", f"${eq - 10000:+,.2f}",
          help="Equity = Cash + Qty * Precio actual")
st.metric("Cash", f"${paper.cash:,.2f}")
st.metric("Posición", f"{paper.qty:.6f} {symbol[:3]}")

if paper.qty > 0:
    st.info(f"📍 LONG {paper.qty:.6f} {symbol[:3]} a ${paper.cash/paper.qty:,.2f} media", icon="📍")

# Columna derecha: señal ML
st.markdown("### Señal ML")
if ml.get("ok"):
    prec = ml.get("precision", 0) or 0

    # ⚠️ Probabilidad direccional: ¿sube o baja el precio?
    pu = _safe_float(ml.get("prob_up"), 0.5)
    pd_ = _safe_float(ml.get("prob_down"), 0.5)
    hz = ml.get("horizon_velas")
    hz_txt = f" · próximas {hz} velas" if hz else ""
    st.markdown(f"**Predicción{hz_txt}**")
    pc1, pc2 = st.columns(2)
    arrow_up = "📈" if pu >= 0.5 else "📉"
    pc1.metric("SUBE", f"{pu:.1%}", help="Probabilidad de que el precio suba")
    pc2.metric("BAJA", f"{pd_:.1%}", help="Probabilidad de que el precio baje")
    st.progress(pu)
    dir_color = "#50fa7b" if pu >= 0.5 else "#ff5555"
    st.markdown(
        f"<div style='text-align:center;font-weight:700;color:{dir_color};'>"
        f"{arrow_up} Dirección más probable: {'SUBE' if pu >= 0.5 else 'BAJA'}</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    extra = ""
    if ml.get("note"):
        extra += f" | {ml['note']}"
    header = f"{ml['label']} p={ml['proba']:.2f}"
    if prec > 0:
        header += f" hist={prec:.0%}"
    if "COMPRAR" in ml["label"]:
        st.success(f"{header}{extra}")
    elif "VENDER" in ml["label"]:
        st.error(f"{header}{extra}")
    else:
        st.warning(f"{header}{extra}")
    st.caption(f"Modelo: {ml.get('path', 'N/A')}")
else:
    st.info(f"ML: {ml.get('label', 'Modelo no entrenado')} | {ml.get('note', '')}")
    st.caption("Sin predicción direccional disponible (modelo no cargado o sin datos).")

# ═══════════════════════════════════════
# 3. Equity curve + drawdown
# ═══════════════════════════════════════
st.markdown('<div class="section-title">💰 Equity Curve & Drawdown</div>', unsafe_allow_html=True)

ec_df = build_equity_curve(paper)
dd_df = drawdown_curve(ec_df)

if ec_df.empty:
    st.info("No hay trades cerrados. Haz BUY y luego SELL para ver los gráficos.", icon="📝")
else:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.6, 0.4],
        subplot_titles=("Equity Curve", "Drawdown %"),
    )
    fig.add_trace(go.Scatter(
        x=ec_df["ts"], y=ec_df["equity"],
        mode="lines", line=dict(color="#50fa7b", width=2),
        name="Equity", fill="tozeroy", fillcolor="#50fa7b22",
        hovertemplate="<b>%{x:%Y-%m-%d %H:%M}</b><br>Equity: $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    if not dd_df.empty:
        fig.add_trace(go.Scatter(
            x=dd_df["ts"], y=dd_df["drawdown"],
            mode="lines", line=dict(color="#ff5555", width=1.5),
            name="Drawdown", fill="tozeroy", fillcolor="#ff555522",
            hovertemplate="<b>Drawdown: %{y:.2f}%</b><extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#2a2a3a", row=2, col=1)

    max_dd = -dd_df["drawdown"].max() if not dd_df.empty else 0.0
    st.caption(f"📉 Max Drawdown: {max_dd:.2f}% | Total trades cerrados: {len(trades_df)}")

    fig.update_layout(
        height=300,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, height=300)

# ═══════════════════════════════════════
# 4. Reward:Risk + Mensual
# ═══════════════════════════════════════
st.markdown('<div class="section-title">📊 Reward:Risk & Meses</div>', unsafe_allow_html=True)

rr_data = risk_reward_summary(paper.trades)
if rr_data["rr_list"]:
    rr_list = rr_data["rr_list"]
    avg_rr = rr_data["avg_rr"]
    colors = ["#50fa7b" if r >= 0 else "#ff5555" for r in rr_list]
    fig_rr = go.Figure()
    fig_rr.add_trace(go.Bar(
        x=list(range(1, len(rr_list) + 1)),
        y=rr_list,
        marker_color=colors,
        name="RR trades",
        hovertemplate="Trade %{x}: %{y:.2f}R<extra></extra>",
    ))
    fig_rr.add_hline(
        y=avg_rr, line_dash="dash", line_color="#8be9fd",
        annotation_text=f"Avg: {avg_rr:.2f}", annotation_position="top",
    )
    fig_rr.update_layout(
        height=180,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=25, b=0),
        xaxis_title="Trade Nº",
        yaxis_title="R",
        showlegend=False,
    )
    st.plotly_chart(fig_rr, use_container_width=True, height=180)

    st.markdown('<div style="display:flex; gap:16px;">')
    st.metric("Avg RR", f"{avg_rr:.2f}")
    st.metric("Median RR", f"{rr_data['median_rr']:.2f}")
    st.metric("Std RR", f"{rr_data['rr_std']:.2f}")
    st.metric("Trades", f"{len(rr_data['rr_list'])}")
    st.markdown("</div>")
else:
    st.info("Ejecuta trades para ver Reward:Risk", icon="📝")

st.markdown("### Meses")
mdf = monthly_stats(paper.trades)
if mdf.empty:
    st.info("No hay suficientes trades para estadísticas mensuales", icon="📝")
else:
    st.dataframe(
        mdf,
        use_container_width=True,
        hide_index=True,
        height=180,
        column_config={
            "PnL": st.column_config.NumberColumn("PnL", format="$%.2f"),
            "Avg Win": st.column_config.NumberColumn("Avg Win", format="$%.2f"),
            "Avg Loss": st.column_config.NumberColumn("Avg Loss", format="$%.2f"),
        },
    )

st.markdown("</div>")

