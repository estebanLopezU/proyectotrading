"""Dashboard Fase 1 - Trading Prototype en vivo.
Framework: Streamlit + Plotly. Datos: ccxt Binance.
Correr: streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import SETTINGS  # noqa: E402
from src.indicators import add_indicators, latest_signal  # noqa: E402
from src.ingest import fetch_ticker_24h, get_ohlcv  # noqa: E402
from src.utils import fmt_money, fmt_pct, setup_logger  # noqa: E402

setup_logger()
st.set_page_config(page_title="Trading Prototype - Vivo", layout="wide", page_icon="📈")

st.sidebar.title("Control Trader")
symbol = st.sidebar.text_input("Simbolo (ccxt)", value=SETTINGS.symbol)
timeframe = st.sidebar.selectbox("Temporalidad", list(SETTINGS.timeframes), index=0)
refresh = st.sidebar.select_slider("Auto-refresh (s)", options=[5, 15, 30, 60], value=15)
if st.sidebar.button("Recargar ahora"):
    st.cache_data.clear()

st_autorefresh(interval=refresh * 1000, key="live_refresh")
st.title(f"Trading Prototype - {symbol} {timeframe} EN VIVO")
st.caption("Fase 1 MVP: visualizacion + estadisticas. Educativo.")


@st.cache_data(ttl=10, show_spinner=False)
def load(symbol_: str, timeframe_: str):
    res = get_ohlcv(symbol_, timeframe_)
    df = add_indicators(res.df)
    tick = fetch_ticker_24h(symbol_)
    return df, res.stale, res.source, res.latency_ms, tick


try:
    with st.spinner("Conectando a Binance..."):
        df, stale, source, latency_ms, tick = load(symbol, timeframe)
except Exception as e:
    st.error(f"Sin datos ni cache. Revisa internet o simbolo: {e}")
    st.stop()

if df.empty:
    st.warning("Sin velas.")
    st.stop()

last = df.iloc[-1]
price = float(tick.get("last") or last["close"])
sig = latest_signal(df)

c1, c2, c3, c4, c5 = st.columns(5)
pct = float(tick.get("pct_24h", 0) or 0)
c1.metric("Precio actual", fmt_money(price), fmt_pct(pct))
c2.metric("Alto 24h", fmt_money(float(tick.get("high_24h") or df["high"].tail(24).max())))
c3.metric("Bajo 24h", fmt_money(float(tick.get("low_24h") or df["low"].tail(24).min())))
c4.metric("RSI 14", f"{float(last.get('rsi14', 0) or 0):.1f}")
atrp = last.get("atr_pct", float("nan"))
c5.metric("ATR %", f"{float(atrp):.2f}%" if pd.notna(atrp) else "-")

estado = "STALE (cache)" if stale else "LIVE"
st.info(f"Estado: {estado} - fuente={source} - lat={latency_ms:.0f}ms - velas={len(df)}")
st.write(f"Senal regla: **{sig['label']}** score={sig['score']} - " + " | ".join(sig["reasons"]))

plot_df = df.tail(200).copy()
fig = go.Figure()
fig.add_trace(go.Candlestick(x=plot_df["timestamp"], open=plot_df["open"], high=plot_df["high"], low=plot_df["low"], close=plot_df["close"], name="Velas"))
for col in ["sma20", "sma50", "bb_high", "bb_mid", "bb_low"]:
    if col in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df[col], mode="lines", name=col))
fig.update_layout(title=f"{symbol} {timeframe} Velas+SMA+Bollinger", xaxis_rangeslider_visible=False, height=520, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

g1, g2 = st.columns(2)
with g1:
    fv = go.Figure()
    fv.add_trace(go.Bar(x=plot_df["timestamp"], y=plot_df["volume"], name="Vol"))
    fv.update_layout(title="Volumen", height=300, template="plotly_dark")
    st.plotly_chart(fv, use_container_width=True)
with g2:
    fr = go.Figure()
    fr.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["rsi14"], mode="lines", name="RSI14"))
    fr.add_hline(y=70, line_dash="dash")
    fr.add_hline(y=30, line_dash="dash")
    fr.update_layout(title="RSI 14", height=300, template="plotly_dark", yaxis_range=[0, 100])
    st.plotly_chart(fr, use_container_width=True)

fm = go.Figure()
fm.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["macd"], mode="lines", name="MACD"))
fm.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["macd_signal"], mode="lines", name="Signal"))
fm.add_trace(go.Bar(x=plot_df["timestamp"], y=plot_df["macd_hist"], name="Hist"))
fm.update_layout(title="MACD 12/26/9", height=300, template="plotly_dark")
st.plotly_chart(fm, use_container_width=True)

st.subheader("Ultimas 10 velas")
show = df.tail(10)[["timestamp", "open", "high", "low", "close", "volume", "sma20", "sma50", "rsi14", "macd", "atr14", "is_closed"]]
st.dataframe(show, use_container_width=True)

st.subheader("Prediccion ML Fase 2+3")
try:
    from src.ml_panel import ml_signal
    ml = ml_signal(df, symbol, timeframe)
    if ml["ok"]:
        extra = f" | {ml.get('note', '')}" if ml.get("note") else f" | {ml.get('path', '')}"
        if "COMPRAR" in ml["label"]:
            st.success(f"{ml['label']} p_subida={ml['proba']:.2f} acc_test={ml['acc']:.3f}{extra}")
        elif "VENDER" in ml["label"]:
            st.error(f"{ml['label']} p_subida={ml['proba']:.2f} acc_test={ml['acc']:.3f}{extra}")
        else:
            st.warning(f"{ml['label']} p_subida={ml['proba']:.2f} acc_test={ml['acc']:.3f}{extra}")
        from src.alerts import send_telegram
        if ml["proba"] >= 0.70 or ml["proba"] <= 0.30:
            r = send_telegram(f"{symbol} {timeframe} {ml['label']} p={ml['proba']:.2f} precio={float(last['close']):.2f}")
            if r["ok"]:
                st.toast("Alerta Telegram enviada")
    else:
        st.info(f"ML no disponible: {ml['label']}. Entrena: python scripts/calibrate.py")
except Exception as e:
    st.info(f"ML no disponible: {e}")

st.subheader("Paper-trading simulado ($10,000)")
pc = st.columns(3)
if "paper_cash" not in st.session_state:
    st.session_state.paper_cash = 10000.0
    st.session_state.paper_qty = 0.0
    st.session_state.paper_log = []
px = float(last["close"])
eq = st.session_state.paper_cash + st.session_state.paper_qty * px
pc[0].metric("Equity sim", f"${eq:,.2f}")
pc[1].metric("Cash", f"${st.session_state.paper_cash:,.2f}")
pc[2].metric("BTC sim", f"{st.session_state.paper_qty:.6f}")
b1, b2 = st.columns(2)
if b1.button("Comprar sim (todo)"):
    if st.session_state.paper_qty == 0 and st.session_state.paper_cash > 0:
        q = (st.session_state.paper_cash / px) * 0.99925
        st.session_state.paper_qty = q
        st.session_state.paper_cash = 0.0
        st.session_state.paper_log.append(f"BUY {px:.2f} x{q:.6f}")
        st.rerun()
if b2.button("Vender sim (todo)"):
    if st.session_state.paper_qty > 0:
        st.session_state.paper_cash = st.session_state.paper_qty * px * 0.99925
        st.session_state.paper_log.append(f"SELL {px:.2f} -> ${st.session_state.paper_cash:,.2f}")
        st.session_state.paper_qty = 0.0
        st.rerun()
if st.session_state.paper_log:
    st.write(st.session_state.paper_log[-5:])

st.caption("Prototipo educativo Fase 1+2+3. No ejecuta ordenes reales.")
