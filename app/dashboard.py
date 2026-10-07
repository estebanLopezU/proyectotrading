"""Dashboard Fase 1 - Trading Prototype en vivo.
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
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import SETTINGS  # noqa: E402
from src.indicators import add_indicators, latest_signal  # noqa: E402
from src.ingest import fetch_ticker_24h, get_ohlcv  # noqa: E402
from src.utils import fmt_money, fmt_pct, setup_logger  # noqa: E402
from src.alerts import send_telegram  # noqa: E402


def _safe_float(x, default: float = 0.0) -> float:
    """Convert to float, returning default if None or NaN."""
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return default
    return fx if not math.isnan(fx) else default

setup_logger()
st.set_page_config(page_title="Trading Prototype - Vivo", layout="wide", page_icon="📈")

st.sidebar.title("Control Trader")
symbol = st.sidebar.text_input("Simbolo (ccxt)", value=SETTINGS.symbol)
timeframe = st.sidebar.selectbox("Temporalidad", list(SETTINGS.timeframes), index=0)
refresh = st.sidebar.select_slider("Auto-refresh (s)", options=[5, 15, 30, 60], value=15)

# Opcion 2: integrar API + Dashboard
ml_source = st.sidebar.radio(
    "Fuente de prediccion ML",
    ["Local", "API (/predict)"],
    help="Local usa el modelo directo. API consume FastAPI en /predict.",
)
api_base = ""
if ml_source == "API (/predict)":
    api_base = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")

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
pct = _safe_float(tick.get("pct_24h"), 0)
c1.metric("Precio actual", fmt_money(price), fmt_pct(pct))
c2.metric("Alto 24h", fmt_money(float(tick.get("high_24h") or df["high"].tail(24).max())))
c3.metric("Bajo 24h", fmt_money(float(tick.get("low_24h") or df["low"].tail(24).min())))
c4.metric("RSI 14", f"{_safe_float(last.get('rsi14'), 0):.1f}")
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
    if ml_source == "API (/predict)":
        # Opcion 2: consumir FastAPI /predict
        import json
        import urllib.request
        url = f"{api_base.rstrip('/')}/predict?symbol={symbol}&timeframe={timeframe}"
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        ml = data.get("ml", {})
        st.caption(f"Fuente: API {url} | RSI={data.get('rsi14')} | precio={data.get('price')}")
    else:
        from src.ml_panel import ml_signal
        ml = ml_signal(df, symbol, timeframe)

    if ml.get("ok"):
        prec = ml.get("precision", 0) or 0

        # ⚠️ Probabilidad direccional: ¿sube o baja el precio?
        pu = _safe_float(ml.get("prob_up"), 0.5)
        pd_ = _safe_float(ml.get("prob_down"), 0.5)
        hz = ml.get("horizon_velas")
        hz_txt = f" (próximas {hz} velas de {timeframe})" if hz else ""
        st.markdown(f"**Predicción de precio{hz_txt}**")
        pc1, pc2 = st.columns(2)
        pc1.metric("📈 SUBE", f"{pu:.1%}", help="Probabilidad de que el precio suba")
        pc2.metric("📉 BAJA", f"{pd_:.1%}", help="Probabilidad de que el precio baje")
        st.progress(pu)
        dir_color = "green" if pu >= 0.5 else "red"
        st.markdown(
            f"<p style='text-align:center;font-weight:700;color:{dir_color};'>"
            f"{'📈' if pu >= 0.5 else '📉'} Dirección más probable: "
            f"{'SUBE' if pu >= 0.5 else 'BAJA'}</p>",
            unsafe_allow_html=True,
        )

        extra = ""
        if ml.get("note"):
            extra += f" | {ml['note']}"
        if ml.get("calibrated"):
            extra += f" | umbral [{ml.get('thr_lo',0):.2f},{ml.get('thr_hi',0):.2f}]"
        header = f"{ml['label']} p={ml['proba']:.2f}"
        if prec > 0:
            header += f" precision_historica={prec:.0%}"
        if "COMPRAR" in ml["label"]:
            st.success(f"{header}{extra}")
        elif "VENDER" in ml["label"]:
            st.error(f"{header}{extra}")
        else:
            st.warning(f"{header}{extra}")
        # Estadistica de probabilidad (analitica)
        st.caption(
            f"Modelo: {ml.get('path','')} | ACC base={ml.get('acc',0):.3f} | "
            f"Umbral de disparo: >=80% precision validada en test"
        )
        min_proba = float(os.getenv("ALERT_MIN_PROBA", "0.70"))
        if ("COMPRAR" in ml["label"] or "VENDER" in ml["label"]) and prec >= 0.80:
            r = send_telegram(f"{symbol} {timeframe} {ml['label']} p={ml['proba']:.2f} prec={prec:.0%} precio={float(last['close']):.2f}")
            if r["ok"]:
                st.toast("Alerta Telegram enviada")
            elif r.get("why", "").startswith("sin credenciales"):
                pass  # sin .env configurado: silencioso
    else:
        st.info(f"ML no disponible: {ml.get('label','Modelo no entrenado')}. Entrena: python scripts/train_v2.py")
except Exception as e:
    st.info(f"ML no disponible ({ml_source}): {e}")

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
