# Trading Prototype — Fases 1-4 completas

Prototipo educativo: ve BTC/USDT de **Binance en vivo**, calcula indicadores,
predice con ML, hace backtest y paper-trading. **No ejecuta ordenes reales.**

## Framework
- UI: **Streamlit + Plotly** (`app/dashboard.py`)
- API: **FastAPI** (`api/main.py`): `/health` `/price` `/predict`
- Datos: **ccxt Binance** + cache + respaldo Yahoo (`src/`)
- ML: scikit-learn RandomForest calibrado + filtro ATR

## Correr (Windows PowerShell)
```powershell
cd C:\proyectos\proyectotrading
pip install -r requirements.txt
pytest -q
streamlit run app/dashboard.py
# API aparte (opcional, para Fuente ML = API en el sidebar):
uvicorn api.main:app --reload --port 8000
# Entrenar modelo (si dice "Modelo no entrenado"):
python scripts/train_model.py --symbol BTC/USDT --timeframe 15m --limit 1500
# Probar alertas Telegram:
python scripts/test_alert.py
```

## Hostear en la nube (Fase 4)
**Opcion A - Streamlit Community Cloud (gratis, dashboard):**
1. Sube el repo a GitHub (ya hecho: estebanLopezU/proyectotrading)
2. Ve a https://share.streamlit.io -> "Create app" -> conecta el repo
3. En "Main file path" pon: `app/dashboard.py`
4. En "Secrets" (opcional) agrega TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
5. Deploy -> te da una URL publica

**Opcion B - Render (gratis, API + dashboard):**
1. Ve a https://render.com -> "New +" -> "Blueprint"
2. Conecta el repo (usa el `render.yaml` incluido)
3. Deploy automatico: API en :8000 y Dashboard en :8501

**Opcion C - Docker local:**
```powershell
docker build -t trading-prototype .
docker run -p 8501:8501 -p 8000:8000 trading-prototype
```

## Evidencias validadas
- Tests: 7/7 passed (indicadores + fase2 + api health)
- Binance LIVE verificado con ccxt (ticker + OHLCV + predict fuente=live)
- Modelo calibrado 15m ACC 0.71 con filtro ATR (ver scripts/calibrate.py)
- Backtest 1h honesto: empate tecnico con MDD controlado

## Estructura
```
config/settings.py
src/ingest.py indicators.py features.py ml_panel.py paper.py alerts.py utils.py fallback_yf.py
app/dashboard.py
api/main.py
scripts/demo_offline.py train_model.py predict_live.py backtest.py calibrate.py
tests/ models/ data/
```

## Opciones implementadas
- **1. Modelo ML mejorado**: +5 features nuevas (macd_abs, bb_width, close_vs_sma50, momentum_5, vol_5), hiperparametros CLI (--n_estimators --max_depth --min_samples_leaf), class_weight balanced. ACC 0.24 -> 0.61 con 1500 velas.
- **2. API + Dashboard**: sidebar "Fuente de prediccion ML" -> Local o API (/predict) con FastAPI corriendo en :8000.
- **3. Alertas Telegram**: .env (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID + ALERT_MIN_PROBA), script `scripts/test_alert.py`.
- **4. Hostear**: `.streamlit/config.toml`, `render.yaml`, `Dockerfile` (instrucciones arriba).

## Aviso
Educativo. El backtest no garantiza futuro. Nunca operes con dinero real
basandote solo en este prototipo.


