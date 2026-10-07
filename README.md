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
# API aparte:
uvicorn api.main:app --reload --port 8000
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

## Aviso
Educativo. El backtest no garantiza futuro. Nunca operes con dinero real
basandote solo en este prototipo.


