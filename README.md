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

## Mejoras implementadas

### Correcciones (P0)
- **Bug fix**: `latest_features` en `scorer.py` ahora calcula indicadores antes de usarlos
- **Unificación**: `features.py` es ahora un wrapper de `features_v2.py` (sin duplicación)

### Tests (P1)
- **Tests para `scorer.py`**: 6 tests cubriendo Wilson LB, score_signal y load_bundle
- **CI/CD**: GitHub Actions con tests, lint y type checking

### Modularización (P1)
- **Dashboard modular**: Componentes separados en `app/components/`
  - `header.py` - Header con precio y estado
  - `indicators.py` - Panel de indicadores técnicos
  - `chart.py` - Gráfico de velas + volumen
  - `ml_panel.py` - Señales ML validadas
  - `paper_trading.py` - Paper trading simulado
  - `multi_symbol.py` - Comparación multi-símbolo
  - `backtest.py` - Backtest interactivo
  - `feature_importance.py` - Feature importance visual

### Optimización (P2)
- **`st.fragment`**: Dashboard dividido en fragmentos para evitar reruns innecesarios
- **`st.cache_resource`**: Modelo ML cacheado para no recargar en cada rerun
- **Tema en config.toml**: Configuración centralizada del tema oscuro

### Features nuevas (P3)
- **Multi-símbolo**: Compara BTC/ETH/SOL y otros en el mismo dashboard
- **Backtest interactivo**: Configura capital, lookback y ejecuta backtest con equity curve
- **Feature Importance**: Visualiza qué features pesan más en la predicción del modelo

### Configuración (P1)
- **`.env.example`**: Archivo de configuración completo con todas las variables de entorno

## Estructura actualizada
```
config/settings.py
src/ingest.py indicators.py features.py features_v2.py scorer.py paper.py alerts.py utils.py
app/
  dashboard.py
  components/
    header.py indicators.py chart.py ml_panel.py
    paper_trading.py multi_symbol.py backtest.py feature_importance.py
    ml_model_cache.py
api/main.py
scripts/demo_offline.py train_model.py predict_live.py backtest.py calibrate.py
tests/ models/ data/
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

## Aviso
Educativo. El backtest no garantiza futuro. Nunca operes con dinero real
basandote solo en este prototipo.
