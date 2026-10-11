"""Visualización de feature importance del modelo ML."""
from __future__ import annotations

import joblib
import plotly.graph_objects as go
import streamlit as st


def render_feature_importance(symbol: str, timeframe: str) -> None:
    """Renderiza las features más importantes del modelo ML."""
    st.subheader("Feature Importance", icon=":material/insights:")

    # Buscar modelo
    safe = symbol.replace("/", "_").replace(":", "_")
    patterns = [
        f"models/signal_{safe}_{timeframe}_h*.joblib",
        f"models/signal_{safe}_*.joblib",
        "models/cal_*.joblib",
        "models/rf_*.joblib",
    ]

    model_path = None
    for pattern in patterns:
        hits = sorted(__import__("glob").glob(pattern))
        if hits:
            model_path = hits[-1]
            break

    if not model_path:
        st.info("No hay modelo entrenado para mostrar feature importance.")
        return

    try:
        bundle = joblib.load(model_path)
        model = bundle.get("model")
        cols = bundle.get("cols", [])

        # Obtener importancias
        if hasattr(model, "feature_importances_"):
            # RandomForest directo
            importances = model.feature_importances_
        elif hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            # CalibratedClassifierCV
            importances = model.calibrated_classifiers_[0].feature_importances_
        else:
            st.info("El modelo no tiene feature_importances_.")
            return

        if not cols or len(cols) != len(importances):
            # Si no hay nombres de columnas, usar índices
            cols = [f"Feature {i}" for i in range(len(importances))]

        # Crear DataFrame y ordenar
        import pandas as pd
        df = pd.DataFrame({"feature": cols, "importance": importances})
        df = df.sort_values("importance", ascending=True)

        # Gráfico de barras horizontal
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation='h',
            marker_color="#60A5FA",
        ))
        fig.update_layout(
            template="plotly_dark",
            title="Features más importantes para la predicción",
            xaxis_title="Importancia relativa",
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
            height=max(300, len(cols) * 25),
            margin=dict(l=150),
        )
        st.plotly_chart(fig, width="stretch")

        # Mostrar nota
        st.caption(
            "Las features con mayor peso son las que más influyen en la predicción del modelo. "
            "Esto ayuda a entender qué patrones de mercado está aprendiendo."
        )

    except Exception as e:  # noqa: BLE001
        st.error(f"Error al cargar feature importance: {e}")
