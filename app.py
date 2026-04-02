import streamlit as st
import pandas as pd
import plotly.express as px
from eda import VisualizationService as FrameworkMineria 
from generator import DataManager
from models import ALL_MODELS, MODEL_PARAMS
from predictor import Predictor

st.set_page_config(page_title="Framework Minería Avanzada", layout="wide", page_icon="📊")

if "data_manager" not in st.session_state: st.session_state.data_manager = DataManager()
if "selected_models" not in st.session_state: st.session_state.selected_models = []

dm = st.session_state.data_manager
df = dm.get_data()

def render_eda(df):
    if df is None:
        archivo = st.file_uploader("Subir Dataset", type=["csv"])
        if archivo: 
            dm.load_data(archivo)
            st.rerun()
        return

    miner = FrameworkMineria(df)
    t1, t2, t3 = st.tabs(["📊 Visualización", "🧪 Minería Avanzada", "📂 Explorador"])
    
    with t1:
        var = st.selectbox("Variable", miner.numeric_columns())
        st.plotly_chart(miner.histogram(var), use_container_width=True)
        st.plotly_chart(miner.correlation_heatmap(), use_container_width=True)

    with t2:
        st.subheader("Detección de Anomalías")
        if st.button("Ejecutar Isolation Forest"):
            miner.deteccionAnomalias()
            st.success("Anomalías procesadas.")
        
        st.divider()
        st.subheader("⏳ Meta 2: Dinámica Temporal")
        col_f = st.selectbox("Columna Fecha", df.columns)
        col_v = st.selectbox("Variable Bajas", df.columns)
        if st.button("Analizar Estacionalidad"):
            ts, mae = miner.pronostico_y_error(col_f, col_v)
            st.metric("MAE (Error de Pronóstico)", f"{mae:.4f}")
            fig_dec = miner.obtener_descomposicion_plotly(col_f, col_v)
            if fig_dec: st.plotly_chart(fig_dec, use_container_width=True)

    with t3: st.dataframe(df)

def render_model_selection():
    tipo = st.radio("Tipo", ["Clasificación", "Regresión"], horizontal=True)
    st.session_state.selected_models = st.multiselect("Modelos", ALL_MODELS.get(tipo, []))

def render_parameters():
    for m in st.session_state.selected_models:
        with st.expander(f"⚙️ {m}"):
            p_cfg = MODEL_PARAMS.get(m, {}).get("basic", {})
            for p, val in p_cfg.items():
                k = f"{m}_{p}"
                if val["type"] == "choice": st.selectbox(p, val["options"], key=k)
                else: st.slider(p, float(val["min"]), float(val["max"]), float(val["default"]), key=k)

def render_prediction():
    if not st.session_state.selected_models:
        st.warning("Seleccione modelos primero.")
        return
    
    if st.button("🚀 Ejecutar Entrenamiento Final (Meta 1)", type="primary"):
        pred_engine = Predictor(df, st.session_state.selected_models, st.session_state)
        res = pred_engine.train_all()
        
        # --- RESTAURACIÓN DE META 1: ROBUSTEZ PREDICTIVA ---
        df_res = pd.DataFrame(res).T.reset_index()
        df_res.columns = ['Modelo', 'Accuracy', 'AUC_ROC', 'CV_Stability']
        df_plot = df_res.copy()
        df_plot['AUC_ROC'] = df_plot['AUC_ROC'].astype(float)
        df_plot['CV_Stability'] = df_plot['CV_Stability'].astype(float)

        st.subheader("📊 Meta 1: Robustez Predictiva")
        st.table(df_res)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(df_plot, x='Modelo', y='AUC_ROC', title="Capacidad de Prevención (AUC)", color='AUC_ROC'), use_container_width=True)
        with c2:
            st.plotly_chart(px.line(df_plot, x='Modelo', y='CV_Stability', title="Estabilidad (Validación Cruzada)", markers=True), use_container_width=True)

# SIDEBAR
with st.sidebar:
    menu = st.radio("Navegación", ["📊 EDA & Minería", "🤖 Selección de Modelos", "⚙️ Perillaje", "📈 Resultados"])

if menu == "📊 EDA & Minería": render_eda(df)
elif menu == "🤖 Selección de Modelos": render_model_selection()
elif menu == "⚙️ Perillaje": render_parameters()
elif menu == "📈 Resultados": render_prediction()