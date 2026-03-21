import streamlit as st
import pandas as pd
# Importamos los servicios del Framework
from eda import VisualizationService as FrameworkMineria 
from generator import DataManager, CleaningService
from models import ALL_MODELS, MODEL_PARAMS
from predictor import Predictor

# ─── CONFIGURACIÓN DE PÁGINA (Tema 4: GUI) ───────────────────────────────────
st.set_page_config(
    page_title="Framework Minería Avanzada - ULead", 
    layout="wide", 
    page_icon="📊"
)

# Inyección de CSS para mejorar la legibilidad del menú lateral
st.markdown("""
    <style>
        /* Tamaño de las opciones del radio button en el sidebar */
        [data-testid="stSidebarNav"] span, .st-emotion-cache-1647z7l {
            font-size: 20px !important;
            font-weight: 600 !important;
            margin-bottom: 8px !important;
        }
        /* Título de la sección de navegación */
        [data-testid="stWidgetLabel"] p {
            font-size: 22px !important;
            color: #FF4B4B !important;
            font-weight: bold !important;
        }
        /* Estilo para los tabs */
        .stTabs [data-baseweb="tab"] p {
            font-size: 18px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ─── ESTADO DE LA SESIÓN ─────────────────────────────────────────────────────
if "data_manager" not in st.session_state:
    st.session_state.data_manager = DataManager()
if "show_new_dataset_modal" not in st.session_state:
    st.session_state.show_new_dataset_modal = False
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []

dm = st.session_state.data_manager
df = dm.get_data()

# ─── FUNCIONES DE NAVEGACIÓN Y CARGA ─────────────────────────────────────────
def load_new_dataset(file):
    """Procesa la carga de un nuevo archivo limpiando estados previos"""
    if file:
        # 1. Actualizar el DataManager
        st.session_state.data_manager.load_data(file)
        
        # 2. Resetear selecciones para evitar conflictos de dimensiones
        st.session_state.selected_models = []
        st.session_state.show_new_dataset_modal = False
        
        # 3. Limpiar parámetros viejos del session_state
        keys_to_del = [k for k in st.session_state.keys() if "Classifier_" in k or "Regressor_" in k]
        for k in keys_to_del:
            del st.session_state[k]
            
        st.toast("✅ Dataset cargado con éxito", icon="🚀")
        st.rerun()

# ─── RENDERIZADO DE PANTALLAS ────────────────────────────────────────────────
def render_eda(df):
    if df is None:
        st.header("📂 Carga de Datos")
        st.info("Bienvenido. Por favor, sube un archivo CSV para activar las funciones del Framework.")
        archivo = st.file_uploader("Subir Dataset", type=["csv"], key="init_uploader")
        if archivo: load_new_dataset(archivo)
        return

    miner = FrameworkMineria(df)
    cleaner = CleaningService(df)
    
    st.subheader("🩺 Diagnóstico de Calidad")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", df.shape[0])
    c2.metric("Variables", df.shape[1])
    c3.metric("Duplicados", miner.duplicated_count())
    c4.metric("% Nulos", f"{miner.null_percentage().mean():.2f}%")
    
    t1, t2, t3 = st.tabs(["📊 Visualización", "🧪 Minería Avanzada", "📂 Explorador"])
    
    with t1:
        cols = miner.numeric_columns()
        if cols:
            var = st.selectbox("Variable para análisis", cols)
            ca, cb = st.columns(2)
            ca.plotly_chart(miner.histogram(var), use_container_width=True)
            cb.plotly_chart(miner.boxplot(var), use_container_width=True)
            st.plotly_chart(miner.correlation_heatmap(), use_container_width=True)

    with t2:
        st.subheader("Detección de Anomalías (Tema 7)")
        cont = st.slider("Porcentaje de Contaminación", 0.01, 0.20, 0.05)
        if st.button("Ejecutar Isolation Forest"):
            miner.deteccionAnomalias(contaminacion=cont)
            st.success("Anomalías marcadas en el dataset.")
            st.dataframe(df.head(10))

    with t3:
        st.dataframe(df, use_container_width=True)

def render_model_selection():
    if df is None:
        st.warning("⚠️ Cargue datos primero")
        return
    
    st.subheader("🎯 Definición del Objetivo")
    tipo = st.radio("¿Qué tipo de problema desea resolver?", 
                    ["Clasificación", "Regresión", "Agrupamiento (Clustering)"], 
                    horizontal=True)
    
    modelos = ALL_MODELS.get(tipo, [])
    st.session_state.selected_models = st.multiselect(
        f"Seleccione modelos de {tipo}:",
        modelos,
        default=[m for m in st.session_state.selected_models if m in modelos]
    )

def render_parameters():
    if df is None or not st.session_state.selected_models:
        st.warning("⚠️ Seleccione modelos en la pestaña anterior")
        return
    
    for m in st.session_state.selected_models:
        with st.expander(f"⚙️ Configurar {m}", expanded=True):
            p_cfg = MODEL_PARAMS.get(m, {}).get("basic", {})
            for p, val in p_cfg.items():
                k = f"{m}_{p}"
                if val["type"] == "int":
                    st.slider(p, val["min"], val["max"], val["default"], key=k)
                elif val["type"] == "float":
                    st.slider(p, float(val["min"]), float(val["max"]), float(val["default"]), key=k)

def render_prediction():
    if df is None or not st.session_state.selected_models:
        st.error("❌ Configuración incompleta.")
        return
    
    if st.button("🚀 Ejecutar Entrenamiento Final", type="primary", use_container_width=True):
        with st.spinner("Aplicando perillaje y entrenando modelos..."):
            pred = Predictor(df, st.session_state.selected_models, st.session_state)
            res = pred.train_all()
            st.success("✅ ¡Proceso finalizado!")
            st.table(pd.DataFrame(res).T)

# ─── SIDEBAR Y RUTEO ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Lead University")
    st.caption("BCD 7213 - Minería Avanzada")
    st.divider()
    
    menu = st.radio("Navegación", 
                    ["📊 EDA & Minería", "🤖 Selección de Modelos", "⚙️ Perillaje", "📈 Resultados"])
    
    st.divider()
    if st.button("🆕 Cargar Nuevo Dataset", use_container_width=True):
        st.session_state.show_new_dataset_modal = True

# Lógica de pantallas
if menu == "📊 EDA & Minería":
    render_eda(df)
elif menu == "🤖 Selección de Modelos":
    render_model_selection()
elif menu == "⚙️ Perillaje":
    render_parameters()
elif menu == "📈 Resultados":
    render_prediction()

# Modal de carga (se activa por el botón del sidebar)
if st.session_state.show_new_dataset_modal:
    st.markdown("---")
    with st.container():
        st.subheader("📂 Cambiar Dataset")
        nuevo = st.file_uploader("Seleccione nuevo archivo CSV", type=["csv"], key="modal_up")
        col1, col2 = st.columns(2)
        if col1.button("Confirmar Carga", type="primary"):
            load_new_dataset(nuevo)
        if col2.button("Cancelar"):
            st.session_state.show_new_dataset_modal = False
            st.rerun()