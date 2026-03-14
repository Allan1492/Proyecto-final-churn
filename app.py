import streamlit as st
import pandas as pd
from eda import EDAService, VisualizationService
from generator import DataManager, CleaningService
from models import ALL_MODELS, MODEL_PARAMS
from predictor import Predictor


st.set_page_config(page_title="Predicción de Deserción de Clientes (Churn) mediante Analítica y Machine Learning", layout="wide", page_icon="📊")

# ─── ESTADO INICIAL ──────────────────────────────────────────────────────────
if "data_manager" not in st.session_state:
    st.session_state.data_manager = DataManager()
if "show_new_dataset_modal" not in st.session_state:
    st.session_state.show_new_dataset_modal = False

dm = st.session_state.data_manager
df = dm.get_data()

# ─── FUNCIONES AUXILIARES ────────────────────────────────────────────────────
def load_new_dataset(file):
    """Carga un nuevo dataset y resetea el estado"""
    if file:
        st.session_state.data_manager = DataManager()
        st.session_state.data_manager.load_data(file)
        st.session_state.show_new_dataset_modal = False
        st.session_state.selected_models = []
        st.success("✅ Dataset cargado exitosamente!")
        st.rerun()
    else:
        st.error("❌ Selecciona un archivo CSV")

def render_dataset_info(df):
    """Muestra información compacta del dataset"""
    if df is not None:
        st.caption(f"📄 {df.shape[0]} filas × {df.shape[1]} columnas")

def render_eda(df):
    """Renderiza la sección de EDA"""
    if df is None:
        st.info("👆 Carga un dataset desde el sidebar para comenzar")
        with st.expander("📂 O cargar aquí"):
            if uploaded := st.file_uploader("Sube un CSV", type=["csv"], key="eda_uploader"):
                dm.load_data(uploaded)
                st.rerun()
        return

    eda, cleaner, viz = EDAService(df), CleaningService(df), VisualizationService(df)
    
    # Diagnóstico
    st.subheader("🩺 Diagnóstico")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", df.shape[0]); c2.metric("Columnas", df.shape[1]); c3.metric("Duplicados", eda.duplicated_count())
    
    # Limpieza interactiva
    with st.expander("🧹 Limpieza", expanded=False):
        if eda.duplicated_count() > 0 and st.button("Eliminar duplicados"):
            dm.update_data(cleaner.remove_duplicates()); st.rerun()
        if cols := st.multiselect("🗑️ Columnas a eliminar", df.columns):
            dm.update_data(cleaner.drop_columns(cols)); st.rerun()
        if num_cols := eda.numeric_columns():
            col, method = st.columns(2)
            with col: c = st.selectbox("Imputar", num_cols); m = st.selectbox("Método", ["mean","median","mode"])
            if st.button("Aplicar imputación"): dm.update_data(cleaner.fill_nulls(c, m)); st.rerun()
    
    # Control de versiones
    with st.expander("🔄 Historial"):
        c1, c2, c3 = st.columns(3)
        if c1.button("⬅ Undo"): dm.undo(); st.rerun()
        if c2.button("➡ Redo"): dm.redo(); st.rerun()
        if c3.button("🔄 Reset"): dm.reset_to_original() and st.success("Restablecido"); st.rerun()
    
    # Visualizaciones
    with st.expander("📈 Visualizaciones", expanded=True):
        if eda.numeric_columns():
            nc = st.selectbox("Variable numérica", eda.numeric_columns(), key="num_viz")
            st.plotly_chart(viz.histogram(nc), use_container_width=True)
            st.plotly_chart(viz.boxplot(nc), use_container_width=True)
        if eda.categorical_columns():
            cc = st.selectbox("Variable categórica", eda.categorical_columns(), key="cat_viz")
            st.plotly_chart(viz.bar_chart(cc), use_container_width=True)
        st.plotly_chart(viz.correlation_heatmap(), use_container_width=True)
    
    # Dataset y descarga
    with st.expander("💾 Dataset actual", expanded=False):
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        c1, c2 = st.columns([1, 3])
        c1.download_button("📥 CSV", csv, "dataset_actualizado.csv", "text/csv", use_container_width=True)
        c2.info(f"Memoria: {df.memory_usage(deep=True).sum()/1024**2:.2f} MB")

def render_model_selection():
    """Renderiza selección de modelos"""
    if df is None:
        st.warning("⚠️ Carga un dataset primero"); return
    all_models = [m for v in ALL_MODELS.values() for m in v]
    st.session_state.selected_models = st.multiselect("🤖 Selecciona modelos", all_models, help="Puedes elegir varios para comparar")

def render_parameters():
    """Renderiza parámetros de modelos seleccionados"""
    if df is None or not st.session_state.get("selected_models"):
        st.warning("⚠️ Selecciona modelos primero"); return
    for model in st.session_state.selected_models:
        with st.expander(f"⚙️ {model}", expanded=False):
            for pname, cfg in MODEL_PARAMS.get(model, {}).get("basic", {}).items():
                key = f"{model}_{pname}"
                if cfg["type"] == "int":
                    st.slider(pname, cfg["min"], cfg["max"], cfg["default"], key=key)
                elif cfg["type"] == "float":
                    st.slider(pname, cfg["min"], cfg["max"], float(cfg["default"]), key=key)
                elif cfg["type"] == "categorical":
                    st.selectbox(pname, cfg["options"], index=cfg["options"].index(cfg["default"]), key=key)

def render_prediction():
    """Renderiza entrenamiento y resultados"""
    if df is None or not st.session_state.get("selected_models"):
        st.warning("⚠️ Configura dataset y modelos primero"); return
    if st.button("🚀 Entrenar modelos", type="primary", use_container_width=True):
        with st.spinner("🔄 Procesando..."):
            predictor = Predictor(df, st.session_state.selected_models, st.session_state)
            results = predictor.train_all()
            st.success("✅ Entrenamiento completado")
            with st.expander("📊 Resultados detallados", expanded=True):
                st.json(results)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(" Bienvenidos al portal")
    st.markdown("Predicción de Deserción de Clientes (Churn) Machine Learning")
    st.divider()
    
    if st.button("🆕 Nuevo Dataset", use_container_width=True):
        st.session_state.show_new_dataset_modal = True
    
    st.divider()
    menu = st.radio("🧭 Navegación", ["📊 EDA", "🤖 Modelos", "⚙️ Parámetros", "📈 Predicción"], label_visibility="collapsed")
    st.divider()
    render_dataset_info(df)

# ─── MODAL CARGA DATASET ─────────────────────────────────────────────────────
if st.session_state.show_new_dataset_modal:
    st.markdown("<div style='position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:999'></div>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("<div style='background:white;padding:20px;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.15)'>", unsafe_allow_html=True)
            st.subheader("📂 Cargar Dataset")
            st.warning("⚠️ Esto reiniciará el análisis actual")
            new_file = st.file_uploader("Selecciona CSV", type=["csv"], key="modal_uploader")
            c1, c2 = st.columns(2)
            if c1.button("✅ Aceptar", use_container_width=True): load_new_dataset(new_file)
            if c2.button("❌ Cancelar", use_container_width=True): 
                st.session_state.show_new_dataset_modal = False; st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# ─── CONTENIDO PRINCIPAL ─────────────────────────────────────────────────────
st.title(f"{menu}")
match menu:
    case "📊 EDA": render_eda(df)
    case "🤖 Modelos": render_model_selection()
    case "⚙️ Parámetros": render_parameters()
    case "📈 Predicción": render_prediction()