import streamlit as st
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN DE MODELOS Y PARÁMETROS ---
MODEL_PARAMS = {
    "Random Forest Classifier": {
        "basic": {
            "n_estimators": {"type": "int", "min": 10, "max": 500, "default": 100},
            "max_depth": {"type": "int", "min": 2, "max": 15, "default": 6},
            "min_samples_split": {"type": "int", "min": 2, "max": 10, "default": 2},
            "criterion": {"type": "choice", "options": ["gini", "entropy"], "default": "gini"}
        }
    },
    "Logistic Regression": {
        "basic": {
            "C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0},
            "max_iter": {"type": "int", "min": 100, "max": 500, "default": 100}
        }
    },
    "XGBoost Classifier": {
        "basic": {
            "learning_rate": {"type": "float", "min": 0.01, "max": 0.3, "default": 0.1},
            "n_estimators": {"type": "int", "min": 50, "max": 300, "default": 100},
            "max_depth": {"type": "int", "min": 2, "max": 10, "default": 4}
        }
    },
    "SVM": {
        "basic": {
            "C": {"type": "float", "min": 0.1, "max": 5.0, "default": 1.0},
            "kernel": {"type": "choice", "options": ["linear", "rbf", "poly"], "default": "rbf"}
        }
    },
    "KNN Classifier": {
        "basic": {
            "n_neighbors": {"type": "int", "min": 1, "max": 30, "default": 5}
        }
    }
}

# Importación de módulos locales
from eda import VisualizationService
from supervisado import ModelosClasificacion
from no_supervisado import SegmentadorClientes, DetectorFraude

# --- 2. CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="Minería Avanzada | Alan Montes", layout="wide", page_icon="📊")

st.markdown("<h1 style='text-align: center;'>🚀 Framework de Minería de Datos Avanzada</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Desarrollado por: <b>Alan Montes</b> | LEAD University</p>", unsafe_allow_html=True)
st.divider()

# --- 3. GESTIÓN DE ESTADO Y DATOS ---
if 'df' not in st.session_state:
    st.session_state.df = None

def cargar_datos():
    archivo = st.file_uploader("📂 Cargar Dataset de Churn (CSV)", type=["csv"])
    if archivo is not None:
        st.session_state.df = pd.read_csv(archivo)
        st.success("¡Datos cargados correctamente!")
        st.rerun()

if st.session_state.df is None:
    cargar_datos()
else:
    df = st.session_state.df
    
    menu = st.sidebar.radio(
        "📚 Unidades del Curso",
        ["1. Diagnóstico y EDA", 
         "2. Supervisado", 
         "3. Series Temporales", 
         "4. No Supervisado (Clustering)"]
    )

    # --- 4. LÓGICA DE LAS UNIDADES ---

    if menu == "1. Diagnóstico y EDA":
        st.header("🔍 Análisis Exploratorio y Calidad")
        viz = VisualizationService(df)
        metrics = viz.get_metrics()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", f"{metrics['shape'][0]:,}")
        c2.metric("Duplicados", metrics['duplicates'])
        c3.metric("% Nulos", f"{metrics['nulls']:.2f}%")
        
        st.divider()
        num_cols = viz.numeric_columns()
        if num_cols:
            col_hist = st.selectbox("Seleccione variable para analizar distribución", num_cols)
            st.plotly_chart(viz.histogram(col_hist), use_container_width=True)
            st.plotly_chart(viz.correlation_heatmap(), use_container_width=True)
        else:
            st.warning("No hay columnas numéricas suficientes.")

    elif menu == "2. Supervisado":
        st.header("🤖 Laboratorio de Modelado (Meta 1)")
        
        target = st.selectbox("Seleccione Variable Objetivo (Target)", df.columns, index=len(df.columns)-1)
        
        st.sidebar.subheader("⚙️ Configuración de Algoritmos")
        modelos_disponibles = list(MODEL_PARAMS.keys())
        seleccionados = st.multiselect("Seleccione modelos", modelos_disponibles, default=["Random Forest Classifier"])

        dict_hiperparametros = {}
        for m_name in seleccionados:
            st.sidebar.markdown(f"---")
            st.sidebar.write(f"**Parámetros: {m_name}**")
            config = MODEL_PARAMS[m_name]["basic"]
            params_del_modelo = {}
            for p_name, p_details in config.items():
                ukey = f"{m_name}_{p_name}"
                if p_details["type"] == "int":
                    params_del_modelo[p_name] = st.sidebar.slider(p_name, p_details["min"], p_details["max"], p_details["default"], key=ukey)
                elif p_details["type"] == "float":
                    params_del_modelo[p_name] = st.sidebar.slider(p_name, float(p_details["min"]), float(p_details["max"]), float(p_details["default"]), step=0.01, key=ukey)
                elif p_details["type"] == "choice":
                    params_del_modelo[p_name] = st.sidebar.selectbox(p_name, p_details["options"], index=p_details["options"].index(p_details["default"]), key=ukey)
            dict_hiperparametros[m_name] = params_del_modelo

        if st.button("🚀 Iniciar Entrenamiento Masivo"):
            with st.spinner("Entrenando modelos y evaluando estabilidad..."):
                try:
                    sup = ModelosClasificacion(df, target)
                    resultados = []
                    
                    mejor_acc = -1
                    mejor_mod_obj = None
                    nombre_ganador = ""

                    for m_name in seleccionados:
                        params = dict_hiperparametros[m_name]
                        
                        if m_name == "Random Forest Classifier": m = sup.ejecutar_rf(**params)
                        elif m_name == "Logistic Regression": m = sup.ejecutar_logistica(**params)
                        elif m_name == "XGBoost Classifier": m = sup.ejecutar_xgb(**params)
                        elif m_name == "SVM": m = sup.ejecutar_svm(**params)
                        elif m_name == "KNN Classifier": m = sup.ejecutar_knn(**params)

                        met = sup.obtener_metricas_finales(m)
                        met["Algoritmo"] = m_name
                        resultados.append(met)
                        
                        # Guardar el mejor para validación cruzada
                        if met["Accuracy"] > mejor_acc:
                            mejor_acc = met["Accuracy"]
                            mejor_mod_obj = m
                            nombre_ganador = m_name

                    # Resultados en tabla
                    st.subheader("📊 Comparativa de Rendimiento (Test Set)")
                    st.table(pd.DataFrame(resultados).set_index("Algoritmo"))

                    # Comparativa de Estabilidad
                    if mejor_mod_obj:
                        st.divider()
                        st.subheader(f"⚖️ Estabilidad del Ganador: {nombre_ganador}")
                        
                        cv_res = sup.ejecutar_validacion_cruzada(mejor_mod_obj, cv=5)
                        
                        c_m1, c_m2 = st.columns(2)
                        with c_m1:
                            st.write("### 📈 Métricas de Control")
                            st.metric("Split Único", f"{mejor_acc:.4f}")
                            st.metric("Promedio K-Fold", f"{cv_res['AUC_Promedio']:.4f}", 
                                      delta=f"{cv_res['AUC_Promedio'] - mejor_acc:.4f}")
                            st.caption("Un delta negativo indica que el split inicial fue algo optimista.")

                        with c_m2:
                            labels = ["Split"] + [f"Fold {i+1}" for i in range(len(cv_res["Iteraciones"]))]
                            vals = [mejor_acc] + cv_res["Iteraciones"]
                            st.bar_chart(pd.DataFrame({"Prueba": labels, "Score": vals}), x="Prueba", y="Score")

                        st.info(f"**Análisis:** Desviación estándar de **{cv_res['AUC_Desviacion']:.4f}**. "
                                f"Muestra qué tan consistente es el modelo con datos no vistos.")

                except Exception as e:
                    st.error(f"Error en el entrenamiento: {e}")

    elif menu == "3. Series Temporales":
        st.header("⏳ Meta 2: Dinámica Temporal")
        viz = VisualizationService(df)
        c1, c2 = st.columns(2)
        col_f = c1.selectbox("Columna Temporal", df.columns)
        col_v = c2.selectbox("Métrica a Analizar", df.columns)
        periodo = st.slider("Periodo de Estacionalidad", 2, 30, 7)
        
        if st.button("🔍 Descomponer Serie"):
            fig, dia = viz.obtener_descomposicion_plotly(col_f, col_v, periodo)
            if fig:
                st.success(f"Día con Mayor Tendencia Detectado: **{dia}**")
                st.plotly_chart(fig, use_container_width=True)

    elif menu == "4. No Supervisado (Clustering)":
        st.header("🧪 Segmentación y Detección de Anomalías")
        tab1, tab2 = st.tabs(["K-Means (Segmentación)", "Isolation Forest (Anomalías)"])
        
        with tab1:
            k = st.slider("Número de Clústeres (K)", 2, 8, 3)
            if st.button("Generar Segmentos"):
                seg = SegmentadorClientes(df)
                df_res = seg.generar_clusters(k)
                st.write("Vista previa de segmentos:")
                st.dataframe(df_res.head(10))
                
        with tab2:
            contam = st.slider("Ratio de Contaminación", 0.01, 0.15, 0.05)
            if st.button("Ejecutar Detección"):
                det = DetectorFraude(df)
                df_anom = det.detectar_anomalias(contam)
                n_anom = len(df_anom[df_anom['Es_Anomalia'] == -1])
                st.warning(f"Se identificaron {n_anom} anomalías.")

# --- 5. ACCIONES ---
if st.sidebar.button("🗑️ Reiniciar Sesión"):
    st.session_state.df = None
    st.rerun()