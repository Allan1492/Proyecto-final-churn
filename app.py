
import streamlit as st
import pandas as pd
import numpy as np
from models import ALL_MODELS, MODEL_PARAMS
from eda import VisualizationService
from supervisado import ModelosClasificacion
from no_supervisado import AnalisisSegmentacion, AnalisisAnomalias, ReglasAsociacion

st.set_page_config(page_title="Minería Avanzada", layout="wide", page_icon="📊")

def load_css(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
load_css("style.css")



st.markdown("<h1 style='text-align: center;'> Framework de Minería de Datos Avanzada</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Analisis Predictivo de la Desercion de Clientes mediante Modelos de Clasificacion y Series Temporales | LEAD University</p>", 
    unsafe_allow_html=True
)
st.divider()

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

        if st.button(" Iniciar Entrenamiento Masivo"):
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
                    st.subheader(" Comparativa de Rendimiento (Test Set)")
                    st.table(pd.DataFrame(resultados).set_index("Algoritmo"))

                    # Comparativa de Estabilidad
                    if mejor_mod_obj:
                        st.divider()
                        st.subheader(f"📊 Estabilidad del Ganador: {nombre_ganador}")
                        
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
        
        if st.button(" Descomponer Serie"):
            fig, dia = viz.obtener_descomposicion_plotly(col_f, col_v, periodo)
            if fig:
                st.success(f"Día con Mayor Tendencia Detectado: **{dia}**")
                st.plotly_chart(fig, use_container_width=True)

    elif menu == "4. No Supervisado (Clustering)":
        st.header(" Segmentación, Anomalías y Asociación")
        
        # 1. Definición de pestañas (Tabs)
        tab1, tab2 = st.tabs([
            "K-Means (Segmentación)", 
            "Reglas de Asociación"
        ])
        
        with tab1:
            k = st.slider("Número de Clústeres (K)", 2, 8, 3)
            if st.button("Ejecutar Clustering"):
                try:
                    seg = AnalisisSegmentacion(df) 
                    df_res, modelo = seg.ejecutar_kmeans(k)
                    st.success(f"Segmentación completada para K={k}")
                    st.dataframe(df_res.head(10))
                except Exception as e:
                    st.error(f"Error en Clustering: {e}")
                

        with tab2:
            st.subheader(" Análisis de Afinidad (Market Basket)")
            st.write("Descubre qué servicios de Tico Mart suelen contratarse juntos.")
            
            # Sliders para ajustar el algoritmo Apriori
            c_a, c_b = st.columns(2)
            soporte = c_a.slider("Soporte Mínimo", 0.01, 0.5, 0.05)
            confianza = c_b.slider("Confianza Mínima", 0.1, 1.0, 0.5)
            
            if st.button("Generar Reglas"):
                with st.spinner("Calculando asociaciones..."):
                    try:
                        # Instancia de la nueva clase que agregamos
                        asociador = ReglasAsociacion(df)
                        reglas = asociador.generar_reglas_asociacion(soporte, confianza)
                        
                        if not reglas.empty:
                            st.success(f"Se detectaron {len(reglas)} reglas significativas.")
                            
                            reglas_viz = reglas.copy()
                            reglas_viz['antecedents'] = reglas_viz['antecedents'].apply(lambda x: ', '.join(list(x)))
                            reglas_viz['consequents'] = reglas_viz['consequents'].apply(lambda x: ', '.join(list(x)))
                            
                            st.dataframe(reglas_viz.style.background_gradient(subset=['lift'], cmap='YlGn'))
                        else:
                            st.warning("No se encontraron reglas con esos parámetros. Intenta bajar el Soporte.")
                    except Exception as e:
                        st.error(f"Error en Reglas: {e}")
                        st.info("Asegúrate de haber instalado mlxtend con: pip install mlxtend")

# --- 5. ACCIONES ---
if st.sidebar.button("🗑️ Reiniciar Sesión"):
    st.session_state.df = None
    st.rerun()