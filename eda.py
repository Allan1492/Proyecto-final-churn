import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.tsa.seasonal import seasonal_decompose

class EDAService:
    """Clase para Análisis Exploratorio de Datos (Base)"""
    def __init__(self, df):
        self.df = df

    def shape(self): return self.df.shape
    def dtypes(self): return self.df.dtypes.to_frame("Tipo")
    def null_values(self): return self.df.isnull().sum()
    def null_percentage(self): return self.df.isnull().mean() * 100
    def null_summary(self):
        return pd.DataFrame({"Nulos": self.null_values(), "%": self.null_percentage()})
    def summary_statistics(self): return self.df.describe(include="all")
    def duplicated_count(self): return self.df.duplicated().sum()
    def numeric_columns(self): return self.df.select_dtypes(include="number").columns.tolist()
    def categorical_columns(self): return self.df.select_dtypes(exclude="number").columns.tolist()

# --- CATEGORÍA 1: APRENDIZAJE SUPERVISADO ---
class MineriaSupervisada(EDAService):
    def validacionCruzada(self, modelo, X_cols, y_col, cv=5):
        X = self.df[X_cols]
        y = self.df[y_col]
        scores = cross_val_score(modelo, X, y, cv=cv)
        return scores

    def graficoAUC_ROC(self, modelo, X_test, y_test):
        probabilidades = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilidades)
        fpr, tpr, _ = roc_curve(y_test, probabilidades)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title("Curva ROC")
        ax.legend()
        return fig

# --- CATEGORÍA 2: APRENDIZAJE NO SUPERVISADO ---
class MineriaNoSupervisada(EDAService):
    def webMiningEstatico(self, url, etiqueta):
        try:
            respuesta = requests.get(url)
            soup = BeautifulSoup(respuesta.text, 'html.parser')
            elementos = soup.find_all(etiqueta)
            return [e.text.strip() for e in elementos]
        except Exception as e:
            return [f"Error: {e}"]

    def deteccionAnomalias(self, contaminacion=0.05):
        num_df = self.df.select_dtypes(include=["number"])
        if num_df.empty: return self.df
        model = IsolationForest(contamination=contaminacion, random_state=42)
        self.df['anomalia'] = model.fit_predict(num_df)
        return self.df

# --- CATEGORÍA 3: AGRUPAMIENTO Y SERIES TEMPORALES ---
class AgrupamientoYSeries(EDAService):
    def analisisSeriesTiempo(self, columna_fecha, columna_valor, periodo=7):
        df_temp = self.df.copy()
        df_temp[columna_fecha] = pd.to_datetime(df_temp[columna_fecha])
        df_ts = df_temp.set_index(columna_fecha).sort_index()
        descomposicion = seasonal_decompose(df_ts[columna_valor], period=periodo)
        return descomposicion.plot()

    def analisisCohorte(self, col_id_usuario, col_fecha_evento):
        df = self.df.copy()
        df['periodo_evento'] = pd.to_datetime(df[col_fecha_evento]).dt.to_period('M')
        df['cohorte'] = df.groupby(col_id_usuario)[col_fecha_evento].transform('min').dt.to_period('M')
        cohorte_data = df.groupby(['cohorte', 'periodo_evento']).agg(n_clientes=(col_id_usuario, 'nunique')).reset_index()
        return cohorte_data

# --- CATEGORÍA 4: VISUALIZACIÓN Y DASHBOARD ---
class VisualizationService(MineriaSupervisada, MineriaNoSupervisada, AgrupamientoYSeries):
    def histogram(self, col): 
        return px.histogram(self.df, x=col, title=f"Distribución de {col}", template="plotly_white")
    
    def boxplot(self, col): 
        return px.box(self.df, y=col, title=f"Análisis de Outliers: {col}", template="plotly_white")
    
    # NUEVO: Método que pide tu app.py
    def correlation_heatmap(self):
        """Genera la matriz de correlación interactiva para el dashboard"""
        corr = self.df.corr(numeric_only=True)
        return px.imshow(corr, text_auto=True, title="Matriz de Correlación de Pearson", color_continuous_scale='RdBu_r')

    def resumenEjecutivoDashboard(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.heatmap(self.df.corr(numeric_only=True), cmap='coolwarm', ax=ax1)
        self.df.isna().sum().plot(kind='bar', ax=ax2)
        return fig