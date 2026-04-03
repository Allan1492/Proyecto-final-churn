import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

class EDAService:
    """Clase base para diagnóstico de datos"""
    def __init__(self, df):
        self.df = df

    def get_metrics(self):
        return {
            "shape": self.df.shape,
            "duplicates": self.df.duplicated().sum(),
            "nulls": self.df.isnull().mean().mean() * 100
        }
        
    def numeric_columns(self): 
        return self.df.select_dtypes(include="number").columns.tolist()

class AgrupamientoYSeries(EDAService):
    """Maneja la Meta 2: Dinámica Temporal"""
    def obtener_descomposicion_plotly(self, columna_fecha, columna_valor, periodo=7):
        df_temp = self.df.copy()
        df_temp[columna_fecha] = pd.to_datetime(df_temp[columna_fecha])
        df_ts = df_temp.groupby(columna_fecha)[columna_valor].count().resample('D').sum().fillna(0)
        
        if len(df_ts) < periodo * 2: return None, "Datos insuficientes"

        result = seasonal_decompose(df_ts, period=periodo)
        
        # Identificación de día crítico (Traducción lógica)
        dias_es = {'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles', 
                   'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
        seasonal_df = pd.DataFrame({'s': result.seasonal})
        top_day_en = seasonal_df.index.day_name()[0] 
        dia_critico = dias_es.get(top_day_en, top_day_en)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                          subplot_titles=("Observado", "Tendencia", "Estacionalidad", "Residuos"))
        
        fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed, name="Real"), row=1, col=1)
        fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, name="Tendencia"), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, name="Ciclo"), row=3, col=1)
        fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='markers', name="Anomalía"), row=4, col=1)
        
        fig.update_layout(height=900, template="plotly_white", title=f"Análisis Temporal - Día Crítico: {dia_critico}")
        return fig, dia_critico

class VisualizationService(AgrupamientoYSeries):
    """Servicios de graficación para la UI"""
    def histogram(self, col): 
        return px.histogram(self.df, x=col, title=f"Distribución: {col}", template="plotly_white")
        
    def correlation_heatmap(self):
        corr = self.df.corr(numeric_only=True)
        return px.imshow(corr, text_auto=True, title="Matriz de Correlación", color_continuous_scale='RdBu_r')