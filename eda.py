import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.tsa.seasonal import seasonal_decompose

class EDAService:
    def __init__(self, df):
        self.df = df

    def duplicated_count(self): return self.df.duplicated().sum()
    def null_percentage(self): return self.df.isnull().mean() * 100
    def numeric_columns(self): return self.df.select_dtypes(include="number").columns.tolist()

class MineriaSupervisada(EDAService):
    def validacionCruzada(self, modelo, X, y, cv=5):
        return cross_val_score(modelo, X, y, cv=cv)

class MineriaNoSupervisada(EDAService):
    def deteccionAnomalias(self, contaminacion=0.05):
        num_df = self.df.select_dtypes(include=["number"])
        if num_df.empty: return self.df
        model = IsolationForest(contamination=contaminacion, random_state=42)
        self.df['anomalia'] = model.fit_predict(num_df)
        return self.df

class AgrupamientoYSeries(EDAService):
    def obtener_descomposicion_plotly(self, columna_fecha, columna_valor, periodo=7):
        df_temp = self.df.copy()
        df_temp[columna_fecha] = pd.to_datetime(df_temp[columna_fecha])
        # Agrupamos por día para tener una serie continua
        df_ts = df_temp.groupby(columna_fecha)[columna_valor].count().resample('D').sum().fillna(0)
        
        if len(df_ts) < periodo * 2: return None

        result = seasonal_decompose(df_ts, period=periodo)
        fig = make_subplots(rows=4, cols=1, subplot_titles=("Observado", "Tendencia", "Estacionalidad", "Residuos"))
        fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed, name="Obs"), row=1, col=1)
        fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, name="Trend"), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, name="Seas"), row=3, col=1)
        fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, name="Resid"), row=4, col=1)
        fig.update_layout(height=800, title_text="Meta 2: Descomposición Temporal", showlegend=False)
        return fig

    def pronostico_y_error(self, columna_fecha, columna_valor):
        df_temp = self.df.copy()
        df_temp[columna_fecha] = pd.to_datetime(df_temp[columna_fecha])
        df_ts = df_temp.groupby(columna_fecha)[columna_valor].count().resample('D').sum().fillna(0)
        pronostico = df_ts.shift(1).fillna(df_ts.mean())
        mae = (df_ts - pronostico).abs().mean()
        return df_ts, mae

class VisualizationService(MineriaSupervisada, MineriaNoSupervisada, AgrupamientoYSeries):
    def histogram(self, col): return px.histogram(self.df, x=col, title=f"Distribución de {col}")
    def correlation_heatmap(self):
        corr = self.df.corr(numeric_only=True)
        return px.imshow(corr, text_auto=True, title="Matriz de Correlación", color_continuous_scale='RdBu_r')