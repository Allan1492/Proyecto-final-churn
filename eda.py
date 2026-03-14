import pandas as pd
import plotly.express as px


class EDAService:

    def __init__(self, df):
        self.df = df

    def shape(self):
        return self.df.shape

    def dtypes(self):
        return self.df.dtypes.to_frame("Tipo")

    def null_values(self):
        return self.df.isnull().sum()

    def null_percentage(self):
        return self.df.isnull().mean() * 100

    def null_summary(self):
        return pd.DataFrame({
            "Nulos": self.null_values(),
            "%": self.null_percentage()
        })

    def summary_statistics(self):
        return self.df.describe(include="all")

    def duplicated_count(self):
        return self.df.duplicated().sum()

    def numeric_columns(self):
        return self.df.select_dtypes(include="number").columns.tolist()

    def categorical_columns(self):
        return self.df.select_dtypes(exclude="number").columns.tolist()


class VisualizationService:

    def __init__(self, df):
        self.df = df

    def histogram(self, col):
        return px.histogram(self.df, x=col, title=f"Distribución de {col}")

    def boxplot(self, col):
        return px.box(self.df, y=col, title=f"Boxplot de {col}")

    def bar_chart(self, col):
        counts = self.df[col].value_counts().reset_index()
        counts.columns = [col, "Frecuencia"]
        return px.bar(counts, x=col, y="Frecuencia", title=f"Frecuencia de {col}")

    def correlation_heatmap(self):
        corr = self.df.corr(numeric_only=True)
        return px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de Correlación")