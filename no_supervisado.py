import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MineriaNoSupervisada:
    """
    Clase Base que documenta el Aprendizaje No Supervisado.
    Tema 1 y 2 del Sílabo: Descubrimiento de patrones y reducción de complejidad.
    """
    def __init__(self, df):
        self.df = df.copy()
        # Filtramos solo datos numéricos para algoritmos de distancia
        self.numeric_data = self.df.select_dtypes(include=['number']).fillna(0)
        self.scaler = StandardScaler()
        self.data_scaled = None

    def _escalar_datos(self):
        """Preprocesamiento esencial para algoritmos basados en distancias (K-Means, PCA)."""
        self.data_scaled = self.scaler.fit_transform(self.numeric_data)
        return self.data_scaled

class AnalisisSegmentacion(MineriaNoSupervisada):
    """
    Documenta el TEMA 1: Algoritmos de Agrupamiento (Clustering).
    Incluye métodos de particionamiento, jerárquicos y basados en densidad.
    """
    
    def ejecutar_kmeans(self, n_clusters=3):
        """Tema: K-Means (Algoritmos de Particionamiento)"""
        data = self._escalar_datos()
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['Segmento_KMeans'] = model.fit_predict(data)
        return self.df, model

    def ejecutar_dbscan(self, eps=0.5, min_samples=5):
        """Tema: DBSCAN (Algoritmos basados en densidad)"""
        data = self._escalar_datos()
        model = DBSCAN(eps=eps, min_samples=min_samples)
        self.df['Segmento_DBSCAN'] = model.fit_predict(data)
        return self.df

    def ejecutar_jerarquico(self, n_clusters=3):
        """Tema: Clustering Jerárquico (Agglomerative)"""
        data = self._escalar_datos()
        model = AgglomerativeClustering(n_clusters=n_clusters)
        self.df['Segmento_Jerarquico'] = model.fit_predict(data)
        return self.df

class ReduccionDimensionalidad(MineriaNoSupervisada):
    """
    Documenta el TEMA 2: Reducción de Dimensionalidad.
    """
    def ejecutar_pca(self, n_components=2):
        """
        Tema: Análisis de Componentes Principales (PCA).
        Transforma el espacio de características para visualización o eficiencia.
        """
        data = self._escalar_datos()
        pca = PCA(n_components=n_components)
        componentes = pca.fit_transform(data)
        
        columnas_pca = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(data=componentes, columns=columnas_pca)
        
        # Retornamos la varianza explicada para documentar la pérdida de información
        varianza = pca.explained_variance_ratio_
        return df_pca, varianza

class AnalisisAnomalias(MineriaNoSupervisada):
    """
    Documenta técnicas de detección de Outliers (Anomalías).
    Tema transversal en Minería de Datos para limpieza y seguridad.
    """
    def detectar_isolation_forest(self, contaminacion=0.05):
        """Algoritmo basado en árboles para identificar puntos aislados."""
        iso = IsolationForest(contamination=contaminacion, random_state=42)
        # -1 es anomalía, 1 es normal
        self.df['Es_Anomalia'] = iso.fit_predict(self.numeric_data)
        return self.df

class ReglasAsociacion:
    """
    Documenta el TEMA: Análisis de Afinidad (Market Basket Analysis).
    Reservado para futuras implementaciones de algoritmos como Apriori.
    """
    def __init__(self, df):
        self.df = df
    
    def ejecutar_apriori(self):
        """Espacio para documentar el análisis de reglas de asociación."""
        pass