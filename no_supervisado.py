from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class MineriaNoSupervisada:
    def __init__(self, df):
        self.df = df
        self.numeric_data = df.select_dtypes(include=['number']).fillna(0)
        self.scaler = StandardScaler()

class SegmentadorClientes(MineriaNoSupervisada):
    """Clase para Clustering K-Means"""
    def generar_clusters(self, n_clusters=3):
        data_scaled = self.scaler.fit_transform(self.numeric_data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data_scaled)
        self.df['Segmento_ID'] = clusters
        return self.df

class DetectorFraude(MineriaNoSupervisada):
    """Clase para Detección de Anomalías (Isolation Forest)"""
    def detectar_anomalias(self, contaminacion=0.05):
        iso = IsolationForest(contamination=contaminacion, random_state=42)
        anomalias = iso.fit_predict(self.numeric_data)
        # Convertir -1 (anomalía) y 1 (normal) a formato legible
        self.df['Es_Anomalia'] = anomalias
        return self.df