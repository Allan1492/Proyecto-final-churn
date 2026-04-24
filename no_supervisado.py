import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

class AnalisisSegmentacion:
    """Clase para análisis de clustering con K-Means"""
    
    def __init__(self, df):
        self.df = df.copy()
        
    def ejecutar_kmeans(self, k=3):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        
        df_num = self.df.select_dtypes(include=[np.number])
        
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_num.dropna())
        
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        
        df_result = self.df.iloc[:len(clusters)].copy()
        df_result['Cluster'] = clusters
        
        return df_result, kmeans


class AnalisisAnomalias:
    """Clase para detección de anomalías con Isolation Forest"""
    
    def __init__(self, df):
        self.df = df.copy()
        
    def detectar_isolation_forest(self, contamination=0.05):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        
        df_num = self.df.select_dtypes(include=[np.number])
        
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_num.dropna())
        
        
        iso = IsolationForest(contamination=contamination, random_state=42)
        predicciones = iso.fit_predict(X_scaled)
        
        df_result = self.df.iloc[:len(predicciones)].copy()
        df_result['Es_Anomalia'] = predicciones
        
        return df_result


class ReglasAsociacion:
    """
    Clase para análisis de reglas de asociación (Algoritmo Apriori)
    Ideal para Market Basket Analysis
    """
    
    def __init__(self, df):
        """
        Inicializa la clase con el DataFrame
        df: DataFrame con datos transaccionales o binarios
        """
        self.df = df.copy()
        self.df_binario = None
        
    def _preparar_datos_binarios(self, columnas_categoricas=None):
        """
        Convierte datos categóricos a formato binario para Apriori
        """
        if columnas_categoricas is None:
            columnas_categoricas = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not columnas_categoricas:
            
            columnas_numericas = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if columnas_numericas:
                self.df_binario = (self.df[columnas_numericas] > self.df[columnas_numericas].median()).astype(int)
                return self.df_binario
        
        
        try:
            
            transacciones = []
            for _, row in self.df[columnas_categoricas].iterrows():
                items = [str(val) for val in row.dropna().unique() if pd.notna(val)]
                if items:
                    transacciones.append(items)
            
            if transacciones:
                te = TransactionEncoder()
                te_ary = te.fit(transacciones).transform(transacciones)
                self.df_binario = pd.DataFrame(te_ary, columns=te.columns_)
                return self.df_binario
        except:
            pass
        
        
        self.df_binario = pd.get_dummies(self.df[columnas_categoricas])
        return self.df_binario
    
    def generar_reglas_asociacion(self, soporte_min=0.05, confianza_min=0.5, metrica='lift'):
        
        if self.df_binario is None:
            self._preparar_datos_binarios()
        
        if self.df_binario is None or self.df_binario.empty:
            raise ValueError("No se pudieron preparar los datos para análisis de asociación. Verifica que el DataFrame tenga columnas categóricas o transaccionales.")
        
        
        itemsets_frecuentes = apriori(
            self.df_binario, 
            min_support=soporte_min, 
            use_colnames=True,
            max_len=2  
        )
        
        if itemsets_frecuentes.empty:
          
            return pd.DataFrame(columns=['antecedents', 'consequents', 'antecedent support', 
                                       'consequent support', 'support', 'confidence', 'lift'])
        
        
        reglas = association_rules(
            itemsets_frecuentes, 
            metric="confidence", 
            min_threshold=confianza_min
        )
        
        if not reglas.empty:
          
            if 'lift' not in reglas.columns:
                reglas['lift'] = reglas['confidence'] / reglas['consequent support']
            
            
            if metrica in reglas.columns:
                reglas = reglas.sort_values(by=metrica, ascending=False)
            
            columnas_output = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
            columnas_existentes = [c for c in columnas_output if c in reglas.columns]
            reglas = reglas[columnas_existentes]
        
        return reglas.reset_index(drop=True)
    
    def obtener_top_reglas(self, soporte_min=0.05, confianza_min=0.5, top_n=10):
        """
        Obtiene las N mejores reglas según lift
        """
        reglas = self.generar_reglas_asociacion(soporte_min, confianza_min)
        if not reglas.empty and 'lift' in reglas.columns:
            return reglas.head(top_n)
        return reglas