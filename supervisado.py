import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, mean_squared_error, r2_score, mean_absolute_error)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier 
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression

class MineriaSupervisada:
   
    def __init__(self, df):
        self.df = df.copy()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = StandardScaler()

    def _limpiar_fuga_datos(self, df_input):
        """
        Elimina variables que causan Data Leakage y IDs.
        """
        columnas_a_eliminar = [
            'CustomerID', 'Fecha_Interaccion', 'Fecha_Inicio_Contrato', 
            'Last Interaction', 'Payment Delay', 'Total Spend'
        ]
        existentes = [c for c in columnas_a_eliminar if c in df_input.columns]
        return df_input.drop(columns=existentes)

class ModelosClasificacion(MineriaSupervisada):
    """
    Documenta el TEMA 3: Modelos de Clasificación y Aprendizaje Profundo.
    """
    def __init__(self, df, target):
        super().__init__(df)
        self.target = target
        self._preparar_datos()

    def _preparar_datos(self):
        """Preprocesamiento completo: Limpieza, Encoding, Split y Escalado."""
        df_ml = self._limpiar_fuga_datos(self.df.copy())
        df_ml = df_ml.dropna()

        # Separación de características y objetivo
        X = df_ml.drop(columns=[self.target])
        X = pd.get_dummies(X, drop_first=True)

       
        le_target = LabelEncoder()
        y = le_target.fit_transform(df_ml[self.target].astype(str))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def obtener_metricas_finales(self, modelo):
        """Calcula el rendimiento del modelo en el conjunto de prueba, incluyendo AUC."""
        preds = modelo.predict(self.X_test)
        
        
        if hasattr(modelo, "predict_proba"):
            probs = modelo.predict_proba(self.X_test)[:, 1]
        else:
            
            probs = preds 

        return {
            "Accuracy": round(accuracy_score(self.y_test, preds), 4),
            "Precision": round(precision_score(self.y_test, preds, zero_division=0), 4),
            "Recall": round(recall_score(self.y_test, preds, zero_division=0), 4),
            "F1-Score": round(f1_score(self.y_test, preds, zero_division=0), 4), # Se agregó coma faltante
            "AUC-ROC": round(roc_auc_score(self.y_test, probs), 4) # Ahora 'probs' sí existe
        }

    def ejecutar_rf(self, **kwargs):
        model = RandomForestClassifier(**kwargs, random_state=42)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_xgb(self, **kwargs):
        params = {'random_state': 42, 'eval_metric': 'logloss'}
        params.update(kwargs)
        model = XGBClassifier(**params)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_red_neuronal(self, **kwargs):
        
        params = {'random_state': 42, 'max_iter': 500}
        params.update(kwargs)
        model = MLPClassifier(**params)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_logistica(self, **kwargs):
        model = LogisticRegression(**kwargs, random_state=42)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_svm(self, **kwargs):
        
        model = SVC(**kwargs, probability=True, random_state=42)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_knn(self, **kwargs):
        model = KNeighborsClassifier(**kwargs)
        return model.fit(self.X_train, self.y_train)
    
    def ejecutar_validacion_cruzada(self, modelo, cv=5):
        scores = cross_val_score(modelo, self.X_train, self.y_train, cv=cv, scoring='roc_auc')
        return {
            "AUC_Promedio": round(scores.mean(), 4),
            "AUC_Desviacion": round(scores.std(), 4),
            "Iteraciones": scores.tolist()
        }

class ModelosRegresion(MineriaSupervisada):
    """
    Documenta el TEMA 4: Regresión y Estimación Numérica.
    """
    def __init__(self, df, target_numerico):
        super().__init__(df)
        self.target = target_numerico
        self._preparar_regresion()

    def _preparar_regresion(self):
        df_reg = self.df.dropna()
        X = pd.get_dummies(df_reg.drop(columns=[self.target]), drop_first=True)
        y = df_reg[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def ejecutar_lineal(self):
        model = LinearRegression()
        return model.fit(self.X_train, self.y_train)

    def obtener_metricas_regresion(self, modelo):
        preds = modelo.predict(self.X_test)
        return {
            "R2_Score": round(r2_score(self.y_test, preds), 4),
            "MAE": round(mean_absolute_error(self.y_test, preds), 4),
            "MSE": round(mean_squared_error(self.y_test, preds), 4)
        }