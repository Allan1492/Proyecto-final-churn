import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             mean_squared_error, r2_score, mean_absolute_error)

# Modelos de Clasificación (Tema 3 del Sílabo)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Redes Neuronales Artificiales (Contenido Avanzado - Tema 3)
from sklearn.neural_network import MLPClassifier

# Modelos de Regresión (Tema 4 del Sílabo)
from sklearn.linear_model import LinearRegression

class MineriaSupervisada:
    """
    Clase Base: Documenta la fase de Preprocesamiento y Limpieza.
    Asegura que todos los modelos utilicen datos estandarizados y sin fuga.
    """
    def __init__(self, df):
        self.df = df.copy()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = StandardScaler()

    def _limpiar_fuga_datos(self, df_input):
        """
        Elimina variables que causan Data Leakage y IDs.
        Crucial para la validez técnica del proyecto.
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

        # Codificación de etiquetas (Churn/No Churn -> 0/1)
        le_target = LabelEncoder()
        y = le_target.fit_transform(df_ml[self.target].astype(str))

        # Split estratificado
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Escalado (Esencial para Logística, SVM, KNN y Redes Neuronales)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def obtener_metricas_finales(self, modelo):
        """Calcula el rendimiento del modelo en el conjunto de prueba."""
        preds = modelo.predict(self.X_test)
        return {
            "Accuracy": round(accuracy_score(self.y_test, preds), 4),
            "Precision": round(precision_score(self.y_test, preds, zero_division=0), 4),
            "Recall": round(recall_score(self.y_test, preds, zero_division=0), 4),
            "F1-Score": round(f1_score(self.y_test, preds, zero_division=0), 4)
        }

    # --- MÉTODOS DE ENTRENAMIENTO FLEXIBLES (**kwargs) ---

    def ejecutar_rf(self, **kwargs):
        """Tema: Random Forest (Bagging). Soporta n_estimators, max_depth, min_samples_split, etc."""
        model = RandomForestClassifier(**kwargs, random_state=42)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_xgb(self, **kwargs):
        """Tema: XGBoost (Boosting). Soporta learning_rate, gamma, subsample, etc."""
        # Se asegura de incluir parámetros de seguridad para XGBoost
        params = {'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False}
        params.update(kwargs)
        model = XGBClassifier(**params)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_red_neuronal(self, **kwargs):
        """Tema: Redes Neuronales (MLP). Soporta hidden_layer_sizes, activation, max_iter."""
        params = {'random_state': 42, 'max_iter': 500}
        params.update(kwargs)
        model = MLPClassifier(**params)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_logistica(self, **kwargs):
        """Tema: Regresión Logística. Soporta C, penalty, solver."""
        model = LogisticRegression(**kwargs, random_state=42)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_svm(self, **kwargs):
        """Tema: SVM. Soporta C, kernel, gamma."""
        model = SVC(**kwargs, probability=True, random_state=42)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_knn(self, **kwargs):
        """Tema: KNN. Soporta n_neighbors, weights."""
        model = KNeighborsClassifier(**kwargs)
        return model.fit(self.X_train, self.y_train)
    
    def ejecutar_validacion_cruzada(self, modelo, cv=5):
        """Tema: Validación de Modelos (K-Fold Cross Validation)."""
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
        """Regresión Lineal Múltiple."""
        model = LinearRegression()
        return model.fit(self.X_train, self.y_train)

    def obtener_metricas_regresion(self, modelo):
        """Métricas de error para variables continuas."""
        preds = modelo.predict(self.X_test)
        return {
            "R2_Score": round(r2_score(self.y_test, preds), 4),
            "MAE": round(mean_absolute_error(self.y_test, preds), 4),
            "MSE": round(mean_squared_error(self.y_test, preds), 4)
        }

class AnalisisTemporal:
    """
    Documenta el TEMA 5: Introducción a las Series Temporales.
    """
    def __init__(self, df, col_fecha, col_valor):
        self.df = df.copy()
        self.col_fecha = col_fecha
        self.col_valor = col_valor

    def generar_serie_agregada(self, frecuencia='M'):
        """Transforma datos a formato de serie de tiempo."""
        self.df[self.col_fecha] = pd.to_datetime(self.df[self.col_fecha])
        serie = self.df.set_index(self.col_fecha)[self.col_valor].resample(frecuencia).sum()
        return serie