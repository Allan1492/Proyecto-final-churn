import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
class MineriaSupervisada:
    def __init__(self, df):
        self.df = df.copy()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = StandardScaler()

class ModelosClasificacion(MineriaSupervisada):
    def __init__(self, df, target):
        super().__init__(df)
        self.target = target
        self._preparar_datos()

    def _preparar_datos(self):
        """Preprocesamiento con limpieza de Data Leakage y Fechas."""
        df_ml = self.df.copy()
        
        # 1. Definir columnas de fuga y técnicas (IDs, Fechas, Respuestas obvias)
        # Eliminamos 'Payment Delay' y 'Support Calls' porque inflan artificialmente el Accuracy
        columnas_a_eliminar = [
            'CustomerID', 'Payment Delay', 'Support Calls', 
            'Fecha_Interaccion', 'Fecha_Inicio_Contrato', 'Last Interaction'
        ]
        
        # Solo eliminamos si existen en el dataframe actual
        existentes = [c for c in columnas_a_eliminar if c in df_ml.columns]
        df_ml = df_ml.drop(columns=existentes)

        # 2. Encoding de variables categóricas (Gender, Subscription Type, etc.)
        le = LabelEncoder()
        for col in df_ml.select_dtypes(include=['object']).columns:
            if col != self.target:
                df_ml[col] = le.fit_transform(df_ml[col].astype(str))

        # 3. Separación de X e y
        X = df_ml.drop(columns=[self.target])
        y = df_ml[self.target]

        # 4. Split Train/Test con Estratificación (mantiene la proporción de Churn)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 5. Escalado de características
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def obtener_metricas_finales(self, modelo):
        """Cálculo de métricas de rendimiento."""
        preds = modelo.predict(self.X_test)
        return {
            "Accuracy": round(accuracy_score(self.y_test, preds), 4),
            "Precision": round(precision_score(self.y_test, preds, zero_division=0), 4),
            "Recall": round(recall_score(self.y_test, preds, zero_division=0), 4),
            "F1-Score": round(f1_score(self.y_test, preds, zero_division=0), 4)
        }

    # Métodos de entrenamiento (se mantienen igual que antes...)
    def ejecutar_rf(self, n_estimators=100, max_depth=10, min_samples_split=2, criterion='gini'):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                       min_samples_split=min_samples_split, criterion=criterion, random_state=42)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_logistica(self, C=1.0, max_iter=100):
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_xgb(self, learning_rate=0.1, n_estimators=100, max_depth=6):
        model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, 
                              max_depth=max_depth, random_state=42, eval_metric='logloss')
        return model.fit(self.X_train, self.y_train)

    def ejecutar_svm(self, C=1.0, kernel='rbf'):
        model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
        return model.fit(self.X_train, self.y_train)

    def ejecutar_knn(self, n_neighbors=5):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        return model.fit(self.X_train, self.y_train)
    
    def ejecutar_validacion_cruzada(self, modelo, cv=5):
        # Usamos 'roc_auc' como métrica principal según tu requerimiento
        scores = cross_val_score(modelo, self.X_train, self.y_train, cv=cv, scoring='roc_auc')
        
        return {
            "AUC_Promedio": round(scores.mean(), 4),
            "AUC_Desviacion": round(scores.std(), 4),
            "Iteraciones": scores.tolist()
    }