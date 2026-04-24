import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report


from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def optimizacion_profunda():
    try:
        df = pd.read_csv('Dataset_Churn.csv')
    except FileNotFoundError:
        print("Error: No se encontró el archivo Dataset_Churn.csv")
        return

    columnas_fuga = [
        'CustomerID', 'Fecha_Interaccion', 'Fecha_Inicio_Contrato', 
        'Last Interaction', 'Support Calls', 'Payment Delay', 'Total Spend'
    ]
    df_ml = df.drop(columns=[c for c in columnas_fuga if c in df.columns])
    
   
    target = 'Churn'
    X = pd.get_dummies(df_ml.drop(columns=[target]), drop_first=True)
    y = pd.get_dummies(df_ml[target], drop_first=True).iloc[:, 0] if df_ml[target].dtype == 'object' else df_ml[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
   
    modelos_config = {
        'XGBoost_Deep': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
            'params': {
                'n_estimators': [100, 300, 500],           # Más árboles para capturar sutilezas
                'max_depth': [3, 4, 5, 6],                # Exploramos profundidades intermedias
                'learning_rate': [0.01, 0.05, 0.1],       # Tasas más bajas requieren más árboles pero son más precisas
                'subsample': [0.7, 0.8, 0.9],             # Muestreo aleatorio para reducir varianza
                'colsample_bytree': [0.7, 0.8],           # Selección de columnas por árbol
                'gamma': [0, 0.1, 0.2]                    # Regularización para evitar sobreajuste
            }
        }
    }

    print("Iniciando Optimización Profunda...")
    print("Esto puede tardar unos minutos debido al volumen de combinaciones...")

    for nombre, config in modelos_config.items():
        
        grid = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=5, 
            scoring='roc_auc', 
            n_jobs=-1, 
            verbose=1
        )
        grid.fit(X_train_scaled, y_train)
        
        # Evaluación
        mejor_modelo = grid.best_estimator_
        auc_test = roc_auc_score(y_test, mejor_modelo.predict_proba(X_test_scaled)[:, 1])
        
        print("\n" + "="*50)
        print(f"RESULTADOS OPTIMIZACIÓN PROFUNDA: {nombre}")
        print("="*50)
        print(f"Mejor AUC en Validación (CV): {grid.best_score_:.4f}")
        print(f"AUC Final en Test Set: {auc_test:.4f}")
        print(f"Mejores Hiperparámetros encontrados:")
        for param, value in grid.best_params_.items():
            print(f"  - {param}: {value}")
        
        print("\nReporte Detallado:")
        print(classification_report(y_test, mejor_modelo.predict(X_test_scaled)))

if __name__ == "__main__":
    optimizacion_profunda()