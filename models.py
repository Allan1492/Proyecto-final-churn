ALL_MODELS = {
    "Clasificación": [
        "Logistic Regression", 
        "Random Forest Classifier", 
        "XGBoost Classifier", 
        "SVM", 
        "KNN Classifier"
    ],
    "Regresión": [
        "Linear Regression", 
        "Random Forest Regressor", 
        "XGBoost Regressor", 
        "SVR"
    ],
    "Agrupamiento (Clustering)": [
        "K-Means", 
        "Agglomerative Clustering", 
        "DBSCAN"
    ]
}


MODEL_PARAMS = {
    "Logistic Regression": {
        "basic": {
            "C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0},
            "max_iter": {"type": "int", "min": 100, "max": 1000, "default": 100}
        }
    },
    "Random Forest Classifier": {
        "basic": {
            "n_estimators": {"type": "int", "min": 10, "max": 1000, "default": 100},
            "max_depth": {"type": "int", "min": 2, "max": 100, "default": 10},
            "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2},
            "criterion": {"type": "choice", "options": ["gini", "entropy"], "default": "gini"}
        }
    },
    "XGBoost Classifier": {
        "basic": {
            "learning_rate": {"type": "float", "min": 0.01, "max": 0.3, "default": 0.1},
            "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 100},
            "max_depth": {"type": "int", "min": 2, "max": 20, "default": 6}
        }
    },
    "SVM": {
        "basic": {
            "C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0},
            "kernel": {"type": "choice", "options": ["linear", "poly", "rbf", "sigmoid"], "default": "rbf"}
        }
    },
    "KNN Classifier": {
        "basic": {
            "n_neighbors": {"type": "int", "min": 1, "max": 50, "default": 5}
        }
    }
}