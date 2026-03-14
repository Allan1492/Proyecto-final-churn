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
        "SVR",
        "KNN Regressor"
    ]
}

MODEL_PARAMS = {

    "Logistic Regression": {
        "basic": {
            "C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0},
            "max_iter": {"type": "int", "min": 50, "max": 500, "default": 100}
        }
    },

    "Random Forest Classifier": {
        "basic": {
            "n_estimators": {"type": "int", "min": 10, "max": 300, "default": 100},
            "max_depth": {"type": "int", "min": 2, "max": 50, "default": 10}
        }
    }
}