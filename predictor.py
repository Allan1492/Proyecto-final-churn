from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd 

class Predictor:
    def __init__(self, df, selected_models, params):
        self.df = df
        self.models = selected_models
        self.params = params

    def train_all(self):
        # 1. TRUCO MÁGICO: Convertir texto a números (Encoding)
        # pd.get_dummies convierte 'Female'/'Male' en 0s y 1s automáticamente
        df_numerico = pd.get_dummies(self.df, drop_first=True)

        # 2. Ahora sí, separamos usando el nuevo dataframe numérico
        X = df_numerico.iloc[:, :-1]
        y = df_numerico.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 # Agregamos random_state para que sea estable
        )

        resultados = {}
        # ... resto de tu código (for model_name in self.models, etc.)

        for model_name in self.models:

            if model_name == "Logistic Regression":
                model = LogisticRegression(
                    C=self.params.get("C", 1.0),
                    max_iter=self.params.get("max_iter", 100)
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                resultados[model_name] = {
                    "accuracy": accuracy_score(y_test, pred)
                }

            elif model_name == "Random Forest Classifier":
                model = RandomForestClassifier(
                    n_estimators=self.params.get("n_estimators", 100),
                    max_depth=self.params.get("max_depth", 10)
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                resultados[model_name] = {
                    "accuracy": accuracy_score(y_test, pred)
                }

        return resultados