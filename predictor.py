from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd 

class Predictor:
    def __init__(self, df, selected_models, params):
        self.df = df
        self.models = selected_models
        self.params = params

    def train_all(self):
        df_numerico = pd.get_dummies(self.df, drop_first=True)
        X = df_numerico.iloc[:, :-1]
        y = df_numerico.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        resultados = {}

        for model_name in self.models:
         
            
            if model_name == "Logistic Regression":
                model = LogisticRegression(
                    C=self.params.get(f"{model_name}_C", 1.0),
                    max_iter=int(self.params.get(f"{model_name}_max_iter", 100))
                )

            elif model_name == "Random Forest Classifier":
                model = RandomForestClassifier(
                    n_estimators=int(self.params.get(f"{model_name}_n_estimators", 100)),
                    max_depth=int(self.params.get(f"{model_name}_max_depth", 10)),
                    min_samples_split=int(self.params.get(f"{model_name}_min_samples_split", 2)),
                    random_state=42
                )

            elif model_name == "XGBoost Classifier":
                model = XGBClassifier(
                    n_estimators=int(self.params.get(f"{model_name}_n_estimators", 100)),
                    learning_rate=self.params.get(f"{model_name}_learning_rate", 0.1),
                    max_depth=int(self.params.get(f"{model_name}_max_depth", 6)),
                    use_label_encoder=False,
                    eval_metric='logloss'
                )

            elif model_name == "SVM":
                model = SVC(
                    C=self.params.get(f"{model_name}_C", 1.0),
                    kernel=self.params.get(f"{model_name}_kernel", "rbf"),
                    probability=True # Necesario para calcular AUC
                )

            elif model_name == "KNN Classifier":
                model = KNeighborsClassifier(
                    n_neighbors=int(self.params.get(f"{model_name}_n_neighbors", 5))
                )
            
            else:
                continue 

            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            
            prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred
            
            resultados[model_name] = {
                "Accuracy": f"{accuracy_score(y_test, pred):.4f}",
                "AUC (ROC)": f"{roc_auc_score(y_test, prob):.4f}",
                "CV Stability": f"{cross_val_score(model, X, y, cv=5).mean():.4f}"
            }

        return resultados