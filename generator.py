import pandas as pd

class StateManager:
    """ Manejo de historial para Undo/Redo """

    def __init__(self):
        self.history = []
        self.future = []

    def save(self, df):
        self.history.append(df.copy())
        self.future = []  # limpiar futuros cuando hay cambio nuevo

    def undo(self):
        if len(self.history) > 1:
            self.future.append(self.history.pop())
            return self.history[-1].copy()
        return None

    def redo(self):
        if self.future:
            df = self.future.pop()
            self.history.append(df.copy())
            return df
        return None


class DataManager:

    def __init__(self):
        self._df = None
        self._original_df = None  # Guardar copia del dataset original
        self.state = StateManager()

    def load_data(self, file):
        self._df = pd.read_csv(file)
        self._original_df = self._df.copy()  # Guardar copia del original
        self.state.save(self._df)

    def get_data(self):
        return self._df

    def update_data(self, df):
        self._df = df
        self._df.reset_index(drop=True, inplace=True)
        self.state.save(self._df)

    def undo(self):
        df = self.state.undo()
        if df is not None:
            self._df = df

    def redo(self):
        df = self.state.redo()
        if df is not None:
            self._df = df
            
    def reset_to_original(self):
        """Restablece el dataset a su estado original"""
        if self._original_df is not None:
            self._df = self._original_df.copy()
            self.state.save(self._df)  # Guardar el reset en el historial
            return True
        return False


class CleaningService:

    def __init__(self, df):
        self.df = df

    def remove_duplicates(self):
        return self.df.drop_duplicates()

    def drop_columns(self, cols):
        return self.df.drop(columns=cols)

    def fill_nulls(self, col, method):
        if method == "mean":
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif method == "median":
            self.df[col] = self.df[col].fillna(self.df[col].median())
        elif method == "mode":
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        return self.df

class ModelService:
    """ Esta clase es la que 'mueve las perillas' """
    def __init__(self, df):
        self.df = df

    def entrenar_modelo(self, nombre_modelo, params):
        # Aquí es donde se aplican los hiperparámetros de models.py
        if nombre_modelo == "Random Forest Classifier":
            from sklearn.ensemble import RandomForestClassifier
            modelo = RandomForestClassifier(**params)
            # Lógica de entrenamiento aquí...
            return modelo