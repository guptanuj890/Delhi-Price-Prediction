import sys
import os
import joblib  # or pickle if your model is saved with .pkl
from pandas import DataFrame
from src.exception import MyException
from src.entity.estimator import MyModel  # Ensure this wraps your model or can be replaced by the raw model class

class Proj1Estimator:
    """
    Handles loading and predicting using a local ML model file.
    """

    def __init__(self, model_path: str):
        """
        :param model_path: Local filesystem path to the trained model
        """
        self.model_path = model_path
        self.loaded_model: MyModel = None

    def is_model_present(self) -> bool:
        """
        Checks if the model file exists locally.
        """
        return os.path.exists(self.model_path)

    def load_model(self) -> MyModel:
        """
        Loads the model from a local file.
        """
        try:
            if not self.is_model_present():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            self.loaded_model = joblib.load(self.model_path)
            return self.loaded_model
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: DataFrame):
        """
        Predict using the loaded model.
        """
        try:
            if self.loaded_model is None:
                self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            raise MyException(e, sys)
