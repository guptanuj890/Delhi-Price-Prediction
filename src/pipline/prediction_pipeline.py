import sys
from src.entity.config_entity import HousePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
import pandas as pd
from pandas import DataFrame


class HouseData:
    def __init__(self, Area, BHK, Bathroom, Furnishing, Locality, Parking, Status, Transaction, Type):
        self.Area = Area
        self.BHK = BHK
        self.Bathroom = Bathroom
        self.Furnishing = Furnishing
        self.Locality = Locality
        self.Parking = Parking
        self.Status = Status
        self.Transaction = Transaction
        self.Type = Type

    def to_dataframe(self):
        return pd.DataFrame([{
            "Area": self.Area,
            "BHK": self.BHK,
            "Bathroom": self.Bathroom,
            "Furnishing": self.Furnishing,
            "Locality": self.Locality,
            "Parking": self.Parking,
            "Status": self.Status,
            "Transaction": self.Transaction,
            "Type": self.Type
        }])

    def to_dict(self):
        """
        Returns input data as a dictionary
        """
        logging.info("Creating input dictionary from HouseData")
        try:
            input_data = {
                "Area": [self.Area],
                "BHK": [self.BHK],
                "Bathroom": [self.Bathroom],
                "Furnishing": [self.Furnishing],
                "Locality": [self.Locality],
                "Parking": [self.Parking],
                "Status": [self.Status],
                "Transaction": [self.Transaction],
                "Type": [self.Type]
            }
            return input_data
        except Exception as e:
            raise MyException(e, sys) from e



from src.utils.main_utils import load_object  # Your local model loading utility

class HousePricePredictor:
    def __init__(self, prediction_pipeline_config: HousePredictorConfig = HousePredictorConfig()):
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self.model = load_object(self.prediction_pipeline_config.model_path)
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> list:
        try:
            logging.info("Predicting using locally loaded model")
            return self.model.predict(dataframe)
        except Exception as e:
            raise MyException(e, sys)

