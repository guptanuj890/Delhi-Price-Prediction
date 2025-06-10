import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from src.entity.estimator import MyModel
from src.entity.artifact_entity import ModelTrainerArtifact

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        try:
            # Split train and test into features and target
            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]
            logging.info("Train-test split done.")

            # Initialize RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=self.model_trainer_config._n_estimators,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                max_depth=self.model_trainer_config._max_depth,
                random_state=self.model_trainer_config._random_state
            )

            logging.info("Training RandomForestRegressor...")
            model.fit(x_train, y_train)
            logging.info("Model training complete.")

            # Predictions and metrics
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Creating regression metric artifact
            metric_artifact = RegressionMetricArtifact(mse=mse, mae=mae, r2_score=r2)
            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")

            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Loaded transformed train-test data.")

            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model training and evaluation done.")

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # Check if r2_score meets expected accuracy
            if r2_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expected_accuracy:
                logging.info("Model did not meet the expected R2 score threshold.")
                raise Exception("No model found with R2 score above the expected accuracy.")

            # Save the final model
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object including preprocessing and trained model.")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
                preprocessed_object_file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
