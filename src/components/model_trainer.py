import sys
from typing import Tuple

import numpy as np
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact
)
from src.entity.estimator import MyModel


def tune_model_with_mlflow(
    model,
    param_grid: dict,
    x_train,
    y_train,
    search_type="random",
    n_iter=10,
    scoring="r2",
    cv=3
) -> Tuple[object, dict, float]:
    """
    Performs hyperparameter tuning and logs each trial to MLflow.
    """
    try:
        if search_type == "random":
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, scoring=scoring, cv=cv,
                verbose=1, n_jobs=-1, random_state=42
            )
        else:
            raise ValueError("Only 'random' search is supported in this helper.")

        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        # Log each param to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_score", best_score)

        return best_model, best_params, best_score

    except Exception as e:
        raise MyException(e, sys)


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            print("------------------------------------------------------------")
            print("Starting Model Trainer Component")

            # Load data
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Define param grid
            param_grid = {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [5, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2, 3]
            }

            # Set MLflow URI and experiment
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("DelhiHousePricePrediction")

            with mlflow.start_run():
                logging.info("Started MLflow run for model training")

                base_model = RandomForestRegressor(
                    random_state=self.model_trainer_config._random_state
                )

                tuned_model, best_params, best_cv_score = tune_model_with_mlflow(
                    model=base_model,
                    param_grid=param_grid,
                    x_train=x_train,
                    y_train=y_train,
                    search_type="random",
                    n_iter=10
                )

                # Evaluate
                y_test_pred = tuned_model.predict(x_test)
                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)

                train_r2 = r2_score(y_train, tuned_model.predict(x_train))

                # Check R2 threshold
                if train_r2 < self.model_trainer_config.expected_accuracy:
                    raise Exception("Model did not meet expected R2 threshold")

                # Log metrics
                mlflow.log_metric("train_r2", train_r2)
                mlflow.log_metric("test_r2", test_r2)
                mlflow.log_metric("test_mae", test_mae)
                mlflow.log_metric("test_mse", test_mse)

                # Save model with preprocessing
                preprocessing_obj = load_object(
                    file_path=self.data_transformation_artifact.transformed_object_file_path)
                final_model = MyModel(preprocessing_object=preprocessing_obj,
                                      trained_model_object=tuned_model)

                save_object(self.model_trainer_config.trained_model_file_path, final_model)

                # Optional: log model
                # mlflow.sklearn.log_model(final_model, artifact_path="model")

                logging.info("Model training, evaluation and logging complete.")

                return ModelTrainerArtifact(
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                    metric_artifact=RegressionMetricArtifact(
                        r2_score=test_r2,
                        mse=test_mse,
                        mae=test_mae
                    ),
                    preprocessed_object_file_path=self.data_transformation_artifact.transformed_object_file_path
                )

        except Exception as e:
            raise MyException(e, sys)
