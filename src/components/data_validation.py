import json
import sys
import os
import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            expected_columns = self._schema_config["columns"]
            status = len(dataframe.columns) == len(expected_columns)
            logging.info(f"Column count match: {status}")
            return status
        except Exception as e:
            raise MyException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            expected_columns = set(self._schema_config["columns"].keys())
            missing_columns = expected_columns - set(df.columns)

            if missing_columns:
                logging.info(f"Missing columns: {missing_columns}")
                return False
            return True
        except Exception as e:
            raise MyException(e, sys)

    def validate_column_dtypes(self, df: DataFrame) -> bool:
        try:
            dtype_mismatches = []
            for column, expected_dtype in self._schema_config["columns"].items():
                if column not in df.columns:
                    continue

                actual_dtype = str(df[column].dtype)
                if expected_dtype == "category":
                    if not pd.api.types.is_object_dtype(df[column]):
                        dtype_mismatches.append((column, actual_dtype, expected_dtype))
                elif expected_dtype == "int":
                    if not pd.api.types.is_integer_dtype(df[column]):
                        dtype_mismatches.append((column, actual_dtype, expected_dtype))
                elif expected_dtype == "float":
                    if not pd.api.types.is_float_dtype(df[column]):
                        dtype_mismatches.append((column, actual_dtype, expected_dtype))

            if dtype_mismatches:
                logging.info(f"Column dtype mismatches found: {dtype_mismatches}")
                return False

            return True
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            for df, name in [(train_df, "train"), (test_df, "test")]:
                if not self.validate_number_of_columns(df):
                    validation_error_msg += f"{name} dataframe has incorrect number of columns. "

                if not self.is_column_exist(df):
                    validation_error_msg += f"{name} dataframe is missing required columns. "

                if not self.validate_column_dtypes(df):
                    validation_error_msg += f"{name} dataframe has incorrect data types. "

            validation_status = validation_error_msg.strip() == ""

            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            os.makedirs(os.path.dirname(self.data_validation_config.validation_report_file_path), exist_ok=True)
            with open(self.data_validation_config.validation_report_file_path, "w") as f:
                json.dump(validation_report, f, indent=4)

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg.strip(),
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            logging.info("Data validation completed.")
            logging.info(f"Validation result: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys)
