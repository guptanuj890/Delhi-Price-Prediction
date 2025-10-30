import os
import sys
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Downloads dataset from Kaggle, unzips it, reads the CSV, and stores it in feature store path.
        """
        try:
            logging.info("Downloading dataset from Kaggle")

            # Set Kaggle API credentials (optional if ~/.kaggle/kaggle.json exists)
            kaggle_username = os.environ['KAGGLE_USERNAME']
            kaggle_key = os.environ['KAGGLE_KEY']

            dataset_name = self.data_ingestion_config.kaggle_dataset_name
            download_path = self.data_ingestion_config.kaggle_download_dir

            os.makedirs(download_path, exist_ok=True)

            # Download dataset from Kaggle
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(dataset=dataset_name, path=download_path, unzip=True)

            logging.info(f"Dataset downloaded and unzipped to {download_path}")

            # Read CSV file (assumes only one CSV file is present)
            csv_file_path = next((os.path.join(download_path, f) for f in os.listdir(download_path) if f.endswith('.csv')), None)
            if csv_file_path is None:
                raise Exception("No CSV file found in the dataset.")

            dataframe = pd.read_csv(csv_file_path)
            logging.info(f"DataFrame loaded with shape: {dataframe.shape}")

            # Save to feature store
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False)

            return dataframe

        except Exception as e:
            raise MyException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        try:
            logging.info("Splitting data into train and test sets")

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42 
            )

            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)

            logging.info("Train and test data saved successfully")
        except Exception as e:
            raise MyException(e, sys)


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion process")

            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe)

            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data ingestion completed with artifact: {artifact}")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
