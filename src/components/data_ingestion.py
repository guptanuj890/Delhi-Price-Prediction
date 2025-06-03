import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split
import kaggle
import zipfile

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj_data import ProjData

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Downloads dataset from Kaggle and loads it into a DataFrame.
        """
        try:
            logging.info(f"Downloading dataset from Kaggle")

            # Set Kaggle environment variables (optional if ~/.kaggle/kaggle.json exists)
            os.environ['KAGGLE_USERNAME'] = self.data_ingestion_config.kaggle_username
            os.environ['KAGGLE_KEY'] = self.data_ingestion_config.kaggle_key

            dataset_name = self.data_ingestion_config.kaggle_dataset_name  # e.g. "zynicide/wine-reviews"
            download_path = self.data_ingestion_config.kaggle_download_dir

            os.makedirs(download_path, exist_ok=True)

            # Download the dataset
            import kaggle.api.kaggle_api_extended as kaggle_api
            kaggle_api = kaggle.api.kaggle_api_extended.KaggleApi()
            kaggle_api.authenticate()
            kaggle_api.dataset_download_files(dataset=dataset_name, path=download_path, unzip=True)

            # Find and read the CSV file (assuming single CSV file for simplicity)
            csv_file_path = next((os.path.join(download_path, f) for f in os.listdir(download_path) if f.endswith('.csv')), None)
            if csv_file_path is None:
                raise Exception("No CSV file found in Kaggle dataset")

            dataframe = pd.read_csv(csv_file_path)
            logging.info(f"Shape of dataframe: {dataframe.shape}")

            # Save to feature store
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False)

            return dataframe

        except Exception as e:
            raise MyException(e, sys)

    def split_data_as_train_test(self,dataframe: DataFrame) ->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()

            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e