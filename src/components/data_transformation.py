import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['Per_Sqft'].fillna((df['Per_Sqft'].mean()), inplace=True)
            df['Bathroom'].fillna(df['Bathroom'].mode()[0], inplace=True)
            df['Furnishing'].fillna(df['Furnishing'].mode()[0], inplace=True)
            df['Parking'].fillna(df['Parking'].mode()[0], inplace=True)
            df['Type'].fillna(df['Type'].mode()[0], inplace=True)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['Locality'] = df['Locality'].apply(self.grp_loc)
            return df
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def grp_loc(locality: str) -> str:
        try:
            locality = str(locality).lower()
            if 'rohini' in locality:
                return 'Rohini Sector'
            elif 'dwarka' in locality:
                return 'Dwarka Sector'
            elif 'shahdara' in locality:
                return 'Shahdara'
            elif 'vasant' in locality:
                return 'Vasant Kunj'
            elif 'paschim' in locality:
                return 'Paschim Vihar'
            elif 'alaknanda' in locality:
                return 'Alaknanda'
            elif 'vasundhara' in locality:
                return 'Vasundhara Enclave'
            elif 'punjabi' in locality:
                return 'Punjabi Bagh'
            elif 'kalkaji' in locality:
                return 'Kalkaji'
            elif 'lajpat' in locality:
                return 'Lajpat Nagar'
            elif 'laxmi' in locality:
                return 'Laxmi Nagar'
            elif 'patel' in locality:
                return 'Patel Nagar'
            else:
                return 'Other'
        except Exception as e:
            raise MyException(e, sys)

    
    def read_data(self, file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)


    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            drop_cols = self._schema_config.get('drop_columns', [])
            logging.info(f"Dropping columns: {drop_cols}")
            df = df.drop(columns=drop_cols, errors='ignore')
            return df
        except Exception as e:
            raise MyException(e, sys)

    def _encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            cat_cols = self._schema_config.get('categorical_columns', [])
            logging.info(f"Encoding categorical columns: {cat_cols}")
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            return df
        except Exception as e:
            raise MyException(e, sys)

    # def get_data_transformer_object(self) -> ColumnTransformer:
    #     try:
    #         num_cols = self._schema_config.get('num_features', [])
    #         logging.info(f"Scaling numerical columns: {num_cols}")
    #         pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    #         transformer = ColumnTransformer(transformers=[
    #             ("num", pipeline, num_cols)
    #         ], remainder="passthrough")
    #         return transformer
    #     except Exception as e:
    #         raise MyException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            num_cols = self._schema_config.get('num_features', [])
            cat_cols = self._schema_config.get('categorical_columns', [])

            logging.info(f"Numerical columns: {num_cols}")
            logging.info(f"Categorical columns: {cat_cols}")

            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols)
            ])

            return preprocessor
        except Exception as e:
            raise MyException(e, sys)

    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test datasets
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Drop target column and prepare inputs
            input_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_train_df = train_df[TARGET_COLUMN]

            input_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_test_df = test_df[TARGET_COLUMN]

            # Preprocessing pipeline
            for df in [input_train_df, input_test_df]:
                self._impute_missing_values(df)
                self._feature_engineering(df)
                self._drop_columns(df)
                #self._encode_categorical_columns(df)

            # Create transformer
            preprocessor = self.get_data_transformer_object()

            # Fit on train, transform both
            input_train_arr = preprocessor.fit_transform(input_train_df)
            input_test_arr = preprocessor.transform(input_test_df)

            # Combine features and targets
            train_arr = np.c_[input_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_test_arr, np.array(target_test_df)]

            # Save artifacts
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys)

