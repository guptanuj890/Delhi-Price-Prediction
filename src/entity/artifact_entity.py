from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    validation_report_file_path: str
    
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str
    
@dataclass
class RegressionMetricArtifact:
    mse: float
    mae: float
    r2_score: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: RegressionMetricArtifact
    preprocessed_object_file_path: str

    
@dataclass
class ModelEvaluationArtifact:
    trained_model_r2_score: float
    best_model_r2_score: float
    is_model_accepted: bool
    difference: float

    
# @dataclass
# class ModelPusherArtifact:
#     bucket_name:str
#     s3_model_path:str