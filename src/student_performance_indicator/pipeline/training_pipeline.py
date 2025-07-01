# Import necessary libraries
import os  # For handling file paths
import sys  # For handling system-level errors


import mlflow
import numpy as np  # For numerical operations
import pandas as pd  # For working with tabular data

from student_performance_indicator.components.data_ingestion import DataIngestion
from student_performance_indicator.components.data_tranformation import (
    DataTransformation,
)
from student_performance_indicator.components.data_validation import DataValidation
from student_performance_indicator.components.model_evaluator import ModelEvaluator

# Import pipeline components
from student_performance_indicator.components.model_trainer import ModelTrainer

# Utility functions
from student_performance_indicator.constant.training_pipeline import TARGET_COLUMN
from student_performance_indicator.entity.artifact_entity import (
    DataIngetionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
)

# Import configuration entities
from student_performance_indicator.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)

# Custom error and logging handling
from student_performance_indicator.exception.exception import (
    StudentPerformanceException,
)
from student_performance_indicator.logger.logger import logger
from student_performance_indicator.utils.main_utils.utils import (
    get_latest_artifact_dir,
    load_object,
)

import mlflow, os
if os.getenv("RUNNING_IN_DOCKER") == "1":
    mlflow.set_tracking_uri("http://mlflow:5000")      # inside container
else:
    mlflow.set_tracking_uri("http://localhost:5001")   # running locally
mlflow.set_experiment("StudentPerformance")


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logger.info("start data Ingestion")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artfifact = data_ingestion.initiate_data_ingestion()
            logger.info(
                f"Data Ingestion completed and artifact: {data_ingestion_artfifact}"
            )
            return data_ingestion_artfifact
        except Exception as e:
            logger.error("Error!, Data ingestion failed", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def start_data_validation(self, data_ingestion_artifact=DataIngetionArtifact):
        try:
            self.data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logger.info("start data validation")
            data_validation = DataValidation(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info(
                f"Data validation completed and artifact: {data_validation_artifact}"
            )
            return data_validation_artifact
        except Exception as e:
            logger.error("Error!, Data validation failed", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def start_data_transfomation(self, data_validation_artifact=DataValidationArtifact):
        try:
            self.data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logger.info("start data transfomation")
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logger.info(
                f"Data transformation completed and artifact: {data_transformation_artifact}"
            )
            return data_transformation_artifact
        except Exception as e:
            logger.error("Error!, Data transformation failed", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact=DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logger.info("start model trainer")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logger.info(
                f"""
                Model Trainer completed successfully.
                Model path: {model_trainer_artifact.trained_model_file_path}
                Train r2_score: {model_trainer_artifact.train_metric_artifact.r2_score}
                Test r2_score: {model_trainer_artifact.test_metric_artifact.r2_score}
                """
            )
            return model_trainer_artifact
        except Exception as e:
            logger.error("Error!, Model trainer failed", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def run_training_pipeline(self):
        try:
            mlflow.set_experiment("StudentPerformanceIndicator")
            mlflow.autolog(disable=True)

            with mlflow.start_run(run_name="Full_Training_Pipeline") as run:
                run_id = run.info.run_id
                logger.info(f"MLflow run started with run_id: {run_id}")

                data_ingestion_artifact = self.start_data_ingestion()
                data_validation_artifact = self.start_data_validation(
                    data_ingestion_artifact
                )
                data_transformation_artifact = self.start_data_transfomation(
                    data_validation_artifact
                )
                model_trainer_artifact = self.start_model_trainer(
                    data_transformation_artifact
                )

                logger.info("Run pipeline completed")
            return model_trainer_artifact

        except Exception as e:
            logger.error("failed to run pipeline", exc_info=True)
            raise StudentPerformanceException(e, sys)
