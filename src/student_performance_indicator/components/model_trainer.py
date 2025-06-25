# Import necessary Python libraries
import os  # To create folders and handle file paths
import sys  # For system-specific exception handling

import mlflow
import numpy as np
import pandas as pd  # Used to work with tabular data (DataFrames)

from student_performance_indicator.components.model_evaluator import (
    ModelEvaluator,
)  # Evaluates different models
from student_performance_indicator.constant.training_pipeline import TARGET_COLUMN
from student_performance_indicator.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from student_performance_indicator.entity.config_entity import ModelTrainerConfig
from student_performance_indicator.exception.exception import (
    StudentPerformanceException,
)  # Custom exception

# Project-specific imports
from student_performance_indicator.logger.logger import logger  # For logging messages
from student_performance_indicator.utils.main_utils.utils import (  # To save/load Python objects
    load_object,
    save_object,
)
from student_performance_indicator.utils.ml_utils.model.estimator import NetworkModel

# The name of the column we're predicting

# mlflow.set_registry_uri("https://dagshub.com/iyan-coder/networksecurity.mlflow")
#         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


# ModelTrainer handles training and saving the best ML model
class ModelTrainer:
    """
    This class:
    - Loads the transformed train/test data.
    - Trains models and finds the best one using ModelEvaluator.
    - Saves the best model wrapped with the preprocessor.
    - Returns an artifact with the model path and performance scores.
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        """
        Constructor sets up necessary configs and paths.

        Args:
            model_trainer_config: Where to save the trained model.
            data_transformation_artifact: Where to get transformed train/test data and preprocessor.
        """
        logger.info("Initializing ModelTrainer...")
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def _save_model(self, model, preprocessor):
        """
        Saves the trained model together with the preprocessor into a file.

        Args:
            model: Best trained model.
            preprocessor: The preprocessor used during data transformation.
        """
        try:
            logger.info("Saving the trained model with preprocessor...")

            # Create the directory for saving the model if it doesn't exist
            model_dir = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir, exist_ok=True)

            # feature_columns = load_object(self.data_transformation_artifact.feature_columns_file_path)
            # Wrap model with its preprocessor and column names
            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=model,
            )

            # Save the wrapped model to disk
            save_object(
                self.model_trainer_config.trained_model_file_path, obj=network_model
            )

            logger.info(
                f"Model saved successfully at: {self.model_trainer_config.trained_model_file_path}"
            )

        except Exception as e:
            logger.error("Failed to save model.", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Runs the full training process:
        - Loads transformed data
        - Splits into features and labels
        - Trains and evaluates models
        - Saves best model
        - Returns an artifact with results
        """
        try:
            logger.info("Loading transformed train and test data...")

            # Log basic params
            mlflow.log_param("pipeline_step", "ModelTrainer")
            mlflow.log_param(
                "model_storage_path", self.model_trainer_config.trained_model_file_path
            )

            train_arr = np.load(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = np.load(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            feature_columns = load_object(
                self.data_transformation_artifact.feature_columns_file_path
            )
            logger.info(f"feature_columns = {feature_columns}")

            logger.info(
                "Successfully split features and labels from transformed arrays."
            )

            # Use ModelEvaluator to find the best-performing model

            logger.info("Evaluating models to find the best one...")
            evaluator = ModelEvaluator(self.data_transformation_artifact)
            best_model, train_metric, test_metric = evaluator.evaluate(
                X_train, y_train, X_test, y_test
            )

            # Load the preprocessor used during data transformation
            logger.info("Loading data preprocessor used during transformation...")
            preprocessor = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # Save the model and preprocessor wrapped together
            logger.info("Saving the final trained model...")
            self._save_model(best_model, preprocessor)

            logger.info("Model training completed successfully.")

            saved_final_model = save_object("final_model/model.pkl", best_model)
            logger.info(f"Model.pkl successfully saved at {saved_final_model}")

            # Return model training results
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )

        except Exception as e:
            logger.error("Failed to initiate model trainer.", exc_info=True)
            raise StudentPerformanceException(e, sys)
