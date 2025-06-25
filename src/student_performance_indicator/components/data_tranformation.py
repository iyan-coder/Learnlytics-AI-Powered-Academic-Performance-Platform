import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from student_performance_indicator.constant.training_pipeline import TARGET_COLUMN
from student_performance_indicator.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from student_performance_indicator.entity.config_entity import DataTransformationConfig
from student_performance_indicator.exception.exception import (
    StudentPerformanceException,
)
from student_performance_indicator.logger.logger import logger
from student_performance_indicator.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object,
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            logger.info("Initializing DataTransformation component...")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            logger.error(
                "Error occurred during DataTransformation initialization", exc_info=True
            )
            raise StudentPerformanceException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logger.info(f"Reading data from: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Failed to read data from: {file_path}", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Returns a ColumnTransformer object with pipelines for numerical and categorical features.
        """
        try:
            logger.info("Preparing preprocessing pipelines...")

            # Read datasets to determine column types
            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            feature_train_df = train_df.drop(columns=[TARGET_COLUMN], errors="ignore")
            feature_test_df = test_df.drop(columns=[TARGET_COLUMN], errors="ignore")

            # Dynamically detect columns
            numerical_columns = feature_train_df.select_dtypes(
                include=["number"]
            ).columns.tolist()
            categorical_columns = feature_test_df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            logger.info(f"Detected numerical columns: {numerical_columns}")
            logger.info(f"Detected categorical columns: {categorical_columns}")

            # Define preprocessing steps
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            logger.info("Preprocessing pipelines successfully created.")
            return preprocessor

        except Exception as e:
            logger.error("Error while creating preprocessing object", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Initiating full data transformation process...")

            # Load datasets
            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            logger.info("Splitting data into features and target...")

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logger.info("Getting preprocessing object...")
            preprocessor = self.get_data_transformer_object()

            logger.info("Applying preprocessing on training and testing data...")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Build final arrays (features + target)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save feature columns for later use
            feature_columns = input_feature_train_df.columns.tolist()
            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.feature_columns_file_path
                ),
                exist_ok=True,
            )
            save_object(
                self.data_transformation_config.feature_columns_file_path,
                feature_columns,
            )
            logger.info(
                f"Feature column names saved at: {self.data_transformation_config.feature_columns_file_path}"
            )

            # Preprocessor pusher
            saved_final_preprocessor = save_object(
                "final_model/preprocessor.pkl", preprocessor
            )
            logger.info(
                f"Preprocessor.pkl is succesfully saved at {saved_final_preprocessor}"
            )

            logger.info("Saving transformed data and preprocessing object...")
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )

            logger.info("Data transformation completed successfully.")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                feature_columns_file_path=self.data_transformation_config.feature_columns_file_path,
            )

        except Exception as e:
            logger.error("Error during data transformation process", exc_info=True)
            raise StudentPerformanceException(e, sys)
