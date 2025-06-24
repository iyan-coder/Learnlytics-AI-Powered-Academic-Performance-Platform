# Basic imports to handle files, system errors, and data tables
import os  # Helps us work with file paths and folders
import sys  # Helps with system-specific error messages
import pandas as pd  # To load and work with data tables (DataFrames)
import numpy as np  # Helpful for working with numbers
from scipy.stats import ks_2samp  # Used for checking if two sets of numbers are similar (drift check)
from scipy.stats import chi2_contingency

from student_performance_indicator.entity.artifact_entity import (
    DataIngetionArtifact,
    DataValidationArtifact
)
from student_performance_indicator.entity.config_entity import DataValidationConfig
from student_performance_indicator.constant.training_pipeline import SCHEMA_FILE_PATH
from student_performance_indicator.logger.logger import logger
from student_performance_indicator.exception.exception import StudentPerformanceException

from student_performance_indicator.utils.main_utils.utils import read_yaml_file
from student_performance_indicator.utils.main_utils.utils import write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngetionArtifact, data_validation_config: DataValidationConfig):
        """
        Initializes the DataValidation class with ingestion artifacts and validation config.
        Loads schema from YAML file.
        """
        try:
            logger.info("Initializing DataValidation...")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logger.info("DataValidation initialized successfully.")
        except Exception as e:
            logger.error("Error during DataValidation initialization", exc_info=True)
            raise StudentPerformanceException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Reads a CSV file and returns a DataFrame."""
        try:
            logger.info(f"Reading data from file: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read file", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """Checks if the number of columns matches the schema."""
        try:
            logger.info("Validating number of columns...")
            expected_columns = self._schema_config["columns"]
            result = len(dataframe.columns) == len(expected_columns)
            logger.info(f"Validation result for number of columns: {result}")
            return result
        except Exception as e:
            logger.error("Failed to validate number of columns", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def validate_schema_columns_and_types(self, dataframe: pd.DataFrame) -> bool:
        """Checks if required columns exist and datatypes match the schema."""
        try:
            logger.info("Validating schema column names and data types...")
            expected_columns = self._schema_config["columns"]
            required_columns = self._schema_config["required_columns"]

            for col in required_columns:
                if col not in dataframe.columns:
                    raise StudentPerformanceException(f"Missing required column: {col}", sys)

            for col, expected_dtype in expected_columns.items():
                if col in dataframe.columns:
                    actual_dtype = str(dataframe[col].dtype)
                    if expected_dtype != actual_dtype:
                        raise StudentPerformanceException(
                            f"Column '{col}' expected type {expected_dtype} but got {actual_dtype}", sys)

            logger.info("Schema column and data type validation completed successfully.")
            return True
        except Exception as e:
            logger.error("Schema validation failed", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops duplicate rows in the DataFrame."""
        try:
            logger.info("Checking and dropping duplicate rows...")
            before = df.shape[0]
            df = df.drop_duplicates()
            after = df.shape[0]
            logger.info(f"Duplicates removed: {before - after}")
            return df
        except Exception as e:
            logger.error("Failed to drop duplicates", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def check_nulls(self, df: pd.DataFrame) -> None:
        """Logs a warning if missing values are found, but does not raise."""
        try:
            logger.info("Checking for missing/null values...")
            null_report = df.isnull().sum()
            total_nulls = null_report.sum()

            if total_nulls > 0:
                logger.warning(f"Missing values detected in columns:\n{null_report[null_report > 0]}")
            else:
                logger.info("No missing values found.")
        except Exception as e:
            logger.error("Null check failed", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def is_numerical_columns_exist(self, dataframe: pd.DataFrame) -> bool:
        """Checks if numerical columns exist in the DataFrame."""
        numerical_cols = dataframe.select_dtypes(include=['number']).columns
        if len(numerical_cols) > 0:
            logger.info(f"Numerical columns found: {list(numerical_cols)}")
            return True
        else:
            logger.warning("No numerical columns found.")
            return False

    from scipy.stats import chi2_contingency

    def check_categorical_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, schema: dict) -> dict:
        """
        Checks for categorical drift using the Chi-squared test for each categorical column.
        Logs the drift report with p-values and drift status.
        """
        try:
            logger.info("Checking for categorical feature drift using Chi-squared test...")
            drift_report = {}

            for col in schema.get("categorical_columns", []):
                if col in base_df.columns and col in current_df.columns:
                    base_counts = base_df[col].value_counts()
                    current_counts = current_df[col].value_counts()
                    categories = sorted(set(base_counts.index).union(set(current_counts.index)))

                    base_freq = [base_counts.get(cat, 0) for cat in categories]
                    current_freq = [current_counts.get(cat, 0) for cat in categories]

                    if sum(base_freq) > 0 and sum(current_freq) > 0:
                        chi2, p_value, _, _ = chi2_contingency([base_freq, current_freq])
                        drift_report[col] = {
                            "p_value": float(p_value),
                            "drift_status": bool(p_value < 0.05),
                            "base_distribution": dict(base_counts),
                            "current_distribution": dict(current_counts)
                        }
                        logger.info(f"Categorical drift check completed for column: {col} | p-value: {p_value:.4f}")
                    else:
                        logger.warning(f"Skipped drift check for '{col}' due to insufficient data.")
                else:
                    logger.warning(f"Column '{col}' not found in both datasets. Skipping.")

            logger.info("Categorical drift detection completed.")
            return drift_report

        except Exception as e:
            logger.error("Categorical drift detection failed", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def detect_numerical_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """Performs drift detection on numerical columns using KS test, adds categorical drift as well."""
        try:
            logger.info("Detecting Numerical drift...")
            validation_status = True
            drift_report = {}

            assert base_df.columns.equals(current_df.columns), "Train and test columns do not match"

            for column in base_df.columns:
                if pd.api.types.is_numeric_dtype(base_df[column]):
                    ks_result = ks_2samp(base_df[column].dropna(), current_df[column].dropna())
                    p_value = ks_result.pvalue
                    drift_detected = p_value < threshold
                    drift_report[column] = {
                        "p_value": float(p_value),
                        "drift_status": bool(drift_detected)
                    }
                    if drift_detected:
                        validation_status = False
                    logger.info(f"Numerical drift check completed for column: {column} | p-value: {p_value:.4f}")

            # Save numerical drift
            drift_report_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)

            # Add categorical drift
            categorical_drift = self.check_categorical_drift(base_df, current_df, self._schema_config)
            drift_report.update(categorical_drift)

            write_yaml_file(file_path=drift_report_path, content=drift_report)
            logger.info("Numerical drift detection completed.")
            return validation_status

        except Exception as e:
            logger.error("Failed to detect dataset drift", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Runs the complete data validation pipeline and returns the artifact."""
        try:
            logger.info("Starting data validation process...")

            # Step 1: Load train/test files
            train_df = DataValidation.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = DataValidation.read_data(self.data_ingestion_artifact.test_file_path)

            # Step 2: Schema validation
            if not self.validate_number_of_columns(train_df) or not self.validate_schema_columns_and_types(train_df):
                raise StudentPerformanceException("Train dataframe schema invalid.", sys)
            if not self.validate_number_of_columns(test_df) or not self.validate_schema_columns_and_types(test_df):
                raise StudentPerformanceException("Test dataframe schema invalid.", sys)

            # Step 3: Cleanup (duplicates + nulls)
            train_df = self.drop_duplicates(train_df)
            test_df = self.drop_duplicates(test_df)

            self.check_nulls(train_df)
            self.check_nulls(test_df)

            # Step 4: Ensure numeric features exist
            if not self.is_numerical_columns_exist(train_df):
                raise StudentPerformanceException("Train dataframe lacks numerical columns.", sys)
            if not self.is_numerical_columns_exist(test_df):
                raise StudentPerformanceException("Test dataframe lacks numerical columns.", sys)

            # Step 5: Drift check
            validation_status = self.detect_numerical_drift(train_df, test_df)

            # Step 6: Save validated data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            logger.info("Data validation completed successfully.")

            # Step 7: Return artifact
            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

        except Exception as e:
            logger.error("Data validation process failed.", exc_info=True)
            raise StudentPerformanceException(e, sys)
