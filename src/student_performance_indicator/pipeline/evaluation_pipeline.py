# src/student_performance_indicator/pipelines/model_evaluation_pipeline.py
import os
import sys

import mlflow
import pandas as pd

from student_performance_indicator.components.model_evaluator import ModelEvaluator
from student_performance_indicator.constant.training_pipeline import TARGET_COLUMN
from student_performance_indicator.exception.exception import (
    StudentPerformanceException,
)
from student_performance_indicator.logger.logger import logger
from student_performance_indicator.utils.main_utils.utils import get_latest_artifact_dir

if os.getenv("RUNNING_IN_DOCKER") == "1":
    mlflow.set_tracking_uri("http://mlflow:5000")  # inside container
else:
    mlflow.set_tracking_uri("http://localhost:5001")  # running locally
mlflow.set_experiment("StudentPerformance")


class ModelEvaluationPipeline:
    """
    Stand-alone pipeline that:
        1. Grabs the most-recent artefact folder.
        2. Loads the *validated* train / test CSV files (they keep original column names).
        3. Loads the most-recent trained model (model.pkl).
        4. Evaluates that model on both sets and returns / logs the metrics.
    """

    def __init__(self) -> None:
        try:
            # ────────────────────────────────────────────────────────────────────
            #  Locate latest artefact directory
            # ────────────────────────────────────────────────────────────────────
            latest_artifact_dir = get_latest_artifact_dir()
            if latest_artifact_dir is None:
                raise FileNotFoundError(
                    "No artefact directory found has the pipeline run?"
                )

            # Paths to the validated (raw) CSV files
            self.valid_train_path = os.path.join(
                latest_artifact_dir, "data_validation", "validated", "train.csv"
            )
            self.valid_test_path = os.path.join(
                latest_artifact_dir, "data_validation", "validated", "test.csv"
            )

            # Path to the trained model that ModelTrainer saved
            self.model_path = os.path.join(
                latest_artifact_dir, "model_trainer", "trained_model", "model.pkl"
            )

        except Exception as e:
            logger.error("Failed to initialise ModelEvaluationPipeline", exc_info=True)
            raise StudentPerformanceException(e, sys)

    # ──────────────
    #  Main entry-point
    # ─────────────────
    def run_evaluation_pipeline(self):
        try:
            logger.info("Starting standalone model-evaluation pipeline")

            # ──────────────────────
            # 1. Load validated data
            # ──────────────────────
            train_df = pd.read_csv(self.valid_train_path, encoding="ISO-8859-1")
            test_df = pd.read_csv(self.valid_test_path, encoding="ISO-8859-1")

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # ────────────────────────
            # 2. Evaluate latest model
            # ────────────────────────
            evaluator = ModelEvaluator(
                data_transformation_artifact=None,  # not needed in load-only mode
                model_path=self.model_path,
                mode="load_and_evaluate",
            )

            best_model, train_metric, test_metric = evaluator.evaluate(
                X_train, y_train, X_test, y_test
            )

            # ────────────────────────
            #  3. Log & return results
            # ────────────────────────
            logger.info(
                f" Model evaluation complete\n"
                f"   • Train R² : {train_metric.r2_score:.4f}\n"
                f"   • Test  R² : {test_metric.r2_score:.4f}"
            )

            return best_model, train_metric, test_metric

        except Exception as e:
            logger.error("Model-evaluation pipeline failed", exc_info=True)
            raise StudentPerformanceException(e, sys)
