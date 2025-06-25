import os
import pickle
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from student_performance_indicator.constant.training_pipeline import SCHEMA_FILE_PATH
from student_performance_indicator.exception.exception import (
    StudentPerformanceException,
)
from student_performance_indicator.logger.logger import logger

# ============================
# EVALUATION_MODELS
# ============================


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, BaseEstimator],
    param: Dict[str, Dict[str, Any]],
    skip_training: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, BaseEstimator]]:
    """
    Evaluate multiple models, optionally perform hyperparameter tuning.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, BaseEstimator]]:
            - model_report: metrics like accuracy and best params for each model.
            - trained_models: the trained (or reused) estimators.
    """
    report: Dict[str, Dict[str, Any]] = {}
    trained_models: Dict[str, BaseEstimator] = {}

    try:
        logger.info(" Starting model evaluation...")
        print("Starting model evaluation...")

        # Debug: Show input shapes and model info
        # Debug: Log input shapes and model/meta information
        logger.info(f"X_train.shape: {X_train.shape}")
        logger.info(f"y_train.shape: {y_train.shape}")
        logger.info(f"X_test.shape: {X_test.shape}")
        logger.info(f"y_test.shape: {y_test.shape}")
        logger.info(f"Models received: {list(models.keys())}")
        logger.info(f"Params received: {list(param.keys())}")

        for model_name, model in models.items():
            logger.info(f" Evaluating model: {model_name}")

            if skip_training:
                logger.info(f"Skipping training for: {model_name}")
                y_test_pred = model.predict(X_test)
                model_r2_score = r2_score(y_test, y_test_pred)

                report[model_name] = {
                    "model_r2_score": model_r2_score,
                    "best_params": "Skipped",
                }
                trained_models[model_name] = model
                logger.info(f"Accuracy for {model_name}: {model_r2_score:.4f}")

            else:
                # Safety: Ensure parameter grid exists
                if model_name not in param:
                    raise StudentPerformanceException(
                        f"Missing parameter grid for model: '{model_name}'", sys
                    )

                logger.info(f" Running GridSearchCV for: {model_name}")
                gs = GridSearchCV(model, param[model_name], cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                best_model.fit(X_train, y_train)

                y_test_pred = best_model.predict(X_test)
                model_r2_score = r2_score(y_test, y_test_pred)

                report[model_name] = {
                    "model_r2_score": model_r2_score,
                    "best_params": gs.best_params_,
                }
                trained_models[model_name] = best_model
                logger.info(f"Best accuracy for {model_name}: {model_r2_score:.4f}")

        logger.info("All models evaluated successfully.")
        return report, trained_models

    except Exception as e:
        logger.error("Model evaluation failed.", exc_info=True)
        print("Exception in evaluate_models():", e)
        raise StudentPerformanceException(e, sys)
