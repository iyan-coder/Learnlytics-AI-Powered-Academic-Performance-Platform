# Importing necessary classes and functions
import os
import sys

from sklearn.metrics import mean_absolute_error  # Sklearn regression metrics
from sklearn.metrics import mean_squared_error, r2_score

from student_performance_indicator.entity.artifact_entity import (
    RegressionMetricArtifact,
)  # Custom class to store regression metric results
from student_performance_indicator.exception.exception import (
    StudentPerformanceException,
)
from student_performance_indicator.logger.logger import logger


def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    """
    Calculates common regression metrics (R², RMSE, MAE).

    Args:
        y_true (np.ndarray): Ground-truth numeric targets.
        y_pred (np.ndarray): Predicted numeric targets.

    Returns:
        RegressionMetricArtifact: Object holding r2, rmse, and mae scores.
    """
    try:
        # R² score — proportion of variance explained
        model_r2_score = r2_score(y_true, y_pred)

        # Root Mean Squared Error (penalises larger errors)
        model_root_mean_squared_error = mean_squared_error(
            y_true, y_pred, squared=False
        )

        # Mean Absolute Error (average magnitude of errors)
        model_mean_absolute_error = mean_absolute_error(y_true, y_pred)

        # Pack metrics into dataclass
        regression_metric = RegressionMetricArtifact(
            r2_score=model_r2_score,
            root_mean_squared_error=model_root_mean_squared_error,
            mean_absolute_error=model_mean_absolute_error,
        )

        return regression_metric

    except Exception as e:
        logger.error("Failed to compute regression metrics", exc_info=True)
        raise StudentPerformanceException(e, sys)
