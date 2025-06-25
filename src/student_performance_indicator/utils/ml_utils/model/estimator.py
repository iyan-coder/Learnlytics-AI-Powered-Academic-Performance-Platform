import os
import sys

import pandas as pd

from student_performance_indicator.exception.exception import (
    StudentPerformanceException,
)
from student_performance_indicator.logger.logger import logger


class NetworkModel:
    """
    A minimal, production-ready wrapper for combining a preprocessor and a trained model.
    This version does NOT rely on feature column names and works with both raw and transformed input.
    """

    def __init__(self, preprocessor, model):
        """
        Args:
            preprocessor: A fitted preprocessing pipeline (e.g., ColumnTransformer)
            model: Trained machine learning model

        """
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            logger.error("Failed to initialize NetworkModel", exc_info=True)
            raise StudentPerformanceException(e, sys)

    def predict(self, x):
        """
        Accepts anything array-like or DataFrame-like and returns predictions.
        """
        try:
            # ensure 2-D DataFrame; we ignore external headers
            if not isinstance(x, pd.DataFrame):
                x = pd.DataFrame(x)

            x_transformed = self.preprocessor.transform(x)
            return self.model.predict(x_transformed)

        except Exception as e:
            logger.error("Prediction failed", exc_info=True)
            raise StudentPerformanceException(e, sys)
