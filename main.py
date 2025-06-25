import sys

import numpy as np

from student_performance_indicator.exception.exception import (
    StudentPerformanceException,
)
from student_performance_indicator.logger.logger import logger
from student_performance_indicator.pipeline.evaluation_pipeline import (
    ModelEvaluationPipeline,
)
from student_performance_indicator.pipeline.training_pipeline import TrainingPipeline


def run_pipeline():
    try:
        logger.info(">>>> Starting the Training Pipeline <<<<")
        training_pipeline = TrainingPipeline()
        model_trainer_artifact = training_pipeline.run_training_pipeline()
        logger.info(">>>> Training Pipeline completed successfully <<<<")

        logger.info(">>>> Starting the Model Evaluation Pipeline <<<<")
        evaluation_pipeline = ModelEvaluationPipeline()
        evaluation_pipeline.run_evaluation_pipeline()
        logger.info(">>>> Model Evaluation completed successfully <<<<")

    except StudentPerformanceException as e:
        logger.error("StudentPerformanceException.", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("An unexpected error occurred.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
