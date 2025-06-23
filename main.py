"""
Main pipeline runner script for the Network Security ML project.

This script sequentially executes the pipeline stages:
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation

Each stage uses its respective configuration and component class.
Errors are logged and raised as custom exceptions.

Author: Your Name
Date: YYYY-MM-DD
"""

# Import necessary libraries
import os  # For handling file paths
import sys  # For handling system-level errors
import numpy as np  # For numerical operations
import pandas as pd  # For working with tabular data

# Import pipeline components
from student_performance_indicator.constant import training_pipeline
from student_performance_indicator.components.data_ingestion import DataIngestion

# Import configuration entities
from student_performance_indicator.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig
)
# Custom error and logging handling
from student_performance_indicator.exception.exception import StudentPerformanceException # Custom exception for standardized error handling
from student_performance_indicator.logger.logger import logger  # Logger for tracking pipeline execution



if __name__ == "__main__":
    try:
        # ===============================
        # Step 1: Initialize Configuration
        # ===============================
        # Create the main training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()

        # ====================
        # Step 2: Data Ingestion
        # ====================
        logger.info("Starting data ingestion...")
        # Create and run the data ingestion process
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logger.info("Data ingestion completed.")
        print(data_ingestion_artifact)  # Show what files/paths were created
        
    except Exception as e:
        # Catch and log any error that happens during the pipeline
        logger.error("Pipeline execution failed.", exc_info=True)
        raise StudentPerformanceException(e,sys)
