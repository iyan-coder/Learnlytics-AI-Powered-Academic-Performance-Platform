import os
import sys
import numpy as np
import pandas as pd

"""
Module: training_pipeline.py

This module defines constants used throughout the Machine Learning pipeline
for a Network Security project. Centralizing constants improves maintainability, 
readability, and scalability. These values configure file names, paths, 
dataset handling, and database connectivity.
"""

# ============================================
# General Pipeline Configuration Constants
# ============================================

# Name of the entire ML pipeline (used in logs, directory naming, etc.)
PIPELINE_NAME: str = "StudentPerformanceIndicator"

# Root directory to store all pipeline artifacts (e.g., ingested data, models, logs)
ARTIFACT_DIR: str = "Artifacts"

# Name of the original CSV file expected as input
FILE_NAME: str = "studentperformance_data.csv"

# File names to be generated after train-test split
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# ============================================
# Data Ingestion Related Constants
# ============================================

# MongoDB collection from which data will be pulled
DATA_INGESTION_COLLECTION_NAME: str = "StudentPerformanceData"

# MongoDB database name
DATA_INGESTION_DATABASE_NAME: str = "Iyan-coder"

# Subdirectory inside artifacts folder for storing data ingestion outputs
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Directory to store raw processed data (after minor cleaning but before splitting)
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# Directory to store the train and test split datasets
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Ratio to split the dataset into train and test sets (0.2 = 20% test data)
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

