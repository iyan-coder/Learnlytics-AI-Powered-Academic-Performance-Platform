from datetime import datetime
import os

# Import constants for pipeline configuration (directory names, filenames, etc.)
from student_performance_indicator.constant import training_pipeline

# Optional: print pipeline metadata (can be removed in production)
print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)


# =============================================
# TrainingPipelineConfig: Manages base artifact paths
# =============================================
class TrainingPipelineConfig:
    """
    Configuration class to define base paths and naming for the ML training pipeline.
    It uses a timestamp to uniquely organize artifacts per pipeline execution.
    """

    def __init__(self, timestamp=datetime.now()):
        """
        Initialize pipeline configuration.

        Args:
            timestamp (datetime, optional): Defaults to current time.
                                             Used to generate unique directory for each run.
        """
        # Format timestamp to be filesystem-safe and human-readable
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        # Name of the pipeline (e.g., "network_security_pipeline")
        self.pipeline_name = training_pipeline.PIPELINE_NAME

        # Base name for artifact storage (e.g., "artifact")
        self.artifact_name = training_pipeline.ARTIFACT_DIR

        # Store formatted timestamp for downstream usage
        self.timestamp: str = timestamp

        # Complete path for artifact directory including timestamp
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)

        

# =========================================================
# DataIngestionConfig: Defines config for data ingestion
# =========================================================
class DataIngestionConfig:
    """
    Configuration class for the data ingestion phase.
    Specifies paths for raw data, feature store, and training/testing datasets.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize all required paths for data ingestion.

        Args:
            training_pipeline_config (TrainingPipelineConfig): Base pipeline config.
        """
        # Root folder for all ingestion outputs (e.g., artifact/05_06_2025_10_34_20/data_ingestion/)
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )

        # Full path to store the feature store data (processed raw data)
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME  # e.g., "data.csv"
        )

        # Path to store the ingested training dataset
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Path to store the ingested testing dataset
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )

        # Split ratio for training and test datasets (e.g., 0.2 means 80% train, 20% test)
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION

        # MongoDB collection name used for data ingestion
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME

        # MongoDB database name
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME