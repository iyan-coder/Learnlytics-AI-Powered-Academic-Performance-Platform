# Artifacts are structured outputs from each stage in the ML pipeline.
# They help track outputs and pass necessary data/configs downstream.
# -------------------------------------------------------------
from dataclasses import dataclass  # Provides a decorator and functions for automatically adding special methods


# ========================================================
# DataIngetionArtifact: Holds paths to split data outputs
# ========================================================
@dataclass  # Simplifies the creation of classes for storing data
class DataIngetionArtifact:
    """
    Artifact class for the data ingestion step.

    Attributes:
        trained_file_path (str): Path to the CSV file containing the training dataset.
        test_file_path (str): Path to the CSV file containing the testing dataset.
    """
    trained_file_path: str  # File path to the training dataset after split
    test_file_path: str     # File path to the testing dataset after split
