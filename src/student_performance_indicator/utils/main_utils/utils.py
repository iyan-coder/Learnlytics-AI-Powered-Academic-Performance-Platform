import pandas as pd
import numpy as np
import yaml
import os
import sys

from student_performance_indicator.constant.training_pipeline import SCHEMA_FILE_PATH
from student_performance_indicator.logger.logger import logger
from student_performance_indicator.exception.exception import StudentPerformanceException

def generate_schema_from_csv(csv_path, save_path=SCHEMA_FILE_PATH):
    logger.info("Initializing schema.yaml generation...")

    try:
        df = pd.read_csv(csv_path)
        logger.info("CSV file read successfully")

        schema = {
            "columns": {},
            "required_columns": df.columns.tolist(),
            "categorical_columns": df.select_dtypes(include='object').columns.tolist(),
            "numerical_columns": df.select_dtypes(include= [np.number]).columns.tolist()
        }

        for col in df.columns:
            dtype = str(df[col].dtype).lower()
            if dtype.startswith("int"):
                schema["columns"][col] = "int64"
            elif dtype.startswith("float"):
                schema["columns"][col] = "float64"
            else:
                schema["columns"][col] = "object"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as schema_file:
            yaml.dump(schema, schema_file, sort_keys=False)

        print(f"Schema generated and saved to {save_path}")
        logger.info(f"Schema generated and saved to {save_path}")

    except Exception as e:
        logger.error("Error! schema.yaml generation failed", exc_info=True)
        raise StudentPerformanceException(e, sys)
    

# =====================
# READ YAML FILE METHOD
# =====================

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: Parsed YAML content as a dictionary.

    Raises:
        NetworkSecurityException: If reading or parsing the YAML file fails.
    """
    try:
        # Open the YAML file in binary read mode
        with open(file_path, "rb") as yaml_file:
            # Parse and return content as dictionary
            return yaml.safe_load(yaml_file)
    
    except Exception as e:
        # Log full error traceback for debugging
        logger.error("Could not read_yaml_file", exc_info=True)
        # Raise custom exception with system info
        raise StudentPerformanceException(e, sys)
    

# ======================
# WRITE YAML FILE METHOD
# ======================

def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


def write_yaml_file(file_path: str, content: dict, replace: bool = False) -> None:
    """
    Saves a dictionary to a YAML file, optionally replacing an existing file.

    Args:
        file_path (str): Path to save the YAML file.
        content (dict): Data to write to the YAML file.
        replace (bool): If True, removes file if it already exists.
    """
    try:
        # If replace is enabled and file exists, delete it
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write dictionary content to YAML file
        with open(file_path, "w") as file:
            yaml.dump(
                convert_numpy(content),           # Dictionary to write
                file,                       # File object
                default_flow_style=False,   # Use block YAML style (cleaner)
                sort_keys=False,            # Preserve original dictionary order
                Dumper=yaml.SafeDumper,     # Use safe dumper (safe serialization)
            )

    except Exception as e:
        logger.error("Failed to write yaml file", exc_info=True)
        raise StudentPerformanceException(e, sys)
