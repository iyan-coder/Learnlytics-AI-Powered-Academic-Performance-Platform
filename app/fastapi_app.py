import os
import sys

import certifi
import mlflow
import pandas as pd
import pymongo
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

from student_performance_indicator.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)
from student_performance_indicator.exception.exception import (
    StudentPerformanceException,
)
from student_performance_indicator.logger.logger import logger
from student_performance_indicator.pipeline.evaluation_pipeline import (
    ModelEvaluationPipeline,
)
from student_performance_indicator.pipeline.training_pipeline import TrainingPipeline
from student_performance_indicator.utils.main_utils.utils import (
    get_latest_artifact_dir,
    load_object,
)
from student_performance_indicator.utils.ml_utils.model.estimator import NetworkModel

# === Load environment variables ===
load_dotenv()

# === MLflow Local Tracking Config ===
# If running inside Docker, use host.docker.internal instead of localhost
import os

if os.getenv("RUNNING_IN_DOCKER") == "1":
    mlflow.set_tracking_uri(
        "http://mlflow:5000"
    )  # ✅ Docker container name + internal port
else:
    mlflow.set_tracking_uri(
        "http://localhost:5001"
    )  # ✅ When running locally outside Docker

mlflow.set_experiment("StudentPerformance")

# === MongoDB setup ===
mongo_db_url = os.getenv("MONGODB_URL_KEY")
ca = certifi.where()
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# === FastAPI App Setup ===
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


# === Routes ===
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["pipeline"])
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_training_pipeline()
        return JSONResponse(content={"message": "Training completed successfully!"})
    except Exception as e:
        logger.error("Training failed", exc_info=True)
        raise StudentPerformanceException(e, sys)


@app.get("/evaluate", tags=["pipeline"])
async def evaluate_route():
    try:
        evaluation_pipeline = ModelEvaluationPipeline()
        evaluation_pipeline.run_evaluation_pipeline()
        return JSONResponse(content={"message": "Evaluation completed successfully!"})
    except Exception as e:
        logger.error("Evaluation failed", exc_info=True)
        raise StudentPerformanceException(e, sys)


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        logger.info(f"Model prediction started for file: {file.filename}")

        # Read CSV into DataFrame
        df = pd.read_csv(file.file)

        # Load preprocessing and model
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        # Create model object
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        print(df.iloc[0])

        # Predict
        y_pred = network_model.predict(df)
        print(y_pred)
        df["predicted_column"] = y_pred
        print(df["predicted_column"])

        # Save output CSV
        os.makedirs("prediction_output", exist_ok=True)
        output_path = os.path.join("prediction_output", f"output_{file.filename}")
        df.to_csv(output_path, index=False)

        # Return as HTML
        table_html = df.to_html(classes="table table-striped", index=False)
        logger.info("Model prediction completed successfully.")
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": table_html}
        )

    except Exception as e:
        logger.error("Model prediction failed", exc_info=True)
        raise StudentPerformanceException(e, sys)
