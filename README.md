# 🎓 Student Performance Indicator – Math Score Predictor  
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)  
[![Build Status](https://github.com/<your-user>/student-performance/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-user>/student-performance/actions) 
[![Docker Ready](https://img.shields.io/badge/docker-publish-green)](https://hub.docker.com/) 
[![MLflow Tracking](https://img.shields.io/badge/MLflow-active-orange)](https://mlflow.org) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Predict a student’s *math score* from socio-economic, demographic and academic-behavioural features.**  
> This repository demonstrates an **end-to-end regression pipeline** with **MLflow tracking, DVC/Dagshub data versioning, and multi-framework serving (Streamlit UI, Flask API, FastAPI API) – all fully Docker-ised and CI/CD-enabled on GitHub.  

---

## 🚀 Quick Glance
| Aspect | Details |
|--------|---------|
| **Goal** | Predict continuous `math_score` |
| **Model** | *Linear Regression* (best R²: **0.89 train / 0.867 test**), plus baseline + experiments tracked in MLflow |
| **Metrics** | R², MAE, RMSE |
| **Tracking / Versioning** | MLflow (experiments & model registry), DVC + Dagshub (data/artifacts), GitHub Actions (CI) |
| **Serving** | `app/streamlit_app` • `app/flask_app` • `app/fastapi_app` |
| **Containerisation** | Single Dockerfile; multi-stage build for slim production image |
| **Author** | Gabriel Adebayo – Mechatronics student, project inspired by Krish Naik’s YouTube series |

---

## 🧠 End-to-End Pipeline

main.py -> Orchestrates ⤵
├─ Data Ingestion (raw → /data)
├─ Data Validation (schema & sanity checks)
├─ Data Transformation (preprocessor.pkl)
├─ Model Training (LinearRegression, hyper-params)
├─ Model Evaluation (R², MAE, RMSE, plots)
└─ Model Pushing (model.pkl + preprocessor.pkl → final_model/)
↳ All runs auto-logged in MLflow



---

## 🗂️ Project Structure

student-performance/
│
├── .github/workflows/ci.yml ← Tests + lint + Docker build
├── app/ ← Serving layer (three flavours)
│ ├── streamlit_app/
│ ├── flask_app/
│ ├── fastapi_app/
│ └── templates/ & static/
│
├── src/ ← Python package
│ └── student_performance_indicator/
│ ├── components/ ← ingestion, validation, etc.
│ ├── pipeline/ ← train & predict pipelines
│ ├── utils/
│ ├── logger.py
│ └── exception.py
│
├── data/ ← Raw CSV (⚠ Git-ignored by DVC)
├── artifacts/ ← DVC-tracked intermediate outputs
├── final_model/
│ ├── preprocessor.pkl
│ └── model.pkl
│
├── structure.txt ← Folder tree auto-generated
├── requirements.txt
├── Dockerfile
└── README.md ← you are here

yaml
Copy
Edit

> **Note:** `MLproject` has been removed – experiment entry-points are handled via `main.py` + MLflow.

---

## 📊 EDA Highlights
* Distribution of math scores (target)
* Correlation heat-map of features vs. math score
* Feature impacts of **parental education**, **lunch type**, **test preparation**, etc.

---

## 🔧 How to Run Locally

```bash
# 1 ▶ Clone & enter repo
git clone https://github.com/<your-user>/student-performance.git
cd student-performance

# 2 ▶ Create & activate environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3 ▶ Install deps
pip install -r requirements.txt

# 4 ▶ Train + log run to MLflow
python main.py

# 5 ▶ Launch Streamlit UI
streamlit run app/streamlit_app/app.py

# Build image
docker build -t math-score-predictor .

# Run Streamlit (port 8501) *and* FastAPI (port 8000) via uvicorn
docker run -p 8501:8501 -p 8000:8000 math-score-predictor

```
### 🙏 Acknowledgements

Krish Naik’s YouTube tutorials – foundational inspiration

The open-source community (Scikit-learn, MLflow, DVC, Streamlit, FastAPI, Flask, Docker)

