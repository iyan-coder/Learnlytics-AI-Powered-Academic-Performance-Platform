# 🎓 Learnlytics – AI-Powered Academic Performance Platform

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)

[![Docker Ready](https://img.shields.io/badge/docker-publish-green)](https://hub.docker.com/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-active-orange)](https://mlflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-red)](https://scikit-learn.org/stable/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37.0-ff69b4)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)](https://fastapi.tiangolo.com)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)](https://flask.palletsprojects.com)
[![DVC](https://img.shields.io/badge/DVC-3.50-purple)](https://dvc.org)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-blue)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



## 🚨 Problem
Students’ academic performance is influenced by multiple factors such as parental education, lunch type, and preparation habits.  
Traditional evaluation often misses these socio-economic and behavioral signals, leading to **ineffective interventions** and **missed opportunities for support**.

## ✅ Solution
Learnlytics uses **Machine Learning regression pipelines** to predict student math scores from socio-economic, demographic, and academic-behavioral features.  
By combining **EDA insights, explainability (SHAP), and deployment-ready APIs (Streamlit, Flask, FastAPI)**, the system helps educators:

- Identify at-risk students early.  
- Provide data-driven personalized support.  
- Improve academic outcomes at scale.  

🚀 Designed for **real-world educational impact**, not just academic demos.


---

## 🚀 Project Highlights

| Aspect          | Details                                                                         |
|-----------------|---------------------------------------------------------------------------------|
| **Goal**        | Predict continuous `math_score`                                                 |
| **Model**       | Linear Regression (best R² ≈ **0.89 train / 0.867 test**)                       |
| **Metrics**     | R², MAE, RMSE                                                                   |
| **Tracking**    | MLflow (experiments, params, metrics, model registry)                           |
| **Versioning**  | DVC + DagsHub (datasets & artifacts)                                            |
| **Serving**     | Streamlit UI • Flask API • FastAPI API                                          |
| **Automation**  | GitHub Actions CI/CD – build, test, (optionally) deploy                         |
| **Author**      | _Gabriel Adebayo_ – Mechatronics student       |

---

## 🧠 End-to-End Pipeline

```mermaid
flowchart TD
    A[Raw Data] --> B[Data Validation]
    B --> C[Data Transformation]
    C --> D[Model Training & Hyperparameter Tuning]
    D --> E[Evaluation (R², MAE, RMSE)]
    E --> F[Model Registry & Pushing]
    F --> G[Deployment: Streamlit, Flask, FastAPI]
    G --> H[Monitoring & CI/CD (MLflow + GitHub Actions)]
```

## 🚀 Why It Matters

- **Reliability** → Validation ensures data quality before training.  
- **Reproducibility** → Transformation pipelines + Model Registry keep experiments consistent.  
- **Real-world Usability** → Multi-stack deployment (Streamlit, Flask, FastAPI) with monitoring ensures scalability.  

✅ Designed for **industry-grade workflows**, not just academic demos.  

---

## 📊 Exploratory Data Analysis (EDA) Insights

- **Target Distribution (Math Scores):** Checked spread of student scores → revealed skewness & performance clusters.  
- **Correlation Heatmap:** Found strong correlation between parental education, lunch type, and test prep with math scores.  
- **Feature Impact:**  
  - Students with higher parental education → consistently scored better.  
  - Lunch type (standard vs. free/reduced) showed performance gaps.  
  - Test preparation course completion led to higher median math scores.  

---
## 🗂️ Project Structure

```
learnlytics/
├─ .github/workflows/ci.yml      ← CI/CD pipeline
├─ app/
│   ├─ streamlit_app/            ← Streamlit front-end
│   ├─ flask_app/                ← Flask batch API
│   ├─ fastapi_app/              ← FastAPI async API
│   └─ templates/, static/
├─ src/learnlytics/
│   ├─ components/               ← ingestion, validation, transformation
│   ├─ pipeline/                 ← train & predict pipelines
│   ├─ utils/, logger.py, exception.py
├─ data/                         ← raw CSVs (DVC-ignored)
├─ artifacts/                    ← intermediate outputs (DVC)
├─ final_model/                  ← model.pkl + preprocessor.pkl
├─ Dockerfile                    ← multi-stage build
├─ docker-compose.yaml           ← runs all services + MLflow
├─ requirements.txt
└─ README.md
```
---
## 🐳 Run Everything with Docker

```
# build image & start Streamlit • FastAPI • Flask • MLflow
docker compose up --build -d

# stop + remove containers & volumes
docker compose down -v
```

| Service      | URL                                            | Notes                         |
| ------------ | ---------------------------------------------- | ----------------------------- |
| Streamlit UI | [http://localhost:8501](http://localhost:8501) | interactive web dashboard     |
| FastAPI      | [http://localhost:8000](http://localhost:8000) | `/docs` for Swagger UI        |
| Flask        | [http://localhost:5000](http://localhost:5000) | CSV upload & HTML table       |
| MLflow UI    | [http://localhost:5001](http://localhost:5001) | experiment tracking dashboard |

---
## ⚙️ Continuous Integration / Deployment

| Stage  | Details                                                                                                                                                                                                |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **CI** | Each push to **`main`** runs `.github/workflows/ci.yml` →<br>• checkout code<br>• cache pip<br>• build Docker image (`docker compose build`)<br>• boot stack & smoke-test (`curl localhost:8000/docs`) |
| **CD** | Uncomment the **Deploy** step.<br>Provide server SSH secrets → pulls latest repo, then `docker compose up -d --build` for zero-downtime redeploy.                                                      |
---

## ▶ minimal CI snippe

```
name: Docker CI

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: docker/setup-buildx-action@v2
      - name: Build & launch
        run: |
          docker compose build
          docker compose up -d
      - name: FastAPI smoke test
        run: curl --retry 5 --retry-delay 3 http://localhost:8000/docs
```
## 🔗 MLflow Tracking
1. Local tracking server inside Docker (http://localhost:5001)

2. All parameters, metrics, models and artifacts logged automatically

3. Switch to a remote MLflow backend by editing docker-compose.yaml
