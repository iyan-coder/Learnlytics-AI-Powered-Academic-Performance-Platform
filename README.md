# 🎓 Student Performance Indicator – Math Score Predictor

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![Build Status](https://github.com/<iyan-coder>/student-performance/actions/workflows/ci.yml/badge.svg)](https://github.com/<iyan-coder>/student-performance/actions)
[![Docker Ready](https://img.shields.io/badge/docker-publish-green)](https://hub.docker.com/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-active-orange)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Predict a student’s _math score_ from socio‑economic, demographic and academic‑behavioural features.**  
> This repo shows an **end‑to‑end regression pipeline** with **MLflow tracking, DVC/Dagshub versioning**, and three serving flavours (Streamlit UI, Flask API, FastAPI API) – all **Dockerized** and wired into **GitHub Actions CI/CD**.

---

## 🚀 Project Highlights

| Aspect          | Details                                                                         |
|-----------------|---------------------------------------------------------------------------------|
| **Goal**        | Predict continuous `math_score`                                                 |
| **Model**       | Linear Regression (best R² ≈ **0.89 train / 0.867 test**)                       |
| **Metrics**     | R², MAE, RMSE                                                                   |
| **Tracking**    | MLflow (experiments, params, metrics, model registry)                           |
| **Versioning**  | DVC + DagsHub (datasets & artifacts)                                            |
| **Serving**     | Streamlit UI • Flask API • FastAPI API                                          |
| **Automation**  | GitHub Actions CI/CD – build, test, (optionally) deploy                         |
| **Author**      | _Gabriel Adebayo_ – Mechatronics student, inspired by Krish Naik’s series       |

---

## 🧠 End‑to‑End Pipeline

main.py → orchestrates:
├─ Data Ingestion (raw CSV → /data)
├─ Data Validation (schema + sanity checks)
├─ Data Transformation (preprocessor.pkl)
├─ Model Training (LinearRegression, hyper‑params)
├─ Model Evaluation (R², MAE, RMSE, plots)
└─ Model Pushing (model.pkl & preprocessor.pkl → final_model/)
↳ every run auto‑logs to MLflow


---

## 📊 EDA Insights

* Target distribution of math scores  
* Correlation heat‑map (features vs. score)  
* Feature impacts of **parental education**, **lunch type**, **test preparation** …

---

## 🗂️ Project Structure


student-performance/
├─ .github/workflows/ci.yml ← CI/CD pipeline
├─ app/
│ ├─ streamlit_app/ ← Streamlit front‑end
│ ├─ flask_app/ ← Flask batch API
│ ├─ fastapi_app/ ← FastAPI async API
│ └─ templates/, static/
├─ src/student_performance_indicator/
│ ├─ components/ ← ingestion, validation, …
│ ├─ pipeline/ ← train & predict pipelines
│ ├─ utils/, logger.py, exception.py
├─ data/ ← raw CSVs (DVC‑ignored)
├─ artifacts/ ← intermediate outputs (DVC)
├─ final_model/ ← model.pkl + preprocessor.pkl
├─ Dockerfile ← multi‑stage build
├─ docker-compose.yaml ← runs all services + MLflow
├─ requirements.txt
└─ README.md


---

## 🐳 Run Everything with Docker

```bash
# build image & start Streamlit • FastAPI • Flask • MLflow
docker compose up --build -d

# stop + remove containers & volumes
docker compose down -v
```

| Service      | URL                                            | Notes                         |
| ------------ | ---------------------------------------------- | ----------------------------- |
| Streamlit UI | [http://localhost:8501](http://localhost:8501) | interactive web dashboard     |
| FastAPI      | [http://localhost:8000](http://localhost:8000) | `/docs` for Swagger UI        |
| Flask        | [http://localhost:5000](http://localhost:5000) | CSV upload & HTML table       |
| MLflow UI    | [http://localhost:5001](http://localhost:5001) | experiment tracking dashboard |

### 💻 Run Locally without Docker
```bash
git clone https://github.com/<your-user>/student-performance.git
cd student-performance
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py                                   # train + log to MLflow
streamlit run app/streamlit_app/app.py           # open dashboard

```
### ⚙️ Continuous Integration / Deployment
| Stage  | Details                                                                                                                                                                                                |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **CI** | Each push to **`main`** runs `.github/workflows/ci.yml` →<br>• checkout code<br>• cache pip<br>• build Docker image (`docker compose build`)<br>• boot stack & smoke‑test (`curl localhost:8000/docs`) |
| **CD** | Uncomment the **Deploy** step.<br>Provide server SSH secrets → pulls latest repo, then `docker compose up -d --build` for zero‑downtime redeploy.                                                      |

<details> <summary>▶ minimal CI snippet</summary>

```bash

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
### 🔗 MLflow Tracking
1. Local tracking server inside Docker (http://localhost:5001)

2. All parameters, metrics, models and artifacts logged automatically.

3. Switch to a remote MLflow backend by editing docker-compose.yaml.

### 🙏 Acknowledgements
1. Krish Naik’s YouTube ML tutorials

2. Open‑source community: Scikit‑learn, Streamlit, FastAPI, Flask, DVC, MLflow, Docker, GitHub Actions

### 📜 License
Distributed under the MIT License.

### What you still need to do

1. Replace every **`<your-user>`** with your real GitHub username.
2. Commit + push the new `README.md`.

That’s it—enjoy your fully documented, Docker‑ready, CI/CD‑enabled project!


