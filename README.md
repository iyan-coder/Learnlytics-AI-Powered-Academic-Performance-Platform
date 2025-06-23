### End to End Project for Student Performance Indicator

# Reusable Poetry ML Project Setup

## Step-by-Step Poetry Setup

1. **Create a project folder**
   ```bash
   mkdir student-performance-indicator
   cd student-performance-indicator


```bash
poetry config virtualenvs.in-project true
poetry init

name = "student_performance_indicator"
requires-python = ">=3.8,<3.12"

poetry env use C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe

poetry add pandas numpy scikit-learn joblib streamlit flask python-dotenv


poetry add pandas@^1.5 numpy@^1.23 scikit-learn@^1.2 fastapi@^0.95 uvicorn@^0.22 streamlit@^1.25 xgboost@^1.7 python-dotenv@^1.0 joblib@^1.2


poetry add --group dev black isort flake8

poetry install

poetry self add poetry-plugin-shell

poetry shell

poetry self add poetry-plugin-export

poetry export -f requirements.txt --without-hashes -o requirements.txt


```

``` bash 
type nul > format.bat
```
@echo off
echo Running isort...
poetry run isort .

echo Running black...
poetry run black .

echo Running flake8...
poetry run flake8 .

echo Done
pause
