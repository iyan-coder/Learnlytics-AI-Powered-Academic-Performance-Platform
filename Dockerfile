########################################
# syntax=docker/dockerfile:1.6
# Student‑Performance – shared image
########################################

############ 1 Dependency Layer ############
FROM python:3.11-slim AS deps

WORKDIR /app

# Copy only the requirements file first to maximize cache re‑use
COPY requirements-prod.txt .

# ---- Install Python deps, cache wheels between builds ----
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --timeout=100 --retries=5 --prefer-binary \
        -r requirements-prod.txt

############ 2  Runtime Layer ############
FROM python:3.11-slim

# Make both /app/src *and* /app importable
ENV PYTHONPATH=/app/src:/app
WORKDIR /app

# Copy the installed site‑packages from the deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages \
                 /usr/local/lib/python3.11/site-packages

# Copy your code and model artefact
COPY . .
# If your model lives inside the repo (as you showed), this also copies it:
#   final_model/model.pkl

# Expose ports used by compose
EXPOSE 8501 8000 5000

# Default command (overridden by docker‑compose for FastAPI/Flask)
CMD ["streamlit", "run", "app/streamlite_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
