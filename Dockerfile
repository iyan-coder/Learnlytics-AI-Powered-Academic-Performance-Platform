# syntax=docker/dockerfile:1.6
########################################
# Student‑Performance – shared image
########################################

######## 1️⃣  Dependency layer ########
FROM python:3.11-slim AS deps
WORKDIR /app
COPY requirements-prod.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements-prod.txt

######## 2️⃣  Runtime layer ########
FROM python:3.11-slim
ENV PYTHONPATH=/app/src:/app
WORKDIR /app
# copy libs **and** CLI tools (streamlit, uvicorn, etc.)
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY . .
EXPOSE 8501 8000 5000
CMD ["python", "-m", "http.server", "8000"]   # harmless default; each service overrides
