# ───────────────────────────────────────────────
# syntax=docker/dockerfile:1.6
# Student-Performance – shared image for all apps
# ───────────────────────────────────────────────
FROM python:3.11

WORKDIR /app

# 🔑  Make both /app/src 𝘢𝘯𝘥 /app importable
ENV PYTHONPATH=/app/src:/app

# 1️⃣ Copy dependency list first (build-cache friendly)
COPY requirements-prod.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --timeout=100 --retries=5 --prefer-binary \
    -r requirements-prod.txt

# 3️⃣ Copy the rest of the project
COPY . .

# 4️⃣ Expose all ports the compose file maps
EXPOSE 8501 8000 5000

# 5️⃣ Default entry-point (overridden by compose for FastAPI/Flask)
CMD ["streamlit", "run", "app/streamlite_app.py"]
