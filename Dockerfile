# ───────────────────────────────────────────────
# syntax=docker/dockerfile:1.6   # enables BuildKit extras
# Student-Performance – shared image for all apps
# ───────────────────────────────────────────────
# full image → pre-built wheels, faster

FROM python:3.11      

WORKDIR /app

# 1️⃣ Copy dependency file FIRST so later code edits don’t bust the cache
COPY requirements.txt .

# 2️⃣ Install deps and cache pip wheels between builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# 3️⃣ Copy the rest of the source (now tiny rebuilds)
COPY . .

# 4️⃣ Open all ports used by the three interfaces
EXPOSE 8501 8000 5000

# 5️⃣ Default entry-point (override in docker-compose for FastAPI/Flask)
CMD ["streamlit", "run", "app/streamlit_app/app.py"]
