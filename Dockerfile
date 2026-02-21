# syntax=docker/dockerfile:1

FROM python:3.11-slim

# XGBoost needs OpenMP runtime on Debian/Ubuntu images
RUN apt-get update \
    ; apt-get install -y --no-install-recommends libgomp1 \
    ; rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies first for better layer caching
COPY requirements.txt ./requirements.txt
COPY frontend/requirements.txt ./frontend/requirements.txt
RUN pip install --upgrade pip \
    ; pip install -r requirements.txt

# Copy the app (expects model/ artifacts to exist)
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
