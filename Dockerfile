# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

# XGBoost needs OpenMP runtime on Debian/Ubuntu images
RUN apt-get update \
    ; apt-get install -y --no-install-recommends libgomp1 \
    ; rm -rf /var/lib/apt/lists/*

# Run as non-root for better container security
RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

# Install dependencies first for better layer caching
COPY requirements.txt ./requirements.txt
COPY frontend/requirements.txt ./frontend/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install --prefer-binary --only-binary=:all: -r requirements.txt

# Copy the app (expects model/ artifacts to exist)
COPY --chown=appuser:appuser . .

EXPOSE 8501:8501

USER appuser

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
