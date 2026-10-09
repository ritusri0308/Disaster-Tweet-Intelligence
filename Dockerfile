FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Run as an unprivileged user (data, models and reports are mounted as volumes, never baked in; see .dockerignore)
RUN useradd --create-home --uid 1000 crisislens
COPY --chown=crisislens:crisislens . .
RUN mkdir -p data models reports && chown -R crisislens:crisislens data models reports
USER crisislens

# Expose ports: 8501 (Streamlit), 8000 (FastAPI)
EXPOSE 8501 8000

# Default health check targets the dashboard; docker-compose.yml overrides it for the API service.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Default command: launch Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
