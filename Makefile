.PHONY: help install train pipeline evaluate test lint run api docker-build

help:
	@echo "CrisisLens Platform Makefile"
	@echo "Commands:"
	@echo "  make install    - Install dependencies and spaCy language model"
	@echo "  make train      - Train classification models and pick best"
	@echo "  make pipeline   - Run full ETL, feature extraction, geocoding pipeline"
	@echo "  make evaluate   - Generate calibration, ROC/PR curves, and evaluation artifacts"
	@echo "  make test       - Run full test suite with coverage"
	@echo "  make lint       - Run ruff linter check"
	@echo "  make run        - Launch Streamlit dashboard"
	@echo "  make api        - Launch FastAPI REST service"
	@echo "  make docker-build - Build Docker container"

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

train:
	python train.py

pipeline:
	python run_pipeline.py

evaluate:
	python evaluate.py

test:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check .

run:
	streamlit run app.py

api:
	uvicorn api.main:app --reload --port 8000

docker-build:
	docker build -t crisislens:latest .
