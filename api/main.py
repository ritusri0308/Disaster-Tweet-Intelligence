"""
FastAPI REST Service for CrisisLens.
Provides endpoints for single and batch emergency tweet intelligence extraction,
operator feedback submission, and surge / incident summaries over the processed dataset.

Configuration (environment variables):
    CRISISLENS_API_KEY        If set, protected endpoints require a matching ``X-API-Key`` header.
    CRISISLENS_CORS_ORIGINS   Comma-separated list of allowed browser origins (CORS is off when unset).
"""

from contextlib import asynccontextmanager
import hmac
import logging
import os
import re
import time
import uuid
from typing import List, Dict, Any, Optional
import json

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.config import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    MODEL_PATH,
    DATASET_NAME,
    DISCLAIMER,
    PROCESSED_PARQUET_PATH,
    PROJECT_CATEGORIES,
)
from src.preprocessing import clean_tweet, extract_single_text_features
from src.classifier import predict_with_confidence
from src.ner import extract_locations_batch
from src.geocoding import filter_and_validate_locations
from src.needs import extract_needs
from src.entities import event_year, extract_quantities
from src.urgency import calculate_urgency_breakdown
from src.model_registry import model_health
from src.database import save_feedback
from src.anomaly import detect_event_surges
from src.clustering import cluster_event_incidents
from src.utils import normalize_list_columns

logger = logging.getLogger(__name__)

URGENCY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Global model cache for API
_MODEL_PIPELINE = None
# Processed dataset cache keyed by file modification time
_DATASET_CACHE: Dict[str, Any] = {"mtime": None, "df": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load machine learning model once on startup."""
    global _MODEL_PIPELINE
    if MODEL_PATH.exists():
        try:
            _MODEL_PIPELINE = joblib.load(MODEL_PATH)
            logger.info(f"Loaded classifier model from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            _MODEL_PIPELINE = None
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}")
        _MODEL_PIPELINE = None
    health = model_health(_MODEL_PIPELINE)
    if health["status"] != "healthy":
        logger.warning(f"Model status '{health['status']}': {health['reason']}")
    yield

app = FastAPI(
    title=f"{APP_NAME} API",
    description=f"{APP_TAGLINE}. Production decision-support service for crisis intelligence.",
    version=APP_VERSION,
    lifespan=lifespan
)

_cors_origins = [o.strip() for o in os.environ.get("CRISISLENS_CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
    )

_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")

@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    """Attach a request id (propagating a well-formed client one) and processing time to every response."""
    incoming = request.headers.get("X-Request-ID", "")
    request_id = incoming if _REQUEST_ID_PATTERN.fullmatch(incoming) else uuid.uuid4().hex
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response

def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Enforce the API key only when CRISISLENS_API_KEY is configured."""
    expected = os.environ.get("CRISISLENS_API_KEY", "").strip()
    if expected and not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key.")

# Pydantic Schemas
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Raw social media tweet text.")
    event: Optional[str] = Field(None, description="Optional disaster event identifier for geospatial prior disambiguation.")

class BatchAnalyzeRequest(BaseModel):
    tweets: List[AnalyzeRequest] = Field(..., max_length=500, description="List of tweets to analyze (max 500).")

class GeocodedLocation(BaseModel):
    name: str
    lat: float
    lon: float
    country_code: str
    population: int = 0
    confidence: float
    is_ambiguous: bool = False
    entity_type: str = "city"

class QuantityFigure(BaseModel):
    kind: str
    count: int

class AnalyzeResponse(BaseModel):
    clean_text: str
    category: str
    confidence: float
    probabilities: Dict[str, float]
    urgency_score: float
    urgency_level: str
    urgency_breakdown: List[Dict[str, Any]]
    matched_terms: List[str]
    locations: List[str]
    geocoded_locations: List[GeocodedLocation]
    needs: List[str]
    quantities: List[QuantityFigure]
    model_version: Optional[str] = None

class FeedbackRequest(BaseModel):
    tweet_id: str = Field(..., min_length=1, max_length=64)
    text: str = Field("", max_length=2000, description="Tweet text, stored so the correction can be used for retraining.")
    event: Optional[str] = None
    original_category: str
    corrected_category: str
    original_urgency: str
    corrected_urgency: str
    notes: str = Field("", max_length=1000)
    reviewer: str = Field("", max_length=100)

    @field_validator("original_category", "corrected_category")
    @classmethod
    def _valid_category(cls, value: str) -> str:
        if value not in PROJECT_CATEGORIES:
            raise ValueError(f"must be one of {PROJECT_CATEGORIES}")
        return value

    @field_validator("original_urgency", "corrected_urgency")
    @classmethod
    def _valid_urgency(cls, value: str) -> str:
        if value.upper() not in URGENCY_LEVELS:
            raise ValueError(f"must be one of {URGENCY_LEVELS}")
        return value.upper()

def _require_model():
    """Return the loaded classifier or fail loudly. A triage system must never guess a category."""
    if _MODEL_PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification model is not loaded. Train it with `python train.py` and restart the service."
        )
    return _MODEL_PIPELINE

def _classify(pipeline, cleaned_texts: List[str]) -> List[Dict[str, Any]]:
    """Classify a batch in one forward pass, returning category, confidence and class probabilities."""
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(cleaned_texts)
        classes = [str(c) for c in pipeline.classes_]
        out = []
        for row in probas:
            best = int(row.argmax())
            out.append({
                "category": classes[best],
                "confidence": float(round(row[best], 4)),
                "probabilities": {c: float(round(p, 4)) for c, p in zip(classes, row)},
            })
        return out
    return [
        {"category": cat, "confidence": conf, "probabilities": {cat: conf}}
        for cat, conf in predict_with_confidence(pipeline, cleaned_texts)
    ]

def analyze_items(items: List[AnalyzeRequest]) -> List[AnalyzeResponse]:
    """Core analysis worker: batched classification and NER, then per-item geocoding, needs, and urgency."""
    pipeline = _require_model()
    if not items:
        return []

    texts = [item.text for item in items]
    cleaned = [clean_tweet(t) for t in texts]
    classified = _classify(pipeline, cleaned)
    raw_locations = extract_locations_batch(texts)
    version = model_health(pipeline)["version"]

    responses = []
    for item, clean, cls, raw_locs in zip(items, cleaned, classified, raw_locations):
        features = extract_single_text_features(item.text)
        clean_locs, geocoded = filter_and_validate_locations(raw_locs, event=item.event)
        score, level, terms, breakdown = calculate_urgency_breakdown(
            text=item.text,
            category=cls["category"],
            uppercase_ratio=features["uppercase_ratio"],
            exclamation_count=features["exclamation_count"],
            is_retweet=features["is_retweet"]
        )
        responses.append(AnalyzeResponse(
            clean_text=clean,
            category=cls["category"],
            confidence=cls["confidence"],
            probabilities=cls["probabilities"],
            urgency_score=score,
            urgency_level=level,
            urgency_breakdown=breakdown,
            matched_terms=terms,
            locations=clean_locs,
            geocoded_locations=geocoded,
            needs=extract_needs(item.text),
            quantities=extract_quantities(item.text, reference_year=event_year(item.event)),
            model_version=version,
        ))
    return responses

def process_single_item(item: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze one tweet (kept for backward compatibility)."""
    return analyze_items([item])[0]

def _plain(obj: Any) -> Any:
    """Convert numpy / pandas scalars inside nested structures into plain JSON-safe values."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

def _load_dataset() -> pd.DataFrame:
    """Load the processed dataset, re-reading it only when the file changes."""
    if not PROCESSED_PARQUET_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processed dataset not found. Run `python run_pipeline.py` first."
        )
    mtime = PROCESSED_PARQUET_PATH.stat().st_mtime
    if _DATASET_CACHE["mtime"] != mtime:
        _DATASET_CACHE["df"] = normalize_list_columns(pd.read_parquet(PROCESSED_PARQUET_PATH))
        _DATASET_CACHE["mtime"] = mtime
    return _DATASET_CACHE["df"]

@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    """Liveness plus model state: healthy, degraded (implausibly small model), or unavailable (no model)."""
    info = model_health(_MODEL_PIPELINE)
    return {
        "status": "unavailable" if info["status"] == "missing" else info["status"],
        "app": APP_NAME,
        "version": APP_VERSION,
        "model_loaded": _MODEL_PIPELINE is not None,
        "model_version": info["version"],
        "model_name": info["model_name"],
        "model_vocab_size": info["vocab_size"],
        "detail": info["reason"],
    }

@app.get("/ready")
def ready(response: Response):
    """Readiness probe: 200 only when a healthy model is loaded."""
    info = model_health(_MODEL_PIPELINE)
    if info["status"] != "healthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": info["status"] == "healthy", "status": info["status"], "detail": info["reason"]}

@app.get("/version", status_code=status.HTTP_200_OK)
def version():
    """Version and dataset provenance metadata."""
    return {
        "app_name": APP_NAME,
        "tagline": APP_TAGLINE,
        "version": APP_VERSION,
        "dataset": DATASET_NAME,
        "disclaimer": DISCLAIMER,
        "mode": "historical_replay",
        "model_version": model_health(_MODEL_PIPELINE)["version"],
    }

@app.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK,
          dependencies=[Depends(require_api_key)])
def analyze(request: AnalyzeRequest):
    """Analyze a single tweet text for category, location, need types, reported figures, and urgency."""
    return process_single_item(request)

@app.post("/analyze/batch", response_model=List[AnalyzeResponse], status_code=status.HTTP_200_OK,
          dependencies=[Depends(require_api_key)])
def analyze_batch(request: BatchAnalyzeRequest):
    """Batch analyze up to 500 tweets (validated by the request schema)."""
    return analyze_items(request.tweets)

@app.post("/feedback", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_api_key)])
def submit_feedback(request: FeedbackRequest):
    """Record an operator correction so it can feed `retrain_with_feedback.py`."""
    feedback_id = save_feedback(
        tweet_id=request.tweet_id,
        original_category=request.original_category,
        corrected_category=request.corrected_category,
        original_urgency=request.original_urgency,
        corrected_urgency=request.corrected_urgency,
        user_notes=request.notes,
        tweet_text=request.text,
        event=request.event or "",
        split="live",
        reviewer=request.reviewer,
        model_version=model_health(_MODEL_PIPELINE)["version"] or "",
    )
    return {"feedback_id": feedback_id}

@app.get("/surges", dependencies=[Depends(require_api_key)])
def surges(
    event: Optional[str] = Query(None, description="Restrict to one disaster event id."),
    limit: int = Query(20, ge=1, le=200),
):
    """Statistical volume / critical-share surges detected in the processed dataset."""
    df = _load_dataset()
    if event:
        df = df[df["event"] == event]
    return _plain(detect_event_surges(df)[:limit])

@app.get("/incidents", dependencies=[Depends(require_api_key)])
def incidents(
    event: str = Query(..., description="Disaster event id to cluster (required: clustering is per event)."),
    limit: int = Query(10, ge=1, le=100),
):
    """Ranked clusters of near-duplicate reports for one event."""
    df = _load_dataset()
    event_df = df[df["event"] == event]
    if event_df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No tweets found for event '{event}'.")
    return _plain(cluster_event_incidents(event_df, max_sample_per_event=500)[:limit])
