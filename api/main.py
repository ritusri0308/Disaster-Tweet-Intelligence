"""
FastAPI REST Service for CrisisLens.
Provides endpoints for single and batch emergency tweet intelligence extraction.
"""

from contextlib import asynccontextmanager
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import joblib

from src.config import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    MODEL_PATH,
    DATASET_NAME,
    DISCLAIMER
)
from src.preprocessing import clean_tweet, extract_single_text_features
from src.classifier import predict_with_confidence
from src.ner import extract_locations_single
from src.geocoding import filter_and_validate_locations
from src.needs import extract_needs
from src.urgency import calculate_urgency_breakdown

logger = logging.getLogger(__name__)

# Global model cache for API
_MODEL_PIPELINE = None

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
    yield

app = FastAPI(
    title=f"{APP_NAME} API",
    description=f"{APP_TAGLINE}. Production decision-support service for crisis intelligence.",
    version=APP_VERSION,
    lifespan=lifespan
)

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
    population: int
    confidence: float
    is_ambiguous: bool

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
    geocoded_locations: List[Dict[str, Any]]
    needs: List[str]

def process_single_item(item: AnalyzeRequest) -> AnalyzeResponse:
    """Core analysis worker reusing src/ modules."""
    cleaned = clean_tweet(item.text)
    features = extract_single_text_features(item.text)

    # Classification
    if _MODEL_PIPELINE is not None:
        preds = predict_with_confidence(_MODEL_PIPELINE, [cleaned])
        cat, conf = preds[0]
        if hasattr(_MODEL_PIPELINE, "predict_proba"):
            probas = _MODEL_PIPELINE.predict_proba([cleaned])[0]
            classes = list(_MODEL_PIPELINE.classes_)
            prob_dict = {str(c): float(round(p, 4)) for c, p in zip(classes, probas)}
        else:
            prob_dict = {cat: conf}
    else:
        cat, conf = "General", 0.5
        prob_dict = {"General": 0.5}

    # NER & Geocoding
    raw_locs = extract_locations_single(item.text)
    clean_locs, geocoded = filter_and_validate_locations(raw_locs, event=item.event)

    # Needs
    needs = extract_needs(item.text)

    # Urgency & Waterfall Breakdown
    score, level, terms, breakdown = calculate_urgency_breakdown(
        text=item.text,
        category=cat,
        uppercase_ratio=features["uppercase_ratio"],
        exclamation_count=features["exclamation_count"],
        is_retweet=features["is_retweet"]
    )

    return AnalyzeResponse(
        clean_text=cleaned,
        category=cat,
        confidence=conf,
        probabilities=prob_dict,
        urgency_score=score,
        urgency_level=level,
        urgency_breakdown=breakdown,
        matched_terms=terms,
        locations=clean_locs,
        geocoded_locations=geocoded,
        needs=needs
    )

@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "model_loaded": _MODEL_PIPELINE is not None
    }

@app.get("/version", status_code=status.HTTP_200_OK)
def version():
    """Version and dataset provenance metadata."""
    return {
        "app_name": APP_NAME,
        "tagline": APP_TAGLINE,
        "version": APP_VERSION,
        "dataset": DATASET_NAME,
        "disclaimer": DISCLAIMER,
        "mode": "historical_replay"
    }

@app.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze(request: AnalyzeRequest):
    """Analyze a single tweet text for category, location, need types, and urgency."""
    return process_single_item(request)

@app.post("/analyze/batch", response_model=List[AnalyzeResponse], status_code=status.HTTP_200_OK)
def analyze_batch(request: BatchAnalyzeRequest):
    """Batch analyze up to 500 tweets."""
    if len(request.tweets) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch limit exceeded. Maximum 500 items allowed per request."
        )
    return [process_single_item(item) for item in request.tweets]
