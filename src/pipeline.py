"""
End-to-end data processing, classification, NER, geocoding, needs extraction, urgency scoring, and storage pipeline for CrisisLens.
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import joblib

from src.config import (
    MODEL_PATH,
    PROCESSED_CSV_PATH,
    PROCESSED_PARQUET_PATH,
    DB_PATH,
    CLASSIFICATION_TARGET
)
from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe
from src.classifier import train_and_compare_models, predict_with_confidence
from src.ner import extract_locations_batch
from src.urgency import calculate_urgency
from src.needs import extract_needs_batch, evaluate_needs_against_raw_labels
from src.geocoding import filter_and_validate_locations, generate_geocoding_report
from src.database import save_processed_tweets

logger = logging.getLogger(__name__)

def run_end_to_end_pipeline(
    raw_dir: Optional[Path] = None,
    target_schema: str = CLASSIFICATION_TARGET,
    save_outputs: bool = True
) -> pd.DataFrame:
    """
    Run complete processing pipeline:
    1. Load TSV dataset
    2. Clean text & extract text features
    3. Load or train classification model
    4. Predict categories & confidence scores
    5. Run batch NER & offline geocoding
    6. Extract multi-label need types
    7. Calculate explainable urgency scores & levels
    8. Save to Parquet, CSV, and SQLite DB
    """
    logger.info("Step 1/7: Loading raw dataset...")
    df = load_data(raw_dir=raw_dir, target_schema=target_schema)

    logger.info("Step 2/7: Preprocessing text and extracting text features...")
    df = preprocess_dataframe(df)

    logger.info("Step 3/7: Checking classification model...")
    if not MODEL_PATH.exists():
        logger.info("Trained model not found. Training model suite first...")
        train_df = df[df["split"] == "train"]
        dev_df = df[df["split"] == "dev"]
        test_df = df[df["split"] == "test"]

        if train_df.empty:
            logger.warning("Train split empty. Splitting dataset 80/20 for training...")
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            dev_df = test_df

        train_and_compare_models(train_df, dev_df, test_df)

    pipeline = joblib.load(MODEL_PATH)

    logger.info("Step 4/7: Predicting categories and confidence...")
    texts_to_predict = df["clean_text"].tolist()
    pred_results = predict_with_confidence(pipeline, texts_to_predict)

    df["category"] = [p[0] for p in pred_results]
    df["confidence"] = [p[1] for p in pred_results]

    logger.info("Step 5/7: Extracting locations & geocoding...")
    raw_texts = df["raw_unescaped_text"].tolist()
    raw_locations_list = extract_locations_batch(raw_texts)

    cleaned_locations = []
    geocoded_locations = []
    events = df["event"].tolist()

    for raw_locs, ev in zip(raw_locations_list, events):
        c_locs, g_locs = filter_and_validate_locations(raw_locs, event=ev)
        cleaned_locations.append(c_locs)
        geocoded_locations.append(g_locs)

    df["locations"] = cleaned_locations
    df["geocoded_locations"] = geocoded_locations

    logger.info("Step 6/7: Extracting multi-label need types...")
    df["needs"] = extract_needs_batch(raw_texts)

    logger.info("Step 7/7: Computing urgency scores and levels...")
    urgency_scores = []
    urgency_levels = []
    matched_terms_list = []

    for idx, row in df.iterrows():
        score, level, terms = calculate_urgency(
            text=row["raw_unescaped_text"],
            category=row["category"],
            uppercase_ratio=row.get("uppercase_ratio", 0.0),
            exclamation_count=row.get("exclamation_count", 0),
            is_retweet=row.get("is_retweet", 0)
        )
        urgency_scores.append(score)
        urgency_levels.append(level)
        matched_terms_list.append(terms)

    df["urgency_score"] = urgency_scores
    df["urgency_level"] = urgency_levels
    df["matched_terms"] = matched_terms_list

    if save_outputs:
        logger.info("Saving processed outputs to Parquet, CSV, and SQLite DB...")
        # Save Parquet
        try:
            df.to_parquet(PROCESSED_PARQUET_PATH, index=False)
            logger.info(f"Saved Parquet file to {PROCESSED_PARQUET_PATH}")
        except Exception as e:
            logger.warning(f"Could not save Parquet: {e}")

        df.to_csv(PROCESSED_CSV_PATH, index=False)
        save_processed_tweets(df, DB_PATH)

        # Generate audit reports
        generate_geocoding_report(df)
        evaluate_needs_against_raw_labels(df)

    logger.info("CrisisLens pipeline execution completed successfully!")
    return df
