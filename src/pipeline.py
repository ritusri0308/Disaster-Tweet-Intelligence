"""
End-to-end data processing, classification, NER, urgency scoring, and storage pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import joblib

from src.config import (
    MODEL_PATH,
    PROCESSED_CSV_PATH,
    DB_PATH,
    CLASSIFICATION_TARGET
)
from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe
from src.classifier import train_and_compare_models, predict_with_confidence
from src.ner import extract_locations_batch
from src.urgency import calculate_urgency
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
    5. Run batch NER for location extraction
    6. Calculate explainable urgency scores & levels
    7. Save to CSV and SQLite DB
    """
    logger.info("Step 1/6: Loading raw dataset...")
    df = load_data(raw_dir=raw_dir, target_schema=target_schema)

    logger.info("Step 2/6: Preprocessing text and extracting text features...")
    df = preprocess_dataframe(df)

    logger.info("Step 3/6: Checking classification model...")
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

    logger.info("Step 4/6: Predicting categories and confidence...")
    texts_to_predict = df["clean_text"].tolist()
    pred_results = predict_with_confidence(pipeline, texts_to_predict)

    df["category"] = [p[0] for p in pred_results]
    df["confidence"] = [p[1] for p in pred_results]

    logger.info("Step 5/6: Extracting locations via batch NER...")
    raw_texts = df["raw_unescaped_text"].tolist()
    locations_list = extract_locations_batch(raw_texts)
    df["locations"] = locations_list

    logger.info("Step 6/6: Computing urgency scores and levels...")
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
        logger.info(f"Saving processed outputs to {PROCESSED_CSV_PATH} and SQLite DB...")
        df.to_csv(PROCESSED_CSV_PATH, index=False)
        save_processed_tweets(df, DB_PATH)

    logger.info("Pipeline execution completed successfully!")
    return df
