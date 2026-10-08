"""
Data loading and dataset validation module.
"""

import csv
import html
import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    RAW_DATA_ALT_DIR,
    CATEGORY_MAPPING,
    CLASSIFICATION_TARGET
)

logger = logging.getLogger(__name__)

def derive_timestamp_from_id(tweet_id_str: str) -> Optional[pd.Timestamp]:
    """
    Derive UTC datetime from Twitter Snowflake tweet_id string.
    Formula: timestamp_ms = (int(tweet_id) >> 22) + 1288834974657
    Returns pd.NaT if tweet_id is not numeric or invalid.
    """
    try:
        tweet_id_int = int(str(tweet_id_str).strip())
        if tweet_id_int <= 0:
            return pd.NaT
        timestamp_ms = (tweet_id_int >> 22) + 1288834974657
        return pd.to_datetime(timestamp_ms, unit="ms", utc=True)
    except (ValueError, TypeError, OverflowError):
        return pd.NaT

def find_raw_data_dir() -> Path:
    """Find the path containing raw data TSV files."""
    if RAW_DATA_DIR.exists() and list(RAW_DATA_DIR.rglob("*.tsv")):
        return RAW_DATA_DIR
    if RAW_DATA_ALT_DIR.exists() and list(RAW_DATA_ALT_DIR.rglob("*.tsv")):
        return RAW_DATA_ALT_DIR
    return RAW_DATA_DIR

def parse_filename_split_event(file_path: Path) -> Tuple[str, str]:
    """
    Extract event name and split from TSV filename.
    Filename expected to be: <event>_<split>.tsv where split is train, dev, or test.
    """
    stem = file_path.stem  # e.g., 'canada_wildfires_2016_train'
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1] in {"train", "dev", "test"}:
        split = parts[-1]
        event = "_".join(parts[:-1])
    else:
        split = "unknown"
        event = stem
    return event, split

def load_all_tsvs(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Recursively discover and load all raw TSV files from raw_dir.
    """
    if raw_dir is None:
        raw_dir = find_raw_data_dir()

    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)

    tsv_files = list(raw_dir.rglob("*.tsv"))
    if not tsv_files:
        logger.warning(f"No TSV files found in {raw_dir}. Creating synthetic sample data...")
        from scripts.make_sample_data import generate_sample_data
        generate_sample_data(raw_dir)
        tsv_files = list(raw_dir.rglob("*.tsv"))

    df_list = []
    for filepath in tsv_files:
        event, split = parse_filename_split_event(filepath)
        try:
            df_file = pd.read_csv(
                filepath,
                sep="\t",
                quoting=csv.QUOTE_NONE,
                dtype=str,
                keep_default_na=False
            )
            df_file["event"] = event
            df_file["split"] = split
            df_file["source_file"] = filepath.name
            df_list.append(df_file)
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {e}")

    if not df_list:
        raise ValueError(f"No valid TSV files could be read from {raw_dir}")

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def validate_and_clean_data(df: pd.DataFrame, target_schema: str = CLASSIFICATION_TARGET) -> pd.DataFrame:
    """
    Validate columns, clean raw text, derive timestamp, map labels, drop duplicates.
    """
    required_cols = {"tweet_id", "tweet_text", "class_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Keep tweet_id as string
    df["tweet_id"] = df["tweet_id"].astype(str).str.strip()

    # Drop null / empty text rows
    df["tweet_text"] = df["tweet_text"].astype(str)
    df = df[df["tweet_text"].str.strip() != ""].copy()

    # Unescape HTML entities
    df["raw_unescaped_text"] = df["tweet_text"].apply(lambda t: html.unescape(t))

    # Derive timestamp
    df["timestamp"] = df["tweet_id"].apply(derive_timestamp_from_id)

    # Label processing
    df["raw_label"] = df["class_label"].astype(str).str.strip()

    # Map raw_label to project_5 category
    unmapped_counts = {}
    categories = []
    for label in df["raw_label"]:
        if label in CATEGORY_MAPPING:
            categories.append(CATEGORY_MAPPING[label])
        else:
            categories.append("General")
            unmapped_counts[label] = unmapped_counts.get(label, 0) + 1

    df["category"] = categories

    if unmapped_counts:
        logger.warning(f"Unmapped labels defaulted to 'General': {unmapped_counts}")

    # Set target column based on target_schema
    if target_schema == "raw_10":
        df["target"] = df["raw_label"]
    else:
        df["target"] = df["category"]

    # Simple normalization for deduplication
    norm_text = df["raw_unescaped_text"].str.lower().str.strip()
    df["_norm_text"] = norm_text

    # Deduplicate keeping first occurrence
    initial_len = len(df)
    df = df.drop_duplicates(subset=["_norm_text"]).copy()
    dedup_len = len(df)
    logger.info(f"Deduplicated dataset from {initial_len} to {dedup_len} rows (dropped {initial_len - dedup_len} duplicates).")

    # Confirm train and test splits do not overlap in normalized text
    train_texts = set(df[df["split"] == "train"]["_norm_text"])
    test_mask = (df["split"] == "test") & (df["_norm_text"].isin(train_texts))
    if test_mask.any():
        logger.warning(f"Found {test_mask.sum()} test texts overlapping with train split. Removing them from test set.")
        df = df[~test_mask].copy()

    df = df.drop(columns=["_norm_text"])

    # Log class distribution and flag imbalance
    log_dataset_statistics(df)

    return df

def log_dataset_statistics(df: pd.DataFrame) -> None:
    """Log distribution per split and event, flagging imbalance."""
    logger.info("=== Dataset Statistics ===")
    logger.info(f"Total rows: {len(df)}")

    split_dist = df["split"].value_counts().to_dict()
    logger.info(f"Split distribution: {split_dist}")

    cat_dist = df["category"].value_counts().to_dict()
    logger.info(f"Category distribution: {cat_dist}")

    event_dist = df["event"].value_counts().to_dict()
    logger.info(f"Events ({len(event_dist)}): {event_dist}")

    # Flag class imbalance
    if cat_dist:
        max_cat = max(cat_dist.values())
        min_cat = min(cat_dist.values())
        ratio = max_cat / max(min_cat, 1)
        logger.info(f"Class imbalance ratio (max/min): {ratio:.2f}x")

def load_data(raw_dir: Optional[Path] = None, target_schema: str = CLASSIFICATION_TARGET) -> pd.DataFrame:
    """High-level function to load and validate dataset."""
    raw_df = load_all_tsvs(raw_dir)
    clean_df = validate_and_clean_data(raw_df, target_schema=target_schema)
    return clean_df
