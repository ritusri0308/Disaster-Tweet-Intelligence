"""
SQLite database storage, schema migration, and feedback persistence module for CrisisLens.
Uses parameterized queries exclusively.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Any, Set, Union, Optional
from datetime import datetime, timezone
import pandas as pd

from src.config import DB_PATH
from src.utils import LIST_COLUMNS

logger = logging.getLogger(__name__)

JSON_COLUMNS = LIST_COLUMNS

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS processed_tweets (
    tweet_id TEXT PRIMARY KEY,
    event TEXT,
    split TEXT,
    text TEXT,
    clean_text TEXT,
    raw_label TEXT,
    category TEXT,
    confidence REAL,
    locations TEXT,
    urgency_score REAL,
    urgency_level TEXT,
    matched_terms TEXT,
    timestamp TEXT,
    needs TEXT,
    geocoded_locations TEXT,
    quantities TEXT,
    model_version TEXT
);
"""

CREATE_FEEDBACK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id TEXT NOT NULL,
    original_category TEXT,
    corrected_category TEXT,
    original_urgency TEXT,
    corrected_urgency TEXT,
    user_notes TEXT,
    timestamp TEXT NOT NULL,
    tweet_text TEXT,
    event TEXT,
    split TEXT,
    reviewer TEXT,
    model_version TEXT
);
"""

# Columns added after the first release; init_db() adds any that an older database is missing.
_TWEET_MIGRATION_COLUMNS = {
    "needs": "TEXT",
    "geocoded_locations": "TEXT",
    "quantities": "TEXT",
    "model_version": "TEXT",
}
_FEEDBACK_MIGRATION_COLUMNS = {
    "tweet_text": "TEXT",
    "event": "TEXT",
    "split": "TEXT",
    "reviewer": "TEXT",
    "model_version": "TEXT",
}

def get_db_connection(db_path: Union[str, Path] = DB_PATH) -> sqlite3.Connection:
    """Create and return SQLite connection."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def _add_missing_columns(cursor: sqlite3.Cursor, table: str, wanted: Dict[str, str]) -> None:
    cursor.execute(f"PRAGMA table_info({table});")
    existing = {row["name"] for row in cursor.fetchall()}
    for column, col_type in wanted.items():
        if column not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type};")

def init_db(db_path: Union[str, Path] = DB_PATH) -> None:
    """Initialize SQLite database schema and run backward-compatible migrations."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_SQL)
        cursor.execute(CREATE_FEEDBACK_TABLE_SQL)
        _add_missing_columns(cursor, "processed_tweets", _TWEET_MIGRATION_COLUMNS)
        _add_missing_columns(cursor, "feedback", _FEEDBACK_MIGRATION_COLUMNS)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_tweet ON feedback(tweet_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tweets_event ON processed_tweets(event);")
        conn.commit()
    finally:
        conn.close()
    logger.info(f"Database initialized and migrated at {db_path}")

def _json_field(value: Any) -> str:
    """Serialise a list-like column to JSON text (already-serialised strings pass through)."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, set, tuple)):
        return json.dumps(list(value), default=str)
    if hasattr(value, "tolist"):  # numpy arrays read back from Parquet
        return json.dumps(value.tolist(), default=str)
    return "[]"

def save_processed_tweets(df: pd.DataFrame, db_path: Union[str, Path] = DB_PATH) -> int:
    """
    Save processed DataFrame to SQLite database using parameterized insert queries.
    Overwrites existing records with matching tweet_id.
    """
    init_db(db_path)

    insert_sql = """
    INSERT OR REPLACE INTO processed_tweets (
        tweet_id, event, split, text, clean_text, raw_label, category,
        confidence, locations, urgency_score, urgency_level, matched_terms, timestamp,
        needs, geocoded_locations, quantities, model_version
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    records = []
    for row in df.to_dict("records"):
        records.append((
            str(row.get("tweet_id", "")),
            str(row.get("event", "")),
            str(row.get("split", "")),
            str(row.get("raw_unescaped_text", row.get("tweet_text", ""))),
            str(row.get("clean_text", "")),
            str(row.get("raw_label", "")),
            str(row.get("category", "General")),
            float(row.get("confidence", 1.0)),
            _json_field(row.get("locations", [])),
            float(row.get("urgency_score", 0.0)),
            str(row.get("urgency_level", "LOW")),
            _json_field(row.get("matched_terms", [])),
            str(row.get("timestamp", "")),
            _json_field(row.get("needs", [])),
            _json_field(row.get("geocoded_locations", [])),
            _json_field(row.get("quantities", [])),
            str(row.get("model_version", "") or ""),
        ))

    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.executemany(insert_sql, records)
        conn.commit()
    finally:
        conn.close()

    logger.info(f"Saved {len(records)} records into SQLite DB at {db_path}")
    return len(records)

def load_processed_tweets(db_path: Union[str, Path] = DB_PATH, split: Optional[str] = None) -> pd.DataFrame:
    """
    Load processed tweets from SQLite database into pandas DataFrame.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        if split:
            query = "SELECT * FROM processed_tweets WHERE split = ?"
            df = pd.read_sql_query(query, conn, params=(split,))
        else:
            query = "SELECT * FROM processed_tweets"
            df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    # Parse JSON fields safely
    if not df.empty:
        for col in JSON_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else (x if isinstance(x, list) else []))

    return df

def get_critical_reports(db_path: Union[str, Path] = DB_PATH, min_score: float = 5.5) -> pd.DataFrame:
    """Query high and critical reports (urgency_score >= min_score)."""
    conn = get_db_connection(db_path)
    try:
        query = "SELECT * FROM processed_tweets WHERE urgency_score >= ? ORDER BY urgency_score DESC"
        df = pd.read_sql_query(query, conn, params=(min_score,))
    finally:
        conn.close()
    return df

# Human-in-the-loop Feedback Queries
def save_feedback(
    tweet_id: str,
    original_category: str,
    corrected_category: str,
    original_urgency: str,
    corrected_urgency: str,
    user_notes: str = "",
    db_path: Union[str, Path] = DB_PATH,
    tweet_text: str = "",
    event: str = "",
    split: str = "",
    reviewer: str = "",
    model_version: str = "",
) -> int:
    """
    Store operator correction into feedback table using parameterized query.

    Every submission is kept as an audit row; ``load_latest_feedback`` returns the most recent row per
    tweet. ``tweet_text``, ``event`` and ``split`` are stored so a correction stays usable for retraining
    even when the tweet is not part of the benchmark dataset (for example text submitted through the API).
    """
    init_db(db_path)
    now_utc = datetime.now(timezone.utc).isoformat()
    sql = """
    INSERT INTO feedback (
        tweet_id, original_category, corrected_category,
        original_urgency, corrected_urgency, user_notes, timestamp,
        tweet_text, event, split, reviewer, model_version
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (
            str(tweet_id), str(original_category), str(corrected_category),
            str(original_urgency), str(corrected_urgency), str(user_notes), now_utc,
            str(tweet_text), str(event), str(split), str(reviewer), str(model_version)
        ))
        conn.commit()
        last_id = cursor.lastrowid
    finally:
        conn.close()
    return last_id

def load_all_feedback(db_path: Union[str, Path] = DB_PATH) -> pd.DataFrame:
    """Load all human feedback records (full audit history, newest first)."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC", conn)
    finally:
        conn.close()
    return df

def load_latest_feedback(db_path: Union[str, Path] = DB_PATH) -> pd.DataFrame:
    """Load only the most recent feedback row for each tweet (the operator's current decision)."""
    df = load_all_feedback(db_path)
    if df.empty:
        return df
    return df.drop_duplicates(subset="tweet_id", keep="first").reset_index(drop=True)

def get_reviewed_tweet_ids(db_path: Union[str, Path] = DB_PATH) -> Set[str]:
    """Tweet ids that already have operator feedback, so the review queue can skip them."""
    df = load_all_feedback(db_path)
    return set(df["tweet_id"].astype(str)) if not df.empty else set()

def get_feedback_stats(db_path: Union[str, Path] = DB_PATH) -> Dict[str, Any]:
    """
    Summary statistics of human review feedback.

    ``usable_for_training`` counts category corrections that retraining can apply (anything outside the
    held-out test split); ``evaluation_only`` counts corrections on the test split, which are tracked but
    never trained on so the benchmark stays clean.
    """
    all_df = load_all_feedback(db_path)
    if all_df.empty:
        return {
            "total_feedback_count": 0, "category_corrections": 0, "urgency_corrections": 0,
            "unique_tweets_reviewed": 0, "usable_for_training": 0, "evaluation_only": 0,
        }

    cat_changes = (all_df["original_category"] != all_df["corrected_category"]).sum()
    urg_changes = (all_df["original_urgency"] != all_df["corrected_urgency"]).sum()

    latest = load_latest_feedback(db_path)
    corrected = latest[latest["original_category"] != latest["corrected_category"]]
    split_col = corrected["split"].fillna("") if "split" in corrected.columns else pd.Series("", index=corrected.index)
    evaluation_only = int((split_col == "test").sum())

    return {
        "total_feedback_count": len(all_df),
        "category_corrections": int(cat_changes),
        "urgency_corrections": int(urg_changes),
        "unique_tweets_reviewed": int(len(latest)),
        "usable_for_training": int(len(corrected) - evaluation_only),
        "evaluation_only": evaluation_only,
    }
