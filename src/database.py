"""
SQLite database storage, schema migration, and feedback persistence module for CrisisLens.
Uses parameterized queries exclusively.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
from datetime import datetime, timezone
import pandas as pd

from src.config import DB_PATH

logger = logging.getLogger(__name__)

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
    geocoded_locations TEXT
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
    timestamp TEXT NOT NULL
);
"""

def get_db_connection(db_path: Union[str, Path] = DB_PATH) -> sqlite3.Connection:
    """Create and return SQLite connection."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Union[str, Path] = DB_PATH) -> None:
    """Initialize SQLite database schema and run backward-compatible migrations."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_SQL)
        cursor.execute(CREATE_FEEDBACK_TABLE_SQL)

        # Migration: check if 'needs' and 'geocoded_locations' exist in existing tables
        cursor.execute("PRAGMA table_info(processed_tweets);")
        columns = [row["name"] for row in cursor.fetchall()]
        if "needs" not in columns:
            cursor.execute("ALTER TABLE processed_tweets ADD COLUMN needs TEXT;")
        if "geocoded_locations" not in columns:
            cursor.execute("ALTER TABLE processed_tweets ADD COLUMN geocoded_locations TEXT;")

        conn.commit()
    logger.info(f"Database initialized and migrated at {db_path}")

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
        needs, geocoded_locations
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    records = []
    for _, row in df.iterrows():
        t_id = str(row.get("tweet_id", ""))
        event = str(row.get("event", ""))
        split = str(row.get("split", ""))
        text = str(row.get("raw_unescaped_text", row.get("tweet_text", "")))
        clean_text = str(row.get("clean_text", ""))
        raw_label = str(row.get("raw_label", ""))
        category = str(row.get("category", "General"))
        confidence = float(row.get("confidence", 1.0))

        # Handle JSON fields
        locs = row.get("locations", [])
        if isinstance(locs, (list, set, tuple)):
            locs_json = json.dumps(list(locs))
        elif isinstance(locs, str):
            locs_json = locs
        else:
            locs_json = "[]"

        urgency_score = float(row.get("urgency_score", 0.0))
        urgency_level = str(row.get("urgency_level", "LOW"))

        terms = row.get("matched_terms", [])
        if isinstance(terms, (list, set, tuple)):
            terms_json = json.dumps(list(terms))
        elif isinstance(terms, str):
            terms_json = terms
        else:
            terms_json = "[]"

        ts = str(row.get("timestamp", ""))

        needs = row.get("needs", [])
        if isinstance(needs, (list, set, tuple)):
            needs_json = json.dumps(list(needs))
        elif isinstance(needs, str):
            needs_json = needs
        else:
            needs_json = "[]"

        geo = row.get("geocoded_locations", [])
        if isinstance(geo, (list, set, tuple)):
            geo_json = json.dumps(list(geo))
        elif isinstance(geo, str):
            geo_json = geo
        else:
            geo_json = "[]"

        records.append((
            t_id, event, split, text, clean_text, raw_label, category,
            confidence, locs_json, urgency_score, urgency_level, terms_json, ts,
            needs_json, geo_json
        ))

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(insert_sql, records)
        conn.commit()

    logger.info(f"Saved {len(records)} records into SQLite DB at {db_path}")
    return len(records)

def load_processed_tweets(db_path: Union[str, Path] = DB_PATH, split: Optional[str] = None) -> pd.DataFrame:
    """
    Load processed tweets from SQLite database into pandas DataFrame.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    if split:
        query = "SELECT * FROM processed_tweets WHERE split = ?"
        df = pd.read_sql_query(query, conn, params=(split,))
    else:
        query = "SELECT * FROM processed_tweets"
        df = pd.read_sql_query(query, conn)
    conn.close()

    # Parse JSON fields safely
    if not df.empty:
        for col in ["locations", "matched_terms", "needs", "geocoded_locations"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else (x if isinstance(x, list) else []))

    return df

def get_critical_reports(db_path: Union[str, Path] = DB_PATH, min_score: float = 5.5) -> pd.DataFrame:
    """Query high and critical reports (urgency_score >= min_score)."""
    conn = get_db_connection(db_path)
    query = "SELECT * FROM processed_tweets WHERE urgency_score >= ? ORDER BY urgency_score DESC"
    df = pd.read_sql_query(query, conn, params=(min_score,))
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
    db_path: Union[str, Path] = DB_PATH
) -> int:
    """Store operator correction into feedback table using parameterized query."""
    init_db(db_path)
    now_utc = datetime.now(timezone.utc).isoformat()
    sql = """
    INSERT INTO feedback (
        tweet_id, original_category, corrected_category,
        original_urgency, corrected_urgency, user_notes, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (
            str(tweet_id), str(original_category), str(corrected_category),
            str(original_urgency), str(corrected_urgency), str(user_notes), now_utc
        ))
        conn.commit()
        last_id = cursor.lastrowid
    return last_id

def load_all_feedback(db_path: Union[str, Path] = DB_PATH) -> pd.DataFrame:
    """Load all human feedback records."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC", conn)
    conn.close()
    return df

def get_feedback_stats(db_path: Union[str, Path] = DB_PATH) -> Dict[str, Any]:
    """Summary statistics of human review feedback."""
    df = load_all_feedback(db_path)
    if df.empty:
        return {"total_feedback_count": 0, "category_corrections": 0, "urgency_corrections": 0}

    cat_changes = (df["original_category"] != df["corrected_category"]).sum()
    urg_changes = (df["original_urgency"] != df["corrected_urgency"]).sum()

    return {
        "total_feedback_count": len(df),
        "category_corrections": int(cat_changes),
        "urgency_corrections": int(urg_changes)
    }
