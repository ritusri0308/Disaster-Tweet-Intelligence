"""
SQLite database storage and query helper module using parameterized queries.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
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
    timestamp TEXT
);
"""

def get_db_connection(db_path: Union[str, Path] = DB_PATH) -> sqlite3.Connection:
    """Create and return SQLite connection."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Union[str, Path] = DB_PATH) -> None:
    """Initialize SQLite database schema."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_SQL)
        conn.commit()
    logger.info(f"Database initialized at {db_path}")

def save_processed_tweets(df: pd.DataFrame, db_path: Union[str, Path] = DB_PATH) -> int:
    """
    Save processed DataFrame to SQLite database using parameterized insert queries.
    Overwrites existing records with matching tweet_id.
    """
    init_db(db_path)
    
    insert_sql = """
    INSERT OR REPLACE INTO processed_tweets (
        tweet_id, event, split, text, clean_text, raw_label, category,
        confidence, locations, urgency_score, urgency_level, matched_terms, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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

        records.append((
            t_id, event, split, text, clean_text, raw_label, category,
            confidence, locs_json, urgency_score, urgency_level, terms_json, ts
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
        if "locations" in df.columns:
            df["locations"] = df["locations"].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else [])
        if "matched_terms" in df.columns:
            df["matched_terms"] = df["matched_terms"].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else [])

    return df

def get_critical_reports(db_path: Union[str, Path] = DB_PATH, min_score: float = 5.5) -> pd.DataFrame:
    """
    Query high and critical reports (urgency_score >= min_score).
    """
    conn = get_db_connection(db_path)
    query = "SELECT * FROM processed_tweets WHERE urgency_score >= ? ORDER BY urgency_score DESC"
    df = pd.read_sql_query(query, conn, params=(min_score,))
    conn.close()
    return df
