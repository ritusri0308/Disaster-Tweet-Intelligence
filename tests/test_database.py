"""
Unit tests for database module.
"""

import tempfile
from pathlib import Path

from src.database import (
    init_db,
    save_processed_tweets,
    load_processed_tweets,
    get_critical_reports
)

def test_database_operations(sample_dataframe):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

        init_db(db_path)
        count = save_processed_tweets(sample_dataframe, db_path)
        assert count == len(sample_dataframe)

        df_loaded = load_processed_tweets(db_path)
        assert len(df_loaded) == len(sample_dataframe)

        crit_df = get_critical_reports(db_path, min_score=7.5)
        assert len(crit_df) == 1
        assert crit_df.iloc[0]["urgency_level"] == "CRITICAL"

def test_sql_injection_safety(sample_dataframe):
    sample_df = sample_dataframe.copy()
    sql_injection_str = "'; DROP TABLE processed_tweets; --"
    sample_df.at[0, "tweet_text"] = sql_injection_str
    sample_df.at[0, "raw_unescaped_text"] = sql_injection_str

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
        save_processed_tweets(sample_df, db_path)

        df_loaded = load_processed_tweets(db_path)
        assert len(df_loaded) == len(sample_df)
        assert df_loaded.iloc[0]["text"] == sql_injection_str
