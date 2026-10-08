"""
Unit tests for analytics module.
"""

import pandas as pd
from src.analytics import (
    get_category_counts,
    get_urgency_counts,
    get_category_urgency_crosstab,
    get_top_locations,
    get_volume_over_time,
    generate_situation_summary
)

def test_analytics_counts_and_crosstab(sample_dataframe):
    cats = get_category_counts(sample_dataframe)
    assert not cats.empty
    assert "Emergency" in cats.index

    urgs = get_urgency_counts(sample_dataframe)
    assert not urgs.empty
    assert "CRITICAL" in urgs.index

    ct = get_category_urgency_crosstab(sample_dataframe)
    assert not ct.empty

def test_get_top_locations(sample_dataframe):
    top_locs = get_top_locations(sample_dataframe)
    assert not top_locs.empty
    assert "Houston" in top_locs.index

def test_get_volume_over_time_missing_timestamp():
    df_no_ts = pd.DataFrame({"tweet_id": ["1"], "category": ["Emergency"]})
    vol = get_volume_over_time(df_no_ts)
    assert vol.empty

def test_generate_situation_summary(sample_dataframe):
    summary = generate_situation_summary(sample_dataframe)
    assert summary["total_tweets"] == 2
    assert summary["critical_count"] == 1
    assert "CRITICAL" in summary["summary_text"]
