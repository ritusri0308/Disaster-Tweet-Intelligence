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
    get_per_event_summary,
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
    assert summary["top_location"] == "Houston"
    assert "Houston" in summary["summary_text"]

def test_generate_situation_summary_no_places():
    df_no_places = pd.DataFrame([
        {
            "tweet_id": "1",
            "category": "General",
            "urgency_level": "LOW",
            "locations": [],
            "geocoded_locations": [],
            "timestamp": pd.to_datetime("2017-06-01 06:14:00", utc=True)
        }
    ])
    summary = generate_situation_summary(df_no_places)
    assert summary["top_location"] == "None detected"
    assert "No specific locations detected in the active view." in summary["summary_text"]

def test_get_per_event_summary(sample_dataframe):
    summary_df = get_per_event_summary(sample_dataframe)
    assert not summary_df.empty
    assert "event" in summary_df.columns
    assert "total_tweets" in summary_df.columns

def test_analytics_empty_dataframes():
    empty_df = pd.DataFrame()
    assert get_category_counts(empty_df).empty
    assert get_urgency_counts(empty_df).empty
    assert get_category_urgency_crosstab(empty_df).empty
    assert get_top_locations(empty_df).empty
    assert get_per_event_summary(empty_df).empty
    sum_res = generate_situation_summary(empty_df)
    assert sum_res["total_tweets"] == 0
