"""
Unit tests for data_loader module.
"""

import pytest
import pandas as pd
from src.data_loader import (
    derive_timestamp_from_id,
    load_all_tsvs,
    validate_and_clean_data,
    parse_filename_split_event,
    load_data
)

def test_derive_timestamp_from_id_valid():
    # 870161518553251840 -> 2017-06-01 06:14 UTC
    ts = derive_timestamp_from_id("870161518553251840")
    assert pd.notna(ts)
    assert ts.year == 2017
    assert ts.month == 6
    assert ts.day == 1
    assert ts.hour == 6
    assert ts.minute == 14

def test_derive_timestamp_from_id_invalid():
    ts = derive_timestamp_from_id("invalid_non_numeric_id")
    assert pd.isna(ts)
    
    ts_none = derive_timestamp_from_id(None)
    assert pd.isna(ts_none)

def test_load_all_tsvs_and_nested_discovery(temp_raw_dir):
    df = load_all_tsvs(temp_raw_dir)
    assert not df.empty
    assert "event" in df.columns
    assert "split" in df.columns
    
    events = set(df["event"].unique())
    assert "canada_wildfires_2016" in events
    assert "greece_wildfires_2018" in events
    
    splits = set(df["split"].unique())
    assert {"train", "dev", "test"}.issubset(splits)

def test_validate_and_clean_data(temp_raw_dir):
    raw_df = load_all_tsvs(temp_raw_dir)
    clean_df = validate_and_clean_data(raw_df)
    
    assert "raw_unescaped_text" in clean_df.columns
    assert "category" in clean_df.columns
    assert "timestamp" in clean_df.columns
    assert "&amp;" not in clean_df["raw_unescaped_text"].iloc[0]

def test_validate_missing_columns():
    bad_df = pd.DataFrame({"some_col": [1, 2]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_and_clean_data(bad_df)
