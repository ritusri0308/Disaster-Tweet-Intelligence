"""
Tests that natural-language query filters are actually applied (not just parsed and displayed as chips).
"""

import pandas as pd

from src.nlquery import apply_query_filters, parse_natural_language_query

def _frame():
    base = pd.Timestamp("2017-09-20 00:00:00", tz="UTC")
    rows = []
    for i in range(10):
        rows.append({
            "tweet_id": str(i),
            "event": "hurricane_maria_2017" if i < 6 else "hurricane_harvey_2017",
            "category": "Help Request" if i % 2 == 0 else "Emergency",
            "urgency_level": "CRITICAL" if i % 3 == 0 else "LOW",
            "locations": ["Puerto Rico"] if i < 6 else ["Houston"],
            "needs": ["Water"] if i % 2 == 0 else ["Medical", "Food"],
            "timestamp": base + pd.Timedelta(hours=i * 12),
        })
    return pd.DataFrame(rows)

def test_need_type_filter_is_applied():
    filters = parse_natural_language_query("clean drinking water needed")["filters"]
    assert filters["need_type"] == "Water"
    out, applied = apply_query_filters(_frame(), filters)
    assert len(out) == 5 and all("Water" in n for n in out["needs"])
    assert any("need = Water" in a for a in applied)

def test_relative_window_counts_back_from_the_newest_tweet():
    filters = parse_natural_language_query("emergency help in last 24 hours")["filters"]
    assert filters["date_range"] == "last_24h"
    out, applied = apply_query_filters(_frame(), filters)
    ts = pd.to_datetime(out["timestamp"], utc=True)
    assert ts.max() - ts.min() <= pd.Timedelta(hours=24)
    assert len(out) < len(_frame())
    assert any("last 24 hours" in a for a in applied)

def test_flexible_windows():
    assert parse_natural_language_query("critical in the last 3 days")["filters"]["date_range"] == "last_72h"
    assert parse_natural_language_query("critical in the past 6 hours")["filters"]["date_range"] == "last_6h"
    assert parse_natural_language_query("critical 48h")["filters"]["date_range"] == "last_48h"

def test_date_after_uses_the_events_own_year():
    f = parse_natural_language_query("hurricane maria urgent requests after sep 20")["filters"]
    assert f["date_after"] == "2017-09-20"
    f2 = parse_natural_language_query("harvey before Aug 30")["filters"]
    assert f2["date_before"] == "2017-08-30"

def test_explicit_year_wins_and_yearless_dates_resolve_against_the_data():
    explicit = parse_natural_language_query("reports since October 3 2016")["filters"]
    assert explicit["date_after"] == "2016-10-03"

    yearless = parse_natural_language_query("reports after sep 22")
    assert yearless["filters"]["date_after"] == "09-22"   # no event, no year: resolved when applied
    out, applied = apply_query_filters(_frame(), yearless["filters"])
    assert pd.to_datetime(out["timestamp"], utc=True).min() >= pd.Timestamp("2017-09-22", tz="UTC")
    assert any("on or after 2017-09-22" in a for a in applied)

def test_before_filter():
    out, applied = apply_query_filters(_frame(), {"date_before": "2017-09-22"})
    assert pd.to_datetime(out["timestamp"], utc=True).max() < pd.Timestamp("2017-09-22", tz="UTC")
    assert any("before 2017-09-22" in a for a in applied)

def test_combined_filters_and_applied_list_only_contains_real_filters():
    parsed = parse_natural_language_query("critical help requests in Puerto Rico")
    out, applied = apply_query_filters(_frame(), parsed["filters"])
    assert set(out["urgency_level"]) <= {"CRITICAL"}
    assert set(out["category"]) <= {"Help Request"}
    assert all("Puerto Rico" in locs for locs in out["locations"])
    assert len(applied) == len(parsed["filters"])

def test_numpy_array_columns_are_handled():
    df = _frame()
    df["needs"] = df["needs"].apply(lambda n: __import__("numpy").array(n, dtype=object))
    df["locations"] = df["locations"].apply(lambda n: __import__("numpy").array(n, dtype=object))
    out, _ = apply_query_filters(df, {"need_type": "Medical", "location": "Puerto"})
    assert len(out) > 0

def test_filters_for_missing_columns_are_skipped_not_reported():
    out, applied = apply_query_filters(pd.DataFrame({"x": [1, 2]}), {"category": "Emergency", "need_type": "Water"})
    assert len(out) == 2 and applied == []
