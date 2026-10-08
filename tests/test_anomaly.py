"""
Unit tests for src/anomaly.py.
"""

import pandas as pd
from datetime import datetime, timedelta
from src.anomaly import compute_robust_z_score, detect_event_surges

def test_compute_robust_z_score_flat():
    series = pd.Series([10.0] * 20)
    z = compute_robust_z_score(series)
    assert (z == 0.0).all()

def test_compute_robust_z_score_outlier():
    data = [5.0] * 19 + [100.0]
    series = pd.Series(data)
    z = compute_robust_z_score(series)
    assert z.iloc[-1] > 3.0

def test_detect_event_surges_empty():
    assert detect_event_surges(pd.DataFrame()) == []
    assert detect_event_surges(pd.DataFrame({"foo": [1, 2]})) == []

def test_detect_event_surges_flat_no_alert():
    # Regular 1 tweet per hour over 24 hours
    base_time = datetime(2018, 1, 1, 0, 0, 0)
    rows = []
    for i in range(24):
        rows.append({
            "tweet_id": f"t_{i}",
            "event": "test_event",
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "urgency_level": "LOW"
        })
    df = pd.DataFrame(rows)
    alerts = detect_event_surges(df, z_threshold=2.5)
    assert len(alerts) == 0

def test_detect_event_surges_volume_spike_detected():
    base_time = datetime(2018, 1, 1, 0, 0, 0)
    rows = []
    # 20 hours with 2 tweets each
    for i in range(20):
        for j in range(2):
            rows.append({
                "tweet_id": f"t_{i}_{j}",
                "event": "test_event",
                "timestamp": (base_time + timedelta(hours=i, minutes=j*10)).isoformat(),
                "urgency_level": "LOW"
            })
    # Hour 21 has massive burst of 50 tweets, many CRITICAL
    for j in range(50):
        rows.append({
            "tweet_id": f"burst_{j}",
            "event": "test_event",
            "timestamp": (base_time + timedelta(hours=21, minutes=j)).isoformat(),
            "urgency_level": "CRITICAL"
        })
    df = pd.DataFrame(rows)
    alerts = detect_event_surges(df, z_threshold=2.0)
    assert len(alerts) > 0
    types = [a["alert_type"] for a in alerts]
    assert any("Burst" in t or "Surge" in t for t in types)
