"""
Unit tests for src/briefing.py.
"""

import pandas as pd
from src.briefing import generate_markdown_briefing, generate_html_briefing

def test_generate_markdown_briefing():
    df = pd.DataFrame([
        {"tweet_id": "1", "urgency_level": "CRITICAL", "urgency_score": 8.5, "category": "Emergency"},
        {"tweet_id": "2", "urgency_level": "HIGH", "urgency_score": 6.5, "category": "Help Request"},
        {"tweet_id": "3", "urgency_level": "LOW", "urgency_score": 1.0, "category": "General"}
    ])
    summary_data = {"summary_text": "Severe coastal flooding reported."}
    incidents = [
        {
            "incident_id": "INC-001",
            "size": 5,
            "priority_rank": 8.5,
            "dominant_category": "Emergency",
            "representative_tweet": "Flooding on main street send boat",
            "places": ["Houston"],
            "time_span": "2017-08-26"
        }
    ]
    surges = [{"alert_type": "VOLUME_SPIKE", "event": "harvey_2017", "details": "High volume", "z_score": 3.1}]
    filters = {"Event": "harvey_2017"}

    md = generate_markdown_briefing(df, summary_data, incidents, surges, filters)
    assert "# 🚨 CrisisLens - Emergency Situation Briefing" in md
    assert "Total Disaster Signals Analyzed" in md
    assert "CRITICAL Priority Alerts" in md
    assert "INC-001" in md
    assert "VOLUME_SPIKE" in md

def test_generate_html_briefing_escaping():
    df = pd.DataFrame([
        {"tweet_id": "1", "urgency_level": "CRITICAL", "category": "Emergency"}
    ])
    summary_data = {"summary_text": "<script>alert('bad')</script>"}
    incidents = [
        {
            "incident_id": "INC-001",
            "size": 2,
            "priority_rank": 5.0,
            "dominant_category": "Emergency",
            "representative_tweet": "<img src=x onerror=alert(1)> Urgent help!",
            "places": ["<b>Houston</b>"],
            "time_span": "2017-08-26"
        }
    ]
    surges = []
    filters = {}

    html_out = generate_html_briefing(df, summary_data, incidents, surges, filters)
    assert "<!DOCTYPE html>" in html_out
    assert "CrisisLens" in html_out
    # Check that dangerous tags are properly escaped
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "<img" not in html_out
    assert "&lt;img" in html_out
