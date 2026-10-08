"""
Unit tests for src/clustering.py.
"""

import pandas as pd
from src.clustering import cluster_event_incidents, generate_semantic_2d_projection

def test_cluster_event_incidents_empty():
    assert cluster_event_incidents(pd.DataFrame()) == []
    assert cluster_event_incidents(pd.DataFrame({"foo": [1]})) == []

def test_cluster_event_incidents_groups_duplicates():
    # Construct tweets where 3 are nearly identical and 1 is completely distinct
    df = pd.DataFrame([
        {
            "tweet_id": "1",
            "clean_text": "urgent rescue needed boat flooding houston downtown street water rising",
            "raw_unescaped_text": "Urgent rescue needed boat flooding Houston downtown street water rising!",
            "event": "harvey_2017",
            "category": "Emergency",
            "urgency_score": 8.5,
            "confidence": 0.95,
            "locations": ["Houston"],
            "timestamp": "2017-08-26 12:00:00"
        },
        {
            "tweet_id": "2",
            "clean_text": "urgent rescue needed boat flooding houston downtown street water rising please help",
            "raw_unescaped_text": "Urgent rescue needed boat flooding Houston downtown street water rising please help!",
            "event": "harvey_2017",
            "category": "Emergency",
            "urgency_score": 8.5,
            "confidence": 0.92,
            "locations": ["Houston"],
            "timestamp": "2017-08-26 12:05:00"
        },
        {
            "tweet_id": "3",
            "clean_text": "urgent rescue needed boat flooding houston street water rising save us",
            "raw_unescaped_text": "Urgent rescue needed boat flooding Houston street water rising save us",
            "event": "harvey_2017",
            "category": "Emergency",
            "urgency_score": 8.0,
            "confidence": 0.90,
            "locations": ["Houston"],
            "timestamp": "2017-08-26 12:10:00"
        },
        {
            "tweet_id": "4",
            "clean_text": "sunny beautiful weather in california enjoying vacation today",
            "raw_unescaped_text": "Sunny beautiful weather in California enjoying vacation today",
            "event": "harvey_2017",
            "category": "General",
            "urgency_score": 1.0,
            "confidence": 0.88,
            "locations": [],
            "timestamp": "2017-08-26 12:15:00"
        }
    ])

    incidents = cluster_event_incidents(df, similarity_threshold=0.5)
    assert len(incidents) >= 1
    main_inc = incidents[0]
    assert main_inc["size"] >= 2
    assert "rescue" in " ".join(main_inc["top_terms"]) or "houston" in " ".join(main_inc["top_terms"])
    assert main_inc["dominant_category"] == "Emergency"

def test_generate_semantic_2d_projection():
    df = pd.DataFrame([
        {
            "tweet_id": f"t_{i}",
            "clean_text": f"disaster emergency rescue flood storm relief food medical assistance unit {i}",
            "raw_unescaped_text": f"Disaster emergency rescue flood storm relief food medical assistance unit {i}",
            "category": "Emergency" if i % 2 == 0 else "Help Request",
            "urgency_level": "HIGH",
            "urgency_score": 7.0,
            "event": "harvey_2017"
        }
        for i in range(20)
    ])

    proj_df = generate_semantic_2d_projection(df, max_points=20, n_topics=2)
    assert not proj_df.empty
    assert "x" in proj_df.columns
    assert "y" in proj_df.columns
    assert "topic_label" in proj_df.columns
    assert len(proj_df) == 20
