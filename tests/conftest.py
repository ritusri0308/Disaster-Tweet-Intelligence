"""
Pytest configuration and test fixtures.
"""

import tempfile
from pathlib import Path
import pytest
import pandas as pd

@pytest.fixture
def temp_raw_dir():
    """Create a temporary directory structure mimicking the raw HumAID dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_events_dir = tmp_path / "events_set1"
        raw_events_dir.mkdir(parents=True, exist_ok=True)

        # Standard event
        event1_dir = raw_events_dir / "canada_wildfires_2016"
        event1_dir.mkdir(parents=True, exist_ok=True)

        # Nested event
        nested_dir = raw_events_dir / "hurricane_matthew_2016" / "greece_wildfires_2018"
        nested_dir.mkdir(parents=True, exist_ok=True)

        # Create TSV files with quotes & HTML entities
        df1 = pd.DataFrame([
            {"tweet_id": "870161518553251840", "tweet_text": "SOS! 5 people trapped &amp; injured inside building! #help", "class_label": "injured_or_dead_people"},
            {"tweet_id": "870161518553251841", "tweet_text": "Clean water &amp; medical supplies needed immediately in Houston", "class_label": "requests_or_urgent_needs"},
            {"tweet_id": "870161518553251842", "tweet_text": "Bridge collapsed on highway after severe earthquake", "class_label": "infrastructure_and_utility_damage"}
        ])

        df2 = pd.DataFrame([
            {"tweet_id": "870161518553251843", "tweet_text": "Stay away from coastal zones due to flood advisory", "class_label": "caution_and_advice"},
            {"tweet_id": "870161518553251844", "tweet_text": "Donate to Red Cross hurricane relief campaign. Please RT!", "class_label": "rescue_volunteering_or_donation_effort"},
            {"tweet_id": "invalid_id", "tweet_text": "Praying for everyone affected by the wildfires in Greece", "class_label": "sympathy_and_support"}
        ])

        for split in ["train", "dev", "test"]:
            f1 = event1_dir / f"canada_wildfires_2016_{split}.tsv"
            df1.to_csv(f1, sep="\t", index=False)

            f2 = nested_dir / f"greece_wildfires_2018_{split}.tsv"
            df2.to_csv(f2, sep="\t", index=False)

        yield raw_events_dir

@pytest.fixture
def sample_dataframe():
    """Create a sample processed DataFrame for unit testing."""
    return pd.DataFrame([
        {
            "tweet_id": "870161518553251840",
            "event": "canada_wildfires_2016",
            "split": "test",
            "tweet_text": "SOS! People trapped near Houston Central!",
            "raw_unescaped_text": "SOS! People trapped near Houston Central!",
            "clean_text": "sos people trapped near houston central",
            "raw_label": "injured_or_dead_people",
            "category": "Emergency",
            "target": "Emergency",
            "confidence": 0.95,
            "locations": ["Houston", "Houston Central"],
            "urgency_score": 9.5,
            "urgency_level": "CRITICAL",
            "matched_terms": ["critical_kw:trapped", "signal:sos"],
            "timestamp": pd.to_datetime("2017-06-01 06:14:00", utc=True),
            "uppercase_ratio": 0.25,
            "exclamation_count": 2,
            "is_retweet": 0
        },
        {
            "tweet_id": "870161518553251841",
            "event": "cyclone_idai_2019",
            "split": "test",
            "tweet_text": "Heavy rain advisory tomorrow",
            "raw_unescaped_text": "Heavy rain advisory tomorrow",
            "clean_text": "heavy rain advisory tomorrow",
            "raw_label": "caution_and_advice",
            "category": "Information/Update",
            "target": "Information/Update",
            "confidence": 0.88,
            "locations": [],
            "urgency_score": 2.0,
            "urgency_level": "LOW",
            "matched_terms": ["medium_kw:advisory"],
            "timestamp": pd.to_datetime("2017-06-01 07:00:00", utc=True),
            "uppercase_ratio": 0.05,
            "exclamation_count": 0,
            "is_retweet": 0
        }
    ])
