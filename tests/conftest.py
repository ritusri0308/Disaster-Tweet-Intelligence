"""
Pytest configuration and test fixtures.

IMPORTANT: the three CRISISLENS_* directory variables are redirected to a throwaway temp directory
*before any ``src`` module is imported*. Without this, tests that train models or write reports would
overwrite the real production model in ``models/`` and the evaluation artifacts in ``reports/``.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_ISOLATED_ROOT = Path(tempfile.mkdtemp(prefix="crisislens_tests_"))
os.environ["CRISISLENS_DATA_DIR"] = str(_ISOLATED_ROOT / "data")
os.environ["CRISISLENS_MODELS_DIR"] = str(_ISOLATED_ROOT / "models")
os.environ["CRISISLENS_REPORTS_DIR"] = str(_ISOLATED_ROOT / "reports")
os.environ["CRISISLENS_MIN_VOCAB"] = "5"  # test fixtures train on tiny vocabularies
os.environ.pop("CRISISLENS_API_KEY", None)
os.environ.pop("CRISISLENS_ALLOW_SYNTHETIC", None)
atexit.register(shutil.rmtree, _ISOLATED_ROOT, ignore_errors=True)

import pytest  # noqa: E402
import pandas as pd  # noqa: E402

TINY_MODEL_TEXTS = [
    ("sos people trapped injured under rubble need rescue", "Emergency"),
    ("five killed and many injured in the collapse", "Emergency"),
    ("dead bodies recovered casualties reported hospital full", "Emergency"),
    ("need food and water please help we have nothing", "Help Request"),
    ("urgent need medical supplies and blankets for families", "Help Request"),
    ("please send boats rescue us we are stranded on the roof", "Help Request"),
    ("bridge collapsed road closed power lines down", "Damage Report"),
    ("flooded streets and damaged homes across the district", "Damage Report"),
    ("highway washed out electricity grid is down", "Damage Report"),
    ("official advisory evacuation order issued for coastal zones", "Information/Update"),
    ("shelter opened at the school for evacuees updates follow", "Information/Update"),
    ("volunteers are collecting donations for relief efforts", "Information/Update"),
    ("praying for everyone affected sending love and support", "General"),
    ("thoughts are with the families watching the news at home", "General"),
    ("nice weather today having coffee", "General"),
]

@pytest.fixture(scope="module")
def tiny_model():
    """Train a small classifier and save it at the (isolated) production model path."""
    from src.classifier import build_model_pipeline
    from src.config import MODEL_PATH
    from src.model_registry import save_model_bundle

    pipe = build_model_pipeline("logistic_regression")
    pipe.fit([t for t, _ in TINY_MODEL_TEXTS], [y for _, y in TINY_MODEL_TEXTS])
    save_model_bundle(pipe, MODEL_PATH)
    yield MODEL_PATH
    for p in (MODEL_PATH, MODEL_PATH.with_name(MODEL_PATH.stem + ".meta.json")):
        if p.exists():
            p.unlink()

@pytest.fixture
def rich_dataframe():
    """A realistic processed DataFrame (all pipeline columns) including hostile tweet text."""
    events = ["hurricane_harvey_2017", "hurricane_maria_2017", "kaikoura_earthquake_2016"]
    cats = ["Emergency", "Help Request", "Damage Report", "Information/Update", "General"]
    levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "LOW"]
    places = [
        {"name": "Houston", "lat": 29.76, "lon": -95.36, "country_code": "US", "confidence": 0.95, "entity_type": "city"},
        {"name": "Puerto Rico", "lat": 18.47, "lon": -66.11, "country_code": "PR", "confidence": 0.85, "entity_type": "country"},
        {"name": "Kaikoura", "lat": -42.4, "lon": 173.68, "country_code": "NZ", "confidence": 0.85, "entity_type": "town"},
    ]
    hostile = [
        "<img src=x onerror=alert(1)> URGENT help needed, $5 & *bold* _under_",
        "SOS! 5 people trapped in collapsed building near Houston!",
        "Death toll hits 83 after the quake, at least 12 injured",
        "Need clean drinking water and food in Puerto Rico",
        "Road closed, bridge washed out near Kaikoura",
        "Praying for everyone affected",
    ]
    rows = []
    base = pd.Timestamp("2017-08-26 00:00:00", tz="UTC")
    for i in range(60):
        ev_i = i % 3
        text = hostile[i % len(hostile)]
        rows.append({
            "tweet_id": str(870000000000000000 + i),
            "event": events[ev_i],
            "split": ["test", "train", "dev"][i % 3] if i % 7 else "test",
            "tweet_text": text,
            "raw_unescaped_text": text,
            "clean_text": text.lower(),
            "raw_label": "requests_or_urgent_needs",
            "category": cats[i % 5],
            "target": cats[i % 5],
            "confidence": round(0.35 + (i % 13) * 0.05, 2),
            "locations": [places[ev_i]["name"]],
            "geocoded_locations": [places[ev_i]],
            "needs": ["Water", "Food"] if i % 4 == 0 else [],
            "quantities": [{"kind": "killed", "count": 83}] if i % 6 == 2 else [],
            "matched_terms": ["critical_kw:trapped"] if i % 5 == 0 else [],
            "urgency_score": float([9.5, 6.5, 4.0, 1.0, 0.5][i % 5]),
            "urgency_level": levels[i % 5],
            "timestamp": base + pd.Timedelta(hours=i * 2),
            "uppercase_ratio": 0.1,
            "exclamation_count": i % 3,
            "is_retweet": i % 2,
            "model_version": "testmodel0001",
        })
    return pd.DataFrame(rows)

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
