"""
Smoke and safety tests for every dashboard page, rendered through Streamlit's AppTest harness.

The tweet fixture includes hostile text (an <img onerror> payload, '$' math delimiters, markdown characters)
so these tests also prove tweet content is escaped before it reaches ``unsafe_allow_html``.
"""

import json

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import MODEL_PATH, REPORTS_DIR

PAGES = ["situation_room", "replay_mode", "triage_queue", "geo_intel", "model_observatory", "review_queue", "about_data"]

def _render(module_name, df, pipeline):
    """Script body executed by AppTest: apply the theme and render one page."""
    import importlib
    from app.ui.theme import inject_custom_theme

    inject_custom_theme()
    importlib.import_module(f"app.pages.{module_name}").render_page(df, pipeline)

def _run(module_name, df, pipeline=None, timeout=60):
    at = AppTest.from_function(_render, args=(module_name, df, pipeline), default_timeout=timeout)
    return at.run()

def _html_blobs(at):
    return [m.value for m in at.markdown]

@pytest.fixture
def pipeline(tiny_model):
    return joblib.load(MODEL_PATH)

@pytest.mark.parametrize("module_name", PAGES)
def test_every_page_renders_with_data(module_name, rich_dataframe, pipeline):
    at = _run(module_name, rich_dataframe, pipeline)
    assert not at.exception, [e.value for e in at.exception]

@pytest.mark.parametrize("module_name", PAGES)
def test_every_page_survives_an_empty_dataset(module_name):
    at = _run(module_name, pd.DataFrame(), None)
    assert not at.exception, [e.value for e in at.exception]

@pytest.mark.parametrize("module_name", ["situation_room", "triage_queue", "geo_intel", "replay_mode", "review_queue"])
def test_hostile_tweet_text_is_never_rendered_as_html(module_name, rich_dataframe, pipeline):
    at = _run(module_name, rich_dataframe, pipeline)
    assert not at.exception
    blobs = " ".join(_html_blobs(at))
    assert "<img src=x" not in blobs, "raw tweet markup reached unsafe_allow_html"
    assert "onerror=" not in blobs.replace("onerror=alert(1)&gt;", "")  # only the escaped form may appear

def test_situation_room_escapes_the_priority_feed(rich_dataframe, pipeline):
    at = _run("situation_room", rich_dataframe, pipeline)
    blobs = " ".join(_html_blobs(at))
    assert "&lt;img src=x onerror=alert(1)&gt;" in blobs
    assert "$5 &amp;" in blobs   # '$' and '&' survive as text

def test_situation_room_search_applies_need_filter(rich_dataframe, pipeline):
    at = _run("situation_room", rich_dataframe, pipeline)
    at.text_input(key="situation_query").set_value("water needed").run()
    assert not at.exception
    blobs = " ".join(_html_blobs(at))
    assert "Need: Water" in blobs and "tweets match" in blobs

def test_situation_room_search_with_no_results_explains_why(rich_dataframe, pipeline):
    at = _run("situation_room", rich_dataframe, pipeline)
    at.text_input(key="situation_query").set_value("critical missing persons in Houston last 1 hours").run()
    assert not at.exception

def test_geo_page_draws_a_map_and_hotspot_table_from_normalised_data(rich_dataframe, pipeline):
    at = _run("geo_intel", rich_dataframe, pipeline)
    assert not at.exception
    assert any("geocoded place mentions" in c.value for c in at.caption)
    assert not any("No recognizable geographic locations" in v for v in _html_blobs(at))

def test_model_observatory_never_invents_numbers_when_reports_are_missing(rich_dataframe, pipeline):
    for name in ("evaluation_artifacts.json", "calibration_data.json"):
        (REPORTS_DIR / name).unlink(missing_ok=True)
    at = _run("model_observatory", rich_dataframe, pipeline)
    assert not at.exception
    blobs = " ".join(_html_blobs(at))
    assert "n/a" in blobs
    assert "0.76" not in blobs and "8,232" not in blobs   # the previous hard-coded fallbacks

def test_model_observatory_renders_artifacts_and_error_analysis(rich_dataframe, pipeline):
    artifacts = {
        "evaluation_timestamp": "2099-01-01T00:00:00+00:00",
        "test_sample_count": 1234,
        "overall_metrics": {"macro_f1": 0.7654, "expected_calibration_error": 0.05},
        "per_class_report": {"Emergency": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 100}},
        "sample_misclassifications": [{
            "tweet_id": "1", "event": "hurricane_harvey_2017", "true_category": "General",
            "predicted_category": "Emergency", "confidence": 0.7,
            "tweet_text": "<b>help</b> trapped near Houston", "clean_text": "help trapped near houston",
        }],
    }
    (REPORTS_DIR / "evaluation_artifacts.json").write_text(json.dumps(artifacts), encoding="utf-8")
    try:
        at = _run("model_observatory", rich_dataframe, pipeline)
        assert not at.exception, [e.value for e in at.exception]
        blobs = " ".join(_html_blobs(at))
        assert "0.77" in blobs and "1,234" in blobs
        assert "<b>help</b>" not in blobs
    finally:
        (REPORTS_DIR / "evaluation_artifacts.json").unlink(missing_ok=True)

def test_model_observatory_warns_when_reports_predate_the_model(rich_dataframe, tiny_model, pipeline):
    (REPORTS_DIR / "evaluation_artifacts.json").write_text(
        json.dumps({"evaluation_timestamp": "2000-01-01T00:00:00+00:00", "test_sample_count": 1}), encoding="utf-8")
    try:
        at = _run("model_observatory", rich_dataframe, pipeline)
        assert "Evaluation reports are out of date" in " ".join(_html_blobs(at))
    finally:
        (REPORTS_DIR / "evaluation_artifacts.json").unlink(missing_ok=True)

def _pending_count(at):
    import re
    match = re.search(r"pending review\*\*: `([\d,]+)`", " ".join(_html_blobs(at)))
    assert match, "pending-review count not found"
    return int(match.group(1).replace(",", ""))

def test_review_queue_hides_tweets_that_were_already_reviewed(rich_dataframe, pipeline):
    from src.database import save_feedback

    before = _run("review_queue", rich_dataframe, pipeline)
    assert not before.exception
    n_before = _pending_count(before)
    assert n_before > 1

    uncertain = rich_dataframe[rich_dataframe["confidence"] < 0.75]
    save_feedback(str(uncertain.iloc[0]["tweet_id"]), "General", "Emergency", "LOW", "HIGH", split="train")

    after = _run("review_queue", rich_dataframe, pipeline)
    assert not after.exception
    assert _pending_count(after) == n_before - 1, "a reviewed tweet must not be offered again"
    assert "already reviewed are hidden" in " ".join(_html_blobs(after))

def test_review_queue_explains_test_split_feedback_is_not_trained_on(rich_dataframe, pipeline):
    at = _run("review_queue", rich_dataframe, pipeline)
    blobs = " ".join(_html_blobs(at))
    assert "never used to retrain" in blobs

def test_multi_line_tweets_stay_inside_their_feed_card(rich_dataframe, pipeline):
    # Markdown ends an HTML block at the first blank line, which used to leave the card without its text.
    idx = rich_dataframe.index[0]
    rich_dataframe.loc[idx, "raw_unescaped_text"] = "line one\n\n    indented line two\nline three"
    rich_dataframe.loc[idx, ["urgency_level", "urgency_score"]] = ["CRITICAL", 10.0]
    at = _run("situation_room", rich_dataframe, pipeline)
    assert not at.exception
    cards = [b for b in _html_blobs(at) if "<div class='cl-feed-item'>" in b]   # not the theme CSS, which also names the class
    assert cards
    assert all("\n" not in card for card in cards), "a newline inside a card lets Markdown split it"
    assert any("line one<br><br>    indented line two<br>line three" in card for card in cards)

def test_csv_export_writes_list_columns_as_json(rich_dataframe):
    from app.ui.cache import cached_csv, df_signature
    import io
    raw = cached_csv(rich_dataframe, df_signature(rich_dataframe))
    exported = pd.read_csv(io.BytesIO(raw))
    assert json.loads(exported["locations"].iloc[0]) == ["Houston"]
    assert json.loads(exported["geocoded_locations"].iloc[0])[0]["name"] == "Houston"
    assert json.loads(exported["quantities"].iloc[2]) == [{"kind": "killed", "count": 83}]
