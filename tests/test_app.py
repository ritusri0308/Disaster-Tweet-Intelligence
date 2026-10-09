"""
End-to-end smoke tests for the Streamlit application entry point (app.py) using AppTest.
Data, model, and reports live in the isolated temp directories configured by conftest.py.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.model_registry as model_registry
from src.config import MODEL_PATH, PROCESSED_PARQUET_PATH
from src.model_registry import fingerprint_file

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

@pytest.fixture(autouse=True)
def fresh_streamlit_caches():
    """app.py caches the dataset and model; every test must start from a clean slate."""
    st.cache_data.clear()
    st.cache_resource.clear()
    PROCESSED_PARQUET_PATH.unlink(missing_ok=True)
    yield
    st.cache_data.clear()
    st.cache_resource.clear()
    PROCESSED_PARQUET_PATH.unlink(missing_ok=True)

def _run():
    return AppTest.from_file(APP_PATH, default_timeout=90).run()

def _html(at):
    return " ".join(m.value for m in at.markdown)

def test_app_load_smoke_without_any_data():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    html = _html(at)
    assert "No processed dataset was found" in html
    assert "run_pipeline.py" in html

def test_app_loads_with_dataset_and_model(rich_dataframe, tiny_model):
    rich_dataframe["model_version"] = fingerprint_file(MODEL_PATH)
    rich_dataframe.to_parquet(PROCESSED_PARQUET_PATH, index=False)

    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    html = _html(at)
    assert "Situation Room" in html
    assert "tweets in view" in html
    assert "No processed dataset was found" not in html
    assert "out of sync" not in html and "looks wrong" not in html
    # the sidebar filter widgets are present
    labels = [s.label for s in at.sidebar.selectbox]
    assert {"Data split", "Disaster event", "Category", "Urgency level"} <= set(labels)

def test_app_warns_when_dataset_was_scored_by_a_different_model(rich_dataframe, tiny_model):
    rich_dataframe["model_version"] = "someoldmodel"
    rich_dataframe.to_parquet(PROCESSED_PARQUET_PATH, index=False)
    at = _run()
    assert not at.exception
    html = _html(at)
    assert "Dataset and model are out of sync" in html
    assert "someoldmodel" in html

def test_app_notes_unstamped_datasets(rich_dataframe, tiny_model):
    rich_dataframe.drop(columns=["model_version"]).to_parquet(PROCESSED_PARQUET_PATH, index=False)
    at = _run()
    assert not at.exception
    assert "predates model versioning" in _html(at)

def test_app_warns_when_the_model_is_implausibly_small(rich_dataframe, tiny_model, monkeypatch):
    # This is exactly what happened when a test fixture overwrote the production model.
    monkeypatch.setattr(model_registry, "MIN_EXPECTED_VOCAB", 1_000_000)
    rich_dataframe["model_version"] = fingerprint_file(MODEL_PATH)
    rich_dataframe.to_parquet(PROCESSED_PARQUET_PATH, index=False)
    at = _run()
    assert not at.exception
    assert "Model looks wrong" in _html(at)

def test_app_warns_when_no_model_is_trained(rich_dataframe):
    MODEL_PATH.unlink(missing_ok=True)
    rich_dataframe.to_parquet(PROCESSED_PARQUET_PATH, index=False)
    at = _run()
    assert not at.exception
    assert "No trained model" in _html(at)

def test_personal_details_are_hidden_by_default_and_revealed_only_on_request(rich_dataframe, tiny_model):
    # A real help request from the dataset: a home address and a phone number inside the tweet.
    pii_text = "Please help, this lady needs to be rescue 6210 Charles ave, Port Arthur , tx 713 5427751, 80 year old"
    idx = rich_dataframe.index[1]
    for col in ("raw_unescaped_text", "tweet_text"):
        rich_dataframe.loc[idx, col] = pii_text
    rich_dataframe.loc[idx, "clean_text"] = pii_text.lower()
    rich_dataframe.loc[idx, ["split", "urgency_level", "urgency_score"]] = ["test", "CRITICAL", 10.0]
    rich_dataframe["model_version"] = fingerprint_file(MODEL_PATH)
    rich_dataframe.to_parquet(PROCESSED_PARQUET_PATH, index=False)

    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    html = _html(at)
    assert "[phone]" in html and "[address]" in html
    assert "5427751" not in html and "6210 Charles" not in html
    assert "Port Arthur" in html          # the operational detail (where) is kept

    at.sidebar.toggle(key="show_pii").set_value(True).run()
    assert not at.exception
    assert "5427751" in _html(at)

    at.sidebar.toggle(key="show_pii").set_value(False).run()
    assert "5427751" not in _html(at), "hiding details again must not serve a cached copy that shows them"
