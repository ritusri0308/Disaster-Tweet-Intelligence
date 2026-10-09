"""
Unit tests for pipeline module.
"""

import pandas as pd
import pytest

from src.config import MODEL_PATH
from src.data_loader import load_all_tsvs
from src.model_registry import fingerprint_file
from src.pipeline import run_end_to_end_pipeline

pytestmark = pytest.mark.usefixtures("tiny_model")

def test_run_end_to_end_pipeline(temp_raw_dir):
    df_out = run_end_to_end_pipeline(raw_dir=temp_raw_dir, save_outputs=False)

    assert not df_out.empty
    assert "category" in df_out.columns
    assert "confidence" in df_out.columns
    assert "locations" in df_out.columns
    assert "urgency_score" in df_out.columns
    assert "urgency_level" in df_out.columns
    assert "matched_terms" in df_out.columns
    assert "quantities" in df_out.columns

def test_pipeline_stamps_every_row_with_the_model_version(temp_raw_dir):
    df_out = run_end_to_end_pipeline(raw_dir=temp_raw_dir, save_outputs=False)
    assert set(df_out["model_version"]) == {fingerprint_file(MODEL_PATH)}

def test_pipeline_refuses_to_invent_data_when_raw_files_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="--demo"):
        run_end_to_end_pipeline(raw_dir=tmp_path / "does_not_exist", save_outputs=False)

def test_synthetic_demo_data_only_when_explicitly_requested(tmp_path):
    raw_dir = tmp_path / "demo_raw"
    with pytest.raises(FileNotFoundError):
        load_all_tsvs(raw_dir)
    df = load_all_tsvs(raw_dir, allow_synthetic=True)
    assert isinstance(df, pd.DataFrame) and not df.empty

def test_synthetic_demo_data_via_environment_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("CRISISLENS_ALLOW_SYNTHETIC", "1")
    assert not load_all_tsvs(tmp_path / "demo_env").empty
