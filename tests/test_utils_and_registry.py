"""
Tests for src/utils.py (list-column handling) and src/model_registry.py (model integrity).
"""

import json

import numpy as np
import pandas as pd

from src.classifier import build_model_pipeline
from src.model_registry import (
    dataset_matches_model,
    describe_pipeline,
    fingerprint_file,
    model_health,
    read_model_metadata,
    save_model_bundle,
)
from src.utils import as_list, normalize_list_columns

# ---- as_list / normalize_list_columns -------------------------------------------------------------

def test_as_list_handles_every_storage_format():
    assert as_list(["a", "b"]) == ["a", "b"]
    assert as_list(("a", "b")) == ["a", "b"]
    assert as_list(np.array(["a", "b"])) == ["a", "b"]          # what Parquet returns
    assert as_list('["a", "b"]') == ["a", "b"]                   # what CSV / SQLite return
    assert as_list(None) == []
    assert as_list(float("nan")) == []
    assert as_list(np.float64("nan")) == []
    assert as_list("") == []
    assert as_list("single") == ["single"]
    assert as_list("[not json") == []

def test_as_list_keeps_dict_elements():
    arr = np.array([{"name": "Houston", "lat": 29.7}], dtype=object)
    assert as_list(arr) == [{"name": "Houston", "lat": 29.7}]

def test_parquet_round_trip_is_normalised(tmp_path, rich_dataframe):
    path = tmp_path / "t.parquet"
    rich_dataframe.to_parquet(path, index=False)
    raw = pd.read_parquet(path)
    # The bug this guards: numpy arrays are not `list`, so isinstance(x, list) checks silently saw nothing.
    assert not isinstance(raw["locations"].iloc[0], list)
    clean = normalize_list_columns(raw)
    for col in ("locations", "needs", "matched_terms", "geocoded_locations", "quantities"):
        assert all(isinstance(v, list) for v in clean[col]), col
    assert clean["geocoded_locations"].iloc[0][0]["name"] == "Houston"

# ---- model registry ------------------------------------------------------------------------------

def _fitted():
    pipe = build_model_pipeline("logistic_regression")
    pipe.fit(["sos trapped people", "need food and water", "praying for all", "bridge collapsed road"],
             ["Emergency", "Help Request", "General", "Damage Report"])
    return pipe

def test_fingerprint_changes_with_content_and_is_none_when_missing(tmp_path):
    assert fingerprint_file(tmp_path / "nope.joblib") is None
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert fingerprint_file(a) != fingerprint_file(b)
    assert fingerprint_file(a) == fingerprint_file(a)
    assert len(fingerprint_file(a)) == 12

def test_save_bundle_writes_metadata_sidecar(tmp_path):
    path = tmp_path / "m" / "model.joblib"
    meta = save_model_bundle(_fitted(), path, extra={"selected_model": "logistic_regression"})
    assert path.exists()
    assert meta["model_version"] == fingerprint_file(path)
    on_disk = json.loads((tmp_path / "m" / "model.meta.json").read_text())
    assert on_disk["selected_model"] == "logistic_regression"
    assert on_disk["vocab_size"] == describe_pipeline(_fitted())["vocab_size"] > 0
    assert read_model_metadata(path)["model_version"] == meta["model_version"]
    assert read_model_metadata(tmp_path / "missing.joblib") == {}

def test_model_health_states(tmp_path):
    path = tmp_path / "model.joblib"
    pipe = _fitted()
    save_model_bundle(pipe, path)

    assert model_health(None, path)["status"] == "missing"
    # The toy 4-sentence model is "degraded" under a production-sized vocabulary requirement...
    degraded = model_health(pipe, path, min_vocab=1000)
    assert degraded["status"] == "degraded"
    assert "test fixture" in degraded["reason"]
    # ...and healthy when the requirement is met.
    healthy = model_health(pipe, path, min_vocab=5)
    assert healthy["status"] == "healthy"
    assert healthy["version"] == fingerprint_file(path)

def test_dataset_matches_model_states():
    assert dataset_matches_model(["abc", "abc"], "abc")["state"] == "ok"
    assert dataset_matches_model(["abc", "old"], "abc")["state"] == "stale"
    assert dataset_matches_model([None, "", "nan"], "abc")["state"] == "unstamped"
    assert dataset_matches_model(None, "abc")["state"] == "unstamped"
    assert dataset_matches_model(["abc"], None)["state"] == "unknown"
