"""
Model integrity, versioning, and health helpers for CrisisLens.

A model file is identified by a content fingerprint. The fingerprint is stored next to every
processed dataset (``model_version`` column) so the dashboard can detect when the predictions it
displays were produced by a different model than the one currently loaded, and a metadata sidecar
(``best_classifier.meta.json``) records how the model was trained.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib

from src.config import MIN_EXPECTED_VOCAB, MODEL_PATH

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def fingerprint_file(path: PathLike = MODEL_PATH) -> Optional[str]:
    """Return a short SHA-256 content fingerprint of the model file, or None if it is missing."""
    p = Path(path)
    if not p.exists():
        return None
    digest = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _unwrap_estimator(pipeline: Any) -> Any:
    """Return the final estimator of a pipeline (or the object itself)."""
    steps = getattr(pipeline, "named_steps", None)
    if steps and "classifier" in steps:
        return steps["classifier"]
    return pipeline


def describe_pipeline(pipeline: Any) -> Dict[str, Any]:
    """Summarize a fitted pipeline: estimator family, vocabulary size, and class labels."""
    info: Dict[str, Any] = {"model_name": type(_unwrap_estimator(pipeline)).__name__, "vocab_size": 0, "classes": []}
    steps = getattr(pipeline, "named_steps", {}) or {}
    tfidf = steps.get("tfidf")
    if tfidf is not None:
        vocab = getattr(tfidf, "vocabulary_", None)
        if vocab is not None:
            info["vocab_size"] = len(vocab)
        elif hasattr(tfidf, "get_feature_names_out"):
            try:
                info["vocab_size"] = len(tfidf.get_feature_names_out())
            except Exception:
                pass
    classes = getattr(pipeline, "classes_", None)
    if classes is not None:
        info["classes"] = [str(c) for c in classes]
    return info


def write_model_metadata(
    model_path: PathLike,
    pipeline: Any,
    extra: Optional[Dict[str, Any]] = None,
    meta_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Write the metadata sidecar for a freshly saved model and return it."""
    model_path = Path(model_path)
    meta = {
        "model_version": fingerprint_file(model_path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        **describe_pipeline(pipeline),
        **(extra or {}),
    }
    target = Path(meta_path) if meta_path else model_path.with_name(model_path.stem + ".meta.json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta


def save_model_bundle(
    pipeline: Any,
    model_path: PathLike = MODEL_PATH,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a model and its metadata sidecar together."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return write_model_metadata(model_path, pipeline, extra)


def read_model_metadata(model_path: PathLike = MODEL_PATH) -> Dict[str, Any]:
    """Read the metadata sidecar next to a model; empty dict when absent or unreadable."""
    meta_path = Path(model_path).with_name(Path(model_path).stem + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def model_health(
    pipeline: Any,
    model_path: PathLike = MODEL_PATH,
    min_vocab: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Describe whether the loaded model is trustworthy.

    ``status`` is ``missing`` (nothing loaded), ``degraded`` (loaded but implausibly small, which is
    the signature of a test fixture overwriting the production model), or ``healthy``.
    """
    min_vocab = MIN_EXPECTED_VOCAB if min_vocab is None else min_vocab
    version = fingerprint_file(model_path)
    if pipeline is None:
        return {"status": "missing", "version": version, "model_name": None, "vocab_size": 0,
                "reason": "No trained model could be loaded. Run `python train.py`."}

    info = describe_pipeline(pipeline)
    meta = read_model_metadata(model_path)
    status, reason = "healthy", ""
    if info["vocab_size"] < min_vocab:
        status = "degraded"
        reason = (f"Model vocabulary has only {info['vocab_size']} terms (expected at least {min_vocab}); "
                  "it looks like a test fixture. Re-run `python train.py`.")
    return {
        "status": status,
        "version": version,
        "model_name": info["model_name"],
        "vocab_size": info["vocab_size"],
        "classes": info["classes"],
        "trained_at": meta.get("trained_at"),
        "reason": reason,
    }


def dataset_matches_model(df_versions: Any, live_version: Optional[str]) -> Dict[str, Any]:
    """
    Compare the ``model_version`` values stamped on a processed dataset with the live model.

    Returns ``{"state": "ok" | "stale" | "unstamped" | "unknown", "dataset_versions": [...]}``.
    """
    if live_version is None:
        return {"state": "unknown", "dataset_versions": []}
    if df_versions is None:
        return {"state": "unstamped", "dataset_versions": []}
    versions = sorted({str(v) for v in df_versions if v is not None and str(v) not in ("", "nan", "None")})
    if not versions:
        return {"state": "unstamped", "dataset_versions": []}
    if versions == [live_version]:
        return {"state": "ok", "dataset_versions": versions}
    return {"state": "stale", "dataset_versions": versions}
