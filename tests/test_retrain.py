"""
Tests for the human-feedback retraining loop (retrain_with_feedback.py).
"""

import joblib
import pandas as pd

from retrain_with_feedback import build_feedback_training_set, retrain_model_with_human_feedback
from src.classifier import build_model_pipeline
from src.database import (
    get_feedback_stats,
    get_reviewed_tweet_ids,
    load_latest_feedback,
    save_feedback,
)
from src.model_registry import read_model_metadata, save_model_bundle

TEXTS = {
    "Emergency": ["sos people trapped injured", "killed and injured in collapse", "dead bodies recovered"],
    "Help Request": ["need food and water please", "urgent need medical supplies", "send boats rescue us"],
    "General": ["praying for everyone", "thoughts with the families", "nice weather today"],
}

def _dataset() -> pd.DataFrame:
    rows, n = [], 0
    for split, repeats in (("train", 6), ("dev", 2), ("test", 2)):
        for category, texts in TEXTS.items():
            for text in texts:
                for r in range(repeats):
                    n += 1
                    rows.append({"tweet_id": f"{split}-{n}", "split": split, "event": "e1",
                                 "clean_text": f"{text} {r}", "target": category})
    return pd.DataFrame(rows)

def _fb(tweet_id, original, corrected, split="", text="", event=""):
    return {"tweet_id": tweet_id, "original_category": original, "corrected_category": corrected,
            "split": split, "tweet_text": text, "event": event}

def test_train_split_corrections_relabel_training_data():
    df = _dataset()
    train_id = df[df.split == "train"].iloc[0]["tweet_id"]
    fb = pd.DataFrame([_fb(train_id, "General", "Emergency", split="train")])
    train_df, held_out, summary = build_feedback_training_set(df, fb)
    assert summary["train_relabeled"] == 1
    assert train_df.loc[train_df.tweet_id == train_id, "target"].iloc[0] == "Emergency"
    assert held_out.empty

def test_dev_and_test_corrections_are_held_out_never_trained_on():
    df = _dataset()
    test_row = df[df.split == "test"].iloc[0]
    dev_row = df[df.split == "dev"].iloc[0]
    fb = pd.DataFrame([
        _fb(test_row.tweet_id, "General", "Emergency", split="test"),
        _fb(dev_row.tweet_id, "General", "Help Request", split="dev"),
    ])
    train_df, held_out, summary = build_feedback_training_set(df, fb)
    assert summary["train_relabeled"] == 0
    assert summary["held_out"] == 2
    assert set(held_out["target"]) == {"Emergency", "Help Request"}
    # the training set is exactly the original train split
    assert len(train_df) == (df.split == "train").sum()
    assert test_row.tweet_id not in set(train_df.tweet_id)

def test_live_feedback_with_text_adds_training_rows():
    df = _dataset()
    fb = pd.DataFrame([
        _fb("api-1", "General", "Help Request", split="live", text="Please send a boat, water is rising https://t.co/x"),
        _fb("api-2", "General", "Emergency", split="live", text=""),  # no text: cannot be used
    ])
    train_df, _, summary = build_feedback_training_set(df, fb)
    assert summary["live_added"] == 1
    assert summary["skipped_no_text"] == 1
    added = train_df[train_df.tweet_id == "api-1"].iloc[0]
    assert added["target"] == "Help Request"
    assert "https" not in added["clean_text"]  # cleaned like any other tweet

def test_confirmations_without_a_change_are_ignored():
    df = _dataset()
    train_id = df[df.split == "train"].iloc[0]["tweet_id"]
    fb = pd.DataFrame([_fb(train_id, "Emergency", "Emergency", split="train")])
    _, _, summary = build_feedback_training_set(df, fb)
    assert summary["corrections"] == 0

def test_retrain_skips_when_nothing_usable(tmp_path):
    df = _dataset()
    db = tmp_path / "fb.db"
    model_path = tmp_path / "model.joblib"
    pipe = build_model_pipeline("logistic_regression").fit(df.clean_text, df.target)
    save_model_bundle(pipe, model_path)
    before = model_path.read_bytes()

    test_id = df[df.split == "test"].iloc[0]["tweet_id"]
    save_feedback(test_id, "General", "Emergency", "LOW", "HIGH", db_path=db, split="test")

    promoted = retrain_model_with_human_feedback(
        db_path=db, model_path=model_path, backup_path=tmp_path / "model.bak", df=df
    )
    assert promoted is False
    assert model_path.read_bytes() == before
    assert not (tmp_path / "model.bak").exists()

def test_retrain_with_usable_feedback_runs_and_keeps_a_backup_when_promoted(tmp_path):
    df = _dataset()
    db = tmp_path / "fb.db"
    model_path = tmp_path / "model.joblib"
    backup_path = tmp_path / "model.bak"
    pipe = build_model_pipeline("logistic_regression").fit(df[df.split == "train"].clean_text, df[df.split == "train"].target)
    save_model_bundle(pipe, model_path)
    original = model_path.read_bytes()

    # An operator relabels a training tweet to what it already says: harmless, so the candidate should not regress.
    row = df[(df.split == "train") & (df.target == "Emergency")].iloc[0]
    save_feedback(row.tweet_id, "General", "Emergency", "LOW", "HIGH", db_path=db, split="train")

    promoted = retrain_model_with_human_feedback(db_path=db, model_path=model_path, backup_path=backup_path, df=df)
    assert isinstance(promoted, bool)
    if promoted:
        assert backup_path.read_bytes() == original
        meta = read_model_metadata(model_path)
        assert meta["retrained_with_feedback"] is True
        assert meta["feedback_summary"]["train_relabeled"] == 1
        assert joblib.load(model_path) is not None
    else:
        assert model_path.read_bytes() == original

def test_review_helpers_latest_row_wins_and_stats(tmp_path):
    db = tmp_path / "fb.db"
    save_feedback("t1", "General", "Emergency", "LOW", "HIGH", db_path=db, split="train")
    save_feedback("t1", "General", "Help Request", "LOW", "HIGH", db_path=db, split="train")  # operator changed their mind
    save_feedback("t2", "General", "Emergency", "LOW", "HIGH", db_path=db, split="test")

    latest = load_latest_feedback(db).set_index("tweet_id")
    assert latest.loc["t1", "corrected_category"] == "Help Request"
    assert get_reviewed_tweet_ids(db) == {"t1", "t2"}

    stats = get_feedback_stats(db)
    assert stats["total_feedback_count"] == 3
    assert stats["unique_tweets_reviewed"] == 2
    assert stats["usable_for_training"] == 1   # t1 on train
    assert stats["evaluation_only"] == 1       # t2 on test
