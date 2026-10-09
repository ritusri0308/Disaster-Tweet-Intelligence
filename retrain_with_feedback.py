"""
Human-in-the-loop retraining module for CrisisLens.
Merges operator feedback from SQLite into training data, retrains the model,
and only updates the production model if test Macro-F1 does not regress.

How corrections are applied (by the split the reviewed tweet belongs to):
    train  -> the tweet's training label is replaced by the operator's correction
    live   -> tweets that are not in the benchmark (e.g. submitted through the API) are added as new
              training rows, provided the feedback stored their text
    dev / test -> never trained on, so the benchmark stays clean; they are reported as a held-out
              check of how the baseline and candidate models do on operator-corrected tweets

Usage:
    python retrain_with_feedback.py
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import pandas as pd

from src.config import MODEL_PATH, MODEL_BACKUP_PATH, DB_PATH
from src.database import load_latest_feedback
from src.data_loader import load_data
from src.preprocessing import clean_tweet, preprocess_dataframe
from src.classifier import build_model_pipeline, evaluate_model
from src.model_registry import save_model_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def build_feedback_training_set(
    df: pd.DataFrame,
    feedback_latest: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Apply operator category corrections to the benchmark data.

    ``df`` is the preprocessed dataset (needs ``tweet_id``, ``split``, ``clean_text``, ``target``).
    ``feedback_latest`` has one row per tweet (see ``load_latest_feedback``).

    Returns ``(train_df, held_out_df, summary)`` where ``train_df`` has the corrections applied, and
    ``held_out_df`` holds ``clean_text`` / ``target`` for corrected dev and test tweets (not trained on).
    """
    train_df = df[df["split"] == "train"].copy()
    summary = {"corrections": 0, "train_relabeled": 0, "live_added": 0,
               "held_out": 0, "skipped_no_text": 0, "skipped_unknown": 0}
    empty_held_out = pd.DataFrame(columns=["clean_text", "target"])

    if feedback_latest is None or feedback_latest.empty:
        return train_df, empty_held_out, summary

    corrections = feedback_latest[feedback_latest["original_category"] != feedback_latest["corrected_category"]]
    summary["corrections"] = int(len(corrections))
    if corrections.empty:
        return train_df, empty_held_out, summary

    unique_tweets = df.drop_duplicates(subset="tweet_id")
    by_id = unique_tweets.set_index(unique_tweets["tweet_id"].astype(str))
    held_rows, live_rows = [], []
    id_to_label: Dict[str, str] = {}

    for row in corrections.to_dict("records"):
        tweet_id = str(row["tweet_id"])
        label = row["corrected_category"]
        stored_split = str(row.get("split") or "").strip()

        if tweet_id in by_id.index:
            split = by_id.loc[tweet_id, "split"]
            if split == "train":
                id_to_label[tweet_id] = label
            else:  # dev / test: never train on these
                held_rows.append({"clean_text": by_id.loc[tweet_id, "clean_text"], "target": label})
        elif stored_split in ("", "live", "unknown"):
            text = str(row.get("tweet_text") or "").strip()
            if text:
                live_rows.append({"tweet_id": tweet_id, "clean_text": clean_tweet(text), "target": label,
                                  "event": row.get("event") or "", "split": "train"})
            else:
                summary["skipped_no_text"] += 1
        else:
            summary["skipped_unknown"] += 1  # a benchmark tweet that is no longer in the loaded dataset

    if id_to_label:
        mask = train_df["tweet_id"].astype(str).isin(id_to_label)
        train_df.loc[mask, "target"] = train_df.loc[mask, "tweet_id"].astype(str).map(id_to_label)
        summary["train_relabeled"] = int(mask.sum())

    if live_rows:
        train_df = pd.concat([train_df, pd.DataFrame(live_rows)], ignore_index=True)
        summary["live_added"] = len(live_rows)

    held_out_df = pd.DataFrame(held_rows) if held_rows else empty_held_out
    summary["held_out"] = int(len(held_out_df))
    return train_df, held_out_df, summary

def retrain_model_with_human_feedback(
    db_path: Path = DB_PATH,
    model_path: Path = MODEL_PATH,
    backup_path: Path = MODEL_BACKUP_PATH,
    df: Optional[pd.DataFrame] = None
) -> bool:
    """
    Retrain classification pipeline incorporating human operator feedback.
    Returns True if model was promoted, False otherwise.

    ``df`` is the preprocessed benchmark dataset; it is loaded from disk when omitted.
    """
    logger.info("Loading feedback records from SQLite...")
    feedback_df = load_latest_feedback(db_path)

    if feedback_df.empty:
        logger.info("No human feedback entries found in database. Retraining skipped.")
        return False

    logger.info(f"Loaded feedback for {len(feedback_df)} distinct tweets (latest decision per tweet).")

    if df is None:
        df = preprocess_dataframe(load_data())

    train_df, held_out_df, summary = build_feedback_training_set(df, feedback_df)
    test_df = df[df["split"] == "test"].copy()

    logger.info(
        f"Corrections: {summary['corrections']} | applied to train: {summary['train_relabeled']} | "
        f"added from live feedback: {summary['live_added']} | held out (dev/test, not trained on): {summary['held_out']} | "
        f"skipped (no stored text): {summary['skipped_no_text']}"
    )

    if summary["train_relabeled"] + summary["live_added"] == 0:
        logger.info(
            "No corrections are usable for training (corrections on the dev/test splits are evaluation-only). "
            "Review tweets from the train split, or submit live feedback with text, to enable retraining."
        )
        return False

    # Baseline evaluation on existing production model
    current_model = joblib.load(model_path)
    base_metrics = evaluate_model(current_model, test_df["clean_text"], test_df["target"], split_name="test")
    base_macro_f1 = base_metrics["macro_f1"]
    logger.info(f"Current production test Macro-F1: {base_macro_f1:.4f}")

    # Train new candidate pipeline
    logger.info("Training candidate model on feedback-augmented dataset...")
    candidate_pipe = build_model_pipeline(model_name="logistic_regression")
    candidate_pipe.fit(train_df["clean_text"], train_df["target"])

    candidate_metrics = evaluate_model(candidate_pipe, test_df["clean_text"], test_df["target"], split_name="test")
    cand_macro_f1 = candidate_metrics["macro_f1"]
    logger.info(f"Candidate model test Macro-F1: {cand_macro_f1:.4f} (baseline was {base_macro_f1:.4f})")

    if not held_out_df.empty:
        base_acc = float((current_model.predict(held_out_df["clean_text"]) == held_out_df["target"]).mean())
        cand_acc = float((candidate_pipe.predict(held_out_df["clean_text"]) == held_out_df["target"]).mean())
        logger.info(
            f"Accuracy on {len(held_out_df)} operator-corrected dev/test tweets (not trained on): "
            f"baseline {base_acc:.1%} -> candidate {cand_acc:.1%}"
        )

    if cand_macro_f1 >= base_macro_f1:
        # Promote candidate model
        logger.info("Candidate model improved or matched baseline. Promoting to production!")
        if Path(model_path).exists():
            shutil.copyfile(model_path, backup_path)
            logger.info(f"Backed up previous model to {backup_path}")
        meta = save_model_bundle(candidate_pipe, model_path, extra={
            "selected_model": "logistic_regression",
            "retrained_with_feedback": True,
            "feedback_summary": summary,
            "test_macro_f1": cand_macro_f1,
            "baseline_test_macro_f1": base_macro_f1,
        })
        logger.info(
            f"New model version {meta['model_version']}. Re-run `python run_pipeline.py` so the dashboard "
            "dataset is scored by the promoted model."
        )
        return True

    logger.warning(
        f"Candidate model regressed ({cand_macro_f1:.4f} < {base_macro_f1:.4f}). "
        "Keeping current production model."
    )
    return False

if __name__ == "__main__":
    retrain_model_with_human_feedback()
