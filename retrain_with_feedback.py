"""
Human-in-the-loop retraining module for CrisisLens.
Merges operator feedback from SQLite into training data, retrains the model,
and only updates the production model if test Macro-F1 does not regress.

Usage:
    python retrain_with_feedback.py
"""

import shutil
import logging
from pathlib import Path
import joblib

from src.config import MODEL_PATH, MODEL_BACKUP_PATH, DB_PATH
from src.database import load_all_feedback
from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe
from src.classifier import build_model_pipeline, evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def retrain_model_with_human_feedback(db_path: Path = DB_PATH) -> bool:
    """
    Retrain classification pipeline incorporating human operator feedback.
    Returns True if model was promoted, False otherwise.
    """
    logger.info("Loading feedback records from SQLite...")
    feedback_df = load_all_feedback(db_path)

    if feedback_df.empty:
        logger.info("No human feedback entries found in database. Retraining skipped.")
        return False

    logger.info(f"Loaded {len(feedback_df)} human feedback records.")

    # Load baseline dataset
    df = load_data()
    df = preprocess_dataframe(df)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Baseline evaluation on existing production model
    current_model = joblib.load(MODEL_PATH)
    base_metrics = evaluate_model(current_model, test_df["clean_text"], test_df["target"], split_name="test")
    base_macro_f1 = base_metrics["macro_f1"]
    logger.info(f"Current production test Macro-F1: {base_macro_f1:.4f}")

    # Merge feedback into train_df
    # For feedback with category corrections, update label or append to train
    corrections = feedback_df[feedback_df["original_category"] != feedback_df["corrected_category"]]
    if not corrections.empty:
        id_to_cat = dict(zip(corrections["tweet_id"].astype(str), corrections["corrected_category"]))
        # Update existing train tweets
        train_mask = train_df["tweet_id"].astype(str).isin(id_to_cat)
        train_df.loc[train_mask, "target"] = train_df.loc[train_mask, "tweet_id"].astype(str).map(id_to_cat)
        logger.info(f"Applied {train_mask.sum()} corrections directly into training split.")

    # Train new candidate pipeline
    logger.info("Training candidate model on feedback-augmented dataset...")
    candidate_pipe = build_model_pipeline(model_name="logistic_regression")
    candidate_pipe.fit(train_df["clean_text"], train_df["target"])

    candidate_metrics = evaluate_model(candidate_pipe, test_df["clean_text"], test_df["target"], split_name="test")
    cand_macro_f1 = candidate_metrics["macro_f1"]
    logger.info(f"Candidate model test Macro-F1: {cand_macro_f1:.4f} (baseline was {base_macro_f1:.4f})")

    if cand_macro_f1 >= base_macro_f1:
        # Promote candidate model
        logger.info("Candidate model improved or matched baseline. Promoting to production!")
        if MODEL_PATH.exists():
            shutil.copyfile(MODEL_PATH, MODEL_BACKUP_PATH)
            logger.info(f"Backed up previous model to {MODEL_BACKUP_PATH}")
        joblib.dump(candidate_pipe, MODEL_PATH)
        return True
    else:
        logger.warning(
            f"Candidate model regressed ({cand_macro_f1:.4f} < {base_macro_f1:.4f}). "
            "Keeping current production model."
        )
        return False

if __name__ == "__main__":
    retrain_model_with_human_feedback()
