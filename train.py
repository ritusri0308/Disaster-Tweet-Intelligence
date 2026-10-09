"""
CLI training script and experiment runner.
Usage:
    python train.py
    python train.py --experiments
"""

import argparse
import logging
import time
import pandas as pd
import numpy as np

from src.config import REPORTS_DIR, HELD_OUT_EVENTS
from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe
from src.classifier import (
    build_model_pipeline,
    evaluate_model,
    train_and_compare_models
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def run_experiments(df: pd.DataFrame) -> None:
    """Run all ablation and leave-one-event-out experiments."""
    logger.info("==========================================")
    logger.info("STARTING PIPELINE EXPERIMENTS")
    logger.info("==========================================")

    # 1. Experiment (a): Unigrams vs Unigrams+Bigrams
    logger.info("Exp (a): Unigrams vs Unigrams+Bigrams...")
    exp_a_results = []
    train_df = df[df["split"] == "train"]
    dev_df = df[df["split"] == "dev"]

    for ngrams in [(1, 1), (1, 2)]:
        pipe = build_model_pipeline(model_name="logistic_regression", ngram_range=ngrams)
        pipe.fit(train_df["clean_text"], train_df["target"])
        m = evaluate_model(pipe, dev_df["clean_text"], dev_df["target"], split_name="dev")
        exp_a_results.append({
            "ngram_range": str(ngrams),
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"]
        })
    df_exp_a = pd.DataFrame(exp_a_results)
    df_exp_a.to_csv(REPORTS_DIR / "exp_a_ngrams.csv", index=False)
    logger.info(f"Exp (a) results:\n{df_exp_a}")

    # 2. Experiment (b): Raw vs Cleaned Text
    logger.info("Exp (b): Raw vs Cleaned Text...")
    exp_b_results = []
    for text_col in ["raw_unescaped_text", "clean_text"]:
        pipe = build_model_pipeline(model_name="logistic_regression")
        pipe.fit(train_df[text_col], train_df["target"])
        m = evaluate_model(pipe, dev_df[text_col], dev_df["target"], split_name="dev")
        exp_b_results.append({
            "text_format": text_col,
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"]
        })
    df_exp_b = pd.DataFrame(exp_b_results)
    df_exp_b.to_csv(REPORTS_DIR / "exp_b_text_format.csv", index=False)
    logger.info(f"Exp (b) results:\n{df_exp_b}")

    # 3. Experiment (c): With vs Without Class Balancing
    logger.info("Exp (c): Class Balancing Effect...")
    exp_c_results = []
    for cw in [True, False]:
        pipe = build_model_pipeline(model_name="logistic_regression", use_class_weight=cw)
        pipe.fit(train_df["clean_text"], train_df["target"])
        m = evaluate_model(pipe, dev_df["clean_text"], dev_df["target"], split_name="dev")
        exp_c_results.append({
            "class_weighted": cw,
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"]
        })
    df_exp_c = pd.DataFrame(exp_c_results)
    df_exp_c.to_csv(REPORTS_DIR / "exp_c_class_balancing.csv", index=False)
    logger.info(f"Exp (c) results:\n{df_exp_c}")

    # 4. Experiment (d): Leave-One-Event-Out Cross-Validation
    logger.info("Exp (d): Leave-One-Event-Out Cross-Validation...")
    exp_d_results = []
    f1_scores = []

    for held_out in HELD_OUT_EVENTS:
        if held_out not in df["event"].unique():
            logger.warning(f"Held-out event '{held_out}' not present in dataset. Skipping.")
            continue

        tr = df[df["event"] != held_out]
        te = df[df["event"] == held_out]

        if tr.empty or te.empty:
            continue

        pipe = build_model_pipeline(model_name="logistic_regression")
        pipe.fit(tr["clean_text"], tr["target"])
        m = evaluate_model(pipe, te["clean_text"], te["target"], split_name=held_out)

        f1_scores.append(m["macro_f1"])
        exp_d_results.append({
            "held_out_event": held_out,
            "test_samples": len(te),
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"]
        })

    if f1_scores:
        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
        exp_d_results.append({
            "held_out_event": "MEAN +/- STD",
            "test_samples": sum(r["test_samples"] for r in exp_d_results),
            "macro_f1": f"{mean_f1:.4f} +/- {std_f1:.4f}",
            "accuracy": f"{np.mean([r['accuracy'] for r in exp_d_results[:-1]]):.4f}"
        })

    df_exp_d = pd.DataFrame(exp_d_results)
    df_exp_d.to_csv(REPORTS_DIR / "exp_d_leave_one_event_out.csv", index=False)
    logger.info(f"Exp (d) Leave-One-Event-Out results:\n{df_exp_d}")

    # 5. Experiment (e): Project 5 vs Raw 10 Classification Target
    logger.info("Exp (e): Project 5 vs Raw 10 Target Comparison...")
    exp_e_results = []
    for schema in ["project_5", "raw_10"]:
        df_schema = load_data(target_schema=schema)
        df_schema = preprocess_dataframe(df_schema)
        tr_s = df_schema[df_schema["split"] == "train"]
        dev_s = df_schema[df_schema["split"] == "dev"]

        pipe = build_model_pipeline(model_name="logistic_regression")
        pipe.fit(tr_s["clean_text"], tr_s["target"])
        m = evaluate_model(pipe, dev_s["clean_text"], dev_s["target"], split_name="dev")
        exp_e_results.append({
            "target_schema": schema,
            "num_classes": len(df_schema["target"].unique()),
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"]
        })
    df_exp_e = pd.DataFrame(exp_e_results)
    df_exp_e.to_csv(REPORTS_DIR / "exp_e_target_schema.csv", index=False)
    logger.info(f"Exp (e) results:\n{df_exp_e}")

    # 6. Experiment (f): Word n-grams only vs word + character n-grams
    logger.info("Exp (f): Word vs Word + Character n-grams...")
    exp_f_results = []
    test_df = df[df["split"] == "test"]
    for use_char in [False, True]:
        t0 = time.time()
        pipe = build_model_pipeline(model_name="logistic_regression", use_char_ngrams=use_char)
        pipe.fit(train_df["clean_text"], train_df["target"])
        m_dev = evaluate_model(pipe, dev_df["clean_text"], dev_df["target"], split_name="dev")
        m_test = evaluate_model(pipe, test_df["clean_text"], test_df["target"], split_name="test")
        exp_f_results.append({
            "features": "word + char_wb(2-5)" if use_char else "word (1-2)",
            "dev_macro_f1": m_dev["macro_f1"],
            "test_macro_f1": m_test["macro_f1"],
            "test_accuracy": m_test["accuracy"],
            "train_seconds": round(time.time() - t0, 1)
        })
    df_exp_f = pd.DataFrame(exp_f_results)
    df_exp_f.to_csv(REPORTS_DIR / "exp_f_char_ngrams.csv", index=False)
    logger.info(f"Exp (f) results:\n{df_exp_f}")

    logger.info("All experiments completed successfully! Results saved to reports/")

def main():
    parser = argparse.ArgumentParser(description="Train classification models and optionally run experiments.")
    parser.add_argument("--experiments", action="store_true", help="Run model comparison experiments (ablation & leave-one-event-out).")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data if no HumAID TSV files are found.")
    args = parser.parse_args()

    logger.info("Loading and cleaning raw dataset for training...")
    df = load_data(allow_synthetic=True if args.demo else None)
    df = preprocess_dataframe(df)

    train_df = df[df["split"] == "train"]
    dev_df = df[df["split"] == "dev"]
    test_df = df[df["split"] == "test"]

    if train_df.empty:
        logger.warning("Train split empty. Falling back to 80/20 train/test split.")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        dev_df = test_df

    logger.info("Training model suite (Naive Bayes, Logistic Regression, Linear SVM, Random Forest)...")
    res = train_and_compare_models(train_df, dev_df, test_df)
    logger.info(f"Model training complete. Best model: {res['best_model_name']}")

    if args.experiments:
        run_experiments(df)

if __name__ == "__main__":
    main()
