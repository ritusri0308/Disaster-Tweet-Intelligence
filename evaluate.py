"""
CLI evaluation script for CrisisLens Model Observatory.
Generates comprehensive quantitative evaluation artifacts:
- ROC curves per class (one-vs-rest)
- Precision-Recall curves per class
- Calibration curves & Expected Calibration Error (ECE)
- Per-event test metrics table
- Misclassification error analysis records

Usage:
    python evaluate.py
"""

import json
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report, precision_recall_fscore_support, accuracy_score
)
from sklearn.calibration import calibration_curve

from src.config import (
    MODEL_PATH,
    REPORTS_DIR,
    RANDOM_SEED
)
from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def compute_expected_calibration_error(y_true_binary: np.ndarray, probas: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probas)
    for i in range(n_bins):
        bin_mask = (probas > bin_edges[i]) & (probas <= bin_edges[i + 1])
        bin_count = bin_mask.sum()
        if bin_count > 0:
            bin_acc = y_true_binary[bin_mask].mean()
            bin_conf = probas[bin_mask].mean()
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(round(ece, 4))

def main():
    logger.info("Evaluating CrisisLens model on official test split...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run 'python train.py' first.")

    pipeline = joblib.load(MODEL_PATH)

    # Load data
    df = load_data()
    df = preprocess_dataframe(df)

    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        logger.warning("Test split not found in dataset. Using full dataset for evaluation.")
        test_df = df.copy()

    X_test = test_df["clean_text"].tolist()
    y_test = test_df["target"].tolist()

    # Predict & probas
    preds = pipeline.predict(X_test)
    test_df["predicted_target"] = preds

    classes = list(pipeline.classes_)
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X_test)
    else:
        # Decision function to softmax fallback
        dec = pipeline.decision_function(X_test)
        exp_dec = np.exp(dec - np.max(dec, axis=1, keepdims=True))
        probas = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)

    test_df["confidence"] = np.max(probas, axis=1)

    # 1. Per-Class Metrics
    acc = accuracy_score(y_test, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
    report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)

    # 2. ROC & PR Curves (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=classes)
    roc_curves_data = {}
    pr_curves_data = {}

    for idx, c_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], probas[:, idx])
        roc_auc = auc(fpr, tpr)
        roc_curves_data[c_name] = {
            "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr) // 50)]],
            "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr) // 50)]],
            "auc": round(float(roc_auc), 4)
        }

        precision, recall, _ = precision_recall_curve(y_test_bin[:, idx], probas[:, idx])
        pr_auc = average_precision_score(y_test_bin[:, idx], probas[:, idx])
        pr_curves_data[c_name] = {
            "recall": [round(float(x), 4) for x in recall[::max(1, len(recall) // 50)]],
            "precision": [round(float(x), 4) for x in precision[::max(1, len(precision) // 50)]],
            "average_precision": round(float(pr_auc), 4)
        }

    # 3. Calibration Curves & ECE
    top_pred_probas = np.max(probas, axis=1)
    is_correct = (np.array(preds) == np.array(y_test)).astype(int)
    prob_true, prob_pred = calibration_curve(is_correct, top_pred_probas, n_bins=10)
    ece = compute_expected_calibration_error(is_correct, top_pred_probas, n_bins=10)

    calibration_data = {
        "prob_pred": [round(float(p), 4) for p in prob_pred],
        "prob_true": [round(float(p), 4) for p in prob_true],
        "expected_calibration_error": ece
    }

    # 4. Per-Event Performance Breakdown
    per_event_rows = []
    for ev, grp in test_df.groupby("event"):
        if grp.empty:
            continue
        ev_acc = accuracy_score(grp["target"], grp["predicted_target"])
        _, _, ev_f1, _ = precision_recall_fscore_support(grp["target"], grp["predicted_target"], average="macro", zero_division=0)
        per_event_rows.append({
            "event": ev,
            "test_samples": len(grp),
            "macro_f1": round(ev_f1, 4),
            "accuracy": round(ev_acc, 4)
        })

    df_per_event = pd.DataFrame(per_event_rows).sort_values("test_samples", ascending=False)
    df_per_event.to_csv(REPORTS_DIR / "per_event_metrics.csv", index=False)

    # 5. Error Analysis Records (Sample of misclassified dispatches)
    errors_df = test_df[test_df["target"] != test_df["predicted_target"]].copy()
    sample_errors = []
    for _, row in errors_df.head(100).iterrows():
        sample_errors.append({
            "tweet_id": row["tweet_id"],
            "event": row["event"],
            "true_category": row["target"],
            "predicted_category": row["predicted_target"],
            "confidence": round(float(row["confidence"]), 4),
            "tweet_text": row.get("raw_unescaped_text", row.get("tweet_text", "")),
            "clean_text": row["clean_text"]
        })

    # Save all artifacts
    evaluation_artifacts = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_file": str(MODEL_PATH.name),
        "test_sample_count": len(test_df),
        "random_seed": RANDOM_SEED,
        "classes": classes,
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(w_f1, 4),
            "expected_calibration_error": ece
        },
        "per_class_report": report_dict,
        "error_sample_count": len(errors_df),
        "sample_misclassifications": sample_errors[:40]
    }

    with open(REPORTS_DIR / "evaluation_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_artifacts, f, indent=2)

    with open(REPORTS_DIR / "roc_pr_curves.json", "w", encoding="utf-8") as f:
        json.dump({"roc_curves": roc_curves_data, "pr_curves": pr_curves_data}, f, indent=2)

    with open(REPORTS_DIR / "calibration_data.json", "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)

    logger.info("Evaluation artifacts successfully generated in reports/!")
    logger.info(f"Test Accuracy: {acc:.4f} | Test Macro-F1: {macro_f1:.4f} | ECE: {ece:.4f}")

if __name__ == "__main__":
    main()
