"""
Machine learning classifier module for disaster tweet category prediction.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

from src.config import (
    MODEL_PATH,
    REPORTS_DIR,
    RANDOM_SEED,
    PROJECT_CATEGORIES
)

logger = logging.getLogger(__name__)

def build_model_pipeline(model_name: str = "logistic_regression", ngram_range: Tuple[int, int] = (1, 2), use_class_weight: bool = True) -> Pipeline:
    """
    Build sklearn Pipeline with TF-IDF and specified classifier.
    """
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.95,
        max_features=15000,
        sublinear_tf=True
    )

    cw = "balanced" if use_class_weight else None

    if model_name == "naive_bayes":
        clf = MultinomialNB(alpha=0.1)
    elif model_name == "logistic_regression":
        clf = LogisticRegression(class_weight=cw, random_state=RANDOM_SEED, max_iter=1000)
    elif model_name == "linear_svm":
        base_svc = LinearSVC(class_weight=cw, random_state=RANDOM_SEED, max_iter=2000, dual='auto')
        clf = CalibratedClassifierCV(estimator=base_svc, cv=2)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=RANDOM_SEED, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("classifier", clf)
    ])
    return pipeline

def evaluate_model(pipeline: Pipeline, X: pd.Series, y: pd.Series, split_name: str = "dev") -> Dict[str, Any]:
    """
    Evaluate trained pipeline on dataset X, y.
    Returns metrics dict and classification report.
    """
    y_pred = pipeline.predict(X)
    acc = accuracy_score(y, y_pred)
    
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(y, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y, y_pred, zero_division=0)

    labels = sorted(list(set(y).union(set(y_pred))))
    cm = confusion_matrix(y, y_pred, labels=labels)

    metrics = {
        "split": split_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(w_p, 4),
        "weighted_recall": round(w_r, 4),
        "weighted_f1": round(w_f1, 4),
        "per_class_report": report_dict,
        "classification_report_str": report_str,
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }
    return metrics

def save_confusion_matrix_plot(cm: np.ndarray, labels: List[str], filename: Path) -> None:
    """Save confusion matrix heat map as PNG image."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def train_and_compare_models(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str = "clean_text", target_col: str = "target") -> Dict[str, Any]:
    """
    Train Naive Bayes, Logistic Regression, Linear SVM, and Random Forest.
    Select best model based on dev macro_f1 and save it to models/.
    Save summary comparison table to reports/.
    """
    model_names = ["naive_bayes", "logistic_regression", "linear_svm", "random_forest"]
    results = []
    trained_pipelines = {}

    X_train, y_train = train_df[text_col], train_df[target_col]
    X_dev, y_dev = dev_df[text_col], dev_df[target_col]
    X_test, y_test = test_df[text_col], test_df[target_col]

    best_dev_f1 = -1.0
    best_model_name = None

    for m_name in model_names:
        logger.info(f"Training model: {m_name}...")
        pipeline = build_model_pipeline(model_name=m_name)
        pipeline.fit(X_train, y_train)
        
        dev_metrics = evaluate_model(pipeline, X_dev, y_dev, split_name="dev")
        test_metrics = evaluate_model(pipeline, X_test, y_test, split_name="test")

        dev_f1 = dev_metrics["macro_f1"]
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_model_name = m_name

        trained_pipelines[m_name] = pipeline

        results.append({
            "model_name": m_name,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"]
        })

    # Best pipeline
    best_pipeline = trained_pipelines[best_model_name]
    logger.info(f"Best model selected: {best_model_name} with dev Macro-F1: {best_dev_f1:.4f}")

    # Save best model to disk
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info(f"Saved best model pipeline to {MODEL_PATH}")

    # Generate final dev and test evaluation for best model
    best_dev_metrics = evaluate_model(best_pipeline, X_dev, y_dev, split_name="dev")
    best_test_metrics = evaluate_model(best_pipeline, X_test, y_test, split_name="test")

    # Save confusion matrix plot and CSV for best model on test set
    cm_array = np.array(best_test_metrics["confusion_matrix"])
    labels = best_test_metrics["labels"]
    
    save_confusion_matrix_plot(cm_array, labels, REPORTS_DIR / "best_model_confusion_matrix.png")
    cm_df = pd.DataFrame(cm_array, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    cm_df.to_csv(REPORTS_DIR / "best_model_confusion_matrix.csv")

    # Save metrics table
    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    results_df.to_json(REPORTS_DIR / "model_comparison.json", orient="records", indent=2)

    return {
        "best_model_name": best_model_name,
        "comparison_table": results_df,
        "best_dev_metrics": best_dev_metrics,
        "best_test_metrics": best_test_metrics
    }

def predict_with_confidence(pipeline: Pipeline, texts: List[str]) -> List[Tuple[str, float]]:
    """
    Predict category and confidence for a list of texts using the pipeline.
    Handles models with predict_proba or decision_function fallback.
    """
    if not texts:
        return []

    preds = pipeline.predict(texts)
    
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(texts)
        confidences = np.max(probas, axis=1)
    elif hasattr(pipeline, "decision_function"):
        decisions = pipeline.decision_function(texts)
        if decisions.ndim == 1:
            # Binary case
            confidences = 1.0 / (1.0 + np.exp(-np.abs(decisions)))
        else:
            # Multiclass case: softmax
            exp_dec = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
            softmax = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)
            confidences = np.max(softmax, axis=1)
    else:
        confidences = np.ones(len(preds))

    return [(str(p), float(round(c, 4))) for p, c in zip(preds, confidences)]
