"""
Machine learning classifier module for disaster tweet category prediction.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from sklearn.pipeline import FeatureUnion, Pipeline
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
    RANDOM_SEED
)
from src.model_registry import save_model_bundle

logger = logging.getLogger(__name__)

def build_model_pipeline(
    model_name: str = "logistic_regression",
    ngram_range: Tuple[int, int] = (1, 2),
    use_class_weight: bool = True,
    use_char_ngrams: bool = False
) -> Pipeline:
    """
    Build sklearn Pipeline with TF-IDF and specified classifier.

    ``use_char_ngrams`` adds character n-gram features alongside the word features (a FeatureUnion). It is
    off by default: on the HumAID dev split it gained about +0.7 pt macro-F1 (+1.0 pt on test) for roughly 8x
    the training time, which did not justify changing the production recipe (see ``train.py --experiments``, Exp f).
    """
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.95,
        max_features=15000,
        sublinear_tf=True
    )
    if use_char_ngrams:
        tfidf = FeatureUnion([
            ("word", tfidf),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=60000, sublinear_tf=True)),
        ])

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
    # Imported here so services that only classify (the REST API) do not need the plotting stack installed.
    import matplotlib
    matplotlib.use("Agg")  # plots are only written to files; never open a GUI window (breaks headless/threads)
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# Candidates ordered from simplest / most interpretable to most complex. Differences in dev macro-F1 smaller
# than SELECTION_TOLERANCE are noise on a ~4,400-tweet dev split, so the simpler model wins a tie: it is smaller,
# faster, and explainable (a random forest here was 162 MB and has no per-token weights).
MODEL_PREFERENCE = ["logistic_regression", "linear_svm", "naive_bayes", "random_forest"]
SELECTION_TOLERANCE = 0.005

def select_model(dev_scores: Dict[str, float], tolerance: float = SELECTION_TOLERANCE) -> str:
    """
    Pick the model with the best dev macro-F1; among those within ``tolerance`` of the best,
    take the simplest according to ``MODEL_PREFERENCE``.
    """
    if not dev_scores:
        raise ValueError("No model scores to select from.")
    top = max(dev_scores.values())
    contenders = [name for name, score in dev_scores.items() if score >= top - tolerance]
    return min(contenders, key=lambda n: MODEL_PREFERENCE.index(n) if n in MODEL_PREFERENCE else len(MODEL_PREFERENCE))

def train_and_compare_models(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "clean_text",
    target_col: str = "target",
    model_path: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    selection_tolerance: float = SELECTION_TOLERANCE
) -> Dict[str, Any]:
    """
    Train Naive Bayes, Logistic Regression, Linear SVM, and Random Forest.
    Select best model based on dev macro_f1 (see ``select_model``) and save it (plus a metadata sidecar)
    to ``model_path``. Save summary comparison table and plots to ``reports_dir``.

    ``model_path`` / ``reports_dir`` default to the configured production locations; callers that
    train throwaway models (tests, experiments) must pass their own so production artifacts are
    never overwritten.
    """
    model_path = Path(model_path) if model_path else MODEL_PATH
    reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_names = ["naive_bayes", "logistic_regression", "linear_svm", "random_forest"]
    results = []
    trained_pipelines = {}

    X_train, y_train = train_df[text_col], train_df[target_col]
    X_dev, y_dev = dev_df[text_col], dev_df[target_col]
    X_test, y_test = test_df[text_col], test_df[target_col]

    dev_scores: Dict[str, float] = {}

    for m_name in model_names:
        logger.info(f"Training model: {m_name}...")
        try:
            pipeline = build_model_pipeline(model_name=m_name)
            pipeline.fit(X_train, y_train)
        except ValueError as exc:
            # e.g. a calibrated SVM needs at least two examples of every class
            logger.warning(f"Skipping {m_name}: it could not be fitted on this training set ({exc})")
            continue

        dev_metrics = evaluate_model(pipeline, X_dev, y_dev, split_name="dev")
        test_metrics = evaluate_model(pipeline, X_test, y_test, split_name="test")

        dev_scores[m_name] = dev_metrics["macro_f1"]
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

    if not dev_scores:
        raise ValueError("No candidate model could be trained on the provided training set.")

    # Best pipeline: highest dev macro-F1, preferring the simpler model when candidates are within noise
    best_model_name = select_model(dev_scores, selection_tolerance)
    best_dev_f1 = dev_scores[best_model_name]
    best_pipeline = trained_pipelines[best_model_name]
    top_dev_f1 = max(dev_scores.values())
    logger.info(
        f"Best model selected: {best_model_name} with dev Macro-F1: {best_dev_f1:.4f} "
        f"(top score {top_dev_f1:.4f}; models within {selection_tolerance} are treated as tied)"
    )

    # Generate final dev and test evaluation for best model
    best_dev_metrics = evaluate_model(best_pipeline, X_dev, y_dev, split_name="dev")
    best_test_metrics = evaluate_model(best_pipeline, X_test, y_test, split_name="test")

    # Save best model (with metadata sidecar) to disk
    meta = save_model_bundle(best_pipeline, model_path, extra={
        "selected_model": best_model_name,
        "selection_rule": f"highest dev macro-F1; ties within {selection_tolerance} resolved toward the simpler model",
        "dev_macro_f1": best_dev_metrics["macro_f1"],
        "test_macro_f1": best_test_metrics["macro_f1"],
        "test_accuracy": best_test_metrics["accuracy"],
        "n_train": int(len(train_df)),
        "n_dev": int(len(dev_df)),
        "n_test": int(len(test_df)),
        "random_seed": RANDOM_SEED,
    })
    logger.info(f"Saved best model pipeline to {model_path} (version {meta['model_version']})")

    # Save confusion matrix plot and CSV for best model on test set
    cm_array = np.array(best_test_metrics["confusion_matrix"])
    labels = best_test_metrics["labels"]

    save_confusion_matrix_plot(cm_array, labels, reports_dir / "best_model_confusion_matrix.png")
    cm_df = pd.DataFrame(cm_array, index=[f"True_{lbl}" for lbl in labels], columns=[f"Pred_{lbl}" for lbl in labels])
    cm_df.to_csv(reports_dir / "best_model_confusion_matrix.csv")

    # Save metrics table
    results_df = pd.DataFrame(results)
    results_df.to_csv(reports_dir / "model_comparison.csv", index=False)
    results_df.to_json(reports_dir / "model_comparison.json", orient="records", indent=2)

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

    if hasattr(pipeline, "predict_proba"):
        # One forward pass: for probabilistic classifiers predict() is the argmax of predict_proba().
        probas = pipeline.predict_proba(texts)
        best_idx = np.argmax(probas, axis=1)
        preds = np.asarray(pipeline.classes_)[best_idx]
        confidences = probas[np.arange(len(best_idx)), best_idx]
        return [(str(p), float(round(c, 4))) for p, c in zip(preds, confidences)]

    preds = pipeline.predict(texts)

    if hasattr(pipeline, "decision_function"):
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
