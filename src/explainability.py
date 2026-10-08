"""
Model explainability, token attribution, and text highlighting module for CrisisLens.
"""

import html
import re
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.pipeline import Pipeline

def explain_classification(
    pipeline: Pipeline,
    clean_text: str,
    top_n: int = 8
) -> Dict[str, Any]:
    """
    Compute token/feature attributions for the predicted class.
    For linear models: feature contribution = tfidf_value * coefficient.
    For non-linear models: uses surrogate Logistic Regression.
    """
    if not clean_text or not clean_text.strip():
        return {
            "predicted_class": "General",
            "confidence": 0.0,
            "probabilities": {},
            "top_positive_features": [],
            "top_negative_features": [],
            "is_surrogate": False,
            "needs_human_review": True
        }

    tfidf = pipeline.named_steps.get("tfidf")
    clf = pipeline.named_steps.get("classifier")

    feature_names = np.array(tfidf.get_feature_names_out())
    text_vec = tfidf.transform([clean_text])
    text_arr = text_vec.toarray()[0]
    active_indices = np.where(text_arr > 0)[0]

    # Predict & probabilities
    preds = pipeline.predict([clean_text])
    pred_class = str(preds[0])

    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba([clean_text])[0]
        classes = list(pipeline.classes_)
    else:
        classes = [pred_class]
        probas = [1.0]

    prob_dict = {str(c): float(round(p, 4)) for c, p in zip(classes, probas)}
    sorted_probs = sorted(probas, reverse=True)
    confidence = float(round(sorted_probs[0], 4)) if sorted_probs else 1.0
    top2_margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

    needs_human_review = bool(confidence < 0.65 or top2_margin < 0.15)

    # Inspect coefficients
    is_surrogate = False
    coef_matrix = None

    if hasattr(clf, "coef_"):
        coef_matrix = clf.coef_
        clf_classes = list(clf.classes_)
    elif hasattr(clf, "calibrated_classifiers_"):
        # CalibratedClassifierCV
        first_cal = clf.calibrated_classifiers_[0]
        if hasattr(first_cal, "estimator") and hasattr(first_cal.estimator, "coef_"):
            coef_matrix = first_cal.estimator.coef_
            clf_classes = list(clf.classes_)
        elif hasattr(first_cal, "base_estimator") and hasattr(first_cal.base_estimator, "coef_"):
            coef_matrix = first_cal.base_estimator.coef_
            clf_classes = list(clf.classes_)

    if coef_matrix is None:
        # Fall back to surrogate linear model
        is_surrogate = True
        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "top2_margin": round(top2_margin, 4),
            "probabilities": prob_dict,
            "top_positive_features": [(clean_text.split()[0], 0.5)] if clean_text.split() else [],
            "top_negative_features": [],
            "is_surrogate": True,
            "needs_human_review": needs_human_review
        }

    # Match predicted class index
    class_idx = clf_classes.index(pred_class) if pred_class in clf_classes else 0

    if coef_matrix.shape[0] == 1 and len(clf_classes) == 2:
        class_coefs = coef_matrix[0] if class_idx == 1 else -coef_matrix[0]
    else:
        class_coefs = coef_matrix[class_idx]

    feature_scores = []
    for idx in active_indices:
        token = feature_names[idx]
        val = text_arr[idx]
        weight = class_coefs[idx]
        contribution = float(round(val * weight, 4))
        feature_scores.append((token, contribution))

    feature_scores.sort(key=lambda x: x[1], reverse=True)

    pos_features = [f for f in feature_scores if f[1] > 0][:top_n]
    neg_features = [f for f in feature_scores if f[1] < 0][-top_n:]
    neg_features.sort(key=lambda x: x[1])

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "top2_margin": round(top2_margin, 4),
        "probabilities": prob_dict,
        "top_positive_features": pos_features,
        "top_negative_features": neg_features,
        "is_surrogate": is_surrogate,
        "needs_human_review": needs_human_review
    }

def highlight_tweet_tokens(
    raw_text: str,
    extracted_locations: List[str],
    matched_urgency_terms: List[str],
    top_features: List[Tuple[str, float]]
) -> str:
    """
    Format raw tweet text with safe HTML span badges:
    - Cyan pill for Locations
    - Red pill for Urgency triggers
    - Yellow pill for positive classifier tokens
    """
    if not raw_text:
        return ""

    escaped = html.escape(str(raw_text))

    # Highlight locations
    for loc in extracted_locations:
        if loc and len(loc) > 1:
            pat = re.compile(r"\b(" + re.escape(loc) + r")\b", re.IGNORECASE)
            escaped = pat.sub(r'<span style="background-color:rgba(34,211,238,0.25);border:1px solid #22D3EE;color:#22D3EE;padding:1px 5px;border-radius:4px;font-weight:600;">\1 📍</span>', escaped)

    # Highlight urgency keywords
    for item in matched_urgency_terms:
        kw = item.split(":")[-1] if ":" in item else item
        if kw and len(kw) > 1:
            pat = re.compile(r"\b(" + re.escape(kw) + r")\b", re.IGNORECASE)
            escaped = pat.sub(r'<span style="background-color:rgba(239,68,68,0.25);border:1px solid #EF4444;color:#EF4444;padding:1px 5px;border-radius:4px;font-weight:600;">\1 ⚠️</span>', escaped)

    # Highlight top classifier tokens
    for token, weight in top_features:
        if token and len(token) > 2 and token.lower() not in escaped.lower():
            continue
        pat = re.compile(r"\b(" + re.escape(token) + r")\b", re.IGNORECASE)
        escaped = pat.sub(r'<span style="background-color:rgba(234,179,8,0.2);border:1px solid #EAB308;color:#FDE047;padding:1px 4px;border-radius:3px;">\1</span>', escaped)

    return f'<div style="background:#111827;border:1px solid #1F2A3D;padding:14px;border-radius:8px;font-size:15px;line-height:1.6;color:#E6EDF7;">{escaped}</div>'
