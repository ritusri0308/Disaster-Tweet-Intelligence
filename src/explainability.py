"""
Model explainability, token attribution, and text highlighting module for CrisisLens.
"""

import html
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline

def _display_feature_name(name: str, is_union: bool) -> Optional[str]:
    """
    Name to show for a TF-IDF feature, or None to hide it.

    Pipelines that combine word and character n-grams (a FeatureUnion) prefix names with ``word__`` /
    ``char__``. Character fragments are meaningless to an operator, so only word features are shown.
    """
    if not is_union:
        return name
    if name.startswith("word__"):
        return name[len("word__"):]
    return None

def explain_classification(
    pipeline: Pipeline,
    clean_text: str,
    top_n: int = 8
) -> Dict[str, Any]:
    """
    Compute token/feature attributions for the predicted class.
    For linear models: feature contribution = tfidf_value * coefficient.
    For non-linear models no per-token weights exist: ``is_surrogate`` is True and the feature lists are empty.
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
    is_union = isinstance(tfidf, FeatureUnion)

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
        # Non-linear model (e.g. random forest): there are no per-token weights to report. Return no
        # attributions rather than inventing one; callers check ``is_surrogate`` to explain why.
        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "top2_margin": round(top2_margin, 4),
            "probabilities": prob_dict,
            "top_positive_features": [],
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
        token = _display_feature_name(str(feature_names[idx]), is_union)
        if token is None:
            continue
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

# Highlight kinds in priority order: a place beats an urgency trigger, which beats a classifier token.
_HIGHLIGHT_PRIORITY = ("loc", "urg", "feat")

def highlight_tweet_tokens(
    raw_text: str,
    extracted_locations: List[str],
    matched_urgency_terms: List[str],
    top_features: List[Tuple[str, float]]
) -> str:
    """
    Format raw tweet text with safe HTML highlight spans (styled by the ``cl-hl--*`` theme classes):
    - ``loc``: locations
    - ``urg``: urgency triggers
    - ``feat``: positive classifier tokens

    Matches are located on the *unescaped* text and the output is built in one pass with every segment
    escaped separately, so neither tweet content nor a search term can corrupt the markup.
    """
    if not raw_text:
        return ""

    text = str(raw_text)

    terms_by_kind = {
        "loc": [str(loc) for loc in extracted_locations if loc and len(str(loc)) > 1],
        "urg": [],
        "feat": [str(tok) for tok, _ in top_features if tok and len(str(tok)) > 1],
    }
    for item in matched_urgency_terms:
        kw = str(item).split(":")[-1] if ":" in str(item) else str(item)
        if kw and len(kw) > 1:
            terms_by_kind["urg"].append(kw)

    spans: List[Tuple[int, int, str]] = []
    taken = np.zeros(len(text) + 1, dtype=bool)
    for kind in _HIGHLIGHT_PRIORITY:
        # Longest terms first so "collapsed building" wins over "building"
        for term in sorted(set(terms_by_kind[kind]), key=len, reverse=True):
            pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
            for m in pattern.finditer(text):
                if not taken[m.start():m.end()].any():
                    taken[m.start():m.end()] = True
                    spans.append((m.start(), m.end(), kind))

    def _segment(s: str) -> str:
        # Newlines become <br>: a blank line would end the HTML block when Streamlit renders it as markdown.
        return html.escape(s).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")

    spans.sort()
    out: List[str] = []
    pos = 0
    for start, end, kind in spans:
        out.append(_segment(text[pos:start]))
        out.append(f'<span class="cl-hl cl-hl--{kind}">{_segment(text[start:end])}</span>')
        pos = end
    out.append(_segment(text[pos:]))

    return f'<div class="cl-tweet">{"".join(out)}</div>'
