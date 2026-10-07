"""
Explainable rule-based urgency prioritization engine.
"""

import logging
import re
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import URGENCY_KEYWORDS, URGENCY_THRESHOLDS

logger = logging.getLogger(__name__)

# Donation / Retweet damping patterns
DONATION_PATTERNS = [
    r"\bdonate\b", r"\bdonation\b", r"\bfundraiser\b", r"\btext to donate\b",
    r"\bred cross donation\b", r"\bcampaign\b", r"\brt to spread\b", r"\bplease retweet\b"
]

def calculate_urgency(
    text: str,
    category: str = "General",
    uppercase_ratio: float = 0.0,
    exclamation_count: int = 0,
    is_retweet: int = 0
) -> Tuple[float, str, List[str]]:
    """
    Compute explainable urgency score (0-10), level, and matched terms.

    Thresholds:
      0.0 - 2.9 : LOW
      3.0 - 5.4 : MEDIUM
      5.5 - 7.4 : HIGH
      7.5 - 10.0: CRITICAL
    """
    if text is None or pd.isna(text):
        text_str = ""
    else:
        text_str = str(text)

    text_lower = text_str.lower()
    score = 0.0
    matched_terms = []

    # 1. Critical keyword matches (+3.5 each)
    for kw in URGENCY_KEYWORDS["critical"]:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            score += 3.5
            matched_terms.append(f"critical_kw:{kw}")

    # 2. High keyword matches (+2.0 each)
    for kw in URGENCY_KEYWORDS["high"]:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            score += 2.0
            matched_terms.append(f"high_kw:{kw}")

    # 3. Medium keyword matches (+1.0 each)
    for kw in URGENCY_KEYWORDS["medium"]:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            score += 1.0
            matched_terms.append(f"medium_kw:{kw}")

    # 4. Text feature signals
    if re.search(r"\bsos\b", text_lower):
        if "critical_kw:sos" not in matched_terms:
            score += 2.0
            matched_terms.append("signal:sos")

    if exclamation_count >= 2:
        score += 1.0
        matched_terms.append(f"signal:exclamation_cnt({exclamation_count})")

    if uppercase_ratio > 0.3:
        score += 1.0
        matched_terms.append(f"signal:uppercase_ratio({uppercase_ratio:.2f})")

    if "need medical" in text_lower or "please help" in text_lower:
        score += 1.5
        matched_terms.append("signal:urgent_phrase")

    # 5. Classifier Category Boost
    if category == "Emergency":
        score += 2.5
        matched_terms.append("category_boost:Emergency(+2.5)")
    elif category == "Help Request":
        score += 1.5
        matched_terms.append("category_boost:Help Request(+1.5)")
    elif category == "Damage Report":
        score += 0.5
        matched_terms.append("category_boost:Damage Report(+0.5)")

    # 6. Damping Rule for Retweets / Donation / Fundraising Content
    is_donation = any(re.search(pattern, text_lower) for pattern in DONATION_PATTERNS)
    if is_donation or (is_retweet and "donate" in text_lower):
        score *= 0.5
        matched_terms.append("damping:donation_or_retweet(*0.5)")
        # Cap donation tweets at max 5.4 (MEDIUM)
        if score > 5.4:
            score = 5.4
            matched_terms.append("damping:cap_donation(5.4)")

    # Bound score strictly between 0.0 and 10.0
    final_score = float(np.clip(score, 0.0, 10.0))
    final_score = round(final_score, 1)

    # Determine Urgency Level
    if final_score >= 7.5:
        level = "CRITICAL"
    elif final_score >= 5.5:
        level = "HIGH"
    elif final_score >= 3.0:
        level = "MEDIUM"
    else:
        level = "LOW"

    return final_score, level, matched_terms

def evaluate_urgency(annotation_csv_path: str) -> Dict[str, Any]:
    """
    Evaluate rule-based urgency model against a human-annotated CSV.
    Expected CSV columns: "tweet_text", "category", "human_urgency".
    "human_urgency" values: LOW, MEDIUM, HIGH, CRITICAL.
    """
    df = pd.read_csv(annotation_csv_path)
    if "human_urgency" not in df.columns:
        raise KeyError("CSV must contain 'human_urgency' column for evaluation.")

    df_eval = df.dropna(subset=["human_urgency"]).copy()
    if df_eval.empty:
        logger.warning("No annotated urgency rows found in CSV.")
        return {"accuracy": 0.0, "macro_f1": 0.0}

    pred_levels = []
    for _, row in df_eval.iterrows():
        text = row.get("tweet_text", "")
        cat = row.get("category", "General")
        _, pred_level, _ = calculate_urgency(text, category=cat)
        pred_levels.append(pred_level)

    true_levels = df_eval["human_urgency"].astype(str).str.upper().tolist()

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    acc = accuracy_score(true_levels, pred_levels)
    _, _, macro_f1, _ = precision_recall_fscore_support(true_levels, pred_levels, average="macro", zero_division=0)
    
    labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    cm = confusion_matrix(true_levels, pred_levels, labels=labels)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "annotated_samples": len(df_eval),
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }
