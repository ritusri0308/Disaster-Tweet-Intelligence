"""
Explainable rule-based urgency prioritization engine for CrisisLens.
"""

import logging
import re
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from src.config import URGENCY_KEYWORDS

logger = logging.getLogger(__name__)

# Donation / Retweet damping patterns
DONATION_PATTERNS = [
    r"\bdonate\b", r"\bdonation\b", r"\bfundraiser\b", r"\btext to donate\b",
    r"\bred cross donation\b", r"\bcampaign\b", r"\brt to spread\b", r"\bplease retweet\b"
]

def calculate_urgency_breakdown(
    text: str,
    category: str = "General",
    uppercase_ratio: float = 0.0,
    exclamation_count: int = 0,
    is_retweet: int = 0
) -> Tuple[float, str, List[str], List[Dict[str, Any]]]:
    """
    Compute explainable urgency score (0-10), level, matched terms, and a structured waterfall breakdown.

    Returns:
        final_score: float (0.0 to 10.0)
        level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        matched_terms: List[str]
        breakdown: List[Dict[str, Any]] with keys: signal, points, description
    """
    if text is None or pd.isna(text):
        text_str = ""
    else:
        text_str = str(text)

    text_lower = text_str.lower()
    score = 0.0
    matched_terms = []
    breakdown = []

    # 1. Critical keyword matches (+3.5 each)
    for kw in URGENCY_KEYWORDS["critical"]:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            score += 3.5
            tag = f"critical_kw:{kw}"
            matched_terms.append(tag)
            breakdown.append({
                "signal": f"Critical Keyword ({kw})",
                "points": 3.5,
                "description": f"High-severity casualty/trapped indicator: '{kw}'"
            })

    # 2. High keyword matches (+2.0 each)
    for kw in URGENCY_KEYWORDS["high"]:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            score += 2.0
            tag = f"high_kw:{kw}"
            matched_terms.append(tag)
            breakdown.append({
                "signal": f"High Keyword ({kw})",
                "points": 2.0,
                "description": f"Urgent assistance requirement indicator: '{kw}'"
            })

    # 3. Medium keyword matches (+1.0 each)
    for kw in URGENCY_KEYWORDS["medium"]:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            score += 1.0
            tag = f"medium_kw:{kw}"
            matched_terms.append(tag)
            breakdown.append({
                "signal": f"Medium Keyword ({kw})",
                "points": 1.0,
                "description": f"Advisory / infrastructure impact indicator: '{kw}'"
            })

    # 4. Text feature signals
    if re.search(r"\bsos\b", text_lower):
        if "critical_kw:sos" not in matched_terms:
            score += 2.0
            matched_terms.append("signal:sos")
            breakdown.append({
                "signal": "SOS Emergency Signal",
                "points": 2.0,
                "description": "Distress call acronym identified"
            })

    if exclamation_count >= 2:
        score += 1.0
        matched_terms.append(f"signal:exclamation_cnt({exclamation_count})")
        breakdown.append({
            "signal": "Intense Punctuation",
            "points": 1.0,
            "description": f"{exclamation_count} exclamation marks indicating panic/urgency"
        })

    if uppercase_ratio > 0.3:
        score += 1.0
        matched_terms.append(f"signal:uppercase_ratio({uppercase_ratio:.2f})")
        breakdown.append({
            "signal": "High Uppercase Ratio",
            "points": 1.0,
            "description": f"{uppercase_ratio:.1%} capital letters indicating alarm"
        })

    if "need medical" in text_lower or "please help" in text_lower:
        score += 1.5
        matched_terms.append("signal:urgent_phrase")
        breakdown.append({
            "signal": "Urgent Request Phrase",
            "points": 1.5,
            "description": "Direct human plea detected ('need medical' or 'please help')"
        })

    # 5. Classifier Category Boost
    if category == "Emergency":
        score += 2.5
        matched_terms.append("category_boost:Emergency(+2.5)")
        breakdown.append({
            "signal": "Category Boost (Emergency)",
            "points": 2.5,
            "description": "Casualty / life safety report classification boost"
        })
    elif category == "Help Request":
        score += 1.5
        matched_terms.append("category_boost:Help Request(+1.5)")
        breakdown.append({
            "signal": "Category Boost (Help Request)",
            "points": 1.5,
            "description": "Resource / evacuation request classification boost"
        })
    elif category == "Damage Report":
        score += 0.5
        matched_terms.append("category_boost:Damage Report(+0.5)")
        breakdown.append({
            "signal": "Category Boost (Damage Report)",
            "points": 0.5,
            "description": "Physical infrastructure impact boost"
        })

    # 6. Damping Rule for Retweets / Donation / Fundraising Content
    is_donation = any(re.search(pattern, text_lower) for pattern in DONATION_PATTERNS)
    if is_donation or (is_retweet and "donate" in text_lower):
        damped_loss = -(score * 0.5)
        score *= 0.5
        matched_terms.append("damping:donation_or_retweet(*0.5)")
        breakdown.append({
            "signal": "Damping (Donation/Campaign)",
            "points": round(damped_loss, 2),
            "description": "Score damped by 50% to prevent fundraising tweets escalating to CRITICAL"
        })
        if score > 5.4:
            cap_loss = -(score - 5.4)
            score = 5.4
            matched_terms.append("damping:cap_donation(5.4)")
            breakdown.append({
                "signal": "Damping Cap (Max MEDIUM)",
                "points": round(cap_loss, 2),
                "description": "Capped score at 5.4 threshold for fundraising content"
            })

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

    return final_score, level, matched_terms, breakdown

def calculate_urgency(
    text: str,
    category: str = "General",
    uppercase_ratio: float = 0.0,
    exclamation_count: int = 0,
    is_retweet: int = 0
) -> Tuple[float, str, List[str]]:
    """
    Standard function maintaining original signature for backward compatibility.
    """
    score, level, terms, _ = calculate_urgency_breakdown(
        text=text,
        category=category,
        uppercase_ratio=uppercase_ratio,
        exclamation_count=exclamation_count,
        is_retweet=is_retweet
    )
    return score, level, terms

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
