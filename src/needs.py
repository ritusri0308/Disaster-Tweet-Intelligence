"""
Multi-label humanitarian need-type extraction module for CrisisLens.
"""

import re
import json
import logging
from typing import List, Dict, Any
import pandas as pd

from src.config import NEED_TAXONOMY, REPORTS_DIR

logger = logging.getLogger(__name__)

# Compile regex patterns for fast matching
_NEED_PATTERNS = {}
for need_type, terms in NEED_TAXONOMY.items():
    escaped = [r"\b" + re.escape(t) + r"\b" for t in terms]
    _NEED_PATTERNS[need_type] = re.compile("|".join(escaped), re.IGNORECASE)

def extract_needs(text: str) -> List[str]:
    """
    Extract multi-label humanitarian need types from text using taxonomy regexes.
    Returns sorted list of unique matched need types.
    """
    if not text or not str(text).strip():
        return []

    text_str = str(text)
    matched_needs = []
    for need_type, pattern in _NEED_PATTERNS.items():
        if pattern.search(text_str):
            matched_needs.append(need_type)

    return sorted(matched_needs)

def extract_needs_batch(texts: List[str]) -> List[List[str]]:
    """Batch extraction of need types."""
    return [extract_needs(t) for t in texts]

def evaluate_needs_against_raw_labels(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate need extraction agreement against HumAID raw_label proxies:
    - requests_or_urgent_needs -> expect at least one survival need (Medical, Water, Food, Shelter, Rescue/Evacuation)
    - injured_or_dead_people -> expect Medical or Rescue
    - infrastructure_and_utility_damage -> expect Transport/Roads or Power/Utilities
    - rescue_volunteering_or_donation_effort -> expect Donations/Volunteering or Rescue
    """
    if df.empty or "raw_label" not in df.columns:
        return {"samples": 0}

    results = {}

    # Proxy 1: Urgent needs rows
    urgent_rows = df[df["raw_label"] == "requests_or_urgent_needs"]
    if not urgent_rows.empty:
        survival_needs = {"Medical", "Water", "Food", "Shelter", "Rescue/Evacuation"}
        matched = sum(
            1 for needs in urgent_rows["needs"]
            if any(n in survival_needs for n in (needs if isinstance(needs, list) else []))
        )
        results["urgent_needs_survival_recall"] = round(matched / len(urgent_rows), 4)
        results["urgent_needs_sample_count"] = len(urgent_rows)

    # Proxy 2: Injured or dead people -> Medical
    injury_rows = df[df["raw_label"] == "injured_or_dead_people"]
    if not injury_rows.empty:
        matched = sum(
            1 for needs in injury_rows["needs"]
            if "Medical" in (needs if isinstance(needs, list) else []) or "Rescue/Evacuation" in (needs if isinstance(needs, list) else [])
        )
        results["injury_medical_recall"] = round(matched / len(injury_rows), 4)
        results["injury_sample_count"] = len(injury_rows)

    # Proxy 3: Infrastructure damage -> Power or Transport
    infra_rows = df[df["raw_label"] == "infrastructure_and_utility_damage"]
    if not infra_rows.empty:
        infra_needs = {"Power/Utilities", "Transport/Roads"}
        matched = sum(
            1 for needs in infra_rows["needs"]
            if any(n in infra_needs for n in (needs if isinstance(needs, list) else []))
        )
        results["infra_power_transport_recall"] = round(matched / len(infra_rows), 4)
        results["infra_sample_count"] = len(infra_rows)

    # Proxy 4: Donation / volunteer effort
    donation_rows = df[df["raw_label"] == "rescue_volunteering_or_donation_effort"]
    if not donation_rows.empty:
        matched = sum(
            1 for needs in donation_rows["needs"]
            if "Donations/Volunteering" in (needs if isinstance(needs, list) else [])
        )
        results["donation_recall"] = round(matched / len(donation_rows), 4)
        results["donation_sample_count"] = len(donation_rows)

    report_path = REPORTS_DIR / "needs_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved needs proxy evaluation to {report_path}")
    return results
