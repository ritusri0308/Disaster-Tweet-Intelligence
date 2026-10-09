"""
Multi-label humanitarian need-type extraction module for CrisisLens.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.config import NEED_TAXONOMY, REPORTS_DIR

logger = logging.getLogger(__name__)

def _as_list(value: Any) -> List[str]:
    """Coerce a stored needs value (list, tuple, numpy array, or None) to a plain list."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return []

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

    # Normalise typographic apostrophes so "can’t" matches "can't"
    text_str = str(text).replace("’", "'").replace("‘", "'")
    matched_needs = []
    for need_type, pattern in _NEED_PATTERNS.items():
        if pattern.search(text_str):
            matched_needs.append(need_type)

    return sorted(matched_needs)

def extract_needs_batch(texts: List[str]) -> List[List[str]]:
    """Batch extraction of need types."""
    return [extract_needs(t) for t in texts]

def evaluate_needs_against_raw_labels(df: pd.DataFrame, reports_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Evaluate need extraction agreement against HumAID raw_label proxies:
    - requests_or_urgent_needs -> expect at least one survival need (Medical, Water, Food, Shelter, Rescue/Evacuation)
    - injured_or_dead_people -> expect Medical or Rescue
    - infrastructure_and_utility_damage -> expect Transport/Roads or Power/Utilities
    - rescue_volunteering_or_donation_effort -> expect Donations/Volunteering or Rescue
    - missing_or_found_people -> expect Missing persons
    - displaced_people_and_evacuations -> expect Shelter or Rescue/Evacuation

    These are proxies, not ground truth: a label describes what a tweet is *about*, not which needs it states.
    Results are written to ``reports_dir`` (default: the configured reports directory).
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

    # Proxy 5: Missing / found people -> Missing persons
    missing_rows = df[df["raw_label"] == "missing_or_found_people"]
    if not missing_rows.empty:
        matched = sum(
            1 for needs in missing_rows["needs"]
            if "Missing persons" in _as_list(needs)
        )
        results["missing_persons_recall"] = round(matched / len(missing_rows), 4)
        results["missing_sample_count"] = len(missing_rows)

    # Proxy 6: Displaced people and evacuations -> Shelter or Rescue/Evacuation
    displaced_rows = df[df["raw_label"] == "displaced_people_and_evacuations"]
    if not displaced_rows.empty:
        matched = sum(
            1 for needs in displaced_rows["needs"]
            if any(n in {"Shelter", "Rescue/Evacuation"} for n in _as_list(needs))
        )
        results["displaced_shelter_rescue_recall"] = round(matched / len(displaced_rows), 4)
        results["displaced_sample_count"] = len(displaced_rows)

    # Share of tweets with no detected need at all (coverage)
    results["tweets_without_any_need_pct"] = round(
        100.0 * sum(1 for needs in df["needs"] if not _as_list(needs)) / len(df), 2
    )

    out_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "needs_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved needs proxy evaluation to {report_path}")
    return results
