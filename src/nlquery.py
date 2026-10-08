"""
Deterministic natural-language query parser for CrisisLens.
Translates operator search queries into structured filters without LLM calls.
"""

import re
from typing import Dict, Any, Optional

from src.config import GAZETTEER_LOCATIONS, NEED_TAXONOMY

# Category Synonyms
CATEGORY_SYNONYMS = {
    "Emergency": ["emergency", "emergencies", "casualty", "casualties", "injured", "dead", "fatalities"],
    "Help Request": ["help request", "help requests", "requests", "request", "plea", "urgent need", "needs help", "rescue me"],
    "Damage Report": ["damage report", "damage reports", "damage", "infrastructure", "collapsed", "destruction", "destroyed", "grid down"],
    "Information/Update": ["information", "update", "updates", "advisory", "warning", "caution", "evacuation", "evacuations", "missing", "volunteering"],
    "General": ["general", "sympathy", "support", "condolences", "non-humanitarian", "prayer", "prayers"]
}

# Urgency Synonyms
URGENCY_SYNONYMS = {
    "CRITICAL": ["critical", "sos", "life threatening", "extreme", "fatal", "dying", "most urgent"],
    "HIGH": ["high", "high urgency", "urgent", "severe", "immediate"],
    "MEDIUM": ["medium", "moderate", "warning", "advisory"],
    "LOW": ["low", "low urgency", "minor", "routine", "informational"]
}

# Event keywords
EVENT_KEYWORDS = {
    "canada_wildfires_2016": ["canada wildfires", "fort mcmurray", "canada", "alberta"],
    "cyclone_idai_2019": ["cyclone idai", "idai", "mozambique", "beira", "zimbabwe", "malawi"],
    "ecuador_earthquake_2016": ["ecuador earthquake", "pedernales", "manabi", "ecuador"],
    "greece_wildfires_2018": ["greece wildfires", "athens wildfires", "mati", "greece"],
    "hurricane_harvey_2017": ["hurricane harvey", "houston hurricane", "texas hurricane", "harvey"],
    "hurricane_irma_2017": ["hurricane irma", "florida hurricane", "irma"],
    "hurricane_maria_2017": ["hurricane maria", "puerto rico hurricane", "maria"],
    "hurricane_matthew_2016": ["hurricane matthew", "haiti hurricane", "matthew"],
    "italy_earthquake_aug_2016": ["italy earthquake", "amatrice", "italy"],
    "kaikoura_earthquake_2016": ["kaikoura earthquake", "new zealand earthquake", "kaikoura"],
    "maryland_floods_2018": ["maryland floods", "ellicott city", "maryland"],
    "puebla_mexico_earthquake_2017": ["puebla earthquake", "mexico earthquake", "puebla", "mexico"],
    "srilanka_floods_2017": ["sri lanka floods", "colombo floods", "sri lanka"]
}

def parse_natural_language_query(query: str, anchor_max_date: Optional[str] = "2019-12-31") -> Dict[str, Any]:
    """
    Parse natural language query into structured CrisisLens filter dict:
    - category
    - urgency_level
    - event
    - location
    - need_type
    - date_start
    - date_end
    - is_valid
    - parsed_chips: List[str]
    - message: str
    """
    if not query or not str(query).strip():
        return {
            "is_valid": False,
            "message": "Empty query. Enter keywords like 'critical medical needs in puerto rico'.",
            "parsed_chips": [],
            "filters": {}
        }

    q_lower = query.lower().strip()
    filters = {}
    chips = []

    # 1. Category extraction
    for cat, synonyms in CATEGORY_SYNONYMS.items():
        for syn in synonyms:
            if re.search(r"\b" + re.escape(syn) + r"\b", q_lower):
                filters["category"] = cat
                chips.append(f"Category: {cat}")
                break
        if "category" in filters:
            break

    # 2. Urgency level extraction
    for urg, synonyms in URGENCY_SYNONYMS.items():
        for syn in synonyms:
            if re.search(r"\b" + re.escape(syn) + r"\b", q_lower):
                filters["urgency_level"] = urg
                chips.append(f"Urgency: {urg}")
                break
        if "urgency_level" in filters:
            break

    # 3. Event extraction
    for ev_id, keywords in EVENT_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                filters["event"] = ev_id
                chips.append(f"Event: {ev_id}")
                break
        if "event" in filters:
            break

    # 4. Location extraction (Gazetteer places)
    for loc in GAZETTEER_LOCATIONS:
        if re.search(r"\b" + re.escape(loc.lower()) + r"\b", q_lower):
            filters["location"] = loc
            chips.append(f"Location: {loc}")
            break

    # 5. Need types
    for need_type, terms in NEED_TAXONOMY.items():
        for t in terms:
            if re.search(r"\b" + re.escape(t) + r"\b", q_lower):
                filters["need_type"] = need_type
                chips.append(f"Need: {need_type}")
                break
        if "need_type" in filters:
            break

    # 6. Relative/Absolute date expressions
    if "last 24 hours" in q_lower or "24h" in q_lower:
        filters["date_range"] = "last_24h"
        chips.append("Window: Last 24 Hours")
    elif "last 48 hours" in q_lower or "48h" in q_lower:
        filters["date_range"] = "last_48h"
        chips.append("Window: Last 48 Hours")
    elif "after sep 20" in q_lower:
        filters["date_after"] = "2017-09-20"
        chips.append("After: 2017-09-20")

    is_valid = len(chips) > 0
    msg = f"Extracted {len(chips)} filters." if is_valid else "Could not recognize specific crisis filters from input."

    return {
        "is_valid": is_valid,
        "message": msg,
        "parsed_chips": chips,
        "filters": filters,
        "raw_query": query
    }
