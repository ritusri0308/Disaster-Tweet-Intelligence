"""
Deterministic natural-language query parser for CrisisLens.
Translates operator search queries into structured filters without LLM calls.
"""

import re
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from src.config import GAZETTEER_LOCATIONS, NEED_TAXONOMY
from src.utils import as_list

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

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

_DATE_PATTERN = re.compile(
    r"\b(after|since|before)\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b"
)
_WINDOW_PATTERN = re.compile(r"\b(?:last|past)\s+(\d{1,3})\s*(hours?|hrs?|days?)\b|\b(\d{1,3})\s*h\b")

def _parse_relative_window(q_lower: str) -> Optional[int]:
    """Hours covered by a relative-window phrase such as 'last 24 hours', 'past 3 days' or '48h', else None."""
    m = _WINDOW_PATTERN.search(q_lower)
    if not m:
        return None
    if m.group(1):
        hours = int(m.group(1)) * (24 if m.group(2).startswith("day") else 1)
    else:
        hours = int(m.group(3))
    return hours if 0 < hours <= 24 * 365 else None

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

    # 6. Relative window ("last 24 hours", "last 3 days", "48h") counted back from the newest tweet in view
    window_hours = _parse_relative_window(q_lower)
    if window_hours:
        filters["date_range"] = f"last_{window_hours}h"
        chips.append(f"Window: Last {window_hours} Hours")

    # 7. Absolute dates ("after sep 20", "before Oct 3 2017"). The year comes from the query, else from the
    #    matched event's own year (hurricane_maria_2017 -> 2017), else it is resolved against the data.
    date_match = _DATE_PATTERN.search(q_lower)
    if date_match:
        direction, month_abbr, day, year = date_match.groups()
        month = _MONTHS[month_abbr[:3]]
        if not year and "event" in filters:
            event_year = re.search(r"_(\d{4})$", filters["event"])
            year = event_year.group(1) if event_year else None
        key = "date_before" if direction == "before" else "date_after"
        if year:
            filters[key] = f"{int(year):04d}-{month:02d}-{int(day):02d}"
            chips.append(f"{'Before' if direction == 'before' else 'After'}: {filters[key]}")
        else:
            filters[key] = f"{month:02d}-{int(day):02d}"
            chips.append(f"{'Before' if direction == 'before' else 'After'}: {month_abbr[:3].title()} {int(day)}")

    is_valid = len(chips) > 0
    msg = f"Extracted {len(chips)} filters." if is_valid else "Could not recognize specific crisis filters from input."

    return {
        "is_valid": is_valid,
        "message": msg,
        "parsed_chips": chips,
        "filters": filters,
        "raw_query": query
    }

def _resolve_date_bound(value: str, ts: pd.Series) -> Optional[pd.Timestamp]:
    """Turn 'YYYY-MM-DD' or a year-less 'MM-DD' into a UTC timestamp (year-less uses the newest year in the data)."""
    try:
        if len(value) == 10:
            bound = pd.Timestamp(value)
        else:
            month, day = (int(p) for p in value.split("-"))
            bound = pd.Timestamp(year=int(ts.max().year), month=month, day=day)
    except (ValueError, TypeError):
        return None
    return bound.tz_localize(ts.dt.tz) if ts.dt.tz is not None and bound.tzinfo is None else bound

def apply_query_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply parsed query filters to a tweet DataFrame.

    Returns ``(filtered_df, applied)`` where ``applied`` lists each filter that was actually applied, so the UI
    never advertises a filter that did nothing. Relative windows count back from the newest tweet in ``df``
    (the data is a historical replay, so "now" has no meaning).
    """
    applied: List[str] = []
    out = df

    if "category" in filters and "category" in out.columns:
        out = out[out["category"] == filters["category"]]
        applied.append(f"category = {filters['category']}")
    if "urgency_level" in filters and "urgency_level" in out.columns:
        out = out[out["urgency_level"] == filters["urgency_level"]]
        applied.append(f"urgency = {filters['urgency_level']}")
    if "event" in filters and "event" in out.columns:
        out = out[out["event"] == filters["event"]]
        applied.append(f"event = {filters['event']}")
    if "location" in filters and "locations" in out.columns:
        wanted = filters["location"].lower()
        out = out[out["locations"].apply(lambda locs: any(wanted in str(loc).lower() for loc in as_list(locs)))]
        applied.append(f"location contains '{filters['location']}'")
    if "need_type" in filters and "needs" in out.columns:
        out = out[out["needs"].apply(lambda needs: filters["need_type"] in as_list(needs))]
        applied.append(f"need = {filters['need_type']}")

    wants_time = any(k in filters for k in ("date_range", "date_after", "date_before"))
    if wants_time and "timestamp" in out.columns and not out.empty:
        ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        if ts.notna().any():
            keep = pd.Series(True, index=out.index)
            if "date_range" in filters:
                hours = int(filters["date_range"].replace("last_", "").replace("h", ""))
                keep &= ts >= ts.max() - pd.Timedelta(hours=hours)
                applied.append(f"within the last {hours} hours of the data")
            if "date_after" in filters:
                bound = _resolve_date_bound(filters["date_after"], ts)
                if bound is not None:
                    keep &= ts >= bound
                    applied.append(f"on or after {bound.date()}")
            if "date_before" in filters:
                bound = _resolve_date_bound(filters["date_before"], ts)
                if bound is not None:
                    keep &= ts < bound
                    applied.append(f"before {bound.date()}")
            out = out[keep]

    return out, applied
