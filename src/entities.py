"""
Quantity / casualty-figure extraction for CrisisLens.

Pulls structured, reported figures out of free text ("death toll hits 413", "5 people trapped",
"500 families evacuated") so operators can see the numbers a tweet claims without reading it in full.
These figures are *claims made in tweets*, not verified counts.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# Normalised figure kinds
KIND_MAP = {
    "killed": "killed", "kills": "killed", "killing": "killed", "dead": "killed", "died": "killed",
    "dies": "killed", "death": "killed", "deaths": "killed", "fatalities": "killed",
    "perished": "killed", "lives": "killed",
    "injured": "injured", "hurt": "injured", "wounded": "injured", "injuries": "injured",
    "missing": "missing",
    "trapped": "trapped", "stranded": "trapped", "stuck": "trapped",
    "displaced": "displaced", "evacuated": "displaced", "homeless": "displaced", "evacuees": "displaced",
    "rescued": "rescued",
}

# The negative lookahead stops the regex backtracking into the middle of a number ("72,000" must not match as "72")
_NUM = r"(?P<n>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?!\d|[,.]\d)(?:\s?(?P<mult>million|thousand))?"
_KIND = r"(?P<kind>" + "|".join(sorted(KIND_MAP, key=len, reverse=True)) + r")\b"
_PERSON_NOUN = (r"(?:people|persons|residents|children|kids|families|villagers|tourists|victims|bodies|"
                r"individuals|migrants|workers|others|civilians|souls|lives|more)")
_FILLER = r"(?:(?:have|has|were|are|been|now|confirmed|reported|feared|believed|known\s+to\s+have|also|all)\s+)*"
_QUALIFIER = r"(?:(?:at\s+least|over|more\s+than|about|nearly|around|another|up\s+to|almost|some)\s+)?"
_UNIT_GUARD = r"(?!\s*(?:%|percent|km|kms|miles?|mph|feet|ft|inch(?:es)?|mm|cm|kg|degrees?|hours?|hrs?|days?|minutes?|mins?)\b)"

# "413 people are now known to have died" / "5 people trapped" / "500 families evacuated" / "90 deaths"
_PATTERN_COUNT_FIRST = re.compile(
    _QUALIFIER + _NUM + r"\s+(?:" + _PERSON_NOUN + r"\s+)?" + _FILLER + _KIND, re.IGNORECASE
)
# "death toll hits 413" / "toll rises to 1,000"
_PATTERN_TOLL = re.compile(
    r"(?:death|casualty|fatality|body)\s+(?:toll|count)\s+(?:has\s+|now\s+)*"
    r"(?:hits?|rises?\s+to|rose\s+to|climbs?\s+to|climbed\s+to|reach(?:es|ed)|stands?\s+at|stood\s+at|"
    r"is|at|of|to|up\s+to|passes|passed|tops|topped|now)?\s*[:]?\s*" + _QUALIFIER + _NUM,
    re.IGNORECASE,
)
# A number that is immediately followed by its own figure word belongs to that word, not the one before it:
# in "47 people dead 72,000 rescued" the 72,000 are rescued, they are not "dead 72,000".
_NEXT_FIGURE_WORD = (
    r"(?!\s+(?:" + _PERSON_NOUN + r"\s+)?" + _FILLER + r"(?:" + "|".join(sorted(KIND_MAP, key=len, reverse=True)) + r")\b)"
    r"(?!\s+(?:homes|houses|buildings|structures|businesses|roofs)\b)"
)
# "killed 83" / "kills at least 10"
_PATTERN_VERB_FIRST = re.compile(
    r"\b(?P<kind>killed|kills|killing|dead|injured|missing|trapped|stranded|displaced|evacuated|rescued)\s+"
    + _QUALIFIER + _NUM + _NEXT_FIGURE_WORD + _UNIT_GUARD + r"(?P<noun>\s+" + _PERSON_NOUN + r")?",
    re.IGNORECASE,
)
# "120 homes destroyed"
_PATTERN_STRUCTURES = re.compile(
    _NUM + r"\s+(?:homes|houses|buildings|structures|businesses|roofs)\s+" + _FILLER
    + r"(?:destroyed|damaged|flooded|lost|collapsed|burned|burnt)\b",
    re.IGNORECASE,
)

MAX_PLAUSIBLE_COUNT = 50_000_000

# A year used as a date ("in 1900", "of 1780", "April 18, 1906", "04.10.1963", "1963 - Hurricane Flora") rather than a count
_YEAR_DIGITS = r"1[5-9]\d\d|20[0-3]\d"
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_PATTERN_YEAR_REFERENCE = re.compile(
    r"(?:\b(?:in|of|since|during|from|before|circa)\s+|\b" + _MONTH + r"\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?|\d{1,2}[./-]\d{1,2}[./-])"
    + r"(?P<year_a>" + _YEAR_DIGITS + r")(?!\d)"
    + r"|(?<!\d)(?P<year_b>" + _YEAR_DIGITS + r")(?!\d)\s*[-–—:]\s*(?:hurricane|earthquake|cyclone|flood|typhoon|storm|tsunami|wildfire|fire)",
    re.IGNORECASE,
)

def event_year(event: Any) -> Optional[int]:
    """Year encoded at the end of an event id: 'hurricane_harvey_2017' -> 2017."""
    m = re.search(r"_(\d{4})$", str(event or ""))
    return int(m.group(1)) if m else None

def _refers_to_earlier_year(text: str, reference_year: int) -> bool:
    """True if the text dates something to a year before ``reference_year`` (a historical comparison)."""
    for m in _PATTERN_YEAR_REFERENCE.finditer(text):
        year = m.group("year_a") or m.group("year_b")
        if year and int(year) < reference_year:
            return True
    return False


def _to_int(num: str, mult: Optional[str] = None) -> int:
    value = float(num.replace(",", ""))
    if mult:
        value *= 1_000_000 if mult.lower() == "million" else 1_000
    return int(round(value))


def _looks_like_year(num: str, value: int) -> bool:
    return "," not in num and "." not in num and 1950 <= value <= 2035


def extract_quantities(text: Any, reference_year: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Extract reported figures from text.

    Returns a list like ``[{"kind": "killed", "count": 413}, {"kind": "trapped", "count": 5}]``,
    de-duplicated and capped at six entries, in order of appearance.

    If ``reference_year`` (the year of the disaster being analysed) is given, a tweet that dates something to an
    earlier year ("the 1780 hurricane killed 27,500", "April 18, 1906 ... 3,000 died") is treated as a historical
    comparison and yields no figures. This is a heuristic, not a guarantee.
    """
    if text is None:
        return []
    s = str(text)
    if not s.strip():
        return []
    if reference_year is not None and _refers_to_earlier_year(s, reference_year):
        return []

    candidates: List[Tuple[int, str, int]] = []  # (position in text, kind, count)

    def add(pos: int, kind: str, num: str, mult: Optional[str] = None, allow_year_like: bool = True):
        try:
            count = _to_int(num, mult)
        except ValueError:
            return
        if count < 1 or count > MAX_PLAUSIBLE_COUNT:
            return
        if not allow_year_like and _looks_like_year(num, count):
            return
        candidates.append((pos, kind, count))

    for m in _PATTERN_COUNT_FIRST.finditer(s):
        add(m.start("n"), KIND_MAP[m.group("kind").lower()], m.group("n"), m.group("mult"), allow_year_like=False)
    for m in _PATTERN_TOLL.finditer(s):
        add(m.start("n"), "killed", m.group("n"), m.group("mult"), allow_year_like=False)
    for m in _PATTERN_VERB_FIRST.finditer(s):
        # A bare 4-digit number right after a verb is usually a year ("killed 2017 ...") unless a person noun follows.
        add(m.start("n"), KIND_MAP[m.group("kind").lower()], m.group("n"), m.group("mult"),
            allow_year_like=bool(m.group("noun")))
    for m in _PATTERN_STRUCTURES.finditer(s):
        add(m.start("n"), "structures", m.group("n"), m.group("mult"), allow_year_like=False)

    # Order by where each figure appears in the text, then drop repeats and cap the list
    found: List[Dict[str, Any]] = []
    seen = set()
    for _, kind, count in sorted(candidates, key=lambda c: c[0]):
        if (kind, count) not in seen:
            seen.add((kind, count))
            found.append({"kind": kind, "count": count})
    return found[:6]


def summarize_quantities(quantities: Any) -> Dict[str, int]:
    """Return the largest reported count per kind from a list of figures."""
    summary: Dict[str, int] = {}
    if not isinstance(quantities, (list, tuple)) and not hasattr(quantities, "__iter__"):
        return summary
    for q in quantities:
        if isinstance(q, dict) and "kind" in q and "count" in q:
            summary[q["kind"]] = max(summary.get(q["kind"], 0), int(q["count"]))
    return summary


def format_quantity(q: Dict[str, Any]) -> str:
    """Human-readable label such as '413 killed'."""
    unit = "structures damaged" if q.get("kind") == "structures" else q.get("kind", "")
    return f"{int(q.get('count', 0)):,} {unit}".strip()
