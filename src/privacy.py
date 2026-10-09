"""
Redaction of personal contact details from tweet text.

Help requests often include a phone number or a home address ("this lady needs rescue ... 6210 Charles ave ...
713 5427751"). The dashboard hides these by default; an operator can reveal them deliberately.

What is covered: phone numbers, email addresses, and street addresses written as ``<number> <name> <street word>``.
What is NOT covered: personal names and free-text descriptions of where someone is. Reliable name detection needs a
model rather than patterns, so redaction reduces exposure but is not a guarantee of anonymity.
"""

import re
from typing import Any

import pandas as pd

EMAIL_PLACEHOLDER = "[email]"
PHONE_PLACEHOLDER = "[phone]"
ADDRESS_PLACEHOLDER = "[address]"

# Text columns that may hold tweet content, and the suffix used for their precomputed redacted copies
TEXT_COLUMNS = ["raw_unescaped_text", "tweet_text", "clean_text", "text"]
REDACTED_SUFFIX = "__redacted"

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_PHONE_INTERNATIONAL = re.compile(r"(?<![\w])\+\d[\d\s().-]{7,}\d(?![\w])")
# A comma or dot directly between digits is a thousands/decimal separator ("1,700,000"), but a comma after a number
# in prose ("... 713 5427751, 80 year old") is just punctuation, so only digit-adjacent separators are rejected.
_NOT_PART_OF_NUMBER_BEFORE = r"(?<!\d)(?<!\d,)(?<!\d\.)"
_NOT_PART_OF_NUMBER_AFTER = r"(?!\d)(?!,\d)(?!\.\d)"
_PHONE_NORTH_AMERICAN = re.compile(
    _NOT_PART_OF_NUMBER_BEFORE + r"(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}" + _NOT_PART_OF_NUMBER_AFTER
)
_PHONE_LOCAL = re.compile(_NOT_PART_OF_NUMBER_BEFORE + r"\d{3}[\s.-]\d{4}" + _NOT_PART_OF_NUMBER_AFTER)

_STREET_WORD = (r"(?i:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct|way|highway|hwy|"
                r"parkway|pkwy|place|pl|circle|cir|trail|trl)")
# Words that mean "<number> <these> road" is a quantity or a description, not an address ("50 people on road")
_NOT_A_STREET_NAME = (r"(?!(?i:people|persons|person|families|family|homes|houses|children|kids|residents|others|more|"
                      r"of|on|in|near|at|the|and|to|for|from|with|by|km|miles|feet|hours|days|are|is|was)\b)")
_STREET_ADDRESS = re.compile(
    r"(?<![\w#])\d{1,6}\s+(?:" + _NOT_A_STREET_NAME + r"[A-Za-z][A-Za-z'.-]*\s+){1,3}" + _STREET_WORD + r"\b\.?"
)

def redact_pii(text: Any) -> str:
    """Replace phone numbers, email addresses and street addresses in ``text`` with placeholders."""
    if text is None:
        return ""
    s = str(text)
    if not s:
        return s
    has_digit = any(ch.isdigit() for ch in s)
    if "@" in s:
        s = _EMAIL.sub(EMAIL_PLACEHOLDER, s)
    if has_digit:
        # Addresses first: their house number must not be eaten by a phone pattern
        s = _STREET_ADDRESS.sub(ADDRESS_PLACEHOLDER, s)
        s = _PHONE_INTERNATIONAL.sub(PHONE_PLACEHOLDER, s)
        s = _PHONE_NORTH_AMERICAN.sub(PHONE_PLACEHOLDER, s)
        s = _PHONE_LOCAL.sub(PHONE_PLACEHOLDER, s)
    return s

def add_redacted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute a redacted copy of each text column (``<column>__redacted``) so toggling is instant."""
    df = df.copy()
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col + REDACTED_SUFFIX] = df[col].map(redact_pii)
    return df

def apply_redaction(df: pd.DataFrame, redact: bool = True) -> pd.DataFrame:
    """
    Return the view of ``df`` to display: with ``redact`` the text columns hold the redacted copies, otherwise the
    originals. The helper ``__redacted`` columns never appear in the result, so exports cannot carry both.
    """
    helper_cols = [c for c in df.columns if c.endswith(REDACTED_SUFFIX)]
    if not helper_cols:
        return df
    out = df.copy()
    if redact:
        for col in TEXT_COLUMNS:
            if col + REDACTED_SUFFIX in out.columns:
                out[col] = out[col + REDACTED_SUFFIX]
    return out.drop(columns=helper_cols)
