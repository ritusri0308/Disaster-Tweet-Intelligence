"""
Small shared helpers for CrisisLens.
"""

import json
from typing import Any, Iterable, List

import pandas as pd

# Columns that hold per-tweet lists (locations, needs, matched urgency terms, geocoded places, figures).
LIST_COLUMNS = ["locations", "matched_terms", "needs", "geocoded_locations", "quantities"]

def as_list(value: Any) -> List[Any]:
    """
    Coerce a per-tweet list value to a plain Python list.

    Parquet returns list columns as numpy arrays, CSV/SQLite as JSON strings, and missing values as
    None/NaN. Code that tests ``isinstance(value, list)`` silently treats numpy arrays as empty, so
    every consumer should go through this helper.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    if isinstance(value, bytes):
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, list) else []
            except ValueError:
                return []
        return [text] if text else []
    if hasattr(value, "tolist"):
        out = value.tolist()
        if isinstance(out, list):
            return out
        return [] if out != out else [out]  # scalar NaN -> empty
    return []

def normalize_list_columns(df: pd.DataFrame, columns: Iterable[str] = LIST_COLUMNS) -> pd.DataFrame:
    """Return a copy of ``df`` whose list columns are plain Python lists, whatever format they were loaded from."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(as_list)
    return df
