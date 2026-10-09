"""
Cached wrappers around the expensive analytics used by the dashboard.

Streamlit reruns the whole page on every widget interaction. Surge detection and incident clustering
(O(n^2) similarity per event) used to be recomputed on every click, including "Show more". These wrappers
key the cache on a cheap signature of the tweets in view instead of hashing the whole DataFrame (which holds
list columns that are slow and awkward to hash).
"""

import hashlib
import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.analytics import generate_situation_summary
from src.anomaly import detect_event_surges
from src.briefing import generate_html_briefing
from src.clustering import cluster_event_incidents
from src.utils import as_list

def df_signature(df: pd.DataFrame) -> str:
    """
    Stable fingerprint of what is in view: which tweets (their ids, in order) and whether personal details are shown.
    The privacy setting must be part of the key, otherwise a cached export built while details were visible could
    be served after the operator hides them again (or the reverse).
    """
    if df.empty:
        return "empty"
    ids = df["tweet_id"].astype(str) if "tweet_id" in df.columns else pd.Series(df.index.astype(str))
    digest = hashlib.md5("|".join(ids).encode("utf-8")).hexdigest()
    pii = "shown" if st.session_state.get("show_pii", False) else "hidden"
    return f"{len(df)}:{digest}:pii-{pii}"

@st.cache_data(show_spinner=False, max_entries=24)
def cached_surges(_df: pd.DataFrame, signature: str) -> List[Dict[str, Any]]:
    return detect_event_surges(_df)

@st.cache_data(show_spinner=False, max_entries=24)
def cached_incidents(_df: pd.DataFrame, signature: str, max_sample_per_event: int = 500) -> List[Dict[str, Any]]:
    return cluster_event_incidents(_df, max_sample_per_event=max_sample_per_event)

@st.cache_data(show_spinner=False, max_entries=24)
def cached_summary(_df: pd.DataFrame, signature: str) -> Dict[str, Any]:
    return generate_situation_summary(_df)

@st.cache_data(show_spinner=False, max_entries=8)
def cached_briefing_html(_df: pd.DataFrame, signature: str, filters_key: str) -> str:
    """Executive HTML briefing for the tweets in view (``filters_key`` is the human-readable filter description)."""
    summary = cached_summary(_df, signature)
    incidents = cached_incidents(_df, signature)
    surges = cached_surges(_df, signature)
    filters = {"Active filters": filters_key} if filters_key else {}
    return generate_html_briefing(_df, summary, incidents, surges, filters)

@st.cache_data(show_spinner=False, max_entries=8)
def cached_csv(_df: pd.DataFrame, signature: str) -> bytes:
    """CSV export of the tweets in view (list columns are written as JSON text, so other tools can parse them)."""
    export = _df.copy()
    for col in ("locations", "matched_terms", "needs", "geocoded_locations", "quantities"):
        if col in export.columns:
            export[col] = export[col].apply(lambda v: json.dumps(as_list(v), default=str))
    return export.to_csv(index=False).encode("utf-8")
