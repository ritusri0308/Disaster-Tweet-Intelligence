"""
Analytics and aggregated intelligence module for disaster tweet dashboard.
"""

from collections import Counter
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from src.entities import summarize_quantities
from src.utils import as_list

def get_category_counts(df: pd.DataFrame) -> pd.Series:
    """Get frequency counts of categories."""
    if df.empty or "category" not in df.columns:
        return pd.Series(dtype=int)
    return df["category"].value_counts()

def get_urgency_counts(df: pd.DataFrame) -> pd.Series:
    """Get frequency counts of urgency levels."""
    if df.empty or "urgency_level" not in df.columns:
        return pd.Series(dtype=int)
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    return df["urgency_level"].value_counts().reindex(order).fillna(0).astype(int)

def get_needs_counts(df: pd.DataFrame) -> pd.Series:
    """Frequency of each detected need type, most common first."""
    if df.empty or "needs" not in df.columns:
        return pd.Series(dtype=int)
    counts = Counter(need for needs in df["needs"] for need in as_list(needs))
    return pd.Series(counts, dtype=int).sort_values(ascending=False) if counts else pd.Series(dtype=int)

def get_activity_timeseries(df: pd.DataFrame, freq: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Tweets per time bucket, split into all / critical+high / critical, for the Situation Room trend chart.

    ``freq`` defaults to hourly when the data spans a week or less, otherwise daily. Returns
    ``(frame, freq)`` where the frame has columns ``timestamp``, ``tweets``, ``priority`` (CRITICAL+HIGH) and ``critical``.
    """
    empty = pd.DataFrame(columns=["timestamp", "tweets", "priority", "critical"])
    if df.empty or "timestamp" not in df.columns:
        return empty, freq or "D"

    valid = df.copy()
    valid["ts"] = pd.to_datetime(valid["timestamp"], errors="coerce")
    valid = valid.dropna(subset=["ts"])
    if valid.empty:
        return empty, freq or "D"

    if freq is None:
        freq = "h" if (valid["ts"].max() - valid["ts"].min()) <= pd.Timedelta(days=7) else "D"

    level = valid["urgency_level"] if "urgency_level" in valid.columns else pd.Series("LOW", index=valid.index)
    valid["_priority"] = level.isin(["CRITICAL", "HIGH"]).astype(int)
    valid["_critical"] = (level == "CRITICAL").astype(int)
    grouped = valid.set_index("ts").resample(freq).agg(
        tweets=("tweet_id", "count") if "tweet_id" in valid.columns else ("_priority", "size"),
        priority=("_priority", "sum"),
        critical=("_critical", "sum"),
    ).reset_index().rename(columns={"ts": "timestamp"})
    return grouped, freq

def get_category_urgency_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Compute category x urgency level cross-tabulation table."""
    if df.empty or "category" not in df.columns or "urgency_level" not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df["category"], df["urgency_level"], margins=False)

def get_top_locations(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """
    Explode locations / geocoded_locations column, strip whitespace, remove empty entries, return top N.
    Prefers clean geocoded_locations if present, with fallback to locations.
    Handles list, numpy.ndarray, tuple, set, and JSON-serialized strings.
    """
    if df.empty:
        return pd.Series(dtype=int)

    locs = []

    def _extract_items(val):
        if val is None:
            return []
        if isinstance(val, str):
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                try:
                    import json
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            return [val] if val else []
        if isinstance(val, (list, tuple, set)) or hasattr(val, "__iter__"):
            return list(val)
        return []

    # 1. Try geocoded_locations first (cleanest, filtered)
    if "geocoded_locations" in df.columns:
        for item_list in df["geocoded_locations"]:
            items = _extract_items(item_list)
            for item in items:
                if isinstance(item, dict) and "name" in item:
                    name = str(item["name"]).strip()
                    if name:
                        locs.append(name)
                elif isinstance(item, str) and item.strip():
                    locs.append(item.strip())

    # 2. If no geocoded locations found, fall back to locations column
    if not locs and "locations" in df.columns:
        for item_list in df["locations"]:
            items = _extract_items(item_list)
            for item in items:
                if isinstance(item, dict) and "name" in item:
                    name = str(item["name"]).strip()
                    if name:
                        locs.append(name)
                elif isinstance(item, str) and item.strip():
                    locs.append(item.strip())

    if not locs:
        return pd.Series(dtype=int)

    return pd.Series(locs).value_counts().head(top_n)

def get_volume_over_time(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Resample tweet counts over time based on derived timestamp.
    Handles missing / NaT timestamps gracefully.
    """
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "tweet_count"])

    valid_df = df.copy()
    valid_df["ts"] = pd.to_datetime(valid_df["timestamp"], errors="coerce")
    valid_df = valid_df.dropna(subset=["ts"])

    if valid_df.empty:
        return pd.DataFrame(columns=["timestamp", "tweet_count"])

    resampled = valid_df.set_index("ts").resample(freq).size().reset_index(name="tweet_count")
    resampled.rename(columns={"ts": "timestamp"}, inplace=True)
    return resampled

def get_per_event_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate event comparison summary table: total tweets, emergency %, help request %, critical count.
    """
    if df.empty or "event" not in df.columns:
        return pd.DataFrame()

    summary_rows = []
    for event, group in df.groupby("event"):
        total = len(group)
        em_pct = round((group["category"] == "Emergency").mean() * 100, 1) if "category" in group.columns else 0.0
        hr_pct = round((group["category"] == "Help Request").mean() * 100, 1) if "category" in group.columns else 0.0
        crit_cnt = (group["urgency_level"] == "CRITICAL").sum() if "urgency_level" in group.columns else 0
        top_loc_s = get_top_locations(group, top_n=1)
        top_loc = top_loc_s.index[0] if not top_loc_s.empty else "N/A"

        summary_rows.append({
            "event": event,
            "total_tweets": total,
            "emergency_pct": em_pct,
            "help_request_pct": hr_pct,
            "critical_count": crit_cnt,
            "top_location": top_loc
        })

    return pd.DataFrame(summary_rows).sort_values("total_tweets", ascending=False)

def most_reported_figures(quantities_column: Any) -> Dict[str, Dict[str, int]]:
    """
    For each figure kind, the value cited by the most tweets (ties go to the larger value).

    The consensus value is far more robust than the maximum: one mis-parsed or exaggerated tweet cannot dominate it.
    Returns ``{kind: {"count": value, "tweets": number_of_tweets_citing_it}}``.
    """
    cited: Dict[str, Counter] = {}
    for quantities in quantities_column:
        for kind, count in summarize_quantities(as_list(quantities)).items():
            cited.setdefault(kind, Counter())[count] += 1
    result: Dict[str, Dict[str, int]] = {}
    for kind, counts in cited.items():
        value, tweets = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        result[kind] = {"count": int(value), "tweets": int(tweets)}
    return result

def generate_situation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate template-based situation intelligence summary (No LLM).
    """
    if df.empty:
        return {
            "total_tweets": 0,
            "critical_count": 0,
            "top_location": "None detected",
            "dominant_category": "None",
            "peak_activity_time": "N/A",
            "top_needs": {},
            "reported_figures": {},
            "summary_text": "No data available for situation summary."
        }

    total_tweets = len(df)
    crit_count = (df["urgency_level"] == "CRITICAL").sum() if "urgency_level" in df.columns else 0

    cat_counts = get_category_counts(df)
    dom_category = cat_counts.index[0] if not cat_counts.empty else "Unknown"

    top_loc_series = get_top_locations(df, top_n=1)
    if not top_loc_series.empty:
        top_location = str(top_loc_series.index[0])
        loc_phrase = f"Most frequently referenced location is '{top_location}'."
    else:
        top_location = "None detected"
        loc_phrase = "No specific locations detected in the active view."

    vol_df = get_volume_over_time(df, freq="D")
    if not vol_df.empty:
        max_row = vol_df.loc[vol_df["tweet_count"].idxmax()]
        peak_time = str(max_row["timestamp"].date()) if hasattr(max_row["timestamp"], "date") else str(max_row["timestamp"])
    else:
        peak_time = "N/A"

    top_needs = get_needs_counts(df).head(3)
    needs_phrase = (
        " Most reported needs: " + ", ".join(f"{name} ({count:,})" for name, count in top_needs.items()) + "."
        if not top_needs.empty else ""
    )

    # Figures that tweets claim (reports, not verified counts). Only a single event gives a meaningful answer:
    # across several events one number would mix unrelated disasters.
    reported_figures: Dict[str, Dict[str, int]] = {}
    if "quantities" in df.columns and "event" in df.columns and df["event"].nunique() == 1:
        reported_figures = most_reported_figures(df["quantities"])
    reported = [f"{reported_figures[k]['count']:,} {k} ({reported_figures[k]['tweets']:,} tweets)"
                for k in ("killed", "injured", "missing", "trapped") if k in reported_figures]
    figures_phrase = f" Most-cited figures in tweets (unverified): {', '.join(reported)}." if reported else ""

    summary_text = (
        f"Analyzed {total_tweets:,} disaster tweets. Identified {crit_count:,} CRITICAL emergency alerts "
        f"requiring attention. Dominant activity category is '{dom_category}'. "
        f"{loc_phrase} Peak tweet volume observed on {peak_time}.{needs_phrase}{figures_phrase}"
    )

    return {
        "total_tweets": total_tweets,
        "critical_count": crit_count,
        "top_location": top_location,
        "dominant_category": dom_category,
        "peak_activity_time": peak_time,
        "top_needs": {str(k): int(v) for k, v in top_needs.items()},
        "reported_figures": reported_figures,
        "summary_text": summary_text
    }
