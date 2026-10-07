"""
Analytics and aggregated intelligence module for disaster tweet dashboard.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

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
    counts = df["urgency_level"].value_counts()
    return counts.reindex([l for l in order if l in counts.index or True]).fillna(0).astype(int)

def get_category_urgency_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Compute category x urgency level cross-tabulation table."""
    if df.empty or "category" not in df.columns or "urgency_level" not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df["category"], df["urgency_level"], margins=False)

def get_top_locations(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """
    Explode locations list column, strip whitespace, remove empty entries, return top N.
    """
    if df.empty or "locations" not in df.columns:
        return pd.Series(dtype=int)

    # Explode location lists
    s = df["locations"]
    locs = []
    for loc_list in s:
        if isinstance(loc_list, list):
            for l in loc_list:
                cleaned_l = str(l).strip()
                if cleaned_l:
                    locs.append(cleaned_l)
        elif isinstance(loc_list, str) and loc_list.strip():
            locs.append(loc_list.strip())

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

def generate_situation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate template-based situation intelligence summary (No LLM).
    """
    if df.empty:
        return {
            "total_tweets": 0,
            "critical_count": 0,
            "top_location": "None",
            "dominant_category": "None",
            "peak_activity_time": "N/A",
            "summary_text": "No data available for situation summary."
        }

    total_tweets = len(df)
    crit_count = (df["urgency_level"] == "CRITICAL").sum() if "urgency_level" in df.columns else 0

    cat_counts = get_category_counts(df)
    dom_category = cat_counts.index[0] if not cat_counts.empty else "Unknown"

    top_loc_series = get_top_locations(df, top_n=1)
    top_location = top_loc_series.index[0] if not top_loc_series.empty else "None detected"

    vol_df = get_volume_over_time(df, freq="D")
    if not vol_df.empty:
        max_row = vol_df.loc[vol_df["tweet_count"].idxmax()]
        peak_time = str(max_row["timestamp"].date()) if hasattr(max_row["timestamp"], "date") else str(max_row["timestamp"])
    else:
        peak_time = "N/A"

    summary_text = (
        f"Analyzed {total_tweets:,} disaster tweets. Identified {crit_count:,} CRITICAL emergency alerts "
        f"requiring immediate responder action. Dominant activity category is '{dom_category}'. "
        f"Most frequently referenced location is '{top_location}'. Peak tweet volume observed on {peak_time}."
    )

    return {
        "total_tweets": total_tweets,
        "critical_count": crit_count,
        "top_location": top_location,
        "dominant_category": dom_category,
        "peak_activity_time": peak_time,
        "summary_text": summary_text
    }
