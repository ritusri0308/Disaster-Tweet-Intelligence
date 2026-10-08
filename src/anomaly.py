"""
Surge and anomaly detection module for CrisisLens.
Detects volume spikes and surges in critical tweet ratios using robust rolling statistics (Median/MAD).
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from src.config import ANOMALY_WINDOW_HOURS, ANOMALY_Z_THRESHOLD

logger = logging.getLogger(__name__)

def compute_robust_z_score(series: pd.Series) -> pd.Series:
    """
    Compute robust z-score using median and Median Absolute Deviation (MAD):
    z = 0.6745 * (x - median) / MAD
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad < 1e-6:
        # Fallback to standard deviation if MAD is 0
        std = series.std()
        if std < 1e-6:
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std
    return 0.6745 * (series - median) / mad

def detect_event_surges(
    df: pd.DataFrame,
    window_hours: int = ANOMALY_WINDOW_HOURS,
    z_threshold: float = ANOMALY_Z_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Detect abnormal tweet-volume bursts and critical-share surges per disaster event.
    """
    if df.empty or "timestamp" not in df.columns:
        return []

    valid_df = df.copy()
    valid_df["ts"] = pd.to_datetime(valid_df["timestamp"], errors="coerce")
    valid_df = valid_df.dropna(subset=["ts"]).sort_values("ts")

    if valid_df.empty:
        return []

    alerts = []

    for event, grp in valid_df.groupby("event"):
        if len(grp) < 10:
            continue

        # Resample hourly
        hourly = grp.set_index("ts").resample("1h").agg(
            total_tweets=("tweet_id", "count"),
            critical_tweets=("urgency_level", lambda s: (s == "CRITICAL").sum())
        ).fillna(0)

        if len(hourly) < 6:
            continue

        # Volume robust z-score
        hourly["z_volume"] = compute_robust_z_score(hourly["total_tweets"])
        hourly["crit_share"] = hourly["critical_tweets"] / np.maximum(hourly["total_tweets"], 1)
        hourly["z_crit"] = compute_robust_z_score(hourly["crit_share"])

        # Detect volume spikes
        spikes = hourly[hourly["z_volume"] >= z_threshold]
        for dt, row in spikes.iterrows():
            window_tweets = grp[(grp["ts"] >= dt) & (grp["ts"] < dt + pd.Timedelta(hours=1))]
            text_col = "clean_text" if "clean_text" in window_tweets.columns else ("tweet_text" if "tweet_text" in window_tweets.columns else None)
            text_corpus = " ".join(window_tweets[text_col].astype(str).tolist()) if text_col else ""
            words = [w for w in text_corpus.split() if len(w) > 3]
            top_words = pd.Series(words).value_counts().head(5).index.tolist() if words else []

            alerts.append({
                "alert_type": "Volume Burst",
                "event": event,
                "place": "All",
                "window_start": str(dt),
                "window_end": str(dt + pd.Timedelta(hours=1)),
                "magnitude": int(row["total_tweets"]),
                "z_score": round(float(row["z_volume"]), 2),
                "top_terms": top_words,
                "details": f"Tweet volume peaked at {int(row['total_tweets'])} tweets/hr (z={row['z_volume']:.1f})"
            })

        # Detect critical share surges (minimum 3 critical tweets in hour)
        crit_spikes = hourly[(hourly["z_crit"] >= z_threshold) & (hourly["critical_tweets"] >= 3)]
        for dt, row in crit_spikes.iterrows():
            alerts.append({
                "alert_type": "Critical Share Surge",
                "event": event,
                "place": "All",
                "window_start": str(dt),
                "window_end": str(dt + pd.Timedelta(hours=1)),
                "magnitude": int(row["critical_tweets"]),
                "z_score": round(float(row["z_crit"]), 2),
                "top_terms": ["critical", "emergency", "immediate"],
                "details": f"Critical tweets surged to {row['crit_share']:.1%} of volume ({int(row['critical_tweets'])} critical alerts, z={row['z_crit']:.1f})"
            })

    # Sort alerts by z_score descending
    alerts.sort(key=lambda a: a["z_score"], reverse=True)
    return alerts
