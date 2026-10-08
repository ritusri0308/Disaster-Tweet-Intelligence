"""
Incident clustering and 2D semantic projection module for CrisisLens.
Clusters near-duplicate and storm-like tweets into cohesive emergency incidents.
Generates 2D SVD semantic embeddings for map exploration.
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

def cluster_event_incidents(
    df: pd.DataFrame,
    similarity_threshold: float = 0.65,
    max_sample_per_event: int = 1500
) -> List[Dict[str, Any]]:
    """
    Cluster tweets within disaster events using TF-IDF cosine similarity.
    Scales efficiently by chunking per event.
    Returns list of incident dictionaries ranked by (max_urgency * log(size + 1)).
    """
    if df.empty or "clean_text" not in df.columns:
        return []

    incidents = []
    global_inc_id = 1

    for event, event_df in df.groupby("event"):
        if event_df.empty:
            continue

        sample_df = event_df.head(max_sample_per_event).copy()
        texts = sample_df["clean_text"].fillna("").astype(str).tolist()

        if len(texts) < 2:
            continue

        vec = TfidfVectorizer(max_features=1500, min_df=1, ngram_range=(1, 2))
        try:
            tfidf_mat = vec.fit_transform(texts)
        except Exception:
            continue

        # Cosine similarity matrix
        sim_mat = cosine_similarity(tfidf_mat)

        # Connected components / greedy clustering
        assigned = set()
        for i in range(len(sample_df)):
            if i in assigned:
                continue

            # Neighbors with similarity >= threshold
            neighbors = np.where(sim_mat[i] >= similarity_threshold)[0]
            if len(neighbors) < 2:
                # Require at least 2 tweets to form an incident cluster
                continue

            for n_idx in neighbors:
                assigned.add(n_idx)

            cluster_rows = sample_df.iloc[neighbors]
            size = len(cluster_rows)

            # Representative tweet: closest to centroid or highest confidence
            rep_idx = cluster_rows["confidence"].idxmax() if "confidence" in cluster_rows.columns else cluster_rows.index[0]
            rep_text = cluster_rows.loc[rep_idx].get("raw_unescaped_text", cluster_rows.loc[rep_idx].get("tweet_text", ""))

            # Top terms
            cluster_text = " ".join(cluster_rows["clean_text"].astype(str).tolist())
            words = [w for w in cluster_text.split() if len(w) > 3]
            top_terms = pd.Series(words).value_counts().head(5).index.tolist() if words else []

            # Dominant category & max urgency
            dom_cat = cluster_rows["category"].mode()[0] if "category" in cluster_rows.columns and not cluster_rows["category"].empty else "General"
            max_urg = float(cluster_rows["urgency_score"].max()) if "urgency_score" in cluster_rows.columns else 0.0

            # Combined places
            places = set()
            if "locations" in cluster_rows.columns:
                for l_list in cluster_rows["locations"]:
                    if isinstance(l_list, list):
                        places.update(l_list)

            # Time span
            ts_series = pd.to_datetime(cluster_rows["timestamp"], errors="coerce").dropna()
            if not ts_series.empty:
                time_span = f"{ts_series.min().strftime('%Y-%m-%d %H:%M')} - {ts_series.max().strftime('%Y-%m-%d %H:%M')}"
            else:
                time_span = "N/A"

            # Sample tweet IDs
            tweet_ids = cluster_rows["tweet_id"].astype(str).tolist()[:5]

            incidents.append({
                "incident_id": f"INC-{event[:4].upper()}-{global_inc_id:04d}",
                "event": event,
                "size": size,
                "representative_tweet": rep_text,
                "top_terms": top_terms,
                "dominant_category": dom_cat,
                "max_urgency": max_urg,
                "places": list(places)[:5],
                "time_span": time_span,
                "sample_tweet_ids": tweet_ids,
                "priority_rank": round(max_urg * np.log1p(size), 2)
            })
            global_inc_id += 1

    # Rank incidents by priority (urgency x size)
    incidents.sort(key=lambda x: x["priority_rank"], reverse=True)
    return incidents

def generate_semantic_2d_projection(
    df: pd.DataFrame,
    max_points: int = 5000,
    n_topics: int = 8,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate 2D semantic projection of TF-IDF representations using TruncatedSVD.
    Runs MiniBatchKMeans to extract topic cluster assignments.
    Returns DataFrame with columns: x, y, topic_id, topic_label, text, category, urgency_level, event.
    """
    if df.empty or "clean_text" not in df.columns:
        return pd.DataFrame()

    # Stratified sample by category
    sample_size = min(len(df), max_points)
    sample_df = df.sample(n=sample_size, random_state=random_state).copy()

    texts = sample_df["clean_text"].fillna("").astype(str).tolist()
    vec = TfidfVectorizer(max_features=2500, min_df=2, ngram_range=(1, 2))
    tfidf_mat = vec.fit_transform(texts)

    # 2D Truncated SVD projection
    svd = TruncatedSVD(n_components=2, random_state=random_state)
    coords_2d = svd.fit_transform(tfidf_mat)

    # Topic clustering via MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_topics, random_state=random_state, batch_size=256)
    cluster_labels = kmeans.fit_predict(tfidf_mat)

    # Get top terms per topic cluster
    feature_names = np.array(vec.get_feature_names_out())
    topic_names = {}
    for cluster_id in range(n_topics):
        center = kmeans.cluster_centers_[cluster_id]
        top_idx = center.argsort()[::-1][:3]
        topic_names[cluster_id] = " / ".join(feature_names[top_idx])

    sample_df["x"] = coords_2d[:, 0]
    sample_df["y"] = coords_2d[:, 1]
    sample_df["topic_id"] = cluster_labels
    sample_df["topic_label"] = [topic_names[c] for c in cluster_labels]

    return sample_df[[
        "tweet_id", "x", "y", "topic_id", "topic_label",
        "category", "urgency_level", "urgency_score", "event",
        "clean_text", "raw_unescaped_text"
    ]]
