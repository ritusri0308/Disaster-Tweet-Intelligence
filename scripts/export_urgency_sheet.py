"""
Script to export urgency annotation sheet for hand-labeling and evaluate urgency rules.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPORTS_DIR
from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe
from src.urgency import calculate_urgency

def export_urgency_sheet(n_samples: int = 100) -> Path:
    """Export ~100 stratified random tweets for urgency hand annotation."""
    df = load_data()
    df = preprocess_dataframe(df)

    # Sample stratified across project categories
    samples = []
    per_cat = max(1, n_samples // len(df["category"].unique()))

    for cat, group in df.groupby("category"):
        sampled_group = group.sample(n=min(per_cat, len(group)), random_state=42)
        samples.append(sampled_group)

    sample_df = pd.concat(samples, ignore_index=True)

    rows = []
    for _, row in sample_df.iterrows():
        text = row["raw_unescaped_text"]
        cat = row["category"]
        score, level, terms = calculate_urgency(
            text=text,
            category=cat,
            uppercase_ratio=row.get("uppercase_ratio", 0.0),
            exclamation_count=row.get("exclamation_count", 0),
            is_retweet=row.get("is_retweet", 0)
        )
        rows.append({
            "tweet_id": row["tweet_id"],
            "category": cat,
            "tweet_text": text,
            "rule_urgency_score": score,
            "rule_urgency_level": level,
            "matched_terms": "|".join(terms),
            "human_urgency": ""  # Left empty for hand annotation
        })

    output_path = REPORTS_DIR / "urgency_annotation_sheet.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"Exported {len(out_df)} tweets to urgency annotation sheet at {output_path}")
    return output_path

if __name__ == "__main__":
    sheet_path = export_urgency_sheet()
