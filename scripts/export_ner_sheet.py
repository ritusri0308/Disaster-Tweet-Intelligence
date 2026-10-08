"""
Script to export NER check sheet for hand-verification and evaluate NER performance.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPORTS_DIR
from src.data_loader import load_data
from src.ner import extract_locations_batch

def export_ner_sheet(n_samples: int = 50) -> Path:
    """Export ~50 random tweets with predicted locations for hand verification."""
    df = load_data()
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42).copy()

    raw_texts = sample_df["raw_unescaped_text"].tolist()
    locs_list = extract_locations_batch(raw_texts)

    rows = []
    for _, row, locs in zip(sample_df.iterrows(), sample_df.iterrows(), locs_list):
        r_data = row[1]
        rows.append({
            "tweet_id": r_data["tweet_id"],
            "tweet_text": r_data["raw_unescaped_text"],
            "predicted_locations": ", ".join(locs),
            "correct_locations": "",  # Empty column for hand verification
            "notes": ""
        })

    output_path = REPORTS_DIR / "ner_check_sheet.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"Exported {len(out_df)} tweets to NER check sheet at {output_path}")
    return output_path

if __name__ == "__main__":
    export_ner_sheet()
