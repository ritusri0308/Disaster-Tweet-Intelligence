"""
Text cleaning and feature extraction module.
"""

import html
import re
from typing import Dict, Any, Union
import pandas as pd
import numpy as np

# Regex patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+", re.IGNORECASE)
RT_PATTERN = re.compile(r"^rt\s+@\w+:?\s*", re.IGNORECASE)
HASHTAG_SYMBOL_PATTERN = re.compile(r"#(\w+)", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")

def clean_tweet(text: Any) -> str:
    """
    Clean raw tweet text for NLP classification.
    - Handles None/non-string gracefully
    - Unescapes HTML entities
    - Strips RT prefix
    - Removes URLs and @mentions
    - Strips '#' from hashtags while keeping the word
    - Lowercases text
    - Normalizes whitespace while preserving words and punctuation relevant for classification
    """
    if text is None or pd.isna(text):
        return ""
    
    text = str(text)
    if not text.strip():
        return ""

    # HTML unescape
    text = html.unescape(text)

    # Lowercase
    cleaned = text.lower()

    # Strip RT prefix
    cleaned = RT_PATTERN.sub("", cleaned)

    # Remove URLs
    cleaned = URL_PATTERN.sub("", cleaned)

    # Remove Mentions
    cleaned = MENTION_PATTERN.sub("", cleaned)

    # Replace #hashtag with hashtag word
    cleaned = HASHTAG_SYMBOL_PATTERN.sub(r"\1", cleaned)

    # Normalize whitespace
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    return cleaned

def extract_single_text_features(raw_text: Any) -> Dict[str, Any]:
    """
    Extract extra text features from raw text before lowercasing/cleaning.
    """
    if raw_text is None or pd.isna(raw_text):
        raw_text = ""
    else:
        raw_text = str(raw_text)

    raw_unescaped = html.unescape(raw_text)

    has_url = 1 if URL_PATTERN.search(raw_unescaped) else 0
    is_rt = 1 if RT_PATTERN.search(raw_unescaped.strip()) or raw_unescaped.strip().startswith("RT ") else 0
    exclamation_count = raw_unescaped.count("!")
    
    # Uppercase ratio
    letters = [c for c in raw_unescaped if c.isalpha()]
    if letters:
        uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    else:
        uppercase_ratio = 0.0

    words = raw_unescaped.split()
    word_count = len(words)
    text_length = len(raw_unescaped)

    return {
        "has_url": has_url,
        "is_retweet": is_rt,
        "exclamation_count": exclamation_count,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "word_count": word_count,
        "text_length": text_length
    }

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "raw_unescaped_text") -> pd.DataFrame:
    """
    Apply clean_tweet and extract text features on a pandas DataFrame.
    """
    df = df.copy()
    if text_col not in df.columns:
        if "tweet_text" in df.columns:
            df["raw_unescaped_text"] = df["tweet_text"].apply(lambda t: html.unescape(str(t)))
            text_col = "raw_unescaped_text"
        else:
            raise KeyError(f"Column '{text_col}' not found in DataFrame.")

    df["clean_text"] = df[text_col].apply(clean_tweet)

    # Compute text features
    features_list = df[text_col].apply(extract_single_text_features).tolist()
    features_df = pd.DataFrame(features_list, index=df.index)

    for col in features_df.columns:
        df[col] = features_df[col]

    return df
