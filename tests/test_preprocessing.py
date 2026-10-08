"""
Unit tests for preprocessing module.
"""

import pandas as pd
from src.preprocessing import clean_tweet, extract_single_text_features, preprocess_dataframe

def test_clean_tweet_standard():
    raw = "RT @user: SOS! 5 people trapped in fire near #Houston http://example.com &amp; need help!"
    cleaned = clean_tweet(raw)

    assert "rt @" not in cleaned
    assert "http" not in cleaned
    assert "sos" in cleaned
    assert "trapped" in cleaned
    assert "fire" in cleaned
    assert "houston" in cleaned  # hashtag word kept
    assert "#" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned

def test_clean_tweet_edge_cases():
    assert clean_tweet(None) == ""
    assert clean_tweet("") == ""
    assert clean_tweet(12345) == "12345"
    assert clean_tweet("🔥 Disaster alert! 🚨") == "🔥 disaster alert! 🚨"

def test_extract_text_features():
    raw = "RT @rescue: SOS!! Need urgent medical assistance in Texas! https://t.co/xyz"
    features = extract_single_text_features(raw)

    assert features["is_retweet"] == 1
    assert features["has_url"] == 1
    assert features["exclamation_count"] == 3
    assert features["uppercase_ratio"] > 0.1
    assert features["word_count"] > 5

def test_preprocess_dataframe():
    df = pd.DataFrame({"tweet_text": ["RT @a: SOS! Need help &amp; water!"]})
    res_df = preprocess_dataframe(df)

    assert "clean_text" in res_df.columns
    assert "is_retweet" in res_df.columns
    assert "exclamation_count" in res_df.columns
    assert res_df["clean_text"].iloc[0] == "sos! need help & water!"
