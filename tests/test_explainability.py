"""
Unit tests for src/explainability.py.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.explainability import explain_classification, highlight_tweet_tokens

def test_explain_classification_empty():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ])
    pipe.fit(["emergency rescue", "general news"], ["Emergency", "General"])
    res = explain_classification(pipe, "")
    assert res["predicted_class"] == "General"
    assert res["confidence"] == 0.0

def test_explain_classification_linear():
    texts = [
        "urgent medical help emergency rescue",
        "routine city update daily news road open",
        "urgent doctor injured ambulance",
        "city transit schedule update weather clear"
    ]
    labels = ["Emergency", "General", "Emergency", "General"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ])
    pipe.fit(texts, labels)

    res = explain_classification(pipe, "urgent doctor needed in clinic")
    assert res["predicted_class"] == "Emergency"
    assert "top_positive_features" in res
    assert len(res["top_positive_features"]) > 0
    tokens = [f[0] for f in res["top_positive_features"]]
    assert "urgent" in tokens or "doctor" in tokens
    assert 0.0 <= res["confidence"] <= 1.0

def test_highlight_tweet_tokens_escaping_and_badges():
    raw_text = '<script>alert("hack")</script> Urgent rescue needed in Houston!'
    locations = ["Houston"]
    urgency_terms = ["critical:rescue"]
    top_features = [("urgent", 0.5)]

    html_out = highlight_tweet_tokens(raw_text, locations, urgency_terms, top_features)
    # Check that raw script tag is escaped
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out

    # Check span tags
    assert "Houston" in html_out
    assert "rescue" in html_out
    assert "<span" in html_out
