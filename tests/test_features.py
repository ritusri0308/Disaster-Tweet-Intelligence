"""
Unit tests for features module.
"""

from src.features import get_tfidf_vectorizer

def test_get_tfidf_vectorizer():
    vec = get_tfidf_vectorizer()
    assert vec is not None
    X = vec.fit_transform(["emergency help needed", "road blocked after earthquake"])
    assert X.shape[0] == 2
    assert X.shape[1] > 0
