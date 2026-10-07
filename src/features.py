"""
Feature engineering module for text vectorization and pipeline building.
"""

from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 0.95,
    max_features: int = 15000
) -> TfidfVectorizer:
    """
    Construct a TF-IDF vectorizer.
    Using min_df=1 to support both small test fixtures and large real datasets.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True
    )
