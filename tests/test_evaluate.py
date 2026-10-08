"""
Unit tests for evaluate.py calibration and metrics computation.
"""

import numpy as np
from evaluate import compute_expected_calibration_error

def test_compute_expected_calibration_error_perfect():
    # If confidence matches actual frequency perfectly
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    probas = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1])
    ece = compute_expected_calibration_error(y_true, probas, n_bins=10)
    assert 0.0 <= ece <= 0.2

def test_compute_expected_calibration_error_poor():
    # Overconfident wrong predictions
    y_true = np.zeros(10, dtype=int)
    probas = np.ones(10) * 0.95
    ece = compute_expected_calibration_error(y_true, probas, n_bins=10)
    assert ece > 0.8
