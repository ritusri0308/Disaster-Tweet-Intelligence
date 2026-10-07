"""
Unit tests for urgency scoring engine.
"""

import pytest
from src.urgency import calculate_urgency

def test_urgency_critical():
    text = "SOS! 5 people are trapped inside a collapsed building! Bleeding and dying!"
    score, level, terms = calculate_urgency(text, category="Emergency", uppercase_ratio=0.4, exclamation_count=2)
    
    assert level == "CRITICAL"
    assert score >= 7.5
    assert 0.0 <= score <= 10.0
    assert len(terms) > 0

def test_urgency_low():
    text = "Heavy rain expected tomorrow. Keep umbrellas ready."
    score, level, terms = calculate_urgency(text, category="Information/Update")
    
    assert level == "LOW"
    assert score < 3.0

def test_urgency_donation_damping():
    text = "SOS! Please donate money to Red Cross fundraising campaign to support disaster victims!"
    score, level, terms = calculate_urgency(text, category="Information/Update", is_retweet=1)
    
    assert level != "CRITICAL"
    assert score <= 5.4
    assert any("damping" in t for t in terms)

def test_urgency_monotonicity():
    text_base = "Flood in the area."
    text_extra = "SOS! Trapped! Emergency! Flood in the area."
    
    score_base, _, _ = calculate_urgency(text_base)
    score_extra, _, _ = calculate_urgency(text_extra)
    
    assert score_extra >= score_base

def test_score_bounds():
    text_extreme = "SOS SOS SOS trapped trapped dead dying bleeding casualties collapse fire flood emergency need medical please help"
    score, level, _ = calculate_urgency(text_extreme, category="Emergency", uppercase_ratio=0.9, exclamation_count=10)
    
    assert 0.0 <= score <= 10.0
