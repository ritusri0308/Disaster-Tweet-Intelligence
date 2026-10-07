"""
Unit tests for NER module.
"""

import pytest
from src.ner import extract_locations_single, extract_locations_batch

def test_extract_locations_known_sentence():
    text = "Hurricane Harvey flooded Houston and parts of Texas severely."
    locs = extract_locations_single(text)
    
    assert any("Houston" in loc for loc in locs)
    assert any("Texas" in loc for loc in locs)

def test_extract_locations_empty():
    assert extract_locations_single("") == []
    assert extract_locations_single(None) == []

def test_extract_locations_batch():
    texts = [
        "Severe earthquake reported near Amatrice in Italy.",
        "URGENT! People trapped near Chennai Central station.",
        ""
    ]
    batch_locs = extract_locations_batch(texts)
    assert len(batch_locs) == 3
    assert any("Amatrice" in l or "Italy" in l for l in batch_locs[0])
    assert any("Chennai" in l for l in batch_locs[1])
    assert batch_locs[2] == []
