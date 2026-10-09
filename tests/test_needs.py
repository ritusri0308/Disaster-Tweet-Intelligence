"""
Unit tests for src/needs.py.
"""

import pandas as pd
from src.needs import extract_needs, extract_needs_batch, evaluate_needs_against_raw_labels

def test_extract_needs_empty():
    assert extract_needs("") == []
    assert extract_needs(None) == []
    assert extract_needs("   ") == []

def test_extract_needs_medical():
    text = "We urgently need insulin, bandages, and a doctor at the clinic!"
    needs = extract_needs(text)
    assert "Medical" in needs

def test_extract_needs_water_food():
    text = "People here are starving, no food and drinking water for 3 days."
    needs = extract_needs(text)
    assert "Food" in needs
    assert "Water" in needs

def test_extract_needs_shelter_rescue():
    text = "Family trapped on roof, please send boat or helicopter rescue, need blankets and temporary housing."
    needs = extract_needs(text)
    assert "Rescue/Evacuation" in needs
    assert "Shelter" in needs

def test_extract_needs_power_transport():
    text = "Power outage across the district, transformer blew up and road blocked by downed wire."
    needs = extract_needs(text)
    assert "Power/Utilities" in needs
    assert "Transport/Roads" in needs

def test_extract_needs_missing_persons_donations():
    text = "Missing person alert: haven't heard from uncle since storm. Please donate to relief fund."
    needs = extract_needs(text)
    assert "Missing persons" in needs
    assert "Donations/Volunteering" in needs

def test_extract_needs_negative_case():
    text = "Good morning everyone, having coffee and reading a book."
    assert extract_needs(text) == []

def test_extract_needs_batch():
    texts = [
        "Clean drinking water needed",
        "Just a nice sunset",
        "Trapped in building need rescue"
    ]
    batch = extract_needs_batch(texts)
    assert len(batch) == 3
    assert "Water" in batch[0]
    assert batch[1] == []
    assert "Rescue/Evacuation" in batch[2]

def test_evaluate_needs_against_raw_labels(tmp_path):
    df = pd.DataFrame([
        {
            "raw_label": "requests_or_urgent_needs",
            "needs": ["Food", "Water"]
        },
        {
            "raw_label": "requests_or_urgent_needs",
            "needs": []
        },
        {
            "raw_label": "injured_or_dead_people",
            "needs": ["Medical"]
        }
    ])
    eval_res = evaluate_needs_against_raw_labels(df, reports_dir=tmp_path)
    assert "urgent_needs_survival_recall" in eval_res
    assert eval_res["urgent_needs_survival_recall"] == 0.5
    assert (tmp_path / "needs_evaluation.json").exists()
    assert eval_res["tweets_without_any_need_pct"] == round(100.0 / 3, 2)

def test_extract_needs_expanded_lexicon():
    assert "Transport/Roads" in extract_needs("Roads closed and a bridge washed out near the river")
    assert "Power/Utilities" in extract_needs("Millions are without power after the storm")
    assert "Communication" in extract_needs("Internet down and no cell service since last night")
    assert "Missing persons" in extract_needs("Two people still missing after the flash flood")
    assert "Water" in extract_needs("No water in the neighbourhood for three days")
    # typographic apostrophe is normalised
    assert "Missing persons" in extract_needs("We can’t find my brother anywhere")

def test_extract_needs_does_not_fire_on_neutral_text():
    assert extract_needs("Thoughts and prayers for everyone watching the storm coverage") == []
