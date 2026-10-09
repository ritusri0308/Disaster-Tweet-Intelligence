"""
Unit tests for src/entities.py (reported casualty / quantity figures).
"""

from src.entities import extract_quantities, format_quantity, summarize_quantities

def kinds(text):
    return {(q["kind"], q["count"]) for q in extract_quantities(text)}

def test_death_toll_phrasings():
    assert ("killed", 413) in kinds("Ecuador quake death toll hits 413: At least 413 people are now known to have died")
    assert ("killed", 1000) in kinds("Cyclone Idai may have killed 1,000 people in Mozambique")
    assert ("killed", 83) in kinds("Greece wildfires that killed 83 suspected as arson")
    assert ("killed", 10) in kinds("Hurricane Matthew kills at least 10 while battering Haiti")
    assert ("killed", 90) in kinds("over 90 deaths last week")

def test_people_at_risk_and_displaced():
    assert kinds("SOS! 5 people injured and 2 trapped in collapsed building") == {("injured", 5), ("trapped", 2)}
    assert ("displaced", 500) in kinds("Over 500 families evacuated to temporary shelter")

def test_number_words_and_structures():
    assert ("displaced", 1_200_000) in kinds("More than 1.2 million people displaced by the floods")
    assert kinds("120 homes destroyed and 40 houses flooded") == {("structures", 120), ("structures", 40)}

def test_years_units_and_plain_text_are_not_figures():
    assert extract_quantities("Hurricane Harvey 2017 killed many; landfall in 2017") == []
    assert extract_quantities("Category 4 storm with winds of 130 mph, 5 miles offshore") == []
    assert extract_quantities("Rainfall of 25 inches in 48 hours, 3 days of rain") == []
    assert extract_quantities("Killed in 2017 hurricane") == []
    assert extract_quantities("Praying for everyone affected") == []

def test_stat_lists_attribute_each_number_to_the_word_that_follows_it():
    # Measured failure: "dead 72,000" was read as 72,000 killed, and "72" as the start of "72,000".
    text = "A few #HurricaneHarvey stats: 47 people dead 72,000 rescued 136,000 structures flooded in Harris County"
    assert kinds(text) == {("killed", 47), ("rescued", 72_000), ("structures", 136_000)}

def test_historical_comparisons_are_ignored_when_the_event_year_is_known():
    # Measured failures: these were reported as figures for the current disaster.
    historical = [
        "The San Calixto hurricane of 1780 killed over 27,500 people. Irma killed 68.",
        "A hurricane in Galveston killed 10,000 Texans in 1900. It happens.",
        "04.10.1963 - Hurricane Flora, kills 6,000 in Cuba and Haiti.",
        "On this date, April 18, 1906, a major earthquake struck San Francisco. 3,000 people died.",
        "1963 - Hurricane Flora kills 6,000",
    ]
    for text in historical:
        assert extract_quantities(text, reference_year=2017) == [], text
    # Without a reference year nothing is filtered: the heuristic is opt-in
    assert extract_quantities(historical[1]) != []

def test_a_large_count_is_not_mistaken_for_a_historical_year():
    assert ("killed", 1500) in kinds("Earthquake in 2016 killed 1500 people in Ecuador")
    assert ("killed", 1500) in {(q["kind"], q["count"]) for q in extract_quantities("Maria killed 1,500 people, 2017 was a bad year", reference_year=2017)}
    assert ("killed", 413) in {(q["kind"], q["count"]) for q in extract_quantities("Death toll hits 413 in Ecuador quake", reference_year=2016)}

def test_event_year_helper():
    from src.entities import event_year
    assert event_year("hurricane_harvey_2017") == 2017
    assert event_year("italy_earthquake_aug_2016") == 2016
    assert event_year("no_year_here") is None and event_year(None) is None

def test_empty_and_none():
    assert extract_quantities(None) == []
    assert extract_quantities("   ") == []

def test_duplicates_collapsed_and_capped():
    text = "killed 5. killed 5. killed 5. " + " ".join(f"killed {n}" for n in range(10, 30))
    found = extract_quantities(text)
    assert found.count({"kind": "killed", "count": 5}) == 1
    assert len(found) <= 6

def test_summarize_and_format():
    quantities = [{"kind": "killed", "count": 83}, {"kind": "killed", "count": 413}, {"kind": "injured", "count": 12}]
    assert summarize_quantities(quantities) == {"killed": 413, "injured": 12}
    assert summarize_quantities(None) == {}
    assert format_quantity({"kind": "killed", "count": 1000}) == "1,000 killed"
    assert format_quantity({"kind": "structures", "count": 3}) == "3 structures damaged"
