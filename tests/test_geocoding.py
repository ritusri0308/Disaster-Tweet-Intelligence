"""
Unit tests for src/geocoding.py.
"""

from src.geocoding import (
    clean_location_name,
    is_junk_entity,
    resolve_location,
    geocode_tweet_locations
)

def test_clean_location_name():
    assert clean_location_name("  \"Houston, TX\"  ") == "Houston"
    assert clean_location_name("(Miami)") == "Miami"
    assert clean_location_name("") == ""

def test_is_junk_entity():
    assert is_junk_entity("Truly") is True
    assert is_junk_entity("update") is True
    assert is_junk_entity("today") is True
    assert is_junk_entity("a") is True
    assert is_junk_entity("Houston") is False
    assert is_junk_entity("Athens") is False

def test_resolve_known_city():
    res = resolve_location("Houston", event="hurricane_harvey_2017")
    assert res is not None
    assert res.get("country_code") == "US"
    assert abs(res["lat"] - 29.76) < 1.0
    assert abs(res["lon"] - -95.36) < 1.0

def test_resolve_event_prior_disambiguation():
    # Athens without event vs with greece_wildfires_2018
    res_gr = resolve_location("Athens", event="greece_wildfires_2018")
    assert res_gr is not None
    assert res_gr.get("country_code") == "GR"

def test_junk_returns_none():
    assert resolve_location("Truly") is None
    assert resolve_location("emergency") is None
    assert resolve_location("xyz123nonexistentplace") is None

def test_geocode_tweet_locations():
    locations = ["Houston", "Truly", "UnknownTownXYZ"]
    cleaned_names, geocoded = geocode_tweet_locations(locations, event="hurricane_harvey_2017")
    assert "Houston" in cleaned_names
    assert "Truly" not in cleaned_names
    assert len(geocoded) >= 1
    assert any(g["name"] == "Houston" for g in geocoded)
    assert not any(g["name"].lower() == "truly" for g in geocoded)
