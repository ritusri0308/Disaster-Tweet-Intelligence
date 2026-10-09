"""
Regression tests for geocoding accuracy.

Each case below was a measured failure of the previous resolver, which checked the city index before
countries and states and fell back to coordinates (0, 0) for countries it had no centroid for.
"""

import pytest

from src.geocoding import (
    clean_location_name,
    filter_and_validate_locations,
    is_junk_entity,
    resolve_location,
)

def test_country_names_resolve_to_the_country_not_a_same_named_town():
    # 1,948 "Mexico" mentions used to land on a town in the Philippines, 2,493 "Puerto Rico" ones in Colombia.
    mexico = resolve_location("Mexico", event="puebla_mexico_earthquake_2017")
    assert (mexico["country_code"], mexico["entity_type"]) == ("MX", "country")
    puerto_rico = resolve_location("Puerto Rico", event="hurricane_maria_2017")
    assert (puerto_rico["country_code"], puerto_rico["entity_type"]) == ("PR", "country")
    assert 17.5 < puerto_rico["lat"] < 19.0 and -67.5 < puerto_rico["lon"] < -65.0

def test_us_states_resolve_to_the_state():
    florida = resolve_location("Florida", event="hurricane_irma_2017")
    assert (florida["country_code"], florida["entity_type"]) == ("US", "state")
    virginia = resolve_location("Virginia")
    assert virginia["country_code"] == "US" and virginia["entity_type"] == "state"
    assert 36.0 < virginia["lat"] < 39.5  # not the middle of the USA

def test_no_place_is_ever_plotted_at_null_island():
    for name, event in [("Dominica", "hurricane_maria_2017"), ("Cuba", "hurricane_irma_2017"), ("Japan", None),
                        ("India", None), ("Bahamas", "hurricane_matthew_2016"), ("Dominican Republic", None)]:
        res = resolve_location(name, event=event)
        assert res is not None, name
        assert abs(res["lat"]) > 0.5 or abs(res["lon"]) > 0.5, f"{name} resolved to (0, 0)"

def test_storm_names_and_event_words_are_not_places():
    for name in ["Harvey", "Irma", "Maria", "Hurricane Harvey", "Hurricanes Harvey", "Puerto Rico Hurricane Maria",
                 "Typhoon", "the earthquake"]:
        assert is_junk_entity(name), name
        assert resolve_location(name, event="hurricane_harvey_2017") is None, name

def test_possessives_articles_and_suffixes_are_normalised():
    assert clean_location_name("Puerto Rico’s") == "Puerto Rico"
    assert clean_location_name("the South Island") == "South Island"
    assert resolve_location("Puerto Ricos", event="hurricane_maria_2017")["name"] == "Puerto Rico"   # apostrophe lost upstream
    assert resolve_location("New Zealands", event="kaikoura_earthquake_2016")["name"] == "New Zealand"
    assert resolve_location("Beira city", event="cyclone_idai_2019")["name"] == "Beira"

def test_curated_places_fill_gaps_in_the_city_cache():
    for name, event, country in [("Alberta", "canada_wildfires_2016", "CA"), ("Kaikoura", "kaikoura_earthquake_2016", "NZ"),
                                 ("Amatrice", "italy_earthquake_aug_2016", "IT"), ("Barbuda", "hurricane_irma_2017", "AG"),
                                 ("Virgin Islands", "hurricane_irma_2017", "VI"), ("St. Croix", "hurricane_maria_2017", "VI")]:
        res = resolve_location(name, event=event)
        assert res is not None and res["country_code"] == country, name

def test_event_restricted_curated_entries_do_not_leak_to_other_events():
    assert resolve_location("Harris County", event="hurricane_harvey_2017")["country_code"] == "US"
    assert resolve_location("Harris County", event="maryland_floods_2018") is None
    assert resolve_location("Mati", event="greece_wildfires_2018")["country_code"] == "GR"

def test_macro_regions_are_recognised_but_never_plotted():
    for name in ["Caribbean", "East Coast", "Southern Africa", "South Asia"]:
        assert resolve_location(name, event="hurricane_irma_2017") is None

def test_city_matches_outside_the_events_countries_keep_the_name_but_are_not_plotted():
    # "Palm Beach" only matches a town in Australia; for a Florida hurricane that is almost certainly wrong.
    res = resolve_location("Palm Beach", event="hurricane_irma_2017")
    assert res is not None and res["confidence"] < 0.5
    names, geocoded = filter_and_validate_locations(["Palm Beach", "Houston"], event="hurricane_harvey_2017")
    assert "Palm Beach" in names
    assert [g["name"] for g in geocoded] == ["Houston"]

def test_in_prior_cities_still_resolve_with_high_confidence():
    houston = resolve_location("Houston", event="hurricane_harvey_2017")
    assert houston["entity_type"] == "city" and houston["confidence"] >= 0.9
    athens = resolve_location("Athens", event="greece_wildfires_2018")
    assert athens["country_code"] == "GR"

def test_filter_and_validate_dedupes_aliases_and_drops_junk():
    names, geocoded = filter_and_validate_locations(
        ["Puerto Rico", "Puerto Ricos", "Hurricane Maria", "Metro", "Today"], event="hurricane_maria_2017")
    assert names == ["Puerto Rico"]
    assert len(geocoded) == 1

@pytest.mark.parametrize("name", ["", None, "  ", "xyz123nonexistentplace"])
def test_unresolvable_input_returns_none(name):
    assert resolve_location(name) is None
