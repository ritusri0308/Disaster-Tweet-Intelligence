"""
Unit tests for src/nlquery.py.
Tests >= 15 natural language query phrasings and failure handling.
"""

from src.nlquery import parse_natural_language_query

def test_nlquery_empty_and_whitespace():
    res1 = parse_natural_language_query("")
    assert res1["is_valid"] is False
    assert len(res1["parsed_chips"]) == 0

    res2 = parse_natural_language_query("    ")
    assert res2["is_valid"] is False

    res3 = parse_natural_language_query("random gibberish asdfghjk")
    assert res3["is_valid"] is False

def test_nlquery_15_phrasings():
    # 1. Critical medical needs
    q1 = parse_natural_language_query("critical medical needs in Houston")
    assert q1["is_valid"]
    assert q1["filters"].get("urgency_level") == "CRITICAL"
    assert q1["filters"].get("need_type") == "Medical"
    assert q1["filters"].get("location") == "Houston"

    # 2. Rescue requests in Puerto Rico
    q2 = parse_natural_language_query("urgent rescue requests in Puerto Rico")
    assert q2["is_valid"]
    assert q2["filters"].get("urgency_level") == "HIGH"
    assert q2["filters"].get("category") == "Help Request"
    assert q2["filters"].get("location") == "Puerto Rico"

    # 3. Water supplies needed
    q3 = parse_natural_language_query("drinking water needed in Colombo")
    assert q3["is_valid"]
    assert q3["filters"].get("need_type") == "Water"
    assert q3["filters"].get("location") == "Colombo"

    # 4. Food shortages in Greece
    q4 = parse_natural_language_query("food rations needed in Greece wildfires")
    assert q4["is_valid"]
    assert q4["filters"].get("need_type") == "Food"
    assert q4["filters"].get("event") == "greece_wildfires_2018"

    # 5. Infrastructure and damage reports
    q5 = parse_natural_language_query("bridge collapsed damage report in Nepal or Italy")
    assert q5["is_valid"]
    assert q5["filters"].get("category") == "Damage Report"
    assert q5["filters"].get("location") == "Italy"

    # 6. Power outage in Florida
    q6 = parse_natural_language_query("power outage and blackout in Florida")
    assert q6["is_valid"]
    assert q6["filters"].get("need_type") == "Power/Utilities"
    assert q6["filters"].get("location") == "Florida"

    # 7. Hurricane Harvey evacuation
    q7 = parse_natural_language_query("hurricane harvey evacuation updates")
    assert q7["is_valid"]
    assert q7["filters"].get("event") == "hurricane_harvey_2017"
    assert q7["filters"].get("category") == "Information/Update"

    # 8. Cyclone Idai emergency
    q8 = parse_natural_language_query("cyclone idai casualty and emergency")
    assert q8["is_valid"]
    assert q8["filters"].get("event") == "cyclone_idai_2019"
    assert q8["filters"].get("category") == "Emergency"

    # 9. Low urgency general sympathy
    q9 = parse_natural_language_query("low urgency condolences and prayers")
    assert q9["is_valid"]
    assert q9["filters"].get("urgency_level") == "LOW"
    assert q9["filters"].get("category") == "General"

    # 10. Missing persons search
    q10 = parse_natural_language_query("missing person search in Fort McMurray")
    assert q10["is_valid"]
    assert q10["filters"].get("need_type") == "Missing persons"
    assert q10["filters"].get("location") == "Fort McMurray"

    # 11. Shelter and tents needed in Ecuador
    q11 = parse_natural_language_query("tents and temporary housing in ecuador earthquake")
    assert q11["is_valid"]
    assert q11["filters"].get("need_type") == "Shelter"
    assert q11["filters"].get("event") == "ecuador_earthquake_2016"

    # 12. Road blockage in Maryland
    q12 = parse_natural_language_query("road blocked in ellicott city maryland")
    assert q12["is_valid"]
    assert q12["filters"].get("need_type") == "Transport/Roads"
    assert q12["filters"].get("location") == "Ellicott City"

    # 13. Donations and volunteers in Mexico
    q13 = parse_natural_language_query("volunteer drive and donations in mexico earthquake")
    assert q13["is_valid"]
    assert q13["filters"].get("need_type") == "Donations/Volunteering"
    assert q13["filters"].get("event") == "puebla_mexico_earthquake_2017"

    # 14. Medium urgency advisory
    q14 = parse_natural_language_query("medium urgency advisory in Kaikoura")
    assert q14["is_valid"]
    assert q14["filters"].get("urgency_level") == "MEDIUM"
    assert q14["filters"].get("location") == "Kaikoura"

    # 15. Last 24 hours window filter
    q15 = parse_natural_language_query("emergency help in Houston in last 24 hours")
    assert q15["is_valid"]
    assert q15["filters"].get("location") == "Houston"
    assert q15["filters"].get("date_range") == "last_24h"

    # 16. Date after Sep 20
    q16 = parse_natural_language_query("hurricane maria urgent requests after sep 20")
    assert q16["is_valid"]
    assert q16["filters"].get("event") == "hurricane_maria_2017"
    assert q16["filters"].get("date_after") == "2017-09-20"
