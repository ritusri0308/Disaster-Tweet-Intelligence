"""
Offline geospatial resolution, disambiguation, and NER validation module for CrisisLens.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import geonamescache

from src.config import (
    EVENT_GEO_PRIORS,
    NER_JUNK_WORDS,
    GAZETTEER_LOCATIONS,
    REPORTS_DIR
)

logger = logging.getLogger(__name__)

# Lazy singleton for GeoNamesCache
_GC = None
_CITY_INDEX = None
_COUNTRY_INDEX = None
_US_STATE_INDEX = None
_CACHE: Dict[Tuple[str, Optional[str]], Optional[Dict[str, Any]]] = {}

def get_geonames_cache():
    """Lazy initialize geonamescache and lookup indexes."""
    global _GC, _CITY_INDEX, _COUNTRY_INDEX, _US_STATE_INDEX
    if _GC is None:
        _GC = geonamescache.GeonamesCache()

        # Build lowercase city name -> list of city objects
        cities = _GC.get_cities()
        _CITY_INDEX = {}
        for c in cities.values():
            c_name = c["name"].lower().strip()
            if c_name not in _CITY_INDEX:
                _CITY_INDEX[c_name] = []
            _CITY_INDEX[c_name].append(c)

        # Build country index
        countries = _GC.get_countries()
        _COUNTRY_INDEX = {}
        for c_code, c_data in countries.items():
            _COUNTRY_INDEX[c_data["name"].lower().strip()] = c_data
            _COUNTRY_INDEX[c_data["iso"].lower().strip()] = c_data

        # Build US states index
        us_states = _GC.get_us_states()
        _US_STATE_INDEX = {}
        for s_code, s_data in us_states.items():
            _US_STATE_INDEX[s_data["name"].lower().strip()] = {
                "name": s_data["name"],
                "countrycode": "US",
                "code": s_code
            }
            _US_STATE_INDEX[s_code.lower().strip()] = {
                "name": s_data["name"],
                "countrycode": "US",
                "code": s_code
            }

    return _GC, _CITY_INDEX, _COUNTRY_INDEX, _US_STATE_INDEX

# Coordinates for US states & common territories (centroids)
_STATE_COORDS = {
    "Texas": (31.9686, -99.9018),
    "Florida": (27.6648, -81.5158),
    "Puerto Rico": (18.2208, -66.5901),
    "Maryland": (39.0458, -76.6413),
    "Louisiana": (30.9843, -91.9623),
    "Georgia": (32.1656, -82.9001),
    "North Carolina": (35.7596, -79.0193),
    "South Carolina": (33.8361, -81.1637),
    "California": (36.7783, -119.4179),
    "Alberta": (53.9333, -116.5765)
}

def clean_location_name(name: str) -> str:
    """Normalize place name and remove noise/affixes."""
    if not name:
        return ""
    # Strip leading/trailing punctuation and quotes
    cleaned = re.sub(r"^[\s\"'.,:;()!?-]+|[\s\"'.,:;()!?-]+$", "", str(name))
    # Strip state abbreviations at end like "Houston, TX" -> "Houston"
    cleaned = re.sub(r",\s*[A-Za-z]{2}$", "", cleaned)
    return cleaned.strip()

def is_junk_entity(name: str) -> bool:
    """Check if extracted entity is noise/stopword."""
    norm = clean_location_name(name).lower()
    if not norm or len(norm) <= 2:
        return True
    if norm in NER_JUNK_WORDS:
        return True
    return False

def resolve_location(
    place_name: str,
    event: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Resolve place name string to latitude/longitude using offline GeoNamesCache.
    Disambiguates using population and event-specific country/region priors.
    """
    clean_name = clean_location_name(place_name)
    if not clean_name or is_junk_entity(clean_name):
        return None

    cache_key = (clean_name.lower(), event)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    _, city_idx, country_idx, state_idx = get_geonames_cache()
    norm_lower = clean_name.lower()

    # Prior countries from event
    priors = EVENT_GEO_PRIORS.get(event, {}) if event else {}
    prior_countries = [c.upper() for c in priors.get("countries", [])]

    # 1. Check City Match
    if norm_lower in city_idx:
        candidates = city_idx[norm_lower]
        is_ambiguous = len(candidates) > 1

        # Score candidates by country prior + population
        def score_candidate(city):
            pop = city.get("population", 0)
            in_prior = 1 if city.get("countrycode") in prior_countries else 0
            return (in_prior, pop)

        best = max(candidates, key=score_candidate)
        confidence = 0.95 if best.get("countrycode") in prior_countries else (0.80 if not is_ambiguous else 0.65)

        res = {
            "name": best["name"],
            "lat": float(best["latitude"]),
            "lon": float(best["longitude"]),
            "country_code": best["countrycode"],
            "population": best.get("population", 0),
            "confidence": confidence,
            "is_ambiguous": is_ambiguous,
            "entity_type": "city"
        }
        _CACHE[cache_key] = res
        return res

    # 2. Check US State / Region Match
    if norm_lower in state_idx:
        s_data = state_idx[norm_lower]
        s_name = s_data["name"]
        coords = _STATE_COORDS.get(s_name, (37.0902, -95.7129))
        confidence = 0.90 if "US" in prior_countries or not prior_countries else 0.70

        res = {
            "name": s_name,
            "lat": coords[0],
            "lon": coords[1],
            "country_code": "US",
            "population": 5000000,
            "confidence": confidence,
            "is_ambiguous": False,
            "entity_type": "state"
        }
        _CACHE[cache_key] = res
        return res

    # 3. Check Country Match
    if norm_lower in country_idx:
        c_data = country_idx[norm_lower]
        # Approximate country center
        c_code = c_data["iso"]
        is_prior = c_code.upper() in prior_countries
        confidence = 0.85 if is_prior else 0.70

        # Known country centroids fallback
        country_coords = {
            "US": (37.0902, -95.7129), "NZ": (-40.9006, 174.8860), "MZ": (-18.6657, 35.5296),
            "GR": (39.0742, 21.8243), "IT": (41.8719, 12.5674), "EC": (-1.8312, -78.1834),
            "MX": (23.6345, -102.5528), "LK": (7.8731, 80.7718), "CA": (56.1304, -106.3468),
            "ZW": (-19.0154, 29.1549), "MW": (-13.2543, 34.3015), "HT": (18.9712, -72.2852)
        }
        lat, lon = country_coords.get(c_code, (0.0, 0.0))

        res = {
            "name": c_data["name"],
            "lat": lat,
            "lon": lon,
            "country_code": c_code,
            "population": int(c_data.get("population", 1000000)),
            "confidence": confidence,
            "is_ambiguous": False,
            "entity_type": "country"
        }
        _CACHE[cache_key] = res
        return res

    _CACHE[cache_key] = None
    return None

def filter_and_validate_locations(
    locations: List[str],
    event: Optional[str] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Validate list of extracted locations:
    - Removes junk false positives (e.g., 'Truly', 'Today')
    - Resolves place names to geocoded coordinates
    - Deduplicates aliases ('Houston, TX' vs 'Houston')
    Returns: (cleaned_place_names, geocoded_records)
    """
    if not locations:
        return [], []

    seen_names = set()
    cleaned_places = []
    geocoded_records = []

    for loc in locations:
        c_name = clean_location_name(loc)
        if not c_name or is_junk_entity(c_name):
            continue

        resolved = resolve_location(c_name, event=event)

        # If not resolved and not in known gazetteer, drop if single common word
        if resolved is None and c_name not in GAZETTEER_LOCATIONS:
            if " " not in c_name:
                continue

        canonical = resolved["name"] if resolved else c_name
        if canonical.lower() not in seen_names:
            seen_names.add(canonical.lower())
            cleaned_places.append(canonical)
            if resolved:
                geocoded_records.append(resolved)

    return cleaned_places, geocoded_records

def generate_geocoding_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary report of geocoding coverage and entity cleaning."""
    total_raw_entities = 0
    resolved_entities = 0
    unique_places = set()
    unique_geocoded = set()

    for _, row in df.iterrows():
        raw_locs = row.get("locations", [])
        if isinstance(raw_locs, list):
            total_raw_entities += len(raw_locs)
            for loc in raw_locs:
                unique_places.add(str(loc).strip())

        geo_locs = row.get("geocoded_locations", [])
        if isinstance(geo_locs, list):
            resolved_entities += len(geo_locs)
            for g in geo_locs:
                if isinstance(g, dict) and "name" in g:
                    unique_geocoded.add(g["name"])

    report = {
        "total_tweets": len(df),
        "total_extracted_entity_mentions": total_raw_entities,
        "total_geocoded_mentions": resolved_entities,
        "mention_geocoding_rate_pct": round((resolved_entities / max(total_raw_entities, 1)) * 100, 2),
        "unique_raw_places_detected": len(unique_places),
        "unique_places_geocoded": len(unique_geocoded),
        "place_resolution_rate_pct": round((len(unique_geocoded) / max(len(unique_places), 1)) * 100, 2),
        "junk_entities_filtered_list": sorted(list(NER_JUNK_WORDS))[:15]
    }

    report_path = REPORTS_DIR / "geocoding_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Geocoding report written to {report_path}")
    return report

# Convenient alias
geocode_tweet_locations = filter_and_validate_locations
