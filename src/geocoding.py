"""
Offline geospatial resolution, disambiguation, and NER validation module for CrisisLens.

Resolution order for a place name:
1. Curated places (small towns, islands, counties the GeoNames city cache does not contain)
2. Macro-regions ("Caribbean", "East Coast") which are recognised but never plotted
3. US states, then countries, so "Mexico" is the country and not a town in the Philippines
4. Cities, disambiguated by the disaster event's expected countries and by population
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import geonamescache

from src.config import (
    COUNTRY_CENTROIDS,
    CURATED_PLACES,
    EVENT_GEO_PRIORS,
    GAZETTEER_LOCATIONS,
    MACRO_REGIONS,
    MIN_GEOCODE_CONFIDENCE,
    NER_JUNK_WORDS,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)

# Lazy singleton for GeoNamesCache
_GC = None
_CITY_INDEX = None
_COUNTRY_INDEX = None
_US_STATE_INDEX = None
_STATE_CENTROIDS: Dict[str, Tuple[float, float]] = {}
_CACHE: Dict[Tuple[str, Optional[str]], Optional[Dict[str, Any]]] = {}

# Names that describe the disaster itself ("Hurricane Harvey", "Puerto Rico Hurricane") are not places.
_EVENT_WORD_PATTERN = re.compile(
    r"\b(hurricanes?|cyclones?|typhoons?|tropical\s+storms?|earthquakes?|floods?|flooding|wildfires?|tsunamis?|storms?)\b",
    re.IGNORECASE,
)

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

        # Population-weighted centroid of each US state's cities (data-driven, no hand-typed coordinates)
        sums: Dict[str, List[float]] = {}
        for c in cities.values():
            if c.get("countrycode") != "US":
                continue
            code = c.get("admin1code")
            pop = float(c.get("population", 0) or 0)
            if not code or pop <= 0:
                continue
            acc = sums.setdefault(code, [0.0, 0.0, 0.0])
            acc[0] += float(c["latitude"]) * pop
            acc[1] += float(c["longitude"]) * pop
            acc[2] += pop
        for code, (lat_sum, lon_sum, pop_sum) in sums.items():
            _STATE_CENTROIDS[code] = (lat_sum / pop_sum, lon_sum / pop_sum)

    return _GC, _CITY_INDEX, _COUNTRY_INDEX, _US_STATE_INDEX

# Geographic centres for US states and territories that are better than a population-weighted mean
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
    # Strip a leading article and a trailing possessive: "the South Island" -> "South Island", "Puerto Rico's" -> "Puerto Rico"
    cleaned = re.sub(r"^the\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[’']s$", "", cleaned)
    return cleaned.strip()

def is_junk_entity(name: str) -> bool:
    """Check if extracted entity is noise/stopword or a disaster/storm name rather than a place."""
    norm = clean_location_name(name).lower()
    if not norm or len(norm) <= 2:
        return True
    if norm in NER_JUNK_WORDS:
        return True
    if _EVENT_WORD_PATTERN.search(norm):
        return True
    return False

def _name_variants(clean_name: str) -> List[str]:
    """Candidate spellings to try: the name itself, then forms with lost possessives or a 'city' suffix removed."""
    variants = [clean_name]
    low = clean_name.lower()
    if len(low) > 4 and low.endswith("s"):
        variants.append(clean_name[:-1])  # "Puerto Ricos" (possessive apostrophe lost) -> "Puerto Rico"
    if low.endswith(" city") and len(low) > 8:
        variants.append(clean_name[:-5])  # "Beira city" -> "Beira"
    return variants

def _state_centroid(state_name: str, state_code: str) -> Tuple[float, float]:
    if state_name in _STATE_COORDS:
        return _STATE_COORDS[state_name]
    return _STATE_CENTROIDS.get(state_code, COUNTRY_CENTROIDS["US"])

def _country_centroid(c_data: Dict[str, Any], city_idx: Dict[str, List[Dict[str, Any]]]) -> Optional[Tuple[float, float]]:
    """Country position: curated centroid, else the capital city from GeoNames, else None (never (0, 0))."""
    iso = c_data["iso"]
    if iso in COUNTRY_CENTROIDS:
        return COUNTRY_CENTROIDS[iso]
    capital = str(c_data.get("capital", "")).lower().strip()
    for city in city_idx.get(capital, []):
        if city.get("countrycode") == iso:
            return float(city["latitude"]), float(city["longitude"])
    return None

def _resolve_exact(clean_name: str, event: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolve one spelling of a place name (no variants)."""
    _, city_idx, country_idx, state_idx = get_geonames_cache()
    norm_lower = clean_name.lower()

    priors = EVENT_GEO_PRIORS.get(event, {}) if event else {}
    prior_countries = [c.upper() for c in priors.get("countries", [])]

    # 1. Curated places
    curated = CURATED_PLACES.get(norm_lower)
    if curated and (not curated.get("events") or event in curated["events"]):
        return {
            "name": curated["name"],
            "lat": float(curated["lat"]),
            "lon": float(curated["lon"]),
            "country_code": curated["country_code"],
            "population": 0,
            "confidence": 0.85,
            "is_ambiguous": False,
            "entity_type": curated.get("entity_type", "place")
        }

    # 2. Macro-regions are recognised but cannot be plotted
    if norm_lower in MACRO_REGIONS:
        return None

    # 3a. US states
    if norm_lower in state_idx:
        s_data = state_idx[norm_lower]
        s_name = s_data["name"]
        lat, lon = _state_centroid(s_name, s_data["code"])
        confidence = 0.90 if "US" in prior_countries or not prior_countries else 0.70
        return {
            "name": s_name,
            "lat": lat,
            "lon": lon,
            "country_code": "US",
            "population": 5000000,
            "confidence": confidence,
            "is_ambiguous": False,
            "entity_type": "state"
        }

    # 3b. Countries (so "Mexico" is the country, not a Philippine town of the same name)
    if norm_lower in country_idx and len(norm_lower) > 2:
        c_data = country_idx[norm_lower]
        centroid = _country_centroid(c_data, city_idx)
        if centroid is not None:
            c_code = c_data["iso"]
            is_prior = c_code.upper() in prior_countries
            return {
                "name": c_data["name"],
                "lat": centroid[0],
                "lon": centroid[1],
                "country_code": c_code,
                "population": int(c_data.get("population", 1000000)),
                "confidence": 0.85 if is_prior else 0.70,
                "is_ambiguous": False,
                "entity_type": "country"
            }

    # 4. Cities
    if norm_lower in city_idx:
        candidates = city_idx[norm_lower]
        is_ambiguous = len(candidates) > 1

        # Score candidates by country prior + population
        def score_candidate(city):
            pop = city.get("population", 0)
            in_prior = 1 if city.get("countrycode") in prior_countries else 0
            return (in_prior, pop)

        best = max(candidates, key=score_candidate)
        in_prior = best.get("countrycode") in prior_countries
        if in_prior:
            confidence = 0.95
        elif prior_countries:
            # The event is in known countries and the only match is elsewhere: probably the wrong place.
            confidence = 0.40
        else:
            confidence = 0.65 if is_ambiguous else 0.80

        return {
            "name": best["name"],
            "lat": float(best["latitude"]),
            "lon": float(best["longitude"]),
            "country_code": best["countrycode"],
            "population": best.get("population", 0),
            "confidence": confidence,
            "is_ambiguous": is_ambiguous,
            "entity_type": "city"
        }

    return None

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

    result = None
    for variant in _name_variants(clean_name):
        result = _resolve_exact(variant, event)
        if result is not None:
            break

    _CACHE[cache_key] = result
    return result

def filter_and_validate_locations(
    locations: List[str],
    event: Optional[str] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Validate list of extracted locations:
    - Removes junk false positives (e.g., 'Truly', 'Today', storm names)
    - Resolves place names to geocoded coordinates
    - Keeps the name but drops coordinates for low-confidence matches
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
        if resolved is not None and resolved["confidence"] < MIN_GEOCODE_CONFIDENCE:
            resolved = None

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

def generate_geocoding_report(df: pd.DataFrame, reports_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Generate summary report of geocoding coverage and entity cleaning."""
    total_raw_entities = 0
    resolved_entities = 0
    unique_places = set()
    unique_geocoded = set()

    for raw_locs, geo_locs in zip(
        df["locations"] if "locations" in df.columns else [],
        df["geocoded_locations"] if "geocoded_locations" in df.columns else [],
    ):
        if isinstance(raw_locs, (list, tuple)):
            total_raw_entities += len(raw_locs)
            for loc in raw_locs:
                unique_places.add(str(loc).strip())

        if isinstance(geo_locs, (list, tuple)):
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

    out_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "geocoding_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Geocoding report written to {report_path}")
    return report

# Convenient alias
geocode_tweet_locations = filter_and_validate_locations
