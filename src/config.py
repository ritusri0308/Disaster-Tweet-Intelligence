"""
Configuration module for CrisisLens.
"""

import os
from pathlib import Path

# Branding & Product Metadata
APP_NAME = "CrisisLens"
APP_TAGLINE = "Crisis intelligence from social signals"
APP_VERSION = "1.0.0"
DISCLAIMER = (
    "CrisisLens is a decision-support and research system. "
    "It is not an operational emergency dispatch tool; outputs must be reviewed by a human responder."
)
DATASET_NAME = "HumAID (events_set1, 2016-2019)"
DATA_MODE_NOTICE = "HISTORICAL REPLAY (Historical benchmark data 2016-2019)"

def _env_path(name: str, default: Path) -> Path:
    """Resolve a directory override from the environment (used to isolate tests and deployments)."""
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default

# Base Paths (each data/model/report root can be redirected with CRISISLENS_* variables)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _env_path("CRISISLENS_DATA_DIR", PROJECT_ROOT / "data")
RAW_DATA_DIR = DATA_DIR / "raw" / "events_set1"
RAW_DATA_ALT_DIR = DATA_DIR / "events_set1"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "processed_tweets.csv"
PROCESSED_PARQUET_PATH = PROCESSED_DATA_DIR / "processed_tweets.parquet"
DB_PATH = PROCESSED_DATA_DIR / "disaster_tweets.db"
MODELS_DIR = _env_path("CRISISLENS_MODELS_DIR", PROJECT_ROOT / "models")
MODEL_PATH = MODELS_DIR / "best_classifier.joblib"
MODEL_BACKUP_PATH = MODELS_DIR / "best_classifier.joblib.bak"
REPORTS_DIR = _env_path("CRISISLENS_REPORTS_DIR", PROJECT_ROOT / "reports")

# A trained production model should know far more than a handful of words. Anything below this
# vocabulary size is almost certainly a test/toy fixture and is flagged as degraded.
MIN_EXPECTED_VOCAB = int(os.environ.get("CRISISLENS_MIN_VOCAB", "1000"))

# Synthetic demo data is only generated when explicitly requested (never silently).
ALLOW_SYNTHETIC_ENV = "CRISISLENS_ALLOW_SYNTHETIC"

# Ensure directories exist
for path in [PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Random Seed
RANDOM_SEED = 42

# Classification Target Config: "project_5" or "raw_10"
CLASSIFICATION_TARGET = "project_5"

# Raw label to 5 Project Categories Mapping
CATEGORY_MAPPING = {
    "injured_or_dead_people": "Emergency",
    "requests_or_urgent_needs": "Help Request",
    "infrastructure_and_utility_damage": "Damage Report",
    "caution_and_advice": "Information/Update",
    "displaced_people_and_evacuations": "Information/Update",
    "missing_or_found_people": "Information/Update",
    "other_relevant_information": "Information/Update",
    "rescue_volunteering_or_donation_effort": "Information/Update",
    "not_humanitarian": "General",
    "sympathy_and_support": "General",
    "dont_know_cant_judge": "General",
}

PROJECT_CATEGORIES = [
    "Emergency",
    "Help Request",
    "Damage Report",
    "Information/Update",
    "General"
]

RAW_LABELS = [
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "infrastructure_and_utility_damage",
    "sympathy_and_support",
    "injured_or_dead_people",
    "caution_and_advice",
    "displaced_people_and_evacuations",
    "not_humanitarian",
    "requests_or_urgent_needs",
    "missing_or_found_people"
]

# Held-out events for cross-validation / leave-one-event-out experiment
HELD_OUT_EVENTS = [
    "srilanka_floods_2017",
    "cyclone_idai_2019",
    "hurricane_harvey_2017",
    "kaikoura_earthquake_2016"
]

# Per-Event Country and Region Priors for Disambiguation
EVENT_GEO_PRIORS = {
    "hurricane_harvey_2017": {"countries": ["US"], "regions": ["Texas", "Louisiana", "Houston"]},
    "hurricane_irma_2017": {"countries": ["US", "CU", "BS", "PR", "VI", "VG", "AG", "MF", "SX", "AI", "TC", "HT", "DO"], "regions": ["Florida", "Georgia", "Puerto Rico", "Barbuda", "Virgin Islands"]},
    "hurricane_maria_2017": {"countries": ["US", "DM", "PR", "VI", "DO"], "regions": ["Puerto Rico", "Virgin Islands", "Dominica"]},
    "hurricane_matthew_2016": {"countries": ["US", "HT", "BS", "CU", "DO", "JM"], "regions": ["Florida", "Georgia", "North Carolina", "South Carolina", "Haiti"]},
    "cyclone_idai_2019": {"countries": ["MZ", "ZW", "MW"], "regions": ["Mozambique", "Zimbabwe", "Malawi", "Beira"]},
    "canada_wildfires_2016": {"countries": ["CA"], "regions": ["Alberta", "Fort McMurray"]},
    "greece_wildfires_2018": {"countries": ["GR"], "regions": ["Attica", "Athens", "Mati", "Rafina"]},
    "italy_earthquake_aug_2016": {"countries": ["IT"], "regions": ["Lazio", "Marche", "Umbria", "Amatrice", "Accumoli"]},
    "kaikoura_earthquake_2016": {"countries": ["NZ"], "regions": ["Canterbury", "Marlborough", "Kaikoura", "Wellington"]},
    "puebla_mexico_earthquake_2017": {"countries": ["MX"], "regions": ["Puebla", "Morelos", "Mexico City", "Oaxaca"]},
    "srilanka_floods_2017": {"countries": ["LK"], "regions": ["Western Province", "Southern Province", "Sabaragamuwa", "Colombo", "Galle", "Matara"]},
    "maryland_floods_2018": {"countries": ["US"], "regions": ["Maryland", "Ellicott City", "Baltimore", "Howard County"]},
    "ecuador_earthquake_2016": {"countries": ["EC"], "regions": ["Manabi", "Esmeraldas", "Pedernales", "Portoviejo", "Manta"]}
}

# Stopwords & False-Positive Entities to Filter in NER Geocoding
NER_JUNK_WORDS = {
    "truly", "today", "now", "please", "morning", "tonight", "update", "alert", "help",
    "emergency", "stay", "breaking", "news", "safe", "hope", "prayers", "god", "love",
    "everyone", "thanks", "people", "home", "road", "street", "center", "city", "county",
    "state", "disaster", "storm", "hurricane", "earthquake", "flood", "wildfire", "cyclone",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "am", "pm", "utc", "rt", "psa", "red cross",
    # Storm names are the event, not a place ("Harvey" otherwise geocodes to a town in Illinois)
    "harvey", "irma", "maria", "matthew", "idai"
}

# Continents and broad regions are mentioned in tweets but cannot be pinned to a single point on the map.
MACRO_REGIONS = {
    "asia", "africa", "europe", "america", "americas", "north america", "south america",
    "central america", "latin america", "caribbean", "caribbean islands", "middle east", "oceania",
    "southern africa", "south asia", "southeast asia", "east coast", "west coast", "gulf coast",
    "southern hemisphere", "gulf of mexico", "atlantic", "pacific"
}

# Geocoded places below this confidence keep their name but are not plotted
# (e.g. a small town that only matches outside the event's expected countries).
MIN_GEOCODE_CONFIDENCE = 0.5

# Approximate centroids for countries, used before falling back to the capital city from GeoNames.
COUNTRY_CENTROIDS = {
    "US": (37.0902, -95.7129), "NZ": (-40.9006, 174.8860), "MZ": (-18.6657, 35.5296),
    "GR": (39.0742, 21.8243), "IT": (41.8719, 12.5674), "EC": (-1.8312, -78.1834),
    "MX": (23.6345, -102.5528), "LK": (7.8731, 80.7718), "CA": (56.1304, -106.3468),
    "ZW": (-19.0154, 29.1549), "MW": (-13.2543, 34.3015), "HT": (18.9712, -72.2852)
}

# Hand-curated places that the offline GeoNames cache (cities with population >= 15,000) cannot resolve.
# Coordinates are approximate centroids. An optional "events" list restricts an entry to those disasters
# when the name is ambiguous elsewhere (e.g. "Howard County" exists in many US states).
CURATED_PLACES = {
    "alberta": {"name": "Alberta", "lat": 53.9333, "lon": -116.5765, "country_code": "CA", "entity_type": "region"},
    "virgin islands": {"name": "U.S. Virgin Islands", "lat": 18.3358, "lon": -64.8963, "country_code": "VI", "entity_type": "region"},
    "us virgin islands": {"name": "U.S. Virgin Islands", "lat": 18.3358, "lon": -64.8963, "country_code": "VI", "entity_type": "region"},
    "british virgin islands": {"name": "British Virgin Islands", "lat": 18.4207, "lon": -64.6400, "country_code": "VG", "entity_type": "region"},
    "barbuda": {"name": "Barbuda", "lat": 17.6264, "lon": -61.7706, "country_code": "AG", "entity_type": "island"},
    "amatrice": {"name": "Amatrice", "lat": 42.6283, "lon": 13.2886, "country_code": "IT", "entity_type": "town"},
    "accumoli": {"name": "Accumoli", "lat": 42.6939, "lon": 13.2467, "country_code": "IT", "entity_type": "town"},
    "kaikoura": {"name": "Kaikoura", "lat": -42.4000, "lon": 173.6800, "country_code": "NZ", "entity_type": "town"},
    "south island": {"name": "South Island", "lat": -43.7000, "lon": 171.0000, "country_code": "NZ", "entity_type": "island"},
    "north island": {"name": "North Island", "lat": -38.9000, "lon": 175.8000, "country_code": "NZ", "entity_type": "island"},
    "east cape": {"name": "East Cape", "lat": -37.6900, "lon": 178.5500, "country_code": "NZ", "entity_type": "region"},
    "st. martin": {"name": "Saint Martin", "lat": 18.0708, "lon": -63.0501, "country_code": "MF", "entity_type": "island"},
    "st martin": {"name": "Saint Martin", "lat": 18.0708, "lon": -63.0501, "country_code": "MF", "entity_type": "island"},
    "saint martin": {"name": "Saint Martin", "lat": 18.0708, "lon": -63.0501, "country_code": "MF", "entity_type": "island"},
    "st. maarten": {"name": "Sint Maarten", "lat": 18.0425, "lon": -63.0548, "country_code": "SX", "entity_type": "island"},
    "st maarten": {"name": "Sint Maarten", "lat": 18.0425, "lon": -63.0548, "country_code": "SX", "entity_type": "island"},
    "sint maarten": {"name": "Sint Maarten", "lat": 18.0425, "lon": -63.0548, "country_code": "SX", "entity_type": "island"},
    "st. croix": {"name": "St. Croix", "lat": 17.7290, "lon": -64.7340, "country_code": "VI", "entity_type": "island"},
    "st croix": {"name": "St. Croix", "lat": 17.7290, "lon": -64.7340, "country_code": "VI", "entity_type": "island"},
    "st. thomas": {"name": "St. Thomas", "lat": 18.3381, "lon": -64.8941, "country_code": "VI", "entity_type": "island"},
    "st. john": {"name": "St. John", "lat": 18.3306, "lon": -64.7410, "country_code": "VI", "entity_type": "island",
                 "events": ["hurricane_irma_2017", "hurricane_maria_2017"]},
    "key largo": {"name": "Key Largo", "lat": 25.0865, "lon": -80.4473, "country_code": "US", "entity_type": "town"},
    "florida keys": {"name": "Florida Keys", "lat": 24.8000, "lon": -81.1000, "country_code": "US", "entity_type": "region"},
    "tampa bay": {"name": "Tampa Bay", "lat": 27.7000, "lon": -82.6000, "country_code": "US", "entity_type": "region"},
    "port aransas": {"name": "Port Aransas", "lat": 27.8339, "lon": -97.0611, "country_code": "US", "entity_type": "town"},
    "harris county": {"name": "Harris County", "lat": 29.8575, "lon": -95.3919, "country_code": "US", "entity_type": "county",
                      "events": ["hurricane_harvey_2017"]},
    "fort bend county": {"name": "Fort Bend County", "lat": 29.5300, "lon": -95.7700, "country_code": "US", "entity_type": "county",
                         "events": ["hurricane_harvey_2017"]},
    "brazoria county": {"name": "Brazoria County", "lat": 29.1700, "lon": -95.4300, "country_code": "US", "entity_type": "county",
                        "events": ["hurricane_harvey_2017"]},
    "galveston county": {"name": "Galveston County", "lat": 29.3800, "lon": -94.9000, "country_code": "US", "entity_type": "county",
                         "events": ["hurricane_harvey_2017"]},
    "howard county": {"name": "Howard County", "lat": 39.2504, "lon": -76.9273, "country_code": "US", "entity_type": "county",
                      "events": ["maryland_floods_2018"]},
    "manicaland": {"name": "Manicaland", "lat": -19.0000, "lon": 32.5000, "country_code": "ZW", "entity_type": "region"},
    "manicaland province": {"name": "Manicaland", "lat": -19.0000, "lon": 32.5000, "country_code": "ZW", "entity_type": "region"},
    "manabi": {"name": "Manabi", "lat": -1.0500, "lon": -80.4500, "country_code": "EC", "entity_type": "region"},
    "pedernales": {"name": "Pedernales", "lat": 0.0719, "lon": -80.0525, "country_code": "EC", "entity_type": "town",
                   "events": ["ecuador_earthquake_2016"]},
    "mati": {"name": "Mati", "lat": 38.0500, "lon": 24.0100, "country_code": "GR", "entity_type": "town",
             "events": ["greece_wildfires_2018"]},
    "rafina": {"name": "Rafina", "lat": 38.0224, "lon": 24.0066, "country_code": "GR", "entity_type": "town"},
}

# Gazetteer Place Names for Fallback Location Extraction
GAZETTEER_LOCATIONS = [
    "Houston", "Texas", "Puerto Rico", "Sri Lanka", "Florida", "Greece", "Athens",
    "Italy", "Amatrice", "Ecuador", "Mexico", "Puebla", "Kaikoura", "New Zealand",
    "Chennai", "Chennai Central", "Mumbai", "Delhi", "Kolkata", "Kerala",
    "Colombo", "Beira", "Mozambique", "Zimbabwe", "Malawi", "Carolina",
    "Dominica", "Barbuda", "St. Martin", "Virgin Islands", "Porto Rico", "Naples",
    "Ellicott City", "Maryland", "Fort McMurray", "Alberta", "Haiti", "Manabi"
]

# Urgency Keywords & Weights
URGENCY_KEYWORDS = {
    "critical": [
        "sos", "trapped", "casualty", "casualties", "dying", "dead", "fatalities",
        "under rubble", "drowning", "send help immediately", "bleeding",
        "collapsed building", "life threatening", "stuck inside", "buried"
    ],
    "high": [
        "urgent", "rescue", "help", "need medical", "please help", "evacuate",
        "evacuation", "fire", "flood", "emergency", "supplies needed", "stranded",
        "missing person", "broken leg", "injured", "no food", "no water"
    ],
    "medium": [
        "damage", "warning", "advisory", "power outage", "shelter", "food",
        "water", "blankets", "blocked road", "tree down", "flooded street"
    ],
    # Reports of deaths (e.g. "killed 83", "death toll rises"). Scored once, as severity context rather
    # than a call for dispatch, so they land in MEDIUM unless other urgency signals are also present.
    # ("victims" and "bodies" are deliberately absent: "help the victims of ..." is a donation appeal and
    # "bodies of water" is not a casualty report.)
    "casualty": [
        "killed", "kills", "killing", "death toll", "deaths", "died",
        "bodies recovered", "bodies found", "lives lost", "perished", "toll rises"
    ]
}

# Thresholds for Urgency Levels
URGENCY_THRESHOLDS = {
    "LOW": (0.0, 2.9),
    "MEDIUM": (3.0, 5.4),
    "HIGH": (5.5, 7.4),
    "CRITICAL": (7.5, 10.0)
}

# Theme Palette mappings
SEVERITY_COLORS = {
    "CRITICAL": "#B91C1C",
    "HIGH": "#C2410C",
    "MEDIUM": "#A16207",
    "LOW": "#15803D"
}

CATEGORY_COLORS = {
    "Emergency": "#DC2626",
    "Help Request": "#EA580C",
    "Damage Report": "#7C3AED",
    "Information/Update": "#2563EB",
    "General": "#94A3B8"
}

# Need-Type Lexicons
NEED_TAXONOMY = {
    "Medical": [
        "medical", "doctor", "nurse", "medicine", "insulin", "ambulance", "hospital",
        "clinic", "oxygen", "injured", "bleeding", "paramedic", "first aid", "blood",
        "medical supplies", "medical aid", "medical help", "medical care", "medical assistance",
        "medical team", "medical teams", "medicines", "medication", "wounded", "dialysis",
        "paramedics", "doctors", "nurses", "injuries", "hospitals", "ambulances", "emergency room"
    ],
    "Water": [
        "drinking water", "clean water", "bottled water", "potable water", "thirsty", "water needed",
        "no water", "without water", "need water", "needs water", "water supply", "water supplies",
        "water shortage", "out of water", "run out of water", "fresh water", "safe water",
        "water and food", "food and water", "water bottles", "thirst", "dehydrated",
        "water purification", "water tablets"
    ],
    "Food": [
        "food", "hungry", "meal", "meals", "starving", "rations", "canned food",
        "baby formula", "groceries", "food supplies", "nutrition",
        "no food", "need food", "needs food", "without food", "food shortage", "food aid",
        "food relief", "out of food", "run out of food", "hunger", "food packages", "food packs",
        "food parcels", "baby food"
    ],
    "Shelter": [
        "shelter", "tent", "tents", "tarp", "tarps", "roof", "housing", "blankets",
        "homeless", "displaced", "evacuation center", "temporary housing",
        "need shelter", "shelters", "emergency shelter", "emergency shelters", "relief camp",
        "relief camps", "evacuation centers", "evacuation centre", "evacuation centres",
        "place to stay", "lost their homes", "lost homes", "homes destroyed", "refuge",
        "bedding", "mattresses", "clothing", "clothes", "temporary shelter"
    ],
    "Rescue/Evacuation": [
        "rescue", "trapped", "boat", "boats", "stranded", "evacuate", "evacuation",
        "airlift", "helicopter", "save us", "lifeline", "under rubble", "stuck",
        "rescued", "rescuers", "rescuer", "rescue team", "rescue teams", "rescue operation",
        "rescue operations", "rescue efforts", "search and rescue", "evacuated", "evacuating",
        "evacuees", "evacuation order", "evacuation orders", "mandatory evacuation", "need rescue",
        "help us", "rooftop", "rooftops", "swift water", "lifeboat", "rescue boat", "coast guard",
        "helicopters", "airlifted"
    ],
    "Power/Utilities": [
        "power outage", "electricity", "blackout", "generator", "generators", "no power",
        "downed wire", "transformer", "grid down", "power cut",
        "power outages", "outage", "outages", "without power", "lost power", "power lines",
        "power line", "power is out", "power out", "no electricity", "without electricity",
        "power restored", "power grid", "power failure", "blackouts", "power cuts", "gas leak",
        "gas leaks", "water main", "sewage", "utility crews", "downed power", "power poles"
    ],
    "Transport/Roads": [
        "road blocked", "bridge collapsed", "impassable", "highway closed", "debris on road",
        "blocked route", "flooded road", "traffic shut", "road closed",
        "roads closed", "road closures", "road closure", "roads blocked", "roads flooded",
        "flooded roads", "flooded streets", "streets flooded", "washed out", "bridge washed",
        "bridge out", "bridges collapsed", "bridge damaged", "roads damaged", "road damage",
        "closed roads", "impassable roads", "highways closed", "interstate closed", "airport closed",
        "airports closed", "flights cancelled", "flights canceled", "cancelled flights",
        "canceled flights", "ferries suspended", "traffic jam", "gridlock", "blocked roads",
        "roads are closed", "streets are flooded", "road conditions"
    ],
    "Communication": [
        "cell service", "wifi", "phone signal", "radio", "contact family", "out of service",
        "network down", "no signal", "satellite phone",
        "phone lines", "phones down", "no phone", "no cell", "cell towers", "cell tower",
        "cell phone service", "telephone lines", "internet down", "no internet", "internet outage",
        "communications down", "communication lines", "communication network", "cut off from",
        "unable to reach", "cannot reach", "can't reach", "hotline", "helpline", "emergency number",
        "emergency numbers", "ham radio", "no service", "phone service", "mobile network",
        "mobile networks", "network outage"
    ],
    "Missing persons": [
        "missing person", "missing persons", "haven't heard from", "lost contact",
        "last seen", "find my", "looking for", "missing relative",
        "missing", "unaccounted for", "whereabouts", "can't find", "cannot find", "have you seen",
        "looking for my", "searching for my", "searching for missing", "reunite", "reunited",
        "reunification", "found alive", "safe and found", "missing people", "missing child",
        "missing children", "missing man", "missing woman"
    ],
    "Donations/Volunteering": [
        "donate", "donation", "donations", "volunteers needed", "volunteer drive",
        "blood donation", "fundraiser", "relief fund",
        "donating", "donated", "donors", "donor", "volunteer", "volunteers", "volunteering",
        "volunteer efforts", "relief efforts", "relief effort", "fundraising", "raise money",
        "raising money", "raised money", "gofundme", "red cross", "salvation army", "how to help",
        "ways to help", "way to help", "support relief", "send supplies", "drop off", "drop-off",
        "collection point", "collection drive", "charity", "charities", "text to donate",
        "crowdfunding", "relief supplies", "supply drive"
    ]
}

# Anomaly Detection Defaults
ANOMALY_WINDOW_HOURS = 24
ANOMALY_Z_THRESHOLD = 2.5
