"""
Configuration module for CrisisLens.
"""

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

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "events_set1"
RAW_DATA_ALT_DIR = DATA_DIR / "events_set1"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "processed_tweets.csv"
PROCESSED_PARQUET_PATH = PROCESSED_DATA_DIR / "processed_tweets.parquet"
DB_PATH = PROCESSED_DATA_DIR / "disaster_tweets.db"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_classifier.joblib"
MODEL_BACKUP_PATH = MODELS_DIR / "best_classifier.joblib.bak"
REPORTS_DIR = PROJECT_ROOT / "reports"

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
    "hurricane_irma_2017": {"countries": ["US", "CU", "BS"], "regions": ["Florida", "Georgia", "Puerto Rico"]},
    "hurricane_maria_2017": {"countries": ["US", "DM"], "regions": ["Puerto Rico", "Virgin Islands", "Dominica"]},
    "hurricane_matthew_2016": {"countries": ["US", "HT", "BS", "CU"], "regions": ["Florida", "Georgia", "North Carolina", "South Carolina", "Haiti"]},
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
    "october", "november", "december", "am", "pm", "utc", "rt", "psa", "red cross"
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
    "CRITICAL": "#EF4444",
    "HIGH": "#F97316",
    "MEDIUM": "#EAB308",
    "LOW": "#22C55E"
}

CATEGORY_COLORS = {
    "Emergency": "#EF4444",
    "Help Request": "#F97316",
    "Damage Report": "#8B5CF6",
    "Information/Update": "#06B6D4",
    "General": "#64748B"
}

# Need-Type Lexicons
NEED_TAXONOMY = {
    "Medical": [
        "medical", "doctor", "nurse", "medicine", "insulin", "ambulance", "hospital",
        "clinic", "oxygen", "injured", "bleeding", "paramedic", "first aid", "blood"
    ],
    "Water": [
        "drinking water", "clean water", "bottled water", "potable water", "thirsty", "water needed"
    ],
    "Food": [
        "food", "hungry", "meal", "meals", "starving", "rations", "canned food",
        "baby formula", "groceries", "food supplies", "nutrition"
    ],
    "Shelter": [
        "shelter", "tent", "tents", "tarp", "tarps", "roof", "housing", "blankets",
        "homeless", "displaced", "evacuation center", "temporary housing"
    ],
    "Rescue/Evacuation": [
        "rescue", "trapped", "boat", "boats", "stranded", "evacuate", "evacuation",
        "airlift", "helicopter", "save us", "lifeline", "under rubble", "stuck"
    ],
    "Power/Utilities": [
        "power outage", "electricity", "blackout", "generator", "generators", "no power",
        "downed wire", "transformer", "grid down", "power cut"
    ],
    "Transport/Roads": [
        "road blocked", "bridge collapsed", "impassable", "highway closed", "debris on road",
        "blocked route", "flooded road", "traffic shut", "road closed"
    ],
    "Communication": [
        "cell service", "wifi", "phone signal", "radio", "contact family", "out of service",
        "network down", "no signal", "satellite phone"
    ],
    "Missing persons": [
        "missing person", "missing persons", "haven't heard from", "lost contact",
        "last seen", "find my", "looking for", "missing relative"
    ],
    "Donations/Volunteering": [
        "donate", "donation", "donations", "volunteers needed", "volunteer drive",
        "blood donation", "fundraiser", "relief fund"
    ]
}

# Anomaly Detection Defaults
ANOMALY_WINDOW_HOURS = 24
ANOMALY_Z_THRESHOLD = 2.5
