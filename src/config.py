"""
Configuration module for Disaster Tweet Intelligence System.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "events_set1"
RAW_DATA_ALT_DIR = DATA_DIR / "events_set1"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "processed_tweets.csv"
DB_PATH = PROCESSED_DATA_DIR / "disaster_tweets.db"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_classifier.joblib"
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

# Gazetteer Place Names for Fallback Location Extraction
GAZETTEER_LOCATIONS = [
    "Houston", "Texas", "Puerto Rico", "Sri Lanka", "Florida", "Greece", "Athens",
    "Italy", "Amatrice", "Ecuador", "Mexico", "Puebla", "Kaikoura", "New Zealand",
    "Chennai", "Chennai Central", "Mumbai", "Delhi", "Kolkata", "Kerala",
    "Colombo", "Beira", "Mozambique", "Zimbabwe", "Malawi", "Carolina", "Texas",
    "Dominica", "Barbuda", "St. Martin", "Virgin Islands", "Porto Rico", "Naples"
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
