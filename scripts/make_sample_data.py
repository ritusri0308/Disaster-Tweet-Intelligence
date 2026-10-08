"""
Synthetic dataset generator for testing and fallback when raw data is missing.
"""

import csv
from pathlib import Path
import random

SAMPLE_EVENTS = [
    "canada_wildfires_2016", "cyclone_idai_2019", "ecuador_earthquake_2016",
    "greece_wildfires_2018", "hurricane_harvey_2017", "hurricane_irma_2017",
    "hurricane_maria_2017", "hurricane_matthew_2016", "italy_earthquake_aug_2016",
    "kaikoura_earthquake_2016", "maryland_floods_2018", "puebla_mexico_earthquake_2017",
    "srilanka_floods_2017"
]

SAMPLE_LABELS = [
    "injured_or_dead_people", "requests_or_urgent_needs",
    "infrastructure_and_utility_damage", "caution_and_advice",
    "displaced_people_and_evacuations", "missing_or_found_people",
    "other_relevant_information", "rescue_volunteering_or_donation_effort",
    "not_humanitarian", "sympathy_and_support"
]

SAMPLE_TEXT_TEMPLATES = {
    "injured_or_dead_people": [
        "SOS! 5 people injured and 2 trapped in collapsed building near Houston central.",
        "Emergency! Medical team needed immediately for casualties in Amatrice.",
        "Tragic news: several fatalities reported following severe flooding in Sri Lanka."
    ],
    "requests_or_urgent_needs": [
        "URGENT: Clean drinking water and food supplies desperately needed in Beira!",
        "Please help! People trapped on roof in Texas. Need rescue boat right now.",
        "Need urgent medical supplies and blankets for displaced families in Puebla."
    ],
    "infrastructure_and_utility_damage": [
        "Major bridge collapsed on Highway 101 after earthquake. Road closed.",
        "Power grid is completely down across Puerto Rico following Hurricane Maria.",
        "Severe road damage and tree blockages reported near Athens after wildfires."
    ],
    "caution_and_advice": [
        "ADVISORY: Stay away from low-lying coastal areas due to storm surge warning.",
        "Caution: High flood risk in Maryland. Keep emergency kits ready.",
        "Safety warning: Wildfire smoke levels are extremely high in Alberta."
    ],
    "displaced_people_and_evacuations": [
        "Over 500 families evacuated to temporary shelter in Kaikoura high school.",
        "Evacuation orders issued for coastal zones in Florida ahead of Irma.",
        "Displaced residents gathering at local sports arena needing shelter."
    ],
    "missing_or_found_people": [
        "MISSING PERSON: John Doe, last seen near flood zone in Sri Lanka. Please RT.",
        "Found child near rescue camp in Ecuador. Contact emergency helpline.",
        "Searching for missing relatives in Amatrice after earthquake."
    ],
    "other_relevant_information": [
        "Weather update: Cyclone Idai category 4 storm making landfall near Beira.",
        "Wildfire spread estimated at 10,000 hectares according to satellite report.",
        "Government opens emergency response hotline for affected disaster victims."
    ],
    "rescue_volunteering_or_donation_effort": [
        "Red Cross volunteering drive starting tomorrow in Houston. Sign up now.",
        "Donate to help disaster victims in Mexico earthquake via text or online.",
        "Local volunteers distributing warm food and water in hurricane shelter."
    ],
    "not_humanitarian": [
        "Watching the news coverage of the hurricane from home.",
        "Interesting article on climate change trends over the last decade.",
        "Hoping everyone stays safe during this bad weather week."
    ],
    "sympathy_and_support": [
        "Praying for everyone affected by the earthquake in Italy. Stay strong!",
        "Our thoughts and heartfelt condolences go out to victims of the floods.",
        "Sending love and solidarity to the brave emergency first responders."
    ]
}

def generate_sample_data(target_dir: Path) -> None:
    """Generate synthetic TSV files matching HumAID dataset layout."""
    print("WARNING: Generating synthetic sample dataset for testing/fallback!")
    target_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    tweet_id_base = 870161518553251840

    for event in SAMPLE_EVENTS:
        # Check nested structure
        if event in ["greece_wildfires_2018", "maryland_floods_2018"]:
            event_dir = target_dir / "hurricane_matthew_2016" / event
        else:
            event_dir = target_dir / event

        event_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "dev", "test"]:
            filepath = event_dir / f"{event}_{split}.tsv"
            num_rows = 15 if split == "train" else 5

            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                writer.writerow(["tweet_id", "tweet_text", "class_label"])

                for _ in range(num_rows):
                    tweet_id_base += random.randint(100, 10000)
                    label = random.choice(SAMPLE_LABELS)
                    template = random.choice(SAMPLE_TEXT_TEMPLATES[label])
                    text = f"{template} &amp; stay safe!"
                    writer.writerow([str(tweet_id_base), text, label])

    print(f"Synthetic sample dataset successfully generated at {target_dir}")

if __name__ == "__main__":
    from src.config import RAW_DATA_DIR
    generate_sample_data(RAW_DATA_DIR)
