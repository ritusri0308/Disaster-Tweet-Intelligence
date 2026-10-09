"""
About & Data Governance page for CrisisLens.
Presents dataset citation, humanitarian licensing terms, legal disclaimers,
category taxonomy mappings, and system limitations.
"""

import streamlit as st
from src.config import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    DISCLAIMER
)
from app.ui.components import esc, render_callout

def render_page(df: any = None, pipeline: any = None):
    st.markdown(f"<div class='cl-muted' style='margin-bottom:1rem;'>{esc(APP_NAME)} version {esc(APP_VERSION)} · {esc(APP_TAGLINE)}</div>", unsafe_allow_html=True)

    render_callout(DISCLAIMER, "warn", "Operational decision-support notice:")

    st.markdown("""
    ### 1. System Scope & Benchmark Data
    - **Mode**: **HISTORICAL REPLAY**. All records represent historical disaster benchmark data from **2016 to 2019**. This deployment is not an active social media streaming service.
    - **Geographic Precision**: The dataset does not contain physical GPS coordinates. Map points are resolved via offline entity recognition and GeoNamesCache gazetteer priors (plus a small curated list of towns and islands) to city, regional, or country centroids.
    - **Emergency Definition**: In this taxonomy, the **Emergency** category represents casualty reports (`injured_or_dead_people`).
    - **Reported figures**: Casualty and displacement numbers are extracted from tweet text. They are claims made in tweets, not verified counts.

    ---

    ### 2. Dataset Citation & Humanitarian License
    - **Dataset**: HumAID (events_set1), 13 disaster incidents, ~43,400 tweets.
    - **Licensing**: Licensed exclusively for non-commercial humanitarian computing research. Raw tweet data and internal database files must never be committed to public repositories or redistributed.
    - **Official Citation**:
    > Firoj Alam, Hassan Sajjad, Muhammad Imran, and Ferda Ofli.
    > *HumAID: Human-Annotated Disaster Incident Data from Twitter with Multimodal Features*.
    > Proceedings of the International AAAI Conference on Web and Social Media (ICWSM), 2021.

    ---

    ### 3. Crisis Taxonomy Mapping
    The 10 raw HumAID incident labels are normalized into 5 actionable emergency categories:

    | Raw HumAID Label | Project Category | Description |
    | :--- | :--- | :--- |
    | `injured_or_dead_people` | **Emergency** | Casualty reports (injured or deceased individuals) |
    | `requests_or_urgent_needs` | **Help Request** | Urgent calls for water, food, medical care, or rescue |
    | `infrastructure_and_utility_damage` | **Damage Report** | Damaged roads, collapsed bridges, power grid blackouts |
    | `caution_and_advice` | **Information/Update** | Official advisories, evacuation instructions, warnings |
    | `displaced_people_and_evacuations` | **Information/Update** | Shelter status, temporary camps, evacuated populations |
    | `missing_or_found_people` | **Information/Update** | Reports of missing individuals or reunifications |
    | `other_relevant_information` | **Information/Update** | General disaster news and meteorology reports |
    | `rescue_volunteering_or_donation_effort` | **Information/Update** | Organized volunteer drives and donation campaigns |
    | `not_humanitarian` | **General** | Irrelevant or non-crisis social media content |
    | `sympathy_and_support` | **General** | Condolences, prayers, emotional solidarity |
    | `dont_know_cant_judge` | **General** | Ambiguous or unverifiable dispatches |

    ---

    ### 4. Technical Architecture
    - **Feature Representation**: Sublinear TF-IDF (unigrams + bigrams) over cleaned tweet text.
    - **Classifier**: The best of Naive Bayes, Logistic Regression, Linear SVM and Random Forest by dev macro-F1. When candidates are within 0.005 of each other the simpler model is chosen. Confidence is the model's predicted probability; the Model Observatory shows how well it is calibrated.
    - **NER & Geocoding**: spaCy `en_core_web_sm` coupled with an offline `geonamescache` resolver. Countries and US states take precedence over same-named towns, and per-event geographic priors disambiguate the rest.
    - **Urgency Scoring**: Explainable rule engine producing 0-10 priority scores and a waterfall attribution breakdown. Reports of deaths raise severity but do not by themselves reach CRITICAL, which is reserved for active calls for help.
    - **Humanitarian Needs**: Phrase lexicons for ten need types, evaluated against HumAID labels as proxies in the reports.
    """)
