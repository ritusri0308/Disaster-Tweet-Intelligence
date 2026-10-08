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

def render_page(df: any = None, pipeline: any = None):
    st.markdown(f"<h3 style='margin-top:0;color:#22D3EE;'>ℹ️ About {APP_NAME} & Data Governance</h3>", unsafe_allow_html=True)
    st.markdown(f"**Version**: `{APP_VERSION}` | **Tagline**: *{APP_TAGLINE}*")

    st.markdown(
        f"""
        <div style='background:rgba(234, 179, 8, 0.1);border-left:4px solid #EAB308;padding:12px 16px;border-radius:6px;font-size:0.95rem;color:#FEF08A;margin:1rem 0;'>
            <strong>Operational Decision-Support Disclaimer:</strong><br>
            {DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 1. System Scope & Benchmark Data
    - **Mode**: **HISTORICAL REPLAY**. All records represent historical disaster benchmark data from **2016 to 2019**. This deployment is not a live Twitter/X streaming scraper.
    - **Geographic Precision**: The dataset does not contain physical GPS coordinates. Map points are resolved via offline entity recognition and GeoNamesCache gazetteer priors to city/state centroids.
    - **Emergency Definition**: In this taxonomy, the **Emergency** category represents casualty reports (`injured_or_dead_people`).

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
    | `injured_or_dead_people` | **Emergency** | Casualty dispatches (injured or deceased individuals) |
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
    - **Feature Representation**: Sublinear TF-IDF (unigrams + bigrams), text features (uppercase ratio, punctuation intensity, retweet indicators).
    - **Classifier**: Calibrated Logistic Regression with balanced class weighting, selected via dev Macro-F1.
    - **NER & Geocoding**: spaCy `en_core_web_sm` coupled with an offline `geonamescache` resolver and per-event geographic priors.
    - **Urgency Scoring**: Explainable rule engine producing 0-10 priority scores and waterfall attribution breakdown.
    """)
