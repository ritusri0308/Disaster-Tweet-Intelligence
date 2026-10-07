"""
Streamlit Web Application for Disaster Tweet Intelligence & Emergency Prioritization System.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_CSV_PATH,
    DB_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    PROJECT_CATEGORIES
)
from src.preprocessing import clean_tweet, extract_single_text_features
from src.classifier import predict_with_confidence
from src.ner import extract_locations_single
from src.urgency import calculate_urgency
from src.analytics import (
    get_category_counts,
    get_urgency_counts,
    get_category_urgency_crosstab,
    get_top_locations,
    get_volume_over_time,
    get_per_event_summary,
    generate_situation_summary
)
from src.database import load_processed_tweets
import joblib

# Page configuration
st.set_page_config(
    page_title="Disaster Tweet Intelligence System",
    page_icon="🚨",
    layout="wide"
)

@st.cache_data(ttl=300)
def get_cached_processed_data() -> pd.DataFrame:
    """Load cached processed dataset from SQLite DB or CSV."""
    if DB_PATH.exists():
        df = load_processed_tweets(DB_PATH)
        if not df.empty:
            return df
    if PROCESSED_CSV_PATH.exists():
        df = pd.read_csv(PROCESSED_CSV_PATH)
        # Parse JSON columns if loaded from CSV
        for col in ["locations", "matched_terms"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else (x if isinstance(x, list) else []))
        return df
    return pd.DataFrame()

@st.cache_resource
def get_cached_model():
    """Load cached sklearn classifier pipeline."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def render_dashboard(df: pd.DataFrame):
    st.title("🚨 Emergency Intelligence Dashboard")
    st.markdown("Real-time aggregate analytics and situation intelligence from disaster tweets.")

    if df.empty:
        st.warning("⚠️ Processed dataset not found. Please run `python run_pipeline.py` first to process data!")
        return

    # Sidebar Filters
    st.sidebar.header("Filter Options")
    
    # Split Filter
    available_splits = list(df["split"].unique()) if "split" in df.columns else []
    split_choice = st.sidebar.selectbox("Data Split", options=["test (Default)", "All Splits"] + available_splits, index=0)

    if split_choice == "test (Default)" and "test" in available_splits:
        filtered_df = df[df["split"] == "test"].copy()
    elif split_choice in available_splits:
        filtered_df = df[df["split"] == split_choice].copy()
    else:
        filtered_df = df.copy()

    # Event Filter
    events = ["All Events"] + sorted(list(filtered_df["event"].unique())) if "event" in filtered_df.columns else ["All Events"]
    selected_event = st.sidebar.selectbox("Disaster Event", options=events)
    if selected_event != "All Events":
        filtered_df = filtered_df[filtered_df["event"] == selected_event]

    # Category Filter
    cats = ["All Categories"] + sorted(list(filtered_df["category"].unique())) if "category" in filtered_df.columns else ["All Categories"]
    selected_cat = st.sidebar.selectbox("Project Category", options=cats)
    if selected_cat != "All Categories":
        filtered_df = filtered_df[filtered_df["category"] == selected_cat]

    # Urgency Filter
    urgencies = ["All Urgency Levels", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    selected_urg = st.sidebar.selectbox("Urgency Level", options=urgencies)
    if selected_urg != "All Urgency Levels":
        filtered_df = filtered_df[filtered_df["urgency_level"] == selected_urg]

    # KPI Metrics Row
    total_tweets = len(filtered_df)
    crit_count = (filtered_df["urgency_level"] == "CRITICAL").sum() if "urgency_level" in filtered_df.columns else 0
    help_count = (filtered_df["category"] == "Help Request").sum() if "category" in filtered_df.columns else 0
    top_locs = get_top_locations(filtered_df, top_n=100)
    unique_locs_count = len(top_locs)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Tweets", f"{total_tweets:,}")
    kpi2.metric("CRITICAL Alerts", f"{crit_count:,}", delta_color="inverse")
    kpi3.metric("Help Requests", f"{help_count:,}")
    kpi4.metric("Unique Locations Detected", f"{unique_locs_count:,}")

    st.markdown("---")

    # Situation Summary Box
    summary_data = generate_situation_summary(filtered_df)
    st.info(f"📋 **Situation Summary**: {summary_data['summary_text']}")

    # Charts Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category Distribution")
        cat_counts = get_category_counts(filtered_df)
        if not cat_counts.empty:
            fig_cat = px.bar(
                x=cat_counts.index,
                y=cat_counts.values,
                labels={"x": "Category", "y": "Tweet Count"},
                color=cat_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_cat.update_layout(showlegend=False)
            st.plotly_chart(fig_cat)

    with col2:
        st.subheader("Urgency Level Breakdown")
        urg_counts = get_urgency_counts(filtered_df)
        if not urg_counts.empty:
            colors = {"CRITICAL": "#e74c3c", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "LOW": "#2ecc71"}
            fig_urg = px.pie(
                values=urg_counts.values,
                names=urg_counts.index,
                hole=0.4,
                color=urg_counts.index,
                color_discrete_map=colors
            )
            st.plotly_chart(fig_urg)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Category x Urgency Heatmap")
        ct = get_category_urgency_crosstab(filtered_df)
        if not ct.empty:
            fig_hm = px.imshow(
                ct,
                labels=dict(x="Urgency Level", y="Category", color="Count"),
                text_auto=True,
                color_continuous_scale="YlOrRd"
            )
            st.plotly_chart(fig_hm)

    with col4:
        st.subheader("Top Extracted Locations")
        if not top_locs.empty:
            top_10 = top_locs.head(10).sort_values(ascending=True)
            fig_loc = px.bar(
                x=top_10.values,
                y=top_10.index,
                orientation="h",
                labels={"x": "Frequency", "y": "Location"},
                color_discrete_sequence=["#3498db"]
            )
            st.plotly_chart(fig_loc)
        else:
            st.write("No location entities detected in filtered view.")

    # Timeline Chart
    st.subheader("Tweet Volume Over Time")
    vol_df = get_volume_over_time(filtered_df, freq="D")
    if not vol_df.empty:
        fig_vol = px.line(
            vol_df,
            x="timestamp",
            y="tweet_count",
            labels={"timestamp": "Date (UTC)", "tweet_count": "Tweet Count"},
            markers=True
        )
        st.plotly_chart(fig_vol)
    else:
        st.write("Timeline data unavailable for current selection.")

    # Event Comparison Table
    st.subheader("Per-Event Summary Comparison")
    evt_summary = get_per_event_summary(filtered_df)
    if not evt_summary.empty:
        st.dataframe(evt_summary)

def render_analyze_tweet(pipeline):
    st.title("🔍 Analyze a Tweet")
    st.markdown("Test the classification, location extraction (NER), and urgency prioritization engine on any custom tweet text.")

    examples = [
        "URGENT! People are trapped inside a collapsed building near Chennai Central. Need medical assistance immediately.",
        "Red Cross is organizing a donation drive for hurricane relief in Houston. Please retweet to help!",
        "Severe road damage and power outage reported on Highway 1 following the earthquake in Kaikoura."
    ]

    st.subheader("Click an example or type custom text below:")
    selected_example = st.radio("Sample Tweets:", ["None (Custom Input)"] + examples)

    if selected_example != "None (Custom Input)":
        user_input = selected_example
    else:
        user_input = st.text_area("Enter Tweet Text:", height=100, value="")

    if st.button("Analyze Tweet", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a valid tweet text to analyze.")
            return

        cleaned = clean_tweet(user_input)
        features = extract_single_text_features(user_input)

        if pipeline is not None:
            pred_res = predict_with_confidence(pipeline, [cleaned])
            cat, conf = pred_res[0]
        else:
            cat, conf = "Information/Update", 0.50
            st.info("Using default heuristic model as saved pipeline was not found.")

        locations = extract_locations_single(user_input)

        score, level, terms = calculate_urgency(
            text=user_input,
            category=cat,
            uppercase_ratio=features["uppercase_ratio"],
            exclamation_count=features["exclamation_count"],
            is_retweet=features["is_retweet"]
        )

        st.markdown("### Analysis Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(f"**Cleaned Text**: `{cleaned}`")
            st.markdown(f"**Predicted Category**: `{cat}` (Confidence: `{conf * 100:.1f}%`)")
            st.markdown(f"**Extracted Locations**: `{', '.join(locations) if locations else 'None detected'}`")

        with res_col2:
            st.markdown(f"**Urgency Score**: `{score} / 10.0`")
            level_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
            st.markdown(f"**Urgency Level**: {level_colors.get(level, '')} **{level}**")
            st.markdown(f"**Matched Trigger Terms**: `{', '.join(terms) if terms else 'None'}`")

def render_critical_reports(df: pd.DataFrame):
    st.title("🚨 Critical & High Urgency Reports")
    st.markdown("Prioritized list of emergency alerts requiring immediate attention from responders.")

    if df.empty:
        st.warning("⚠️ No data loaded. Run `python run_pipeline.py` to generate reports.")
        return

    high_crit_df = df[df["urgency_level"].isin(["CRITICAL", "HIGH"])].copy()
    if high_crit_df.empty:
        st.info("No CRITICAL or HIGH urgency reports found in dataset.")
        return

    search_query = st.text_input("Search Reports (Keyword or Location):", "")
    if search_query.strip():
        mask = high_crit_df["raw_unescaped_text"].str.contains(search_query, case=False, na=False)
        high_crit_df = high_crit_df[mask]

    st.write(f"Displaying **{len(high_crit_df)}** prioritized reports sorted by urgency score.")

    display_cols = ["urgency_level", "urgency_score", "category", "confidence", "locations", "raw_unescaped_text", "event", "split"]
    cols_to_show = [c for c in display_cols if c in high_crit_df.columns]

    st.dataframe(high_crit_df[cols_to_show].sort_values("urgency_score", ascending=False))

    csv_data = high_crit_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Critical Reports CSV",
        data=csv_data,
        file_name="critical_disaster_reports.csv",
        mime="text/csv"
    )

def render_model_performance():
    st.title("📊 Model Performance & Experimentation")
    st.markdown("Detailed quantitative evaluation, confusion matrices, and ablation experiments.")

    # Saved Model Comparison Table
    comp_csv = REPORTS_DIR / "model_comparison.csv"
    if comp_csv.exists():
        st.subheader("Classifier Suite Comparison (Dev & Test Sets)")
        comp_df = pd.read_csv(comp_csv)
        st.dataframe(comp_df)
    else:
        st.info("Model comparison results not found. Run `python train.py` to generate model comparison metrics.")

    # Confusion Matrix Image
    cm_img = REPORTS_DIR / "best_model_confusion_matrix.png"
    if cm_img.exists():
        st.subheader("Best Model Test Confusion Matrix")
        st.image(str(cm_img), caption="Confusion Matrix on Test Split", width=600)

    # Experiments Tables
    st.subheader("Ablation & Leave-One-Event-Out Experiments")
    
    exp_files = {
        "(a) N-gram Range (Unigrams vs Bigrams)": REPORTS_DIR / "exp_a_ngrams.csv",
        "(b) Text Cleaning Effect": REPORTS_DIR / "exp_b_text_format.csv",
        "(c) Class Balancing Effect": REPORTS_DIR / "exp_c_class_balancing.csv",
        "(d) Leave-One-Event-Out Cross-Validation": REPORTS_DIR / "exp_d_leave_one_event_out.csv",
        "(e) Classification Target Schema": REPORTS_DIR / "exp_e_target_schema.csv"
    }

    found_any = False
    for exp_title, exp_path in exp_files.items():
        if exp_path.exists():
            found_any = True
            st.markdown(f"#### {exp_title}")
            exp_df = pd.read_csv(exp_path)
            st.dataframe(exp_df)

    if not found_any:
        st.info("Experiments have not been executed yet. Run `python train.py --experiments` to generate experiment tables.")

def render_about():
    st.title("ℹ️ About & System Limitations")
    st.markdown("""
    ### Academic Prototype Disclaimer
    This application is an academic prototype developed for the university course **Advanced Python Programming (CSI3007)**.
    It is designed for humanitarian computing research and educational purposes, **NOT** as an operational emergency dispatch tool.

    ---

    ### Dataset & Citation
    - **Dataset**: HumAID (events_set1), 13 disaster events (~43,400 tweets, 2016–2019).
    - **License**: Licensed exclusively for humanitarian-computing research and non-commercial educational use.
    - **Citation**:
      > Firoj Alam, Hassan Sajjad, Muhammad Imran, and Ferda Ofli. *HumAID: Human-Annotated Disaster Incident Data from Twitter with Multimodal Features*. ICWSM 2021.

    ---

    ### System Scope & Mapping Caveats
    1. **Category Mapping**: The original 10 HumAID raw class labels are mapped into 5 project categories:
       - `Emergency` <- `injured_or_dead_people`
       - `Help Request` <- `requests_or_urgent_needs`
       - `Damage Report` <- `infrastructure_and_utility_damage`
       - `Information/Update` <- `caution_and_advice`, `displaced_people_and_evacuations`, `missing_or_found_people`, `other_relevant_information`, `rescue_volunteering_or_donation_effort`
       - `General` <- `not_humanitarian`, `sympathy_and_support`, `dont_know_cant_judge`
    2. **Class Imbalance**: Information/Update accounts for ~55% of tweets, while Help Request accounts for ~4%. Class-weighted loss and macro-F1 evaluation are employed.
    3. **Emergency Definition**: In this project, "Emergency" specifically refers to casualty reports (injured or dead people).
    4. **Geographic Scope**: Dataset contains historical disasters from 2016–2019. No real-time GPS coordinates are provided in the dataset; map visualizers are omitted per requirements.
    5. **NER & Urgency Limitations**: Location extraction relies on spaCy `en_core_web_sm` and rule gazetteers. Urgency prioritization uses an explainable rule engine with keyword signals, text feature boosts, and retweet/donation damping rules.
    """)

def main():
    st.sidebar.title("🚨 Disaster Intelligence")
    st.sidebar.markdown("CSI3007 Course Project")

    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Analyze a Tweet", "Critical Reports", "Model Performance", "About / Limitations"]
    )

    df = get_cached_processed_data()
    pipeline = get_cached_model()

    if page == "Dashboard":
        render_dashboard(df)
    elif page == "Analyze a Tweet":
        render_analyze_tweet(pipeline)
    elif page == "Critical Reports":
        render_critical_reports(df)
    elif page == "Model Performance":
        render_model_performance()
    elif page == "About / Limitations":
        render_about()

if __name__ == "__main__":
    main()
