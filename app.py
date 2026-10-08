"""
CrisisLens - Command Center Application Entry Point.
A professional crisis intelligence platform turning unstructured disaster social signals
into structured emergency dispatches.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    APP_NAME,
    APP_TAGLINE,
    PROCESSED_PARQUET_PATH,
    PROCESSED_CSV_PATH,
    DB_PATH,
    MODEL_PATH
)
from src.database import load_processed_tweets
from app.ui.theme import inject_custom_theme
from app.ui.components import render_header, render_global_sidebar_filters

# Import Page Renderers
from app.pages import (
    situation_room,
    replay_mode,
    triage_queue,
    geo_intel,
    model_observatory,
    review_queue,
    about_data
)

# Set page configuration
st.set_page_config(
    page_title=f"{APP_NAME} | {APP_TAGLINE}",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply command-center theme
inject_custom_theme()

@st.cache_data(ttl=600)
def load_crisis_dataset() -> pd.DataFrame:
    """Load processed dataset prioritizing fast Parquet format."""
    # 1. Try Parquet
    if PROCESSED_PARQUET_PATH.exists():
        try:
            df = pd.read_parquet(PROCESSED_PARQUET_PATH)
            if not df.empty:
                return df
        except Exception:
            pass

    # 2. Try SQLite
    if DB_PATH.exists():
        try:
            df = load_processed_tweets(DB_PATH)
            if not df.empty:
                return df
        except Exception:
            pass

    # 3. Try CSV
    if PROCESSED_CSV_PATH.exists():
        try:
            df = pd.read_csv(PROCESSED_CSV_PATH)
            for col in ["locations", "matched_terms", "needs", "geocoded_locations"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else (x if isinstance(x, list) else []))
            return df
        except Exception:
            pass

    return pd.DataFrame()

@st.cache_resource
def load_crisis_model():
    """Load cached classification model."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def main():
    # Load assets
    df_raw = load_crisis_dataset()
    pipeline = load_crisis_model()

    # Sidebar Navigation
    st.sidebar.markdown(f"<h2 style='color:#22D3EE;margin-top:0;'>🚨 {APP_NAME}</h2>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p style='font-size:0.8rem;color:#8B9BB4;margin-top:-10px;'>{APP_TAGLINE}</p>", unsafe_allow_html=True)

    page_nav = st.sidebar.radio(
        "Navigation",
        [
            "📍 Situation Room",
            "⏳ Replay Mode",
            "🚨 Priority Triage Queue",
            "🗺️ Geospatial Intel",
            "🔬 Model Observatory",
            "👥 Human Review Queue",
            "ℹ️ About & Data"
        ],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("<hr style='margin:0.75rem 0;border-color:#1F2A3D;'>", unsafe_allow_html=True)

    # Render global sidebar filters
    filtered_df = render_global_sidebar_filters(df_raw)

    # Render top header status strip
    render_header(filtered_df, page_nav)

    # Page routing
    if page_nav == "📍 Situation Room":
        situation_room.render_page(filtered_df, pipeline)
    elif page_nav == "⏳ Replay Mode":
        replay_mode.render_page(filtered_df, pipeline)
    elif page_nav == "🚨 Priority Triage Queue":
        triage_queue.render_page(filtered_df, pipeline)
    elif page_nav == "🗺️ Geospatial Intel":
        geo_intel.render_page(filtered_df, pipeline)
    elif page_nav == "🔬 Model Observatory":
        model_observatory.render_page(filtered_df, pipeline)
    elif page_nav == "👥 Human Review Queue":
        review_queue.render_page(filtered_df, pipeline)
    elif page_nav == "ℹ️ About & Data":
        about_data.render_page(filtered_df, pipeline)

if __name__ == "__main__":
    main()
