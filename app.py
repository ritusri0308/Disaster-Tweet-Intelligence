"""
CrisisLens - Command Center Application Entry Point.
A professional crisis intelligence platform turning unstructured disaster social signals
into structured emergency dispatches.
"""

import sys
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
from src.model_registry import dataset_matches_model, fingerprint_file, model_health
from src.privacy import add_redacted_columns, apply_redaction
from src.utils import normalize_list_columns
from app.ui.theme import inject_custom_theme
from app.ui.components import (
    render_callout,
    render_global_sidebar_filters,
    render_header,
    render_model_banner,
)

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

# Apply command-center light theme
inject_custom_theme()

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise list columns and precompute redacted copies of the tweet text (so the privacy toggle is instant)."""
    return add_redacted_columns(normalize_list_columns(df))

@st.cache_data(ttl=600)
def load_crisis_dataset() -> pd.DataFrame:
    """
    Load processed dataset prioritizing fast Parquet format.

    List columns (locations, needs, ...) are normalised to plain Python lists whichever source is used:
    Parquet returns numpy arrays, which the page code would otherwise silently treat as empty.
    """
    # 1. Try Parquet
    if PROCESSED_PARQUET_PATH.exists():
        try:
            df = pd.read_parquet(PROCESSED_PARQUET_PATH)
            if not df.empty:
                return _prepare(df)
        except Exception:
            pass

    # 2. Try SQLite
    if DB_PATH.exists():
        try:
            df = load_processed_tweets(DB_PATH)
            if not df.empty:
                return _prepare(df)
        except Exception:
            pass

    # 3. Try CSV
    if PROCESSED_CSV_PATH.exists():
        try:
            return _prepare(pd.read_csv(PROCESSED_CSV_PATH))
        except Exception:
            pass

    return pd.DataFrame()

@st.cache_resource
def load_crisis_model(model_version: str):
    """
    Load the classification model. ``model_version`` is the file's content fingerprint and exists only to
    key the cache, so retraining a model is picked up without restarting the app.
    """
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def main():
    # Load assets
    df_raw = load_crisis_dataset()
    live_version = fingerprint_file(MODEL_PATH)
    pipeline = load_crisis_model(live_version or "none")

    model_status = model_health(pipeline)
    dataset_state = dataset_matches_model(
        df_raw["model_version"] if "model_version" in df_raw.columns else None, live_version
    ) if not df_raw.empty else {"state": "unknown", "dataset_versions": []}

    # Privacy: phone numbers, emails and street addresses are hidden unless the operator asks to see them.
    # Redaction happens before filtering so searching cannot reveal a hidden number.
    st.sidebar.toggle(
        "Show phone numbers & addresses", key="show_pii", value=False,
        help="Contact details found in tweet text are hidden by default. Names are not detected, so redaction reduces exposure but does not anonymise."
    )
    df_view = apply_redaction(df_raw, redact=not st.session_state.get("show_pii", False))

    # Sidebar Filter Controls
    filtered_df = render_global_sidebar_filters(df_view)

    def make_page(title: str, module):
        """Page wrapper: header, any model/dataset warnings, then the page itself."""
        def _run():
            render_header(filtered_df, title, model_version=live_version)
            if df_raw.empty:
                render_callout(
                    "No processed dataset was found. Run `python run_pipeline.py` (add `--demo` to use synthetic sample data).",
                    "warn", "No data loaded."
                )
            render_model_banner(model_status, dataset_state)
            module.render_page(filtered_df, pipeline)
        return _run

    pages = [
        st.Page(make_page("Situation Room", situation_room), title="Situation Room", url_path="situation-room",
                icon=":material/dashboard:", default=True),
        st.Page(make_page("Replay Mode", replay_mode), title="Replay Mode", url_path="replay",
                icon=":material/history:"),
        st.Page(make_page("Priority Triage Queue", triage_queue), title="Priority Triage Queue", url_path="triage",
                icon=":material/priority_high:"),
        st.Page(make_page("Geospatial Intel", geo_intel), title="Geospatial Intel", url_path="geo",
                icon=":material/public:"),
        st.Page(make_page("Model Observatory", model_observatory), title="Model Observatory", url_path="models",
                icon=":material/insights:"),
        st.Page(make_page("Human Review Queue", review_queue), title="Human Review Queue", url_path="review",
                icon=":material/fact_check:"),
        st.Page(make_page("About & Data", about_data), title="About & Data", url_path="about",
                icon=":material/info:")
    ]

    nav = st.navigation(pages)
    nav.run()

    # Clean footer line
    st.markdown("<div class='cl-footer'>Decision-support tool, not an emergency dispatch system</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
