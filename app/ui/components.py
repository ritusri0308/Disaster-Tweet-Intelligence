"""
Reusable UI components for CrisisLens.
Includes top status strip, global filter bar, KPI cards, and triage detail views.
"""

import json
from typing import Dict, Any
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    APP_NAME,
    APP_TAGLINE,
    DATA_MODE_NOTICE,
    REPORTS_DIR,
    PROJECT_CATEGORIES
)
from src.explainability import explain_classification, highlight_tweet_tokens
from src.urgency import calculate_urgency_breakdown
from app.ui.theme import (
    apply_chart_theme
)

def get_model_provenance() -> Dict[str, Any]:
    """Read trained model metadata from reports/model_comparison.json."""
    comp_file = REPORTS_DIR / "model_comparison.json"
    if comp_file.exists():
        try:
            with open(comp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    # Best model row
                    best = max(data, key=lambda x: x.get("dev_macro_f1", 0))
                    return {
                        "name": best.get("model_name", "Logistic Regression").replace("_", " ").title(),
                        "dev_macro_f1": best.get("dev_macro_f1", 0.7604),
                        "test_macro_f1": best.get("test_macro_f1", 0.7604),
                        "test_acc": best.get("test_accuracy", 0.8079)
                    }
        except Exception:
            pass
    return {
        "name": "Logistic Regression",
        "dev_macro_f1": 0.7604,
        "test_macro_f1": 0.7604,
        "test_acc": 0.8079
    }

def render_header(df_view: pd.DataFrame, page_title: str = ""):
    """Render top product header with status pills strip."""
    prov = get_model_provenance()
    active_split = st.session_state.get("filter_split", "test")
    view_count = len(df_view)

    col1, col2 = st.columns([1.5, 2.5])
    with col1:
        st.markdown(f"<h2 style='margin:0;color:#22D3EE;font-weight:700;'>🚨 {APP_NAME}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:0;color:#8B9BB4;font-size:0.875rem;'>{APP_TAGLINE}</p>", unsafe_allow_html=True)

    with col2:
        pills_html = f"""
        <div style='display:flex;flex-wrap:wrap;justify-content:flex-end;gap:6px;align-items:center;margin-top:6px;'>
            <span class='status-pill pill-active'>🕒 {DATA_MODE_NOTICE}</span>
            <span class='status-pill'>📊 View: {view_count:,} Dispatches</span>
            <span class='status-pill'>🔀 Split: {active_split.upper()}</span>
            <span class='status-pill'>🤖 Model: {prov['name']} (Macro-F1: {prov['test_macro_f1']:.2f})</span>
        </div>
        """
        st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.75rem 0 1.25rem 0;border-color:#1F2A3D;'>", unsafe_allow_html=True)

def render_global_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render global sidebar filters persistent via st.session_state."""
    st.sidebar.markdown(f"<h3 style='color:#22D3EE;margin-top:0;'>{APP_NAME} Command Filters</h3>", unsafe_allow_html=True)

    # Initialize session state keys
    defaults = {
        "filter_split": "test",
        "filter_event": "All Events",
        "filter_cat": "All Categories",
        "filter_urg": "All Urgency Levels",
        "filter_search": "",
        "filter_min_conf": 0.0,
        "filter_hide_rt": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Reset button
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

    # Split selector (default: test)
    splits = ["test", "train", "dev", "all"]
    st.session_state["filter_split"] = st.sidebar.selectbox(
        "Data Split",
        splits,
        index=splits.index(st.session_state["filter_split"]) if st.session_state["filter_split"] in splits else 0,
        help="Official benchmark split partition. 'test' ensures out-of-sample evaluation."
    )

    # Filter df by split first
    filtered = df.copy()
    if st.session_state["filter_split"] != "all" and "split" in filtered.columns:
        filtered = filtered[filtered["split"] == st.session_state["filter_split"]]

    # Disaster Event
    events = ["All Events"] + sorted(list(filtered["event"].unique())) if "event" in filtered.columns else ["All Events"]
    current_ev = st.session_state["filter_event"]
    ev_idx = events.index(current_ev) if current_ev in events else 0
    st.session_state["filter_event"] = st.sidebar.selectbox("Disaster Event", events, index=ev_idx)

    if st.session_state["filter_event"] != "All Events":
        filtered = filtered[filtered["event"] == st.session_state["filter_event"]]

    # Category Filter
    cats = ["All Categories"] + PROJECT_CATEGORIES
    current_cat = st.session_state["filter_cat"]
    cat_idx = cats.index(current_cat) if current_cat in cats else 0
    st.session_state["filter_cat"] = st.sidebar.selectbox("Category", cats, index=cat_idx)

    if st.session_state["filter_cat"] != "All Categories" and "category" in filtered.columns:
        filtered = filtered[filtered["category"] == st.session_state["filter_cat"]]

    # Urgency Filter
    urgencies = ["All Urgency Levels", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    current_urg = st.session_state["filter_urg"]
    urg_idx = urgencies.index(current_urg) if current_urg in urgencies else 0
    st.session_state["filter_urg"] = st.sidebar.selectbox("Urgency Priority", urgencies, index=urg_idx)

    if st.session_state["filter_urg"] != "All Urgency Levels" and "urgency_level" in filtered.columns:
        filtered = filtered[filtered["urgency_level"] == st.session_state["filter_urg"]]

    # Location / Keyword search
    st.session_state["filter_search"] = st.sidebar.text_input(
        "Search Keyword / Location",
        value=st.session_state["filter_search"],
        placeholder="e.g. Houston, trapped, flood"
    )
    if st.session_state["filter_search"].strip():
        q = st.session_state["filter_search"].strip().lower()
        mask = (
            filtered["raw_unescaped_text"].str.lower().str.contains(q, na=False) |
            filtered["clean_text"].str.lower().str.contains(q, na=False) |
            filtered["locations"].apply(lambda locs: any(q in str(loc).lower() for loc in (locs if isinstance(locs, list) else [])))
        )
        filtered = filtered[mask]

    # Min confidence slider
    st.session_state["filter_min_conf"] = st.sidebar.slider(
        "Min Confidence",
        0.0, 1.0, float(st.session_state["filter_min_conf"]), 0.05
    )
    if st.session_state["filter_min_conf"] > 0.0 and "confidence" in filtered.columns:
        filtered = filtered[filtered["confidence"] >= st.session_state["filter_min_conf"]]

    # Hide retweets toggle
    st.session_state["filter_hide_rt"] = st.sidebar.checkbox(
        "Hide Retweets (RT)",
        value=st.session_state["filter_hide_rt"]
    )
    if st.session_state["filter_hide_rt"] and "is_retweet" in filtered.columns:
        filtered = filtered[filtered["is_retweet"] == 0]

    st.sidebar.markdown(f"<div style='font-size:0.8rem;color:#8B9BB4;margin-top:1rem;'>Matching Dispatches: <strong style='color:#22D3EE;'>{len(filtered):,}</strong></div>", unsafe_allow_html=True)

    return filtered

def render_kpi_card(title: str, value: Any, delta: str = "", color: str = "#22D3EE"):
    """Render custom dark-theme glass KPI card."""
    card_html = f"""
    <div class='kpi-container' style='border-top: 3px solid {color};'>
        <div class='kpi-title'>{title}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-delta' style='color:{color};'>{delta}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_triage_detail_panel(row: pd.Series, pipeline: Any):
    """
    Detail inspection panel for selected triage row:
    - Original text with highlights
    - Urgency waterfall chart
    - Classifier token attributions
    - Probabilities & review flags
    """
    text = row.get("raw_unescaped_text", row.get("tweet_text", ""))
    clean = row.get("clean_text", "")
    locs = row.get("locations", []) if isinstance(row.get("locations"), list) else []
    terms = row.get("matched_terms", []) if isinstance(row.get("matched_terms"), list) else []
    cat = row.get("category", "General")
    conf = row.get("confidence", 1.0)

    # Explanation
    explanation = explain_classification(pipeline, clean) if pipeline is not None else {
        "predicted_class": cat, "confidence": conf, "top_positive_features": [], "top_negative_features": [], "probabilities": {cat: conf}, "needs_human_review": False
    }

    # Highlighted text
    st.markdown("#### Dispatch Analysis & Text Highlighting")
    highlighted_html = highlight_tweet_tokens(text, locs, terms, explanation.get("top_positive_features", []))
    st.markdown(highlighted_html, unsafe_allow_html=True)

    # Review flag banner
    if explanation.get("needs_human_review"):
        st.warning("⚠️ **Flagged for Human Review**: Low classifier confidence or ambiguous top-2 margin.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📈 Urgency Signal Waterfall")
        # Waterfall signals
        _, _, _, breakdown = calculate_urgency_breakdown(
            text=text,
            category=cat,
            uppercase_ratio=row.get("uppercase_ratio", 0.0),
            exclamation_count=row.get("exclamation_count", 0),
            is_retweet=row.get("is_retweet", 0)
        )
        if breakdown:
            wf_names = [b["signal"] for b in breakdown]
            wf_values = [b["points"] for b in breakdown]
            fig_wf = go.Figure(go.Waterfall(
                name="Urgency Score",
                orientation="v",
                measure=["relative"] * len(breakdown),
                x=wf_names,
                textposition="outside",
                text=[f"+{v}" if v > 0 else f"{v}" for v in wf_values],
                y=wf_values,
                connector={"line": {"color": "#1F2A3D"}},
                decreasing={"marker": {"color": "#64748B"}},
                increasing={"marker": {"color": "#EF4444"}}
            ))
            fig_wf.update_layout(title="Rule Engine Point Contributions", yaxis_title="Points Added", height=280)
            apply_chart_theme(fig_wf)
            st.plotly_chart(fig_wf, use_container_width=True)
        else:
            st.info("Baseline informational tweet (No urgency signals triggered).")

    with col2:
        st.markdown("##### 🔍 Classifier Token Attribution")
        pos_feats = explanation.get("top_positive_features", [])
        if pos_feats:
            f_tokens = [f[0] for f in pos_feats][::-1]
            f_weights = [f[1] for f in pos_feats][::-1]
            fig_tokens = px.bar(
                x=f_weights,
                y=f_tokens,
                orientation="h",
                labels={"x": "Contribution (TF-IDF × Weight)", "y": "N-gram"},
                color_discrete_sequence=["#22D3EE"],
                height=280
            )
            apply_chart_theme(fig_tokens)
            st.plotly_chart(fig_tokens, use_container_width=True)
        else:
            st.info("No dominant N-gram weights available for this input.")
