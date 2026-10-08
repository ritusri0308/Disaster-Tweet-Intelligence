"""
Priority Triage Queue and Explainability page for CrisisLens.
Ranked emergency dispatch queue with row selection and granular decision attribution.
"""

import streamlit as st
import pandas as pd
from app.ui.components import render_triage_detail_panel

def render_page(df: pd.DataFrame, pipeline: any):
    st.markdown("<h3 style='margin-top:0;color:#E6EDF7;'>🚨 Priority Triage Queue & Explainability</h3>", unsafe_allow_html=True)
    st.markdown("Ranked emergency queue ordered by urgency score. Inspect individual dispatches to audit AI and rule engine attribution.")

    if df.empty:
        st.warning("No dispatches match the current filter criteria.")
        return

    # Sort queue by urgency_score descending
    sorted_df = df.sort_values(["urgency_score", "confidence"], ascending=[False, False]).copy()

    # Filters within Triage
    col_search, col_need = st.columns([3, 1])
    with col_search:
        queue_search = st.text_input("Filter Triage Dispatches", placeholder="Search text or location in queue...", label_visibility="collapsed")
    with col_need:
        need_filter = st.selectbox(
            "Need Type Filter",
            ["All Needs", "Medical", "Water", "Food", "Shelter", "Rescue/Evacuation", "Power/Utilities", "Transport/Roads", "Communication", "Missing persons", "Donations/Volunteering"],
            label_visibility="collapsed"
        )

    if queue_search.strip():
        q = queue_search.strip().lower()
        sorted_df = sorted_df[
            sorted_df["clean_text"].str.lower().str.contains(q, na=False) |
            sorted_df["raw_unescaped_text"].str.lower().str.contains(q, na=False)
        ]

    if need_filter != "All Needs" and "needs" in sorted_df.columns:
        sorted_df = sorted_df[sorted_df["needs"].apply(lambda n_list: need_filter in (n_list if isinstance(n_list, list) else []))]

    st.markdown(f"**Triage Dispatches in Queue**: `{len(sorted_df):,}`")

    # Selection index
    top_subset = sorted_df.head(100).reset_index(drop=True)

    # Render interactive table
    selected_idx = st.selectbox(
        "Select Dispatch to Inspect (Top 100 in Queue):",
        options=list(range(len(top_subset))),
        format_func=lambda i: f"#{i+1} | [{top_subset.loc[i, 'urgency_level']}] Score {top_subset.loc[i, 'urgency_score']} | {top_subset.loc[i, 'category']} | {top_subset.loc[i, 'raw_unescaped_text'][:70]}..."
    )

    if selected_idx is not None and not top_subset.empty:
        selected_row = top_subset.iloc[selected_idx]
        st.markdown("---")
        render_triage_detail_panel(selected_row, pipeline)

    # Downloads
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        csv_data = sorted_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Triage Queue as CSV",
            data=csv_data,
            file_name="crisislens_triage_queue.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        # Self-contained HTML report
        from src.briefing import generate_html_briefing, generate_situation_summary
        summary = generate_situation_summary(sorted_df)
        html_doc = generate_html_briefing(sorted_df, summary, [], [], {})
        st.download_button(
            "📄 Download Triage Report as HTML",
            data=html_doc.encode("utf-8"),
            file_name="crisislens_triage_report.html",
            mime="text/html",
            use_container_width=True
        )
