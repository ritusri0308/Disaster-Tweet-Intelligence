"""
Priority Triage Queue and Explainability page for CrisisLens.
Ranked emergency dispatch queue with row selection and granular decision attribution.
"""

import pandas as pd
import streamlit as st

from src.config import NEED_TAXONOMY
from src.utils import as_list
from app.ui.cache import cached_briefing_html, cached_csv, df_signature
from app.ui.components import describe_active_filters, render_empty, render_triage_detail_panel

QUEUE_ROWS = 200

SORT_OPTIONS = {
    "Urgency (highest first)": (["urgency_score", "confidence"], [False, False]),
    "Confidence (lowest first)": (["confidence", "urgency_score"], [True, False]),
    "Newest first": (["timestamp", "urgency_score"], [False, False]),
}

def _queue_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Compact display table for the queue (one row per tweet)."""
    return pd.DataFrame({
        "Rank": range(1, len(rows) + 1),
        "Level": rows["urgency_level"].to_numpy(),
        "Score": rows["urgency_score"].to_numpy(dtype=float),
        "Category": rows["category"].to_numpy(),
        "Confidence": rows["confidence"].to_numpy(dtype=float),
        "Places": [", ".join(str(p) for p in as_list(v)[:2]) for v in rows["locations"]] if "locations" in rows else "",
        "Needs": [", ".join(str(n) for n in as_list(v)[:3]) for v in rows["needs"]] if "needs" in rows else "",
        "Report": rows["raw_unescaped_text"].astype(str).to_numpy(),
    })

def render_page(df: pd.DataFrame, pipeline: any):
    st.markdown("<div class='cl-muted' style='margin-bottom:1rem;'>Ranked emergency queue. Select a row to audit the machine-learning classification and rule-engine signal attributions behind it.</div>", unsafe_allow_html=True)

    if df.empty:
        render_empty("No tweets match the current filter criteria.")
        return

    # Filters within Triage
    col_search, col_need, col_sort = st.columns([3, 1.4, 1.6])
    with col_search:
        queue_search = st.text_input("Filter triage reports", placeholder="Search text or location in queue...", label_visibility="collapsed")
    with col_need:
        need_filter = st.selectbox("Need type filter", ["All Needs", *NEED_TAXONOMY], label_visibility="collapsed")
    with col_sort:
        sort_label = st.selectbox("Sort queue", list(SORT_OPTIONS), label_visibility="collapsed")

    sort_cols, ascending = SORT_OPTIONS[sort_label]
    sort_cols = [c for c in sort_cols if c in df.columns]
    sorted_df = df.sort_values(sort_cols, ascending=ascending[:len(sort_cols)], kind="mergesort").copy()

    if queue_search.strip():
        q = queue_search.strip().lower()
        mask = pd.Series(False, index=sorted_df.index)
        for col in ("clean_text", "raw_unescaped_text"):
            if col in sorted_df.columns:
                mask |= sorted_df[col].astype(str).str.lower().str.contains(q, na=False, regex=False)
        if "locations" in sorted_df.columns:
            mask |= sorted_df["locations"].apply(lambda locs: any(q in str(loc).lower() for loc in as_list(locs)))
        sorted_df = sorted_df[mask]

    if need_filter != "All Needs" and "needs" in sorted_df.columns:
        sorted_df = sorted_df[sorted_df["needs"].apply(lambda needs: need_filter in as_list(needs))]

    if sorted_df.empty:
        render_empty("No reports in the queue match this search. Clear the search or the need filter.")
        return

    shown = min(len(sorted_df), QUEUE_ROWS)
    st.markdown(
        f"**Triage reports in queue**: `{len(sorted_df):,}`"
        + (f" · showing the first `{shown}`" if shown < len(sorted_df) else "")
    )

    top_subset = sorted_df.head(QUEUE_ROWS).reset_index(drop=True)
    event = st.dataframe(
        _queue_table(top_subset),
        key="triage_table",
        hide_index=True,
        height=380,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Level": st.column_config.TextColumn(width="small"),
            "Score": st.column_config.ProgressColumn("Urgency", min_value=0, max_value=10, format="%.1f", width="small"),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent", width="small"),
            "Report": st.column_config.TextColumn(width="large"),
        },
    )

    selected_rows = event.selection.rows if event is not None and event.selection else []
    selected_idx = selected_rows[0] if selected_rows and selected_rows[0] < len(top_subset) else 0
    if not selected_rows:
        st.caption("Showing the top-ranked report. Select any row above to inspect it.")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    render_triage_detail_panel(top_subset.iloc[selected_idx], pipeline)

    # Downloads (generated from cached results, so they do not slow down every click)
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    sig = df_signature(sorted_df)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download triage queue as CSV",
            data=cached_csv(sorted_df, sig),
            file_name="crisislens_triage_queue.csv",
            mime="text/csv",
            width="stretch",
            on_click="ignore"
        )
    with c2:
        st.download_button(
            "Download triage report as HTML",
            data=cached_briefing_html(sorted_df, sig, describe_active_filters(queue_search)),
            file_name="crisislens_triage_report.html",
            mime="text/html",
            width="stretch",
            on_click="ignore"
        )
