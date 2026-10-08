"""
Human-in-the-Loop Review Queue page for CrisisLens.
Allows crisis response coordinators to audit low-confidence dispatches,
correct labels, and persist feedback to SQLite for retraining.
"""

import streamlit as st
import pandas as pd

from src.config import PROJECT_CATEGORIES
from src.database import save_feedback, get_feedback_stats, load_all_feedback
from app.ui.components import render_triage_detail_panel

def render_page(df: pd.DataFrame, pipeline: any):
    st.markdown("<h3 style='margin-top:0;color:#E6EDF7;'>👥 Human-in-the-Loop Review Queue</h3>", unsafe_allow_html=True)
    st.markdown("Operator oversight workflow for ambiguous, low-confidence, or boundary dispatches.")

    # Feedback Stats Banner
    stats = get_feedback_stats()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Operator Reviews", f"{stats['total_feedback_count']:,}")
    with c2:
        st.metric("Category Corrections", f"{stats['category_corrections']:,}")
    with c3:
        st.metric("Urgency Corrections", f"{stats['urgency_corrections']:,}")

    st.markdown("<hr style='margin:0.5rem 0 1rem 0;border-color:#1F2A3D;'>", unsafe_allow_html=True)

    if df.empty:
        st.warning("No dispatches found in view.")
        return

    # Filter for low confidence dispatches (< 0.70) or unverified
    reviewable = df[df["confidence"] < 0.75].sort_values("confidence", ascending=True).head(50).reset_index(drop=True)
    if reviewable.empty:
        reviewable = df.sort_values("confidence", ascending=True).head(20).reset_index(drop=True)

    st.markdown(f"**Dispatches Pending Review**: `{len(reviewable)}` (sorted by lowest model confidence)")

    sel_idx = st.selectbox(
        "Select Dispatch to Review:",
        options=list(range(len(reviewable))),
        format_func=lambda i: f"#{i+1} | Conf: {reviewable.loc[i, 'confidence']:.0%} | {reviewable.loc[i, 'category']} | {reviewable.loc[i, 'raw_unescaped_text'][:80]}..."
    )

    if sel_idx is not None and not reviewable.empty:
        target_row = reviewable.iloc[sel_idx]
        st.markdown("---")

        render_triage_detail_panel(target_row, pipeline)

        st.markdown("#### ✏️ Submit Operator Feedback & Label Correction")
        with st.form(f"feedback_form_{target_row['tweet_id']}"):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                cur_cat = target_row.get("category", "General")
                cat_opts = PROJECT_CATEGORIES
                cat_idx = cat_opts.index(cur_cat) if cur_cat in cat_opts else 0
                new_cat = st.selectbox("Corrected Category", cat_opts, index=cat_idx)
            with f_col2:
                cur_urg = target_row.get("urgency_level", "LOW")
                urg_opts = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
                urg_idx = urg_opts.index(cur_urg) if cur_urg in urg_opts else 0
                new_urg = st.selectbox("Corrected Urgency Level", urg_opts, index=urg_idx)

            notes = st.text_input("Reviewer Notes (Optional)", placeholder="e.g., Ambiguous wording, confirmed casualty report on local news")
            submit = st.form_submit_button("💾 Save Correction to Database", type="primary")

            if submit:
                save_feedback(
                    tweet_id=str(target_row["tweet_id"]),
                    original_category=str(cur_cat),
                    corrected_category=str(new_cat),
                    original_urgency=str(cur_urg),
                    corrected_urgency=str(new_urg),
                    user_notes=notes
                )
                st.success(f"Correction saved for Tweet ID {target_row['tweet_id']}! Available for model retraining.")
                st.rerun()

    # View logged feedback history
    st.markdown("---")
    st.markdown("##### 📜 Recent Operator Review Log")
    fb_df = load_all_feedback()
    if not fb_df.empty:
        st.dataframe(fb_df.head(15), use_container_width=True)
    else:
        st.info("No operator review records logged yet.")
