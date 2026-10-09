"""
Human-in-the-Loop Review Queue page for CrisisLens.
Allows crisis response coordinators to audit low-confidence dispatches,
correct labels, and persist feedback to SQLite for retraining.
"""

import pandas as pd
import streamlit as st

from src.config import PROJECT_CATEGORIES
from src.database import get_feedback_stats, get_reviewed_tweet_ids, load_all_feedback, save_feedback
from app.ui.components import (
    render_callout,
    render_empty,
    render_kpi_card,
    render_triage_detail_panel,
    set_flash,
    show_flash,
)

REVIEW_CONFIDENCE_THRESHOLD = 0.75
QUEUE_SIZE = 50
URGENCY_OPTIONS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

def _use_training_pool() -> None:
    st.session_state["filter_split"] = "train"

def _review_priority(df: pd.DataFrame) -> pd.Series:
    """How much a review is worth: uncertain predictions on urgent tweets come first."""
    urgency = df["urgency_score"].fillna(0) if "urgency_score" in df.columns else 0.0
    return (1.0 - df["confidence"].fillna(0)) * (1.0 + urgency / 10.0)

def _queue_table(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Priority": rows["review_priority"].round(2).to_numpy(),
        "Confidence": rows["confidence"].to_numpy(dtype=float),
        "Category": rows["category"].to_numpy(),
        "Urgency": rows["urgency_level"].to_numpy(),
        "Split": rows["split"].to_numpy() if "split" in rows.columns else "",
        "Report": rows["raw_unescaped_text"].astype(str).to_numpy(),
    })

def render_page(df: pd.DataFrame, pipeline: any):
    show_flash()
    st.markdown("<div class='cl-muted' style='margin-bottom:1rem;'>Operator oversight workflow for ambiguous, low-confidence, or boundary dispatches. Uncertain predictions on urgent tweets are listed first.</div>", unsafe_allow_html=True)

    # Feedback Stats Banner
    stats = get_feedback_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Tweets reviewed", f"{stats['unique_tweets_reviewed']:,}", f"{stats['total_feedback_count']:,} submissions logged")
    with c2:
        render_kpi_card("Category corrections", f"{stats['category_corrections']:,}", "Taxonomy adjustments")
    with c3:
        render_kpi_card("Usable for retraining", f"{stats['usable_for_training']:,}", "Train split / live feedback")
    with c4:
        render_kpi_card("Evaluation only", f"{stats['evaluation_only']:,}", "Test-split corrections")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    active_split = st.session_state.get("filter_split", "test")
    if active_split in ("test", "all"):
        render_callout(
            "Corrections on test-split tweets are logged but never used to retrain the model, so the benchmark stays clean. "
            "To create feedback that improves the model, review tweets from the train split.",
            "info", f"You are reviewing the {active_split} split."
        )
        st.button("Switch to the train split", on_click=_use_training_pool, key="review_use_train")

    if df.empty:
        render_empty("No dispatches found in view.")
        return

    # Hide tweets that already have feedback, then rank what is left
    already_reviewed = get_reviewed_tweet_ids()
    pool = df[~df["tweet_id"].astype(str).isin(already_reviewed)].copy()
    hidden = len(df) - len(pool)
    if pool.empty:
        render_empty(f"Every dispatch in view has already been reviewed ({hidden:,}). Widen the sidebar filters to find more.")
    else:
        uncertain = pool[pool["confidence"] < REVIEW_CONFIDENCE_THRESHOLD]
        candidates = uncertain if not uncertain.empty else pool
        candidates = candidates.assign(review_priority=_review_priority(candidates))
        reviewable = candidates.sort_values("review_priority", ascending=False, kind="mergesort").head(QUEUE_SIZE).reset_index(drop=True)

        st.markdown(
            f"**Dispatches pending review**: `{len(candidates):,}` (showing the top `{len(reviewable)}`)"
            + (f" · `{hidden:,}` already reviewed are hidden" if hidden else "")
        )

        event = st.dataframe(
            _queue_table(reviewable), key="review_table", hide_index=True, height=300,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "Priority": st.column_config.NumberColumn(width="small", help="(1 - confidence) x (1 + urgency/10)"),
                "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent", width="small"),
                "Report": st.column_config.TextColumn(width="large"),
            },
        )
        selected = event.selection.rows if event is not None and event.selection else []
        sel_idx = selected[0] if selected and selected[0] < len(reviewable) else 0
        if not selected:
            st.caption("Showing the highest-priority dispatch. Select any row above to review a different one.")

        target_row = reviewable.iloc[sel_idx]
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        render_triage_detail_panel(target_row, pipeline)

        st.markdown("<div class='cl-section'>Submit Operator Feedback & Label Correction</div>", unsafe_allow_html=True)
        split = str(target_row.get("split", ""))
        if split == "test":
            st.caption("This is a test-split tweet: your correction is recorded for evaluation but will not be trained on.")

        with st.form(f"feedback_form_{target_row['tweet_id']}"):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                cur_cat = target_row.get("category", "General")
                cat_idx = PROJECT_CATEGORIES.index(cur_cat) if cur_cat in PROJECT_CATEGORIES else 0
                new_cat = st.selectbox("Corrected category", PROJECT_CATEGORIES, index=cat_idx)
            with f_col2:
                cur_urg = target_row.get("urgency_level", "LOW")
                urg_idx = URGENCY_OPTIONS.index(cur_urg) if cur_urg in URGENCY_OPTIONS else 0
                new_urg = st.selectbox("Corrected urgency level", URGENCY_OPTIONS, index=urg_idx)

            n_col1, n_col2 = st.columns([3, 1.4])
            with n_col1:
                notes = st.text_input("Reviewer notes (optional)", placeholder="e.g., Ambiguous phrasing, confirmed casualty on news")
            with n_col2:
                reviewer = st.text_input("Your name / ID", value=st.session_state.get("reviewer_id", ""), placeholder="optional")
            submit = st.form_submit_button("Save review", type="primary")
            st.caption("If you leave both fields unchanged, this is recorded as a confirmation that the model was right.")

            if submit:
                st.session_state["reviewer_id"] = reviewer
                save_feedback(
                    tweet_id=str(target_row["tweet_id"]),
                    original_category=str(cur_cat),
                    corrected_category=str(new_cat),
                    original_urgency=str(cur_urg),
                    corrected_urgency=str(new_urg),
                    user_notes=notes,
                    tweet_text=str(target_row.get("raw_unescaped_text", "")),
                    event=str(target_row.get("event", "")),
                    split=split,
                    reviewer=reviewer,
                    model_version=str(target_row.get("model_version", "") or ""),
                )
                changed = (new_cat != cur_cat) or (new_urg != cur_urg)
                if split == "test":
                    msg = f"Saved review for tweet {target_row['tweet_id']} (test split: evaluation only)."
                elif changed:
                    msg = f"Correction saved for tweet {target_row['tweet_id']}. It will be used the next time the model is retrained."
                else:
                    msg = f"Confirmation saved for tweet {target_row['tweet_id']}."
                set_flash(msg)
                st.rerun()

    # View logged feedback history
    st.markdown("<div class='cl-section'>Recent Operator Review Log</div>", unsafe_allow_html=True)
    fb_df = load_all_feedback()
    if fb_df.empty:
        render_empty("No operator review records logged yet.")
    else:
        columns = [c for c in ("timestamp", "tweet_id", "original_category", "corrected_category",
                               "original_urgency", "corrected_urgency", "split", "reviewer", "user_notes") if c in fb_df.columns]
        st.dataframe(fb_df[columns].head(15), hide_index=True, width="stretch")
