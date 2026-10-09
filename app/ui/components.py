"""
Reusable UI components for CrisisLens.
Clean, modern light interface components. No hard-coded hex colours (see app/ui/theme.py) and every
piece of tweet-derived text is HTML-escaped before it reaches ``unsafe_allow_html``.
"""

import html
from typing import Any, Dict, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import APP_NAME, PROJECT_CATEGORIES
from src.entities import format_quantity
from src.explainability import explain_classification, highlight_tweet_tokens
from src.urgency import calculate_urgency_breakdown
from src.utils import as_list
from app.ui.theme import (
    apply_chart_theme,
    ACCENT_PRIMARY,
    CONNECTOR_LINE,
    DANGER,
    SUCCESS,
)

# Readable names for event ids whose title-cased form would be wrong
_EVENT_NAME_OVERRIDES = {"srilanka": "Sri Lanka"}

def esc(value: Any) -> str:
    """HTML-escape any value (None becomes an empty string)."""
    return html.escape("" if value is None else str(value))

def esc_text(value: Any) -> str:
    """
    Escape tweet text for embedding in an HTML block passed to ``st.markdown``.

    Markdown ends an HTML block at the first blank line, which used to split any multi-line tweet out of its
    card, so newlines become ``<br>`` (and the caller must not introduce blank lines of its own).
    """
    return esc(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")

def humanize_event(event_id: Any) -> str:
    """'hurricane_harvey_2017' -> 'Hurricane Harvey 2017'. Non-event labels pass through unchanged."""
    text = str(event_id)
    if "_" not in text:
        return text
    words = []
    for word in text.split("_"):
        words.append(_EVENT_NAME_OVERRIDES.get(word, word.capitalize() if not word.isdigit() else word))
    return " ".join(words)

def badge_slug(label: Any) -> str:
    """CSS class suffix for a category label: 'Information/Update' -> 'information-update'."""
    return str(label).lower().replace(" ", "-").replace("/", "-")

def format_timestamp(value: Any) -> str:
    """Compact 'YYYY-MM-DD HH:MM' text for a timestamp-like value."""
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(value)[:16]

def render_callout(text: str, kind: str = "info", title: Optional[str] = None) -> None:
    """Coloured callout box. ``kind`` is one of info / warn / danger / ok. Text is escaped."""
    head = f"<strong>{esc(title)}</strong> " if title else ""
    st.markdown(f"<div class='cl-alert cl-alert--{esc(kind)}'>{head}{esc(text)}</div>", unsafe_allow_html=True)

def render_empty(message: str) -> None:
    """Standard empty-state card."""
    st.markdown(f"<div class='cl-card cl-muted'>{esc(message)}</div>", unsafe_allow_html=True)

def set_flash(message: str, kind: str = "success") -> None:
    """Queue a message that survives the next rerun (st.success followed by st.rerun() is never seen)."""
    st.session_state["_flash"] = (kind, message)

def show_flash() -> None:
    """Show and clear any queued flash message."""
    flash = st.session_state.pop("_flash", None)
    if not flash:
        return
    kind, message = flash
    {"success": st.success, "warning": st.warning, "error": st.error}.get(kind, st.info)(message)
    st.toast(message)

def render_feed_item(row: Any, timestamp: Any = None, show_places: bool = True, show_figures: bool = True) -> None:
    """One tweet in a feed: severity + category badges, places, reported figures, escaped text."""
    urg = str(row.get("urgency_level", "LOW")).upper()
    cat = str(row.get("category", "General"))
    score = row.get("urgency_score", 0)

    chips = ""
    if show_places:
        chips += "".join(f"<span class='cl-chip'>{esc(p)}</span>" for p in as_list(row.get("locations"))[:3])
    if show_figures:
        chips += "".join(
            f"<span class='cl-chip cl-chip--figure'>{esc(format_quantity(q))}</span>"
            for q in as_list(row.get("quantities")) if isinstance(q, dict)
        )

    ts_html = f"<div class='cl-mono cl-muted'>{esc(format_timestamp(timestamp))}</div>" if timestamp is not None else ""
    # One line, no blank lines or indentation: anything else lets Markdown end the HTML block early.
    st.markdown(
        "<div class='cl-feed-item'>"
        "<div class='cl-feed-head'><div>"
        f"<span class='cl-badge cl-badge--{esc(urg.lower())}'>{esc(urg)} {esc(score)}/10</span>"
        f"<span class='cl-badge cl-badge--{badge_slug(cat)}'>{esc(cat)}</span>"
        f"{chips}"
        f"</div>{ts_html}</div>"
        f"<div class='cl-feed-text'>{esc_text(row.get('raw_unescaped_text', row.get('tweet_text', '')))}</div>"
        "</div>",
        unsafe_allow_html=True
    )

def render_alert_card(alert: Dict[str, Any]) -> None:
    """A surge alert: type and event, details, window, place and top terms."""
    title = f"{str(alert.get('alert_type', 'Alert')).replace('_', ' ').title()} · {humanize_event(alert.get('event', ''))}"
    place = alert.get("place")
    place_part = f" · Place: {esc(place)}" if place and place != "All" else ""
    terms = ", ".join(str(t) for t in alert.get("top_terms", []))
    terms_part = f" · Terms: {esc(terms)}" if terms else ""
    st.markdown(
        "<div class='cl-card' style='margin-bottom:0.6rem;'>"
        f"<div class='cl-alert-card__title'>{esc(title)}</div>"
        f"<div class='cl-alert-card__body'>{esc(alert.get('details', ''))}</div>"
        f"<div class='cl-muted' style='margin-top:4px;'>Window: {esc(format_timestamp(alert.get('window_start')))}{place_part}{terms_part}</div>"
        "</div>",
        unsafe_allow_html=True
    )

def render_model_banner(model_status: Dict[str, Any], dataset_state: Dict[str, Any]) -> None:
    """
    Warn when the loaded model is missing/implausible, or when the dataset on screen was scored by a
    different model than the one loaded (so the predictions shown and new analyses disagree).
    """
    status = model_status.get("status")
    if status == "missing":
        render_callout(model_status.get("reason", ""), "danger", "No trained model.")
    elif status == "degraded":
        render_callout(model_status.get("reason", ""), "warn", "Model looks wrong.")

    state = dataset_state.get("state")
    if state == "stale":
        render_callout(
            f"The dataset was scored by model {', '.join(dataset_state.get('dataset_versions', []))} but model "
            f"{model_status.get('version')} is loaded. Re-run `python run_pipeline.py` so the numbers on screen match the loaded model.",
            "warn", "Dataset and model are out of sync."
        )
    elif state == "unstamped":
        render_callout(
            "This dataset has no model version stamp, so it cannot be checked against the loaded model. "
            "Re-run `python run_pipeline.py` to stamp it.",
            "info", "Dataset predates model versioning."
        )

def render_header(df_view: pd.DataFrame, page_title: str = "", model_version: Optional[str] = None):
    """Render top product header with plain title and single calm muted status line."""
    active_split = st.session_state.get("filter_split", "test")
    view_count = len(df_view)

    title_display = page_title if page_title else APP_NAME
    version_part = f" · model {model_version}" if model_version else ""
    st.markdown(f"<div class='cl-page-title'>{esc(title_display)}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='cl-muted' style='margin-bottom:1.25rem;'>Historical data 2016–2019 · {esc(active_split)} split · "
        f"{view_count:,} tweets in view{esc(version_part)}</div>",
        unsafe_allow_html=True
    )

FILTER_DEFAULTS = {
    "filter_split": "test",
    "filter_event": "All Events",
    "filter_cat": "All Categories",
    "filter_urg": "All Urgency Levels",
    "filter_search": "",
    "filter_min_conf": 0.0,
    "filter_hide_rt": False
}

def describe_active_filters(query: str = "") -> str:
    """Human-readable summary of the non-default sidebar filters (and an optional search) for exports."""
    ss = st.session_state
    parts = []
    if ss.get("filter_split", "test") != "all":
        parts.append(f"Split: {ss.get('filter_split', 'test')}")
    if ss.get("filter_event", "All Events") != "All Events":
        parts.append(f"Event: {humanize_event(ss['filter_event'])}")
    if ss.get("filter_cat", "All Categories") != "All Categories":
        parts.append(f"Category: {ss['filter_cat']}")
    if ss.get("filter_urg", "All Urgency Levels") != "All Urgency Levels":
        parts.append(f"Urgency: {ss['filter_urg']}")
    if str(ss.get("filter_search", "")).strip():
        parts.append(f"Keyword: {ss['filter_search'].strip()}")
    if float(ss.get("filter_min_conf", 0.0)) > 0:
        parts.append(f"Min confidence: {ss['filter_min_conf']:.0%}")
    if ss.get("filter_hide_rt"):
        parts.append("Retweets hidden")
    if query.strip():
        parts.append(f"Search: \"{query.strip()}\"")
    return "; ".join(parts)

def _place_matches(locations: Any, needle: str) -> bool:
    return any(needle in str(loc).lower() for loc in as_list(locations))

def render_global_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render global sidebar filters in clean light design."""
    for k, v in FILTER_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if df.empty:
        st.sidebar.info("No processed data loaded yet.")
        return df

    filtered = df.copy()

    with st.sidebar.expander("Filters", expanded=True):
        # 1. Split selector (default: test)
        splits = ["test", "train", "dev", "all"]
        st.session_state["filter_split"] = st.selectbox(
            "Data split",
            splits,
            index=splits.index(st.session_state["filter_split"]) if st.session_state["filter_split"] in splits else 0,
            help="Official benchmark split partition. 'test' ensures out-of-sample evaluation."
        )

        # Filter by split first
        if st.session_state["filter_split"] != "all" and "split" in filtered.columns:
            filtered = filtered[filtered["split"] == st.session_state["filter_split"]]

        # 2. Disaster Event
        events = ["All Events"] + sorted(list(filtered["event"].unique())) if "event" in filtered.columns else ["All Events"]
        current_ev = st.session_state["filter_event"]
        ev_idx = events.index(current_ev) if current_ev in events else 0
        st.session_state["filter_event"] = st.selectbox(
            "Disaster event", events, index=ev_idx,
            format_func=lambda e: e if e == "All Events" else humanize_event(e)
        )

        if st.session_state["filter_event"] != "All Events" and "event" in filtered.columns:
            filtered = filtered[filtered["event"] == st.session_state["filter_event"]]

        # 3. Category Filter
        cats = ["All Categories"] + PROJECT_CATEGORIES
        current_cat = st.session_state["filter_cat"]
        cat_idx = cats.index(current_cat) if current_cat in cats else 0
        st.session_state["filter_cat"] = st.selectbox("Category", cats, index=cat_idx)

        if st.session_state["filter_cat"] != "All Categories" and "category" in filtered.columns:
            filtered = filtered[filtered["category"] == st.session_state["filter_cat"]]

        # 4. Urgency Filter
        urgencies = ["All Urgency Levels", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
        current_urg = st.session_state["filter_urg"]
        urg_idx = urgencies.index(current_urg) if current_urg in urgencies else 0
        st.session_state["filter_urg"] = st.selectbox("Urgency level", urgencies, index=urg_idx)

        if st.session_state["filter_urg"] != "All Urgency Levels" and "urgency_level" in filtered.columns:
            filtered = filtered[filtered["urgency_level"] == st.session_state["filter_urg"]]

        # Nested expander for secondary filters
        with st.expander("More filters", expanded=False):
            st.session_state["filter_search"] = st.text_input(
                "Keyword search",
                value=st.session_state["filter_search"],
                placeholder="Search text or place..."
            )

            st.session_state["filter_min_conf"] = st.slider(
                "Minimum confidence",
                0.0, 1.0, float(st.session_state["filter_min_conf"]), 0.05
            )

            st.session_state["filter_hide_rt"] = st.checkbox(
                "Hide retweets",
                value=st.session_state["filter_hide_rt"]
            )

        if st.button("Reset filters", width="stretch"):
            for k, v in FILTER_DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

    # Apply secondary filters
    if st.session_state["filter_search"].strip():
        q = st.session_state["filter_search"].strip().lower()
        mask = pd.Series(False, index=filtered.index)
        for col in ("raw_unescaped_text", "clean_text"):
            if col in filtered.columns:
                # regex=False: search terms like "c++" or "(" are plain text, not patterns
                mask |= filtered[col].astype(str).str.lower().str.contains(q, na=False, regex=False)
        if "locations" in filtered.columns:
            mask |= filtered["locations"].apply(lambda locs: _place_matches(locs, q))
        filtered = filtered[mask]

    if st.session_state["filter_min_conf"] > 0.0 and "confidence" in filtered.columns:
        filtered = filtered[filtered["confidence"] >= st.session_state["filter_min_conf"]]

    if st.session_state["filter_hide_rt"] and "is_retweet" in filtered.columns:
        filtered = filtered[filtered["is_retweet"] == 0]

    return filtered

def render_kpi_card(title: str, value: Any, delta: str = "", color: str = ""):
    """Render clean light-theme KPI card without top color bars."""
    delta_html = f"<div class='cl-kpi__delta'>{esc(delta)}</div>" if delta else ""
    card_html = f"""
    <div class='cl-kpi'>
        <div class='cl-kpi__label'>{esc(title)}</div>
        <div class='cl-kpi__value'>{esc(value)}</div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_triage_detail_panel(row: pd.Series, pipeline: Any):
    """
    Detail inspection panel for selected triage row:
    - Original text with light token highlights
    - Reported figures and detected needs
    - Urgency waterfall chart in light mode
    - Classifier token attributions
    """
    text = row.get("raw_unescaped_text", row.get("tweet_text", ""))
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    clean = row.get("clean_text", "")
    if not isinstance(clean, str):
        clean = ""
    locs = [str(loc) for loc in as_list(row.get("locations"))]
    terms = [str(t) for t in as_list(row.get("matched_terms"))]
    needs = [str(n) for n in as_list(row.get("needs"))]
    figures = [q for q in as_list(row.get("quantities")) if isinstance(q, dict)]
    # Error-analysis rows carry the model's guess as predicted_category rather than category
    cat = row.get("category") or row.get("predicted_category") or "General"
    conf = row.get("confidence", 1.0)

    # Explanation
    explanation = explain_classification(pipeline, clean) if pipeline is not None else {
        "predicted_class": cat, "confidence": conf, "top_positive_features": [], "top_negative_features": [], "probabilities": {cat: conf}, "needs_human_review": False
    }

    st.markdown("<div class='cl-section'>Text Analysis & Token Highlights</div>", unsafe_allow_html=True)
    highlighted_html = highlight_tweet_tokens(text, locs, terms, explanation.get("top_positive_features", []))
    st.markdown(highlighted_html, unsafe_allow_html=True)
    st.markdown(
        "<div class='cl-muted' style='margin-top:6px;'>"
        "<span class='cl-hl cl-hl--loc'>place</span> "
        "<span class='cl-hl cl-hl--urg'>urgency trigger</span> "
        "<span class='cl-hl cl-hl--feat'>classifier signal</span></div>",
        unsafe_allow_html=True
    )

    detail_chips = "".join(f"<span class='cl-chip cl-chip--figure'>{esc(format_quantity(q))}</span>" for q in figures)
    detail_chips += "".join(f"<span class='cl-chip cl-chip--active'>{esc(n)}</span>" for n in needs)
    if detail_chips:
        st.markdown(f"<div style='margin-top:0.75rem;'>{detail_chips}</div>", unsafe_allow_html=True)

    probs = explanation.get("probabilities") or {}
    if probs:
        top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        st.markdown(
            "<div class='cl-muted' style='margin-top:0.5rem;'>Model probabilities: "
            + " · ".join(f"{esc(k)} {v:.0%}" for k, v in top3) + "</div>",
            unsafe_allow_html=True
        )

    if explanation.get("needs_human_review"):
        st.warning("Flagged for human review: Low classifier confidence or close top-2 margin.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='cl-section'>Urgency Signal Contributions</div>", unsafe_allow_html=True)
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
                connector={"line": {"color": CONNECTOR_LINE}},
                decreasing={"marker": {"color": SUCCESS}},
                increasing={"marker": {"color": DANGER}}
            ))
            fig_wf.update_layout(title="Rule engine points added", yaxis_title="Points", height=280)
            apply_chart_theme(fig_wf)
            st.plotly_chart(fig_wf, width="stretch")
        else:
            st.info("Baseline informational tweet (No urgency signals triggered).")

    with col2:
        st.markdown("<div class='cl-section'>Classifier Token Attribution</div>", unsafe_allow_html=True)
        pos_feats = explanation.get("top_positive_features", [])
        if pos_feats and not explanation.get("is_surrogate"):
            f_tokens = [f[0] for f in pos_feats][::-1]
            f_weights = [f[1] for f in pos_feats][::-1]
            fig_tokens = px.bar(
                x=f_weights,
                y=f_tokens,
                orientation="h",
                labels={"x": "Contribution (TF-IDF × Weight)", "y": "Feature"},
                color_discrete_sequence=[ACCENT_PRIMARY],
                height=280
            )
            apply_chart_theme(fig_tokens)
            st.plotly_chart(fig_tokens, width="stretch")
        elif explanation.get("is_surrogate"):
            st.info("The active classifier is not a linear model, so per-token weights are not available.")
        else:
            st.info("No dominant feature weights available for this input.")
