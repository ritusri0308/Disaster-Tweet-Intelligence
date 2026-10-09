"""
Situation Room page for CrisisLens.
Top command center displaying natural-language search bar, executive KPIs, an activity timeline with
surge markers, active surge alerts, clustered incidents, needs and place breakdowns, and prioritized dispatches.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import get_activity_timeseries, get_needs_counts, get_top_locations
from src.nlquery import apply_query_filters, parse_natural_language_query
from app.ui.cache import cached_briefing_html, cached_incidents, cached_summary, cached_surges, df_signature
from app.ui.components import (
    describe_active_filters,
    esc,
    esc_text,
    humanize_event,
    render_alert_card,
    render_callout,
    render_empty,
    render_feed_item,
    render_kpi_card,
)
from app.ui.theme import ACCENT_PRIMARY, DANGER, SURFACE_COLOR, apply_chart_theme

FEED_LIMIT_KEY = "situation_feed_limit"

def _focus_event(event: str) -> None:
    """Narrow the whole dashboard to one event (runs as a button callback, before the sidebar re-renders)."""
    st.session_state["filter_event"] = event

def _render_activity_timeline(df: pd.DataFrame, surges: list) -> None:
    st.markdown("<div class='cl-section'>Activity timeline</div>", unsafe_allow_html=True)
    series, freq = get_activity_timeseries(df)
    if len(series) < 2:
        render_empty("Not enough timestamped tweets in view to draw a timeline.")
        return

    unit = "hour" if freq == "h" else "day"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["timestamp"], y=series["tweets"], mode="lines", name="All tweets",
        fill="tozeroy", line=dict(color=ACCENT_PRIMARY, width=2),
        hovertemplate="%{x}<br>%{y} tweets<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=series["timestamp"], y=series["priority"], mode="lines", name="Critical + high",
        line=dict(color=DANGER, width=2),
        hovertemplate="%{x}<br>%{y} critical/high<extra></extra>"
    ))

    # Mark the buckets in which a statistical surge was detected
    by_bucket = series.set_index("timestamp")["tweets"]
    marks_x, marks_y, marks_text = [], [], []
    for alert in surges[:25]:
        bucket = pd.Timestamp(alert["window_start"]).floor(freq)
        if bucket in by_bucket.index:
            marks_x.append(bucket)
            marks_y.append(int(by_bucket.loc[bucket]))
            marks_text.append(f"{alert['alert_type']} (z={alert['z_score']})<br>{humanize_event(alert['event'])}")
    if marks_x:
        fig.add_trace(go.Scatter(
            x=marks_x, y=marks_y, mode="markers", name="Detected surge",
            marker=dict(symbol="diamond", size=12, color=DANGER, line=dict(width=1.5, color=SURFACE_COLOR)),
            hovertext=marks_text, hoverinfo="text"
        ))

    apply_chart_theme(fig)
    fig.update_layout(height=300, yaxis_title=f"Tweets per {unit}", xaxis_title="Time (UTC)")
    st.plotly_chart(fig, width="stretch")

def _render_surges(surges: list, filtered_to_one_event: bool) -> None:
    st.markdown("<div class='cl-section'>Surge Alerts</div>", unsafe_allow_html=True)
    if not surges:
        render_empty("No abnormal volume spikes detected in current filtered view.")
        return
    for i, alert in enumerate(surges[:3]):
        render_alert_card(alert)
        if not filtered_to_one_event:
            st.button(
                f"Focus on {humanize_event(alert['event'])}", key=f"focus_{i}_{alert['event']}",
                on_click=_focus_event, args=(alert["event"],), type="tertiary"
            )
    if len(surges) > 3:
        with st.expander(f"Show all surge alerts ({len(surges)})"):
            for alert in surges[3:]:
                render_alert_card(alert)

def _render_incidents(incidents: list) -> None:
    st.markdown("<div class='cl-section'>Incident Groups</div>", unsafe_allow_html=True)
    if not incidents:
        render_empty("No multi-tweet incident clusters found in active view.")
        return
    for inc in incidents[:5]:
        places = ", ".join(str(p) for p in inc.get("places", [])) or "Area general"
        with st.expander(f"{inc['incident_id']} · {inc['dominant_category']} ({inc['size']} related tweets)"):
            st.markdown(
                f"<div class='cl-muted'>Max urgency <strong>{esc(inc['max_urgency'])}/10</strong> · "
                f"Locations: {esc(places)} · Window: {esc(inc['time_span'])}</div>"
                f"<div class='cl-feed-text' style='margin-top:6px;'>“{esc_text(inc['representative_tweet'])}”</div>",
                unsafe_allow_html=True
            )
    if len(incidents) > 5:
        # Streamlit does not allow expanders inside expanders, so the remainder is a table.
        with st.expander(f"Show all incident groups ({len(incidents)})"):
            table = pd.DataFrame([{
                "Incident": i["incident_id"], "Category": i["dominant_category"], "Tweets": i["size"],
                "Max urgency": i["max_urgency"], "Locations": ", ".join(str(p) for p in i.get("places", [])),
                "Window": i["time_span"], "Key report": i["representative_tweet"],
            } for i in incidents])
            st.dataframe(table, hide_index=True, width="stretch")

def _render_breakdowns(df: pd.DataFrame) -> None:
    col_needs, col_places = st.columns(2)
    with col_needs:
        st.markdown("<div class='cl-section'>Reported needs</div>", unsafe_allow_html=True)
        needs = get_needs_counts(df).head(8)
        if needs.empty:
            render_empty("No humanitarian needs detected in the current view.")
        else:
            fig = px.bar(x=needs.values[::-1], y=needs.index[::-1], orientation="h",
                         labels={"x": "Tweets", "y": ""}, color_discrete_sequence=[ACCENT_PRIMARY])
            apply_chart_theme(fig)
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")
    with col_places:
        st.markdown("<div class='cl-section'>Most mentioned places</div>", unsafe_allow_html=True)
        places = get_top_locations(df, top_n=8)
        if places.empty:
            render_empty("No places recognised in the current view.")
        else:
            fig = px.bar(x=places.values[::-1], y=places.index[::-1], orientation="h",
                         labels={"x": "Mentions", "y": ""}, color_discrete_sequence=[ACCENT_PRIMARY])
            apply_chart_theme(fig)
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")

def _render_priority_feed(df: pd.DataFrame) -> None:
    st.markdown("<div class='cl-section'>Priority Feed</div>", unsafe_allow_html=True)
    if FEED_LIMIT_KEY not in st.session_state:
        st.session_state[FEED_LIMIT_KEY] = 5

    crit_df = df[df["urgency_level"].isin(["CRITICAL", "HIGH"])].sort_values("urgency_score", ascending=False, kind="mergesort")
    if crit_df.empty:
        render_empty("No critical or high-priority tweets in the current filtered subset.")
        return

    for _, row in crit_df.head(st.session_state[FEED_LIMIT_KEY]).iterrows():
        render_feed_item(row, timestamp=row.get("timestamp"))

    if len(crit_df) > st.session_state[FEED_LIMIT_KEY]:
        if st.button("Show more", key="btn_show_more_feed"):
            st.session_state[FEED_LIMIT_KEY] += 10
            st.rerun()

def render_page(df: pd.DataFrame, pipeline: any):
    if df.empty:
        render_empty("No tweets match the current filters. Adjust or reset the sidebar filters.")
        return

    # 1. Natural Language Command Bar (Enter applies it)
    nl_query = st.text_input(
        "Search",
        key="situation_query",
        placeholder="Try: critical help requests in Puerto Rico · water needed in the last 48 hours",
        label_visibility="collapsed",
        help="Describe what you are looking for in plain language, then press Enter."
    )

    query_text = nl_query.strip()
    if query_text:
        base_count = len(df)
        parsed = parse_natural_language_query(nl_query)
        if parsed["is_valid"]:
            df, applied = apply_query_filters(df, parsed["filters"])
            chips_html = "".join(f"<span class='cl-chip cl-chip--active'>{esc(c)}</span>" for c in parsed["parsed_chips"])
            st.markdown(
                f"<div style='margin-bottom:1rem;'>{chips_html}"
                f"<span class='cl-muted'> &nbsp;{len(df):,} of {base_count:,} tweets match</span></div>",
                unsafe_allow_html=True
            )
            if df.empty:
                render_callout(
                    "Nothing matches this search inside the current sidebar filters. "
                    "Try resetting the sidebar filters (event, category, urgency) or loosening the query.",
                    "warn", "No results."
                )
                return
        else:
            st.info(parsed["message"])

    sig = df_signature(df)
    surges = cached_surges(df, sig)
    incidents = cached_incidents(df, sig)
    summary_data = cached_summary(df, sig)

    # 2. Executive KPIs Row (Exactly 4 cards)
    total_tweets = len(df)
    crit_count = int((df["urgency_level"] == "CRITICAL").sum())
    high_count = int((df["urgency_level"] == "HIGH").sum())
    help_count = int((df["category"] == "Help Request").sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Total tweets", f"{total_tweets:,}", "Active split")
    with k2:
        render_kpi_card("Critical alerts", f"{crit_count:,}", "Immediate review")
    with k3:
        render_kpi_card("High priority", f"{high_count:,}", "Escalated urgency")
    with k4:
        render_kpi_card("Help requests", f"{help_count:,}", "Assistance needed")

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    # 3. Situation Summary Card + Briefing Export
    col_sum, col_export = st.columns([3.5, 1.5])
    with col_sum:
        st.markdown(
            f"<div class='cl-card'><strong>Situation summary:</strong> {esc(summary_data['summary_text'])}</div>",
            unsafe_allow_html=True
        )
    with col_export:
        filters_desc = describe_active_filters(query_text)
        st.download_button(
            "Export briefing",
            data=cached_briefing_html(df, sig, filters_desc),
            file_name="crisislens_situation_briefing.html",
            mime="text/html",
            width="stretch",
            on_click="ignore"
        )

    # 4. Activity timeline with surge markers
    _render_activity_timeline(df, surges)

    # 5. Two-Column Panels: Surge Alerts & Clustered Incidents
    left_col, right_col = st.columns([1, 1])
    with left_col:
        _render_surges(surges, filtered_to_one_event=st.session_state.get("filter_event", "All Events") != "All Events")
    with right_col:
        _render_incidents(incidents)

    # 6. Needs and places
    _render_breakdowns(df)

    # 7. Priority Feed
    _render_priority_feed(df)
