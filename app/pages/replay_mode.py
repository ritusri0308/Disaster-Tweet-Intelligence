"""
Replay Mode (Historical Time Machine) page for CrisisLens.
Simulates chronological playback of disaster tweet streams based on Twitter Snowflake timestamps.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from app.ui.theme import apply_chart_theme
from app.ui.components import render_kpi_card

def render_page(df: pd.DataFrame, pipeline: any):
    st.markdown("<h3 style='margin-top:0;color:#E6EDF7;'>⏳ Replay Mode (Historical Playback)</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background:rgba(234, 179, 8, 0.1);border-left:4px solid #EAB308;padding:10px 14px;border-radius:6px;font-size:0.875rem;color:#FEF08A;margin-bottom:1rem;'>
            ⚠️ <strong>Historical Benchmark Replay</strong>: This feature replays historical disaster events chronologically
            using UTC timestamps derived from Twitter Snowflake IDs (2016-2019). It is not a real-time monitor.
        </div>
        """,
        unsafe_allow_html=True
    )

    if df.empty or "timestamp" not in df.columns:
        st.warning("No timestamped data available for chronological replay.")
        return

    valid_df = df.copy()
    valid_df["ts"] = pd.to_datetime(valid_df["timestamp"], errors="coerce")
    valid_df = valid_df.dropna(subset=["ts"]).sort_values("ts")

    if valid_df.empty:
        st.warning("All timestamps in the current view are invalid or missing.")
        return

    min_ts = valid_df["ts"].min().to_pydatetime()
    max_ts = valid_df["ts"].max().to_pydatetime()

    if min_ts == max_ts:
        st.info("Single timestamp detected in view. Expand filters to observe time window.")
        return

    # Event selector for focused replay
    events = sorted(list(valid_df["event"].unique()))
    selected_event = st.selectbox("Focus Replay Event", events)
    event_df = valid_df[valid_df["event"] == selected_event].sort_values("ts")

    if event_df.empty:
        st.info("No records for selected event.")
        return

    ev_min_ts = event_df["ts"].min().to_pydatetime()
    ev_max_ts = event_df["ts"].max().to_pydatetime()

    # Time Slider
    st.markdown("##### ⏱️ Playback Horizon Cursor")
    current_time = st.slider(
        "Current Time Window Horizon",
        min_value=ev_min_ts,
        max_value=ev_max_ts,
        value=ev_max_ts,
        format="YYYY-MM-DD HH:mm UTC"
    )

    # Filter window up to current_time
    replay_view = event_df[event_df["ts"] <= current_time]

    # Dynamic KPIs for Replayed Window [start, t]
    tot = len(replay_view)
    crit = (replay_view["urgency_level"] == "CRITICAL").sum()
    high = (replay_view["urgency_level"] == "HIGH").sum()
    help_req = (replay_view["category"] == "Help Request").sum()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Accumulated Dispatches", f"{tot:,}", f"{(tot/len(event_df)):.1%} of total", "#22D3EE")
    with k2:
        render_kpi_card("Critical Alerts", f"{crit:,}", "Immediate danger", "#EF4444")
    with k3:
        render_kpi_card("High Priority", f"{high:,}", "Elevated triage", "#F97316")
    with k4:
        render_kpi_card("Help Requests", f"{help_req:,}", "Needs resources", "#EAB308")

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    # Timeline Chart showing cursor position
    st.markdown("##### 📊 Event Timeline Progression")
    hourly = event_df.set_index("ts").resample("6h").size().reset_index(name="tweets")
    fig_time = px.area(hourly, x="ts", y="tweets", labels={"ts": "Time (UTC)", "tweets": "Tweets / 6hr"})
    fig_time.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="#22D3EE", annotation_text="Cursor")
    apply_chart_theme(fig_time)
    st.plotly_chart(fig_time, use_container_width=True)

    # Replay Feed Ticker (newest first up to current_time)
    st.markdown(f"##### 📜 Recent Dispatches as of {current_time.strftime('%Y-%m-%d %H:%M UTC')}")
    recent = replay_view.sort_values("ts", ascending=False).head(6)
    if not recent.empty:
        for _, row in recent.iterrows():
            badge_color = "#EF4444" if row.get("urgency_level") == "CRITICAL" else ("#F97316" if row.get("urgency_level") == "HIGH" else "#22C55E")
            st.markdown(
                f"""
                <div class='glass-card' style='padding:0.75rem 1rem;margin-bottom:0.4rem;'>
                    <div style='display:flex;justify-content:space-between;'>
                        <div>
                            <span class='status-pill' style='border-color:{badge_color};color:{badge_color};font-weight:700;'>{row.get('urgency_level')} {row.get('urgency_score')}/10</span>
                            <span class='status-pill'>{row.get('category')}</span>
                        </div>
                        <div style='font-family:JetBrains Mono;font-size:0.75rem;color:#22D3EE;'>{row['ts'].strftime('%Y-%m-%d %H:%M UTC')}</div>
                    </div>
                    <div style='margin-top:4px;color:#E6EDF7;font-size:0.9rem;'>{row.get('raw_unescaped_text', '')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No tweets before selected cursor time.")
