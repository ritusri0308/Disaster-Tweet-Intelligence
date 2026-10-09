"""
Replay Mode (Historical Playback) page for CrisisLens.
Simulates chronological playback of disaster tweet streams based on Twitter Snowflake timestamps,
with play / pause / restart controls and surge alerts appearing as they would have been detected.
"""

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import get_activity_timeseries
from app.ui.cache import cached_surges, df_signature
from app.ui.components import (
    esc,
    humanize_event,
    render_callout,
    render_empty,
    render_feed_item,
    render_kpi_card,
)
from app.ui.theme import ACCENT_LIGHT, ACCENT_PRIMARY, DANGER, SURFACE_COLOR, apply_chart_theme

# Seconds a full replay takes at each speed (one frame per second)
SPEED_SECONDS = {"Slow": 120, "Normal": 60, "Fast": 20}

def _toggle_play(at_end: bool, start) -> None:
    if not st.session_state.get("replay_playing") and at_end:
        st.session_state["replay_cursor"] = start  # pressing Play at the end starts over, like a media player
    st.session_state["replay_playing"] = not st.session_state.get("replay_playing", False)

def _restart(start) -> None:
    st.session_state["replay_cursor"] = start
    st.session_state["replay_playing"] = False

def _build_timeline(event_df: pd.DataFrame, cursor, surges: list) -> go.Figure:
    series, freq = get_activity_timeseries(event_df)
    played = series[series["timestamp"] <= pd.Timestamp(cursor)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["timestamp"], y=series["tweets"], mode="lines", name="Full event",
        line=dict(color=ACCENT_LIGHT, width=1.5), hovertemplate="%{x}<br>%{y} tweets<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=played["timestamp"], y=played["tweets"], mode="lines", name="Replayed so far",
        fill="tozeroy", line=dict(color=ACCENT_PRIMARY, width=2.5), hovertemplate="%{x}<br>%{y} tweets<extra></extra>"
    ))

    by_bucket = series.set_index("timestamp")["tweets"]
    marks = [(pd.Timestamp(a["window_start"]).floor(freq), a) for a in surges if pd.Timestamp(a["window_start"]) <= pd.Timestamp(cursor)]
    marks = [(b, a) for b, a in marks if b in by_bucket.index]
    if marks:
        fig.add_trace(go.Scatter(
            x=[b for b, _ in marks], y=[int(by_bucket.loc[b]) for b, _ in marks], mode="markers", name="Surge detected",
            marker=dict(symbol="diamond", size=12, color=DANGER, line=dict(width=1.5, color=SURFACE_COLOR)),
            hovertext=[f"{a['alert_type']} (z={a['z_score']})" for _, a in marks], hoverinfo="text"
        ))

    cursor_ts = pd.Timestamp(cursor)
    fig.add_shape(type="line", x0=cursor_ts, x1=cursor_ts, y0=0, y1=1, yref="paper",
                  line=dict(color=DANGER, width=2, dash="dash"))
    fig.add_annotation(x=cursor_ts, y=1, yref="paper", text="Cursor", showarrow=False, yanchor="bottom",
                       font=dict(color=DANGER))
    apply_chart_theme(fig)
    fig.update_layout(height=300, xaxis_title="Time (UTC)",
                      yaxis_title="Tweets per hour" if freq == "h" else "Tweets per day")
    return fig

def render_page(df: pd.DataFrame, pipeline: any):
    render_callout(
        "This feature replays historical disaster events chronologically using UTC timestamps derived from "
        "Twitter Snowflake IDs (2016–2019). It is an evaluation replay tool.",
        "warn", "Historical benchmark replay."
    )

    if df.empty or "timestamp" not in df.columns:
        render_empty("No timestamped data available for chronological replay.")
        return

    valid_df = df.copy()
    valid_df["ts"] = pd.to_datetime(valid_df["timestamp"], errors="coerce")
    valid_df = valid_df.dropna(subset=["ts"]).sort_values("ts")

    if valid_df.empty:
        render_empty("All timestamps in the current view are invalid or missing.")
        return

    counts = valid_df["event"].value_counts()
    events = list(counts.index)
    selected_event = st.selectbox(
        "Focus replay event", events, key="replay_event",
        format_func=lambda e: f"{humanize_event(e)} ({counts[e]:,} tweets)"
    )
    event_df = valid_df[valid_df["event"] == selected_event].sort_values("ts")

    if event_df.empty:
        render_empty("No records for selected event.")
        return

    ev_min_ts = event_df["ts"].min().to_pydatetime()
    ev_max_ts = event_df["ts"].max().to_pydatetime()

    if ev_min_ts == ev_max_ts:
        render_empty(
            f"All {len(event_df)} records for {humanize_event(selected_event)} share the single timestamp "
            f"({ev_min_ts.strftime('%Y-%m-%d %H:%M UTC')}). Timeline playback requires multiple timestamps."
        )
        st.markdown("<div class='cl-section'>Tweets in Event</div>", unsafe_allow_html=True)
        for _, row in event_df.head(6).iterrows():
            render_feed_item(row, timestamp=row["ts"])
        return

    # Reset the cursor when the event changes (or on first visit); a new event opens fully played
    if st.session_state.get("replay_loaded_event") != selected_event:
        st.session_state["replay_loaded_event"] = selected_event
        st.session_state["replay_cursor"] = ev_max_ts
        st.session_state["replay_playing"] = False
    cursor_now = st.session_state.get("replay_cursor", ev_max_ts)
    at_end = cursor_now >= ev_max_ts

    # Playback controls
    st.markdown("<div class='cl-section'>Playback Horizon</div>", unsafe_allow_html=True)
    playing = bool(st.session_state.get("replay_playing"))
    c_play, c_restart, c_speed = st.columns([1, 1, 2])
    with c_play:
        st.button("Pause" if playing else "Play", key="replay_play_btn", icon=":material/pause:" if playing else ":material/play_arrow:",
                  on_click=_toggle_play, args=(at_end, ev_min_ts), type="primary", width="stretch")
    with c_restart:
        st.button("Restart", key="replay_restart_btn", icon=":material/replay:",
                  on_click=_restart, args=(ev_min_ts,), width="stretch")
    with c_speed:
        speed = st.segmented_control("Speed", list(SPEED_SECONDS), default="Normal", key="replay_speed", label_visibility="collapsed") or "Normal"

    sig = df_signature(event_df)
    surges = cached_surges(event_df, sig)
    step = (ev_max_ts - ev_min_ts) / SPEED_SECONDS[speed]

    # Everything that depends on the cursor lives in a fragment, so each playback frame re-renders only this
    # block instead of the whole app. It ticks once a second while playing and is idle otherwise.
    @st.fragment(run_every=1.0 if playing else None)
    def _playback():
        if st.session_state.get("replay_playing"):
            nxt = min(st.session_state.get("replay_cursor", ev_min_ts) + step, ev_max_ts)
            st.session_state["replay_cursor"] = nxt
            if nxt >= ev_max_ts:
                st.session_state["replay_playing"] = False
                st.rerun()  # full rerun so the Play/Pause button label updates

        current_time = st.slider(
            "Current playback time",
            min_value=ev_min_ts,
            max_value=ev_max_ts,
            key="replay_cursor",
            step=timedelta(minutes=15),
            format="YYYY-MM-DD HH:mm UTC"
        )
        replay_view = event_df[event_df["ts"] <= current_time]

        # Dynamic KPIs for Replayed Window [start, t]
        tot = len(replay_view)
        crit = int((replay_view["urgency_level"] == "CRITICAL").sum())
        high = int((replay_view["urgency_level"] == "HIGH").sum())
        help_req = int((replay_view["category"] == "Help Request").sum())

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card("Accumulated tweets", f"{tot:,}", f"{(tot / len(event_df)):.1%} of event")
        with k2:
            render_kpi_card("Critical alerts", f"{crit:,}", "Immediate review")
        with k3:
            render_kpi_card("High priority", f"{high:,}", "Elevated triage")
        with k4:
            render_kpi_card("Help requests", f"{help_req:,}", "Assistance needed")

        st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

        st.markdown("<div class='cl-section'>Timeline Progression</div>", unsafe_allow_html=True)
        st.plotly_chart(_build_timeline(event_df, current_time, surges), width="stretch")

        st.markdown(
            f"<div class='cl-section'>Recent Tweets as of {esc(current_time.strftime('%Y-%m-%d %H:%M UTC'))}</div>",
            unsafe_allow_html=True
        )
        recent = replay_view.sort_values("ts", ascending=False).head(6)
        if recent.empty:
            render_empty("No tweets before selected cursor time.")
        else:
            for _, row in recent.iterrows():
                render_feed_item(row, timestamp=row["ts"])

    _playback()
