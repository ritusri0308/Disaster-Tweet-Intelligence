"""
Situation Room page for CrisisLens.
Top command center displaying natural-language search bar, executive KPIs,
active surge alerts, clustered incidents, and prioritized dispatches.
"""

import streamlit as st
import pandas as pd

from src.nlquery import parse_natural_language_query
from src.anomaly import detect_event_surges
from src.clustering import cluster_event_incidents
from src.analytics import generate_situation_summary
from src.briefing import generate_html_briefing
from app.ui.components import render_kpi_card

def render_page(df: pd.DataFrame, pipeline: any):
    st.markdown("<h3 style='margin-top:0;color:#E6EDF7;'>📍 Situation Room</h3>", unsafe_allow_html=True)

    # 1. Natural Language Command Bar
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        nl_query = st.text_input(
            "Natural Language Command Bar",
            placeholder="e.g., 'critical help requests in puerto rico' or 'damage reports harvey last 48 hours'",
            label_visibility="collapsed"
        )
    with col_btn:
        st.button("⚡ Execute Query", use_container_width=True)

    if nl_query.strip():
        parsed = parse_natural_language_query(nl_query)
        if parsed["is_valid"]:
            chips_str = " &nbsp;|&nbsp; ".join([f"<span class='status-pill pill-active'>{c}</span>" for c in parsed["parsed_chips"]])
            st.markdown(f"<div style='margin-bottom:1rem;'>{chips_str}</div>", unsafe_allow_html=True)
            # Apply parsed filters
            f = parsed["filters"]
            if "category" in f and "category" in df.columns:
                df = df[df["category"] == f["category"]]
            if "urgency_level" in f and "urgency_level" in df.columns:
                df = df[df["urgency_level"] == f["urgency_level"]]
            if "event" in f and "event" in df.columns:
                df = df[df["event"] == f["event"]]
            if "location" in f and "locations" in df.columns:
                loc_lower = f["location"].lower()
                df = df[df["locations"].apply(lambda locs: any(loc_lower in str(loc).lower() for loc in (locs if isinstance(locs, list) else [])))]
        else:
            st.info(f"ℹ️ {parsed['message']}")

    # 2. Executive KPIs Row
    total_tweets = len(df)
    crit_count = (df["urgency_level"] == "CRITICAL").sum() if "urgency_level" in df.columns else 0
    high_count = (df["urgency_level"] == "HIGH").sum() if "urgency_level" in df.columns else 0
    help_count = (df["category"] == "Help Request").sum() if "category" in df.columns else 0
    median_conf = float(df["confidence"].median()) if "confidence" in df.columns and not df.empty else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Total Dispatches", f"{total_tweets:,}", "Indexed Signals", "#22D3EE")
    with k2:
        render_kpi_card("Critical Alerts", f"{crit_count:,}", "Immediate Action", "#EF4444")
    with k3:
        render_kpi_card("High Urgency", f"{high_count:,}", "Escalated Priority", "#F97316")
    with k4:
        render_kpi_card("Help Requests", f"{help_count:,}", "Survival Pleadings", "#EAB308")
    with k5:
        render_kpi_card("Median Confidence", f"{median_conf:.1%}", "Calibrated Model", "#22C55E")

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    # 3. Situation Summary Box + Briefing Export
    summary_data = generate_situation_summary(df)
    col_sum, col_export = st.columns([3.5, 1.5])
    with col_sum:
        st.info(f"📋 **Situation Intelligence**: {summary_data['summary_text']}")
    with col_export:
        # Briefing download buttons
        surge_alerts = detect_event_surges(df)
        incidents = cluster_event_incidents(df, max_sample_per_event=500)
        html_briefing = generate_html_briefing(df, summary_data, incidents, surge_alerts, {})

        st.download_button(
            "📄 Export HTML Briefing",
            data=html_briefing.encode("utf-8"),
            file_name="crisislens_situation_briefing.html",
            mime="text/html",
            use_container_width=True
        )

    # 4. Two-Column Panels: Surge Alerts & Top Clustered Incidents
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("#### 🚨 Statistical Surge Alerts (Volume & Severity)")
        if surge_alerts:
            for alert in surge_alerts[:5]:
                st.markdown(
                    f"""
                    <div class='glass-card' style='border-left: 3px solid #EF4444;margin-bottom:0.6rem;padding:0.75rem;'>
                        <div style='font-size:0.8rem;color:#EF4444;font-weight:700;'>{alert['alert_type'].upper()} • {alert['event']}</div>
                        <div style='font-size:0.9rem;color:#E6EDF7;margin-top:2px;'>{alert['details']}</div>
                        <div style='font-size:0.75rem;color:#8B9BB4;margin-top:4px;'>Window: {alert['window_start']} | Top Terms: {', '.join(alert.get('top_terms', []))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown("<div class='glass-card' style='color:#8B9BB4;'>No abnormal volume spikes detected in current filtered view.</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("#### 📦 Clustered Incident Groups")
        if incidents:
            for inc in incidents[:4]:
                places_str = ", ".join(inc.get("places", [])) or "Area general"
                with st.expander(f"{inc['incident_id']} • {inc['dominant_category']} ({inc['size']} related tweets)"):
                    st.markdown(f"**Max Urgency**: `{inc['max_urgency']}/10` | **Locations**: `{places_str}`")
                    st.markdown(f"**Key Dispatches**: \"{inc['representative_tweet']}\"")
                    st.markdown(f"**Time Window**: {inc['time_span']}")
        else:
            st.markdown("<div class='glass-card' style='color:#8B9BB4;'>No multi-tweet incident clusters found in active view.</div>", unsafe_allow_html=True)

    # 5. Live-style Ticker of High & Critical Priority Dispatches
    st.markdown("#### ⚡ Priority Dispatch Feed")
    crit_df = df[df["urgency_level"].isin(["CRITICAL", "HIGH"])].sort_values("urgency_score", ascending=False).head(8)
    if not crit_df.empty:
        for _, row in crit_df.iterrows():
            badge_color = "#EF4444" if row.get("urgency_level") == "CRITICAL" else "#F97316"
            places = ", ".join(row.get("locations", [])) if isinstance(row.get("locations"), list) else "Unknown"
            st.markdown(
                f"""
                <div class='glass-card' style='padding:0.85rem 1rem;margin-bottom:0.5rem;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <div>
                            <span class='status-pill' style='border-color:{badge_color};color:{badge_color};font-weight:bold;'>{row.get('urgency_level')} {row.get('urgency_score')}/10</span>
                            <span class='status-pill'>{row.get('category')}</span>
                            <span class='status-pill'>📍 {places}</span>
                        </div>
                        <div style='font-family:JetBrains Mono;font-size:0.75rem;color:#8B9BB4;'>{str(row.get('timestamp', ''))[:16]}</div>
                    </div>
                    <div style='margin-top:0.5rem;font-size:0.925rem;color:#E6EDF7;'>{row.get('raw_unescaped_text', '')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No critical dispatches in current filtered subset.")
