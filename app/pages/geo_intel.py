"""
Geospatial Intelligence page for CrisisLens.
Displays token-free offline geocoded map of emergency reports and location hotspots.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from app.ui.theme import apply_chart_theme

def render_page(df: pd.DataFrame, pipeline: any):
    st.markdown("<h3 style='margin-top:0;color:#E6EDF7;'>🗺️ Geospatial Intelligence</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background:rgba(34,211,238,0.1);border-left:4px solid #22D3EE;padding:10px 14px;border-radius:6px;font-size:0.875rem;color:#E6EDF7;margin-bottom:1rem;'>
            📍 <em>Approximate locations geocoded from place names mentioned in tweets (GeoNames); not device GPS.</em>
        </div>
        """,
        unsafe_allow_html=True
    )

    if df.empty or "geocoded_locations" not in df.columns:
        st.warning("No geocoded dispatches available in current view.")
        return

    # Extract all geocoded records into a flattened DataFrame
    records = []
    for _, row in df.iterrows():
        geos = row.get("geocoded_locations", [])
        if isinstance(geos, list):
            for g in geos:
                if isinstance(g, dict) and "lat" in g and "lon" in g:
                    records.append({
                        "name": g.get("name", "Unknown"),
                        "lat": float(g["lat"]),
                        "lon": float(g["lon"]),
                        "country": g.get("country_code", ""),
                        "category": row.get("category", "General"),
                        "urgency_score": float(row.get("urgency_score", 0.0)),
                        "urgency_level": str(row.get("urgency_level", "LOW")),
                        "event": row.get("event", "General")
                    })

    if not records:
        st.info("No recognizable geographic locations found in filtered tweets.")
        return

    geo_df = pd.DataFrame(records)

    # Aggregate by place
    agg_places = geo_df.groupby(["name", "lat", "lon", "country"]).agg(
        mention_count=("urgency_score", "count"),
        max_urgency=("urgency_score", "max"),
        mean_urgency=("urgency_score", "mean")
    ).reset_index().sort_values("mention_count", ascending=False)

    col_map, col_table = st.columns([3, 1.8])

    with col_map:
        st.markdown("##### 🌍 Incident Density & Geospatial Distribution")
        fig_map = px.scatter_geo(
            agg_places,
            lat="lat",
            lon="lon",
            hover_name="name",
            size="mention_count",
            color="max_urgency",
            color_continuous_scale="YlOrRd",
            size_max=30,
            labels={"mention_count": "Reports", "max_urgency": "Max Urgency"},
            projection="natural earth"
        )
        fig_map.update_geos(
            showcountries=True,
            countrycolor="#1F2A3D",
            showland=True,
            landcolor="#111827",
            showocean=True,
            oceancolor="#0B0F17",
            showlakes=False
        )
        apply_chart_theme(fig_map)
        fig_map.update_layout(height=480, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    with col_table:
        st.markdown("##### 📍 Location Hotspot Ranking")
        st.dataframe(
            agg_places[["name", "country", "mention_count", "max_urgency"]].rename(
                columns={
                    "name": "Location",
                    "country": "Country",
                    "mention_count": "Reports",
                    "max_urgency": "Max Urgency"
                }
            ),
            use_container_width=True,
            height=450
        )
