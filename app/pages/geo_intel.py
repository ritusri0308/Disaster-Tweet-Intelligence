"""
Geospatial Intelligence page for CrisisLens.
Displays token-free offline geocoded map of emergency reports and location hotspots, with drill-down
from a hotspot to the tweets that mention it.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CATEGORY_COLORS
from app.ui.components import esc, humanize_event, render_callout, render_empty, render_feed_item
from app.ui.theme import BORDER_COLOR, LAND_COLOR, SURFACE_COLOR, apply_chart_theme

COLOR_MODES = ["Max urgency", "Dominant category", "Mentions"]

def _flatten_geocoded(df: pd.DataFrame) -> pd.DataFrame:
    """One row per geocoded place mention, with the tweet's category / urgency / event attached."""
    needed = [c for c in ("geocoded_locations", "category", "urgency_score", "urgency_level", "event") if c in df.columns]
    if "geocoded_locations" not in needed:
        return pd.DataFrame()

    exploded = df[needed].explode("geocoded_locations").dropna(subset=["geocoded_locations"])
    exploded = exploded[exploded["geocoded_locations"].apply(lambda g: isinstance(g, dict) and "lat" in g and "lon" in g)]
    if exploded.empty:
        return pd.DataFrame()

    places = pd.DataFrame(exploded["geocoded_locations"].tolist(), index=exploded.index)
    out = pd.concat([exploded.drop(columns="geocoded_locations"), places], axis=1)
    out["lat"] = out["lat"].astype(float)
    out["lon"] = out["lon"].astype(float)
    if "country_code" not in out:
        out["country_code"] = ""
    out["urgency_score"] = out["urgency_score"].astype(float) if "urgency_score" in out else 0.0
    return out

def _aggregate_places(geo_df: pd.DataFrame) -> pd.DataFrame:
    agg = geo_df.groupby(["name", "lat", "lon", "country_code"]).agg(
        mentions=("urgency_score", "count"),
        max_urgency=("urgency_score", "max"),
        mean_urgency=("urgency_score", "mean"),
        critical=("urgency_level", lambda s: int((s == "CRITICAL").sum())),
        dominant_category=("category", lambda s: s.mode().iat[0] if not s.mode().empty else "General"),
    ).reset_index()
    agg["max_urgency"] = agg["max_urgency"].round(1)
    agg["mean_urgency"] = agg["mean_urgency"].round(1)
    return agg.sort_values(["mentions", "max_urgency"], ascending=False).reset_index(drop=True)

def _view_bounds(places: pd.DataFrame, coverage: float = 0.9):
    """
    Map window covering the places that account for ``coverage`` of all mentions, padded.
    A handful of far-away mentions (e.g. 'Japan' in a Texas storm) should not zoom the map out to the whole planet.
    """
    top = places.sort_values("mentions", ascending=False)
    cumulative = top["mentions"].cumsum() / top["mentions"].sum()
    core = top[cumulative.shift(fill_value=0) < coverage]
    lat_lo, lat_hi = core["lat"].min(), core["lat"].max()
    lon_lo, lon_hi = core["lon"].min(), core["lon"].max()
    lat_pad = max((lat_hi - lat_lo) * 0.25, 3.0)
    lon_pad = max((lon_hi - lon_lo) * 0.25, 4.0)
    lat_range = [max(lat_lo - lat_pad, -85), min(lat_hi + lat_pad, 85)]
    lon_range = [max(lon_lo - lon_pad, -180), min(lon_hi + lon_pad, 180)]
    return lat_range, lon_range, (lon_range[1] - lon_range[0]) > 150

def _build_map(places: pd.DataFrame, color_mode: str):
    plot_df = places.assign(marker_size=np.log1p(places["mentions"]) + 1)
    common = dict(
        lat="lat", lon="lon", hover_name="name", size="marker_size", size_max=26, projection="natural earth",
        hover_data={"mentions": True, "max_urgency": True, "critical": True, "marker_size": False, "lat": False, "lon": False},
        labels={"mentions": "Mentions", "max_urgency": "Max urgency", "critical": "Critical tweets"},
    )
    if color_mode == "Dominant category":
        fig = px.scatter_geo(plot_df, color="dominant_category", color_discrete_map=CATEGORY_COLORS, **common)
    elif color_mode == "Mentions":
        fig = px.scatter_geo(plot_df, color="mentions", color_continuous_scale="Blues", **common)
    else:
        fig = px.scatter_geo(plot_df, color="max_urgency", color_continuous_scale="Reds", range_color=(0, 10), **common)

    lat_range, lon_range, world_view = _view_bounds(places)
    fig.update_geos(
        showcountries=True, countrycolor=BORDER_COLOR,
        showland=True, landcolor=LAND_COLOR,
        showocean=True, oceancolor=SURFACE_COLOR,
        showlakes=False,
        projection_type="natural earth" if world_view else "mercator",
        lataxis_range=None if world_view else lat_range,
        lonaxis_range=None if world_view else lon_range,
    )
    apply_chart_theme(fig)
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=20, b=0))
    return fig

def _tweets_for_place(df: pd.DataFrame, place_name: str) -> pd.DataFrame:
    mask = df["geocoded_locations"].apply(
        lambda places: any(isinstance(g, dict) and g.get("name") == place_name for g in places)
    )
    return df[mask].sort_values("urgency_score", ascending=False, kind="mergesort")

def render_page(df: pd.DataFrame, pipeline: any):
    render_callout(
        "Approximate locations geocoded from place names mentioned in tweets (GeoNamesCache); not device GPS coordinates.",
        "info"
    )

    if df.empty or "geocoded_locations" not in df.columns:
        render_empty("No geocoded dispatches available in current view.")
        return

    geo_df = _flatten_geocoded(df)
    if geo_df.empty:
        render_empty("No recognizable geographic locations found in filtered tweets.")
        return

    places = _aggregate_places(geo_df)
    st.caption(f"{len(geo_df):,} geocoded place mentions across {len(places):,} distinct places.")

    color_mode = st.segmented_control("Colour markers by", COLOR_MODES, default=COLOR_MODES[0], key="geo_color_mode") or COLOR_MODES[0]

    col_map, col_table = st.columns([3, 1.8])
    with col_map:
        st.markdown("<div class='cl-section'>Geospatial Distribution</div>", unsafe_allow_html=True)
        st.plotly_chart(_build_map(places, color_mode), width="stretch")

    with col_table:
        st.markdown("<div class='cl-section'>Location Hotspots</div>", unsafe_allow_html=True)
        st.caption("Select a row to list the tweets that mention it.")
        table = places[["name", "country_code", "mentions", "max_urgency"]].rename(columns={
            "name": "Location", "country_code": "Country", "mentions": "Reports", "max_urgency": "Max urgency"
        })
        selection = st.dataframe(
            table, key="geo_hotspots", hide_index=True, height=420,
            on_select="rerun", selection_mode="single-row",
            column_config={"Max urgency": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f")},
        )

    rows = selection.selection.rows if selection is not None and selection.selection else []
    if rows and rows[0] < len(places):
        place = places.iloc[rows[0]]
        tweets = _tweets_for_place(df, place["name"])
        st.markdown(
            f"<div class='cl-section'>Reports mentioning {esc(place['name'])}</div>"
            f"<div class='cl-muted' style='margin-bottom:0.75rem;'>{len(tweets):,} tweets · "
            f"{int(place['critical']):,} critical · showing the most urgent</div>",
            unsafe_allow_html=True
        )
        for _, row in tweets.head(6).iterrows():
            render_feed_item(row, timestamp=row.get("timestamp"))
        if tweets["event"].nunique() > 1:
            st.caption("Events: " + ", ".join(humanize_event(e) for e in tweets["event"].unique()[:4]))
