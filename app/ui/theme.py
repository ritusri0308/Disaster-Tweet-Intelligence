"""
Light design system, design tokens, and Plotly styling for CrisisLens.
Clean, modern, fresh, minimalist, light interface with large readable type.

This module is the ONLY place colours are defined. Pages and components reference the tokens below
(or the ``cl-*`` CSS classes) and never contain hex literals; tests/test_ui_tokens.py enforces that.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# Design Token Constants
BG_COLOR = "#F8FAFC"
SURFACE_COLOR = "#FFFFFF"
SUBTLE_FILL = "#F1F5F9"
BORDER_COLOR = "#E2E8F0"
TEXT_PRIMARY = "#0F172A"
TEXT_MUTED = "#475569"
ACCENT_PRIMARY = "#2563EB"
ACCENT_HOVER = "#1D4ED8"
ACCENT_SOFT = "#EFF6FF"
ACCENT_LIGHT = "#93C5FD"
ACCENT_BORDER = "#BFDBFE"

# Status tokens
DANGER = "#DC2626"
DANGER_TEXT = "#B91C1C"
DANGER_SOFT = "#FEE2E2"
DANGER_BORDER = "#FECACA"
WARN_TEXT = "#A16207"
WARN_SOFT = "#FEF9C3"
WARN_BORDER = "#FEF08A"
SUCCESS = "#15803D"
SUCCESS_SOFT = "#DCFCE7"
SUCCESS_BORDER = "#BBF7D0"
ORANGE_SOFT = "#FFEDD5"
ORANGE_TEXT = "#C2410C"
ORANGE_BORDER = "#FED7AA"

# Chart tokens
GRID_LINE = "#EEF2F6"
CONNECTOR_LINE = "#CBD5E1"
NEUTRAL_LINE = "#94A3B8"
LAND_COLOR = "#F1F5F9"

# Severity Badges (tinted bg + dark text)
SEVERITY_BADGES = {
    "CRITICAL": {"bg": DANGER_SOFT, "text": DANGER_TEXT},
    "HIGH": {"bg": ORANGE_SOFT, "text": ORANGE_TEXT},
    "MEDIUM": {"bg": WARN_SOFT, "text": WARN_TEXT},
    "LOW": {"bg": SUCCESS_SOFT, "text": SUCCESS}
}

SEVERITY_COLORS = {
    "CRITICAL": DANGER_TEXT,
    "HIGH": ORANGE_TEXT,
    "MEDIUM": WARN_TEXT,
    "LOW": SUCCESS
}

# Category Colors (calm color-blind safe)
CATEGORY_COLORS = {
    "Emergency": DANGER,
    "Help Request": "#EA580C",
    "Damage Report": "#7C3AED",
    "Information/Update": ACCENT_PRIMARY,
    "General": NEUTRAL_LINE
}

CATEGORY_BADGES = {
    "Emergency": {"bg": DANGER_SOFT, "text": DANGER},
    "Help Request": {"bg": ORANGE_SOFT, "text": "#EA580C"},
    "Damage Report": {"bg": "#F3E8FF", "text": "#7C3AED"},
    "Information/Update": {"bg": ACCENT_SOFT, "text": ACCENT_PRIMARY},
    "General": {"bg": SUBTLE_FILL, "text": "#64748B"}
}

CUSTOM_CSS = f"""
<style>
:root {{
    --cl-bg: {BG_COLOR};
    --cl-surface: {SURFACE_COLOR};
    --cl-subtle: {SUBTLE_FILL};
    --cl-border: {BORDER_COLOR};
    --cl-text: {TEXT_PRIMARY};
    --cl-muted: {TEXT_MUTED};
    --cl-accent: {ACCENT_PRIMARY};
    --cl-accent-soft: {ACCENT_SOFT};
}}

html {{
    font-size: 17px;
}}

/* System font stack: no remote font download, so the app renders identically when air-gapped. */
body, [class*="css"], .stApp {{
    font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: {TEXT_PRIMARY};
    background-color: {BG_COLOR};
    line-height: 1.6;
}}

/* Centered layout with generous whitespace */
.block-container {{
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    margin: 0 auto;
}}

/* Hide default menu and footer */
#MainMenu, footer, header {{
    visibility: hidden;
}}

/* Typography */
h1, .cl-page-title {{
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: {TEXT_PRIMARY} !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em !important;
}}

h2, h3, .cl-section {{
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: {TEXT_PRIMARY} !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
    letter-spacing: -0.01em !important;
}}

.cl-muted {{
    color: {TEXT_MUTED};
    font-size: 0.95rem;
    line-height: 1.5;
}}

.cl-mono {{
    font-family: ui-monospace, 'JetBrains Mono', SFMono-Regular, Menlo, Consolas, monospace !important;
    font-size: 0.9rem !important;
}}

/* Card Container */
.cl-card {{
    background: {SURFACE_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    margin-bottom: 1.25rem;
}}

/* Callouts */
.cl-alert {{
    border: 1px solid transparent;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
    font-size: 0.95rem;
    line-height: 1.5;
}}
.cl-alert--info {{ background: {ACCENT_SOFT}; border-color: {ACCENT_BORDER}; color: {ACCENT_HOVER}; }}
.cl-alert--warn {{ background: {WARN_SOFT}; border-color: {WARN_BORDER}; color: {WARN_TEXT}; }}
.cl-alert--danger {{ background: {DANGER_SOFT}; border-color: {DANGER_BORDER}; color: {DANGER_TEXT}; }}
.cl-alert--ok {{ background: {SUCCESS_SOFT}; border-color: {SUCCESS_BORDER}; color: {SUCCESS}; }}

/* KPI Metric Cards */
.cl-kpi {{
    background: {SURFACE_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 120px;
}}

.cl-kpi__label {{
    color: {TEXT_MUTED};
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
    text-transform: none;
    letter-spacing: normal;
}}

.cl-kpi__value {{
    font-size: 2.6rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1.1;
    letter-spacing: -0.03em;
    overflow-wrap: anywhere;
}}

.cl-kpi__delta {{
    font-size: 0.875rem;
    color: {TEXT_MUTED};
    margin-top: 0.35rem;
    font-weight: 500;
}}

@media (max-width: 640px) {{
    .cl-kpi__value {{ font-size: 2rem; }}
    .cl-card, .cl-kpi {{ padding: 1rem 1.1rem; }}
}}

/* Soft Badges */
.cl-badge {{
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.875rem;
    font-weight: 600;
    line-height: 1;
    margin-right: 0.5rem;
    border: 1px solid transparent;
}}

.cl-badge--critical {{
    background-color: {SEVERITY_BADGES["CRITICAL"]["bg"]};
    color: {SEVERITY_BADGES["CRITICAL"]["text"]};
    border-color: {DANGER_BORDER};
}}

.cl-badge--high {{
    background-color: {SEVERITY_BADGES["HIGH"]["bg"]};
    color: {SEVERITY_BADGES["HIGH"]["text"]};
    border-color: {ORANGE_BORDER};
}}

.cl-badge--medium {{
    background-color: {SEVERITY_BADGES["MEDIUM"]["bg"]};
    color: {SEVERITY_BADGES["MEDIUM"]["text"]};
    border-color: {WARN_BORDER};
}}

.cl-badge--low {{
    background-color: {SEVERITY_BADGES["LOW"]["bg"]};
    color: {SEVERITY_BADGES["LOW"]["text"]};
    border-color: {SUCCESS_BORDER};
}}

.cl-badge--emergency {{
    background-color: {CATEGORY_BADGES["Emergency"]["bg"]};
    color: {CATEGORY_BADGES["Emergency"]["text"]};
}}

.cl-badge--help-request {{
    background-color: {CATEGORY_BADGES["Help Request"]["bg"]};
    color: {CATEGORY_BADGES["Help Request"]["text"]};
}}

.cl-badge--damage-report {{
    background-color: {CATEGORY_BADGES["Damage Report"]["bg"]};
    color: {CATEGORY_BADGES["Damage Report"]["text"]};
}}

.cl-badge--information-update {{
    background-color: {CATEGORY_BADGES["Information/Update"]["bg"]};
    color: {CATEGORY_BADGES["Information/Update"]["text"]};
}}

.cl-badge--general {{
    background-color: {CATEGORY_BADGES["General"]["bg"]};
    color: {CATEGORY_BADGES["General"]["text"]};
}}

/* Chips */
.cl-chip {{
    display: inline-flex;
    align-items: center;
    background-color: {SUBTLE_FILL};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 0.875rem;
    color: {TEXT_MUTED};
    margin: 2px 4px 2px 0;
}}

.cl-chip--active {{
    background-color: {ACCENT_SOFT};
    border-color: {ACCENT_BORDER};
    color: {ACCENT_PRIMARY};
    font-weight: 500;
}}

.cl-chip--figure {{
    background-color: {DANGER_SOFT};
    border-color: {DANGER_BORDER};
    color: {DANGER_TEXT};
    font-weight: 600;
}}

/* Feed item card */
.cl-feed-item {{
    background: {SURFACE_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}}

.cl-feed-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 8px;
}}

.cl-feed-text {{
    overflow-wrap: anywhere;
}}

/* Surge / alert cards */
.cl-alert-card__title {{
    font-weight: 600;
    color: {DANGER};
    font-size: 0.95rem;
}}

.cl-alert-card__body {{
    margin-top: 4px;
}}

/* Tweet text with token highlights */
.cl-tweet {{
    background: {SURFACE_COLOR};
    border: 1px solid {BORDER_COLOR};
    padding: 16px;
    border-radius: 12px;
    font-size: 1rem;
    line-height: 1.7;
    color: {TEXT_PRIMARY};
    overflow-wrap: anywhere;
}}

.cl-hl {{
    padding: 2px 6px;
    border-radius: 6px;
    font-weight: 600;
    border: 1px solid transparent;
}}
.cl-hl--loc {{ background-color: {ACCENT_SOFT}; border-color: {ACCENT_BORDER}; color: {ACCENT_HOVER}; }}
.cl-hl--urg {{ background-color: {DANGER_SOFT}; border-color: {DANGER_BORDER}; color: {DANGER_TEXT}; }}
.cl-hl--feat {{ background-color: {WARN_SOFT}; border-color: {WARN_BORDER}; color: {WARN_TEXT}; font-weight: 500; }}

/* Clean buttons */
.stButton > button {{
    border-radius: 10px !important;
    min-height: 44px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.15s ease-in-out !important;
}}

.stButton > button[kind="primary"] {{
    background-color: {ACCENT_PRIMARY} !important;
    color: {SURFACE_COLOR} !important;
    border: 1px solid {ACCENT_PRIMARY} !important;
}}

.stButton > button[kind="primary"]:hover {{
    background-color: {ACCENT_HOVER} !important;
    border-color: {ACCENT_HOVER} !important;
}}

.stButton > button[kind="secondary"] {{
    background-color: {SURFACE_COLOR} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {BORDER_COLOR} !important;
}}

/* Inputs & Selectboxes */
.stTextInput > div > div > input,
.stSelectbox > div > div {{
    background-color: {SURFACE_COLOR} !important;
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    color: {TEXT_PRIMARY} !important;
}}

/* Alert / Info boxes */
div[data-testid="stNotification"] {{
    border-radius: 10px !important;
    border-width: 1px !important;
    border-style: solid !important;
    padding: 1rem 1.25rem !important;
    font-size: 0.95rem !important;
}}

/* DataFrames */
.stDataFrame {{
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}

/* Footer note */
.cl-footer {{
    color: {TEXT_MUTED};
    font-size: 0.875rem;
    text-align: center;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid {BORDER_COLOR};
}}
</style>
"""

def inject_custom_theme():
    """Inject CrisisLens light CSS into Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def setup_plotly_theme():
    """Register and apply CrisisLens custom Plotly light template."""
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=TEXT_PRIMARY,
            size=14
        ),
        title=dict(
            font=dict(color=TEXT_PRIMARY, size=15),
            x=0.0,
            xanchor="left"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(color=TEXT_MUTED, size=13)
        ),
        xaxis=dict(
            gridcolor=GRID_LINE,
            linecolor="rgba(0,0,0,0)",
            showline=False,
            zerolinecolor=GRID_LINE,
            tickfont=dict(color=TEXT_MUTED, size=13),
            title=dict(font=dict(color=TEXT_PRIMARY, size=14))
        ),
        yaxis=dict(
            gridcolor=GRID_LINE,
            linecolor="rgba(0,0,0,0)",
            showline=False,
            zerolinecolor=GRID_LINE,
            tickfont=dict(color=TEXT_MUTED, size=13),
            title=dict(font=dict(color=TEXT_PRIMARY, size=14))
        ),
        bargap=0.35,
        hoverlabel=dict(
            bgcolor=SURFACE_COLOR,
            bordercolor=BORDER_COLOR,
            font=dict(family="Inter, -apple-system, sans-serif", color=TEXT_PRIMARY, size=14)
        ),
        colorway=[
            ACCENT_PRIMARY,
            DANGER,
            "#EA580C",
            "#7C3AED",
            "#059669",
            "#D97706",
            "#64748B"
        ]
    )
    pio.templates["crisislens"] = template
    pio.templates.default = "crisislens"

setup_plotly_theme()

def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply unified light styling and clean layout."""
    fig.update_layout(
        template="crisislens",
        margin=dict(l=15, r=15, t=30, b=25),
        hovermode="closest"
    )
    # Ensure line widths are clean
    for trace in fig.data:
        if hasattr(trace, "line") and trace.line is not None:
            if not getattr(trace.line, "width", None):
                trace.line.width = 2.5
    return fig
