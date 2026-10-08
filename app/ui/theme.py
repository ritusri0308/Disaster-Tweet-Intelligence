"""
Design system, command-center dark theme, and Plotly styling for CrisisLens.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# Theme Palette Tokens
BG_COLOR = "#0B0F17"
SURFACE_COLOR = "#111827"
ELEVATED_COLOR = "#162033"
BORDER_COLOR = "#1F2A3D"
TEXT_PRIMARY = "#E6EDF7"
TEXT_MUTED = "#8B9BB4"
ACCENT_CYAN = "#22D3EE"

SEVERITY_COLORS = {
    "CRITICAL": "#EF4444",
    "HIGH": "#F97316",
    "MEDIUM": "#EAB308",
    "LOW": "#22C55E"
}

CATEGORY_COLORS = {
    "Emergency": "#EF4444",
    "Help Request": "#F97316",
    "Damage Report": "#8B5CF6",
    "Information/Update": "#06B6D4",
    "General": "#64748B"
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {TEXT_PRIMARY};
}}

code, pre, .mono {{
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Clean up Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

/* Glass card styling */
.glass-card {{
    background: {SURFACE_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 14px;
    padding: 1.25rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    margin-bottom: 1rem;
}}

/* KPI Metric Cards */
.kpi-container {{
    background: {SURFACE_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    padding: 1rem 1.25rem;
    position: relative;
    overflow: hidden;
}}

.kpi-container::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: {ACCENT_CYAN};
}}

.kpi-title {{
    color: {TEXT_MUTED};
    font-size: 0.825rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}}

.kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1.2;
}}

.kpi-delta {{
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 0.25rem;
}}

/* Status Pills */
.status-pill {{
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    border: 1px solid {BORDER_COLOR};
    background: {ELEVATED_COLOR};
    color: {TEXT_MUTED};
    margin-right: 0.5rem;
}}

.pill-active {{
    border-color: rgba(34, 211, 238, 0.4);
    color: {ACCENT_CYAN};
    background: rgba(34, 211, 238, 0.1);
}}

/* Table adjustments */
.stDataFrame {{
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}}
</style>
"""

def inject_custom_theme():
    """Inject CrisisLens custom CSS into Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def setup_plotly_theme():
    """Register and apply CrisisLens custom Plotly dark template."""
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=12),
        title=dict(font=dict(color=TEXT_PRIMARY, size=14)),
        legend=dict(font=dict(color=TEXT_MUTED, size=11)),
        xaxis=dict(
            gridcolor=BORDER_COLOR,
            linecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_MUTED),
            title=dict(font=dict(color=TEXT_MUTED))
        ),
        yaxis=dict(
            gridcolor=BORDER_COLOR,
            linecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_MUTED),
            title=dict(font=dict(color=TEXT_MUTED))
        ),
        hoverlabel=dict(
            bgcolor=SURFACE_COLOR,
            bordercolor=BORDER_COLOR,
            font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY)
        )
    )
    pio.templates["crisislens"] = template
    pio.templates.default = "crisislens"

setup_plotly_theme()

def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply unified styling and remove modebar clutter."""
    fig.update_layout(
        template="crisislens",
        margin=dict(l=20, r=20, t=35, b=25),
        hovermode="closest"
    )
    return fig
