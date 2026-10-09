"""
Briefing export module for CrisisLens.
Generates executive Markdown and self-contained HTML situation reports without LLM calls.

Every dynamic value in the HTML report is escaped; tweet text is untrusted input.
"""

import html
from collections import Counter
from typing import Dict, Any, List
import pandas as pd

from src.config import APP_NAME, APP_TAGLINE, DISCLAIMER, DATASET_NAME
from src.entities import format_quantity
from src.utils import as_list

def _count(df: pd.DataFrame, column: str, value: str) -> int:
    return int((df[column] == value).sum()) if column in df.columns else 0

def _top_needs(df: pd.DataFrame, n: int = 5) -> List[tuple]:
    """Most common detected need types as (need, count) pairs."""
    if "needs" not in df.columns or df.empty:
        return []
    counts = Counter(need for needs in df["needs"] for need in as_list(needs))
    return counts.most_common(n)

def _critical_dispatches(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The n highest-scoring CRITICAL tweets."""
    if df.empty or "urgency_level" not in df.columns:
        return pd.DataFrame()
    crit_df = df[df["urgency_level"] == "CRITICAL"]
    if not crit_df.empty and "urgency_score" in crit_df.columns:
        crit_df = crit_df.sort_values("urgency_score", ascending=False)
    return crit_df.head(n)

def _figures_text(row: pd.Series) -> str:
    return ", ".join(format_quantity(q) for q in as_list(row.get("quantities", [])) if isinstance(q, dict))

def generate_markdown_briefing(
    df: pd.DataFrame,
    summary_data: Dict[str, Any],
    top_incidents: List[Dict[str, Any]],
    surge_alerts: List[Dict[str, Any]],
    active_filters: Dict[str, Any]
) -> str:
    """Generate executive Markdown situation briefing."""
    total = len(df)
    critical_cnt = _count(df, "urgency_level", "CRITICAL")
    high_cnt = _count(df, "urgency_level", "HIGH")
    help_cnt = _count(df, "category", "Help Request")

    lines = [
        f"# 🚨 {APP_NAME} - Emergency Situation Briefing",
        f"**Tagline**: {APP_TAGLINE} | **Dataset**: {DATASET_NAME}",
        f"> **Notice**: {DISCLAIMER}",
        "",
        "## 1. Executive Summary & KPIs",
        f"- **Total Disaster Signals Analyzed**: {total:,}",
        f"- **CRITICAL Priority Alerts**: {critical_cnt:,}",
        f"- **HIGH Priority Alerts**: {high_cnt:,}",
        f"- **Help / Rescue Requests**: {help_cnt:,}",
        f"- **Key Intelligence Summary**: {summary_data.get('summary_text', 'N/A')}",
    ]
    needs = _top_needs(df)
    if needs:
        lines.append("- **Most Reported Needs**: " + ", ".join(f"{n} ({c:,})" for n, c in needs))
    lines.extend(["", "## 2. Active View & Applied Filters"])

    if active_filters:
        for k, v in active_filters.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- *No restrictive filters applied (Full dataset view)*")

    lines.extend([
        "",
        "## 3. Detected Surge Alerts & Volume Bursts"
    ])

    if surge_alerts:
        for idx, a in enumerate(surge_alerts[:5], 1):
            lines.append(f"{idx}. **{a.get('alert_type', 'Alert')}** ({a.get('event', 'General')}): {a.get('details', '')} [z={a.get('z_score', 'N/A')}]")
    else:
        lines.append("- *No abnormal statistical volume surges detected in current window.*")

    lines.extend([
        "",
        "## 4. Top Clustered Incidents"
    ])

    if top_incidents:
        for idx, inc in enumerate(top_incidents[:5], 1):
            places_str = ", ".join(inc.get("places", [])) or "Unspecified"
            lines.append(
                f"### {idx}. [{inc.get('incident_id')}] {inc.get('dominant_category')} (Max Urgency: {inc.get('max_urgency')}/10)\n"
                f"- **Cluster Size**: {inc.get('size')} related messages\n"
                f"- **Impacted Locations**: {places_str}\n"
                f"- **Representative Text**: \"{inc.get('representative_tweet')}\"\n"
            )
    else:
        lines.append("- *No multi-tweet incident clusters found in active view.*")

    lines.extend([
        "",
        "## 5. Top 10 Critical Dispatches Requiring Action"
    ])

    crit_df = _critical_dispatches(df)
    if not crit_df.empty:
        for idx, (_, row) in enumerate(crit_df.iterrows(), 1):
            text = row.get("raw_unescaped_text", row.get("tweet_text", ""))
            places = ", ".join(str(p) for p in as_list(row.get("locations", [])))
            terms = ", ".join(str(t) for t in as_list(row.get("matched_terms", [])))
            figures = _figures_text(row)
            lines.append(
                f"{idx}. **[{row.get('urgency_level')}] Score {row.get('urgency_score')}/10** | *{row.get('category')}* (Conf: {row.get('confidence', 1.0):.0%})\n"
                f"   - **Places**: {places or 'None'}\n"
                f"   - **Text**: \"{text}\"\n"
                + (f"   - **Reported figures**: {figures}\n" if figures else "")
                + f"   - **Triggers**: `{terms}`\n"
            )
    else:
        lines.append("- *No critical alerts found.*")

    lines.extend([
        "",
        "---",
        "### Legal & Data Limitations",
        "- Data derived from historical HumAID benchmark (2016-2019). Coordinates geocoded from place mentions, not GPS.",
        "- Reported figures are claims made in tweets and are not verified counts.",
        f"- Generated automatically by {APP_NAME}."
    ])

    return "\n".join(lines)

_REPORT_CSS = """
  :root { --bg:#F8FAFC; --surface:#FFFFFF; --border:#E2E8F0; --text:#0F172A; --muted:#475569; --accent:#2563EB; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--text); margin: 0; padding: 32px 16px; line-height: 1.55; }
  .container { max-width: 920px; margin: 0 auto; }
  h1 { font-size: 1.7rem; margin: 0 0 4px; letter-spacing: -0.02em; }
  h2 { font-size: 1.2rem; margin: 2rem 0 0.75rem; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
  .tagline { color: var(--muted); margin-bottom: 20px; }
  .notice { background: #FEF9C3; border: 1px solid #FEF08A; color: #854D0E; padding: 12px 16px; border-radius: 10px; margin-bottom: 20px; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0; }
  .kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
  .kpi .label { color: var(--muted); font-size: 0.9rem; font-weight: 600; }
  .kpi .value { font-size: 2rem; font-weight: 700; letter-spacing: -0.03em; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; page-break-inside: avoid; }
  .muted { color: var(--muted); font-size: 0.95rem; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 8px; font-weight: 600; font-size: 0.9rem; margin-right: 6px; }
  .badge-critical { background: #FEE2E2; color: #B91C1C; }
  .badge-high { background: #FFEDD5; color: #C2410C; }
  .badge-cat { background: #EFF6FF; color: #1D4ED8; }
  table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  ul { padding-left: 1.2rem; }
  footer { margin-top: 2rem; color: var(--muted); font-size: 0.9rem; }
  @media print {
    body { background: #FFFFFF; padding: 0; }
    .card, .kpi { box-shadow: none; }
    h2 { page-break-after: avoid; }
  }
"""

def _e(value: Any) -> str:
    """Escape any value for safe inclusion in HTML."""
    return html.escape(str(value if value is not None else ""))

def generate_html_briefing(
    df: pd.DataFrame,
    summary_data: Dict[str, Any],
    top_incidents: List[Dict[str, Any]],
    surge_alerts: List[Dict[str, Any]],
    active_filters: Dict[str, Any]
) -> str:
    """Generate a self-contained, printable executive HTML situation report."""
    total = len(df)
    critical_cnt = _count(df, "urgency_level", "CRITICAL")
    high_cnt = _count(df, "urgency_level", "HIGH")
    help_cnt = _count(df, "category", "Help Request")

    parts: List[str] = []

    # KPIs
    parts.append(
        '<div class="kpis">'
        f'<div class="kpi"><div class="label">Total signals</div><div class="value">{total:,}</div></div>'
        f'<div class="kpi"><div class="label">Critical</div><div class="value">{critical_cnt:,}</div></div>'
        f'<div class="kpi"><div class="label">High priority</div><div class="value">{high_cnt:,}</div></div>'
        f'<div class="kpi"><div class="label">Help requests</div><div class="value">{help_cnt:,}</div></div>'
        '</div>'
    )

    # Executive summary + needs
    needs = _top_needs(df)
    needs_html = (
        f'<p class="muted"><strong>Most reported needs:</strong> {_e(", ".join(f"{n} ({c:,})" for n, c in needs))}</p>'
        if needs else ""
    )
    parts.append(
        '<h2>Executive summary</h2>'
        f'<div class="card"><p>{_e(summary_data.get("summary_text", ""))}</p>{needs_html}</div>'
    )

    # Filters
    if active_filters:
        items = "".join(f"<li><strong>{_e(k)}</strong>: {_e(v)}</li>" for k, v in active_filters.items())
        parts.append(f"<h2>Active view</h2><div class=\"card\"><ul>{items}</ul></div>")

    # Surge alerts
    parts.append("<h2>Surge alerts</h2>")
    if surge_alerts:
        rows = "".join(
            f"<tr><td>{_e(a.get('alert_type', 'Alert'))}</td><td>{_e(a.get('event', ''))}</td>"
            f"<td>{_e(a.get('details', ''))}</td><td>{_e(a.get('z_score', ''))}</td></tr>"
            for a in surge_alerts[:5]
        )
        parts.append(
            '<div class="card"><table><thead><tr><th>Type</th><th>Event</th><th>Details</th><th>z</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
        )
    else:
        parts.append('<div class="card muted">No abnormal statistical volume surges detected in the current window.</div>')

    # Incidents
    parts.append("<h2>Top incident clusters</h2>")
    if top_incidents:
        for inc in top_incidents[:5]:
            places = ", ".join(str(p) for p in inc.get("places", [])) or "Unspecified"
            parts.append(
                '<div class="card">'
                f'<div><span class="badge badge-cat">{_e(inc.get("dominant_category", ""))}</span>'
                f'<strong>{_e(inc.get("incident_id", ""))}</strong> &middot; {_e(inc.get("size", ""))} related messages'
                f' &middot; max urgency {_e(inc.get("max_urgency", ""))}/10</div>'
                f'<div class="muted">Locations: {_e(places)} &middot; Window: {_e(inc.get("time_span", ""))}</div>'
                f'<p>&ldquo;{_e(inc.get("representative_tweet", ""))}&rdquo;</p>'
                '</div>'
            )
    else:
        parts.append('<div class="card muted">No multi-tweet incident clusters found in the active view.</div>')

    # Critical dispatches
    parts.append("<h2>Critical dispatches requiring action</h2>")
    crit_df = _critical_dispatches(df)
    if not crit_df.empty:
        rows = []
        for _, row in crit_df.iterrows():
            text = row.get("raw_unescaped_text", row.get("tweet_text", ""))
            places = ", ".join(str(p) for p in as_list(row.get("locations", []))) or "None"
            figures = _figures_text(row)
            rows.append(
                "<tr>"
                f'<td><span class="badge badge-critical">{_e(row.get("urgency_score", ""))}/10</span></td>'
                f'<td>{_e(row.get("category", ""))}</td>'
                f'<td>{_e(text)}<div class="muted">Places: {_e(places)}'
                + (f" &middot; Reported: {_e(figures)}" if figures else "")
                + "</div></td></tr>"
            )
        parts.append(
            '<div class="card"><table><thead><tr><th>Score</th><th>Category</th><th>Report</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )
    else:
        parts.append('<div class="card muted">No critical alerts found.</div>')

    body = "\n".join(parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(APP_NAME)} Situation Briefing</title>
<style>{_REPORT_CSS}</style>
</head>
<body>
<div class="container">
  <h1>{_e(APP_NAME)} &ndash; Emergency Situation Report</h1>
  <div class="tagline">{_e(APP_TAGLINE)} &middot; {_e(DATASET_NAME)}</div>
  <div class="notice"><strong>Decision-support notice:</strong> {_e(DISCLAIMER)}</div>
  {body}
  <footer>
    Data derived from the historical HumAID benchmark (2016&ndash;2019). Coordinates are geocoded from place mentions, not GPS.
    Reported figures are claims made in tweets and are not verified counts. Generated automatically by {_e(APP_NAME)}.
  </footer>
</div>
</body>
</html>"""
