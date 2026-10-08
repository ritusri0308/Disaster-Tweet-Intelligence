"""
Briefing export module for CrisisLens.
Generates executive Markdown and self-contained HTML situation reports without LLM calls.
"""

import html
from typing import Dict, Any, List
import pandas as pd

from src.config import APP_NAME, APP_TAGLINE, DISCLAIMER, DATASET_NAME

def generate_markdown_briefing(
    df: pd.DataFrame,
    summary_data: Dict[str, Any],
    top_incidents: List[Dict[str, Any]],
    surge_alerts: List[Dict[str, Any]],
    active_filters: Dict[str, Any]
) -> str:
    """Generate executive Markdown situation briefing."""
    total = len(df)
    critical_cnt = (df["urgency_level"] == "CRITICAL").sum() if "urgency_level" in df.columns else 0
    high_cnt = (df["urgency_level"] == "HIGH").sum() if "urgency_level" in df.columns else 0
    help_cnt = (df["category"] == "Help Request").sum() if "category" in df.columns else 0

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
        "",
        "## 2. Active View & Applied Filters"
    ]

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

    crit_df = df[df["urgency_level"] == "CRITICAL"] if ("urgency_level" in df.columns and not df.empty) else pd.DataFrame()
    if not crit_df.empty and "urgency_score" in crit_df.columns:
        crit_df = crit_df.sort_values("urgency_score", ascending=False).head(10)
    elif not crit_df.empty:
        crit_df = crit_df.head(10)
    if not crit_df.empty:
        for idx, (_, row) in enumerate(crit_df.iterrows(), 1):
            text = row.get("raw_unescaped_text", row.get("tweet_text", ""))
            places = ", ".join(row.get("locations", [])) if isinstance(row.get("locations"), list) else ""
            terms = ", ".join(row.get("matched_terms", [])) if isinstance(row.get("matched_terms"), list) else ""
            lines.append(
                f"{idx}. **[{row.get('urgency_level')}] Score {row.get('urgency_score')}/10** | *{row.get('category')}* (Conf: {row.get('confidence', 1.0):.0%})\n"
                f"   - **Places**: {places or 'None'}\n"
                f"   - **Text**: \"{text}\"\n"
                f"   - **Triggers**: `{terms}`\n"
            )
    else:
        lines.append("- *No critical alerts found.*")

    lines.extend([
        "",
        "---",
        "### Legal & Data Limitations",
        "- Data derived from historical HumAID benchmark (2016-2019). Coordinates geocoded from place mentions, not GPS.",
        f"- Generated automatically by {APP_NAME}."
    ])

    return "\n".join(lines)

def generate_html_briefing(
    df: pd.DataFrame,
    summary_data: Dict[str, Any],
    top_incidents: List[Dict[str, Any]],
    surge_alerts: List[Dict[str, Any]],
    active_filters: Dict[str, Any]
) -> str:
    """Generate self-contained executive HTML situation report."""
    md_text = generate_markdown_briefing(df, summary_data, top_incidents, surge_alerts, active_filters)
    escaped_summary = html.escape(summary_data.get("summary_text", ""))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{APP_NAME} Situation Briefing</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0B0F17; color: #E6EDF7; margin: 0; padding: 40px; line-height: 1.6; }}
  .container {{ max-width: 900px; margin: 0 auto; background: #111827; border: 1px solid #1F2A3D; border-radius: 12px; padding: 36px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
  h1 {{ color: #22D3EE; margin-top: 0; border-bottom: 1px solid #1F2A3D; padding-bottom: 16px; font-size: 26px; }}
  h2 {{ color: #38BDF8; font-size: 20px; border-bottom: 1px solid #1F2A3D; padding-bottom: 8px; margin-top: 32px; }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 12px; margin-right: 6px; }}
  .badge-critical {{ background: rgba(239, 68, 68, 0.2); border: 1px solid #EF4444; color: #EF4444; }}
  .badge-high {{ background: rgba(249, 115, 22, 0.2); border: 1px solid #F97316; color: #F97316; }}
  .card {{ background: #162033; border: 1px solid #1F2A3D; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .disclaimer {{ background: rgba(234, 179, 8, 0.1); border-left: 4px solid #EAB308; padding: 12px; font-size: 13px; color: #FEF08A; margin-bottom: 24px; border-radius: 4px; }}
  pre {{ background: #0B0F17; padding: 12px; border-radius: 6px; overflow-x: auto; color: #A5F3FC; }}
</style>
</head>
<body>
<div class="container">
  <h1>🚨 {APP_NAME} - Emergency Situation Report</h1>
  <div class="disclaimer">
    <strong>Decision-Support Notice:</strong> {DISCLAIMER}
  </div>
  <div class="card">
    <h3 style="margin-top:0;color:#22D3EE;">Executive Summary</h3>
    <p>{escaped_summary}</p>
    <div>
      <span class="badge badge-critical">Total Signals: {len(df):,}</span>
      <span class="badge badge-critical">Critical: {(df['urgency_level'] == 'CRITICAL').sum() if 'urgency_level' in df.columns else 0:,}</span>
      <span class="badge badge-high">Help Requests: {(df['category'] == 'Help Request').sum() if 'category' in df.columns else 0:,}</span>
    </div>
  </div>
  <pre>{html.escape(md_text)}</pre>
</div>
</body>
</html>"""
