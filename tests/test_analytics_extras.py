"""
Tests for surge details, incident clustering, the HTML briefing, token highlighting, urgency casualty
signals, and the analytics helpers added for the dashboard.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    generate_situation_summary,
    get_activity_timeseries,
    get_needs_counts,
    get_urgency_counts,
)
from src.anomaly import detect_event_surges
from src.briefing import generate_html_briefing, generate_markdown_briefing
from src.clustering import cluster_event_incidents
from src.explainability import highlight_tweet_tokens
from src.urgency import calculate_urgency_breakdown

# ---- surges ---------------------------------------------------------------------------------------

def _surge_frame():
    base = datetime(2018, 1, 1)
    rows = []
    for i in range(20):
        for j in range(2):
            rows.append({"tweet_id": f"t{i}_{j}", "event": "ev", "timestamp": (base + timedelta(hours=i, minutes=j * 10)).isoformat(),
                         "urgency_level": "LOW", "clean_text": "routine weather update", "locations": ["Springfield"]})
    for j in range(50):
        rows.append({"tweet_id": f"b{j}", "event": "ev", "timestamp": (base + timedelta(hours=21, minutes=j)).isoformat(),
                     "urgency_level": "CRITICAL", "clean_text": "bridge collapsed rescue boats needed riverside",
                     "locations": np.array(["Riverside"], dtype=object)})
    return pd.DataFrame(rows)

def test_surge_alerts_report_real_terms_and_places():
    alerts = detect_event_surges(_surge_frame(), z_threshold=2.0)
    crit = [a for a in alerts if a["alert_type"] == "Critical Share Surge"]
    assert crit, "expected a critical-share surge"
    for a in crit:
        # previously a hardcoded ["critical", "emergency", "immediate"] and place "All"
        assert a["top_terms"] != ["critical", "emergency", "immediate"]
        assert "bridge" in a["top_terms"] or "rescue" in a["top_terms"]
        assert a["place"] == "Riverside"
    volume = [a for a in alerts if a["alert_type"] == "Volume Burst"]
    assert volume and volume[0]["place"] == "Riverside"
    assert "need" not in volume[0]["top_terms"]  # stopword filtered

# ---- clustering -----------------------------------------------------------------------------------

def _cluster_frame(n_filler=40):
    rows = []
    for i in range(n_filler):  # many unrelated low-urgency tweets listed first
        rows.append({"tweet_id": f"f{i}", "event": "ev", "clean_text": f"unrelated topic number {i} alpha{i} beta{i}",
                     "raw_unescaped_text": f"unrelated {i}", "category": "General", "urgency_score": 0.5, "confidence": 0.9,
                     "locations": [], "timestamp": "2017-08-26 10:00:00"})
    for i in range(4):  # near-duplicate urgent reports listed last
        rows.append({"tweet_id": f"u{i}", "event": "ev", "clean_text": "people trapped on roof need boat rescue houston street flooding",
                     "raw_unescaped_text": f"URGENT {i} people trapped on roof", "category": "Emergency", "urgency_score": 9.0 + i / 10,
                     "confidence": 0.9, "locations": np.array(["Houston"], dtype=object), "timestamp": "2017-08-26 12:00:00"})
    return pd.DataFrame(rows)

def test_urgent_tweets_late_in_the_file_are_still_clustered():
    # With head(n) sampling and n=10, the four urgent duplicates (rows 40-43) were never even considered.
    incidents = cluster_event_incidents(_cluster_frame(), similarity_threshold=0.6, max_sample_per_event=10)
    assert incidents, "urgent duplicate reports must form an incident"
    top = incidents[0]
    assert top["dominant_category"] == "Emergency"
    assert top["size"] == 4
    assert top["places"] == ["Houston"]

def test_each_tweet_belongs_to_at_most_one_incident():
    incidents = cluster_event_incidents(_cluster_frame(), similarity_threshold=0.6, max_sample_per_event=60)
    ids = [tid for inc in incidents for tid in inc["sample_tweet_ids"]]
    assert len(ids) == len(set(ids))

# ---- briefing -------------------------------------------------------------------------------------

def _briefing_inputs():
    df = pd.DataFrame([
        {"tweet_id": "1", "urgency_level": "CRITICAL", "urgency_score": 9.0, "category": "Emergency", "confidence": 0.9,
         "raw_unescaped_text": "<b>SOS</b> 5 trapped & hurt", "locations": ["Houston"], "matched_terms": ["critical_kw:trapped"],
         "needs": ["Rescue/Evacuation", "Medical"], "quantities": [{"kind": "trapped", "count": 5}]},
        {"tweet_id": "2", "urgency_level": "LOW", "urgency_score": 1.0, "category": "General", "confidence": 0.8,
         "raw_unescaped_text": "praying", "locations": [], "matched_terms": [], "needs": [], "quantities": []},
    ])
    incidents = [{"incident_id": "INC-1", "size": 3, "max_urgency": 9.0, "dominant_category": "Emergency",
                  "representative_tweet": "boats needed", "places": ["Houston"], "time_span": "2017-08-26"}]
    surges = [{"alert_type": "Volume Burst", "event": "ev", "details": "peak", "z_score": 3.2}]
    return df, {"summary_text": "Summary."}, incidents, surges

def test_html_briefing_is_real_structured_html_not_a_pre_dump():
    html_out = generate_html_briefing(*_briefing_inputs(), {"Event": "ev"})
    assert "<pre>" not in html_out
    for fragment in ("<table>", "Critical dispatches", "Surge alerts", "Top incident clusters", "@media print", "5 trapped"):
        assert fragment in html_out, fragment
    assert "Rescue/Evacuation" in html_out or "Medical" in html_out  # needs summary
    # untrusted tweet text is escaped, never injected
    assert "<b>SOS</b>" not in html_out
    assert "&lt;b&gt;SOS&lt;/b&gt;" in html_out

def test_markdown_briefing_includes_needs_and_figures():
    md = generate_markdown_briefing(*_briefing_inputs(), {})
    assert "Most Reported Needs" in md
    assert "Reported figures" in md and "5 trapped" in md

def test_html_briefing_handles_an_empty_view():
    out = generate_html_briefing(pd.DataFrame(), {"summary_text": ""}, [], [], {})
    assert "No critical alerts found" in out and "<!DOCTYPE html>" in out

# ---- highlighting ---------------------------------------------------------------------------------

def test_highlight_cannot_be_corrupted_by_markup_like_search_terms():
    # Previously terms like "span", "class", "loc" were substituted into already-inserted <span class=...> markup.
    out = highlight_tweet_tokens("Flood near Houston, span loc class", ["Houston"], ["high_kw:flood"],
                                 [("span", 0.4), ("loc", 0.3), ("class", 0.2)])
    assert out.count('<span class="cl-hl') == out.count("</span>")
    assert 'class="cl-hl cl-hl--loc">Houston</span>' in out
    assert 'class="cl-hl cl-hl--urg">Flood</span>' in out
    assert 'cl-hl--feat">span</span>' in out
    # the only tags in the output are our own spans inside one wrapper div
    import re
    tags = re.findall(r"<[^>]+>", out)
    assert all(t.startswith(("<span class=\"cl-hl", "</span", "<div class=\"cl-tweet", "</div")) for t in tags), tags

def test_highlight_escapes_everything_and_does_not_split_entities():
    out = highlight_tweet_tokens("AT&T <script>alert(1)</script> amp", [], [], [("amp", 0.5), ("script", 0.5)])
    assert "<script>" not in out
    assert "&amp;T" in out                      # the ampersand is escaped exactly once
    assert "&<span" not in out                  # a feature token "amp" must not slice the &amp; entity

def test_highlight_prefers_places_over_urgency_over_features_and_longest_match():
    out = highlight_tweet_tokens("collapsed building in Fort McMurray", ["Fort McMurray"],
                                 ["critical_kw:collapsed building"], [("building", 0.2), ("fort", 0.1)])
    assert 'cl-hl--loc">Fort McMurray</span>' in out
    assert 'cl-hl--urg">collapsed building</span>' in out
    assert 'cl-hl--feat' not in out             # "building" and "fort" were already covered

def test_highlight_empty():
    assert highlight_tweet_tokens("", ["x"], [], []) == ""

# ---- urgency --------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Greece wildfires that killed 83 suspected as arson",
    "Ecuador quake death toll hits 413",
    "Over 90 deaths reported after the fires",
])
def test_casualty_reports_are_no_longer_scored_below_medium(text):
    score, level, terms, breakdown = calculate_urgency_breakdown(text, category="Emergency")
    assert level in {"MEDIUM", "HIGH", "CRITICAL"}, (score, level)
    assert any(t.startswith("casualty_kw:") for t in terms)
    casualty = [b for b in breakdown if b["signal"].startswith("Casualty Report")]
    assert len(casualty) == 1 and casualty[0]["points"] == 2.0   # scored once however many terms match

def test_casualty_signal_does_not_turn_news_into_a_dispatch():
    score, level, _, _ = calculate_urgency_breakdown("Death toll rises to 413, 413 died, victims named", category="Emergency")
    assert level != "CRITICAL"

def test_casualty_bonus_is_not_stacked_on_a_critical_keyword():
    # "dead" already scores as a critical keyword; "killed" must not be counted on top of it.
    # Before this rule such death-toll news reached 8.0 (CRITICAL); it now keeps the keyword-only score.
    text = "16 dead in PR as of 2 days ago, 26 killed in Mati"
    score, level, terms, breakdown = calculate_urgency_breakdown(text, category="Emergency")
    assert "critical_kw:dead" in terms
    assert not any(t.startswith("casualty_kw:") for t in terms)
    assert not any(b["signal"].startswith("Casualty Report") for b in breakdown)
    assert level != "CRITICAL"

    # ...while a report with no other signal is still lifted out of LOW
    _, level2, terms2, _ = calculate_urgency_breakdown("Greece wildfires that killed 83", category="Emergency")
    assert level2 == "MEDIUM" and any(t.startswith("casualty_kw:") for t in terms2)

def test_breakdown_still_sums_to_the_score():
    # Chosen so the score does not saturate at 10: the waterfall must add up exactly.
    text = "Roads flooded and 3 killed in the storm"
    score, _, _, breakdown = calculate_urgency_breakdown(text, category="Damage Report", exclamation_count=0)
    assert 0 < score < 10
    assert round(sum(b["points"] for b in breakdown), 1) == score

# ---- analytics helpers ----------------------------------------------------------------------------

def test_activity_timeseries_picks_hourly_for_short_spans_and_daily_for_long(rich_dataframe):
    short, freq = get_activity_timeseries(rich_dataframe)
    assert freq == "h"
    assert short["tweets"].sum() == len(rich_dataframe)
    assert short["priority"].sum() == int(rich_dataframe["urgency_level"].isin(["CRITICAL", "HIGH"]).sum())
    assert short["critical"].sum() == int((rich_dataframe["urgency_level"] == "CRITICAL").sum())

    long_df = rich_dataframe.copy()
    long_df["timestamp"] = pd.Timestamp("2017-01-01", tz="UTC") + pd.to_timedelta(np.arange(len(long_df)) * 5, unit="D")
    daily, freq2 = get_activity_timeseries(long_df)
    assert freq2 == "D"
    assert daily["tweets"].sum() == len(long_df)

def test_activity_timeseries_empty_inputs():
    frame, _ = get_activity_timeseries(pd.DataFrame())
    assert frame.empty and list(frame.columns) == ["timestamp", "tweets", "priority", "critical"]
    assert get_activity_timeseries(pd.DataFrame({"timestamp": ["not a date"], "tweet_id": ["1"]}))[0].empty

def test_needs_counts_handle_numpy_arrays(rich_dataframe):
    df = rich_dataframe.copy()
    df["needs"] = df["needs"].apply(lambda n: np.array(n, dtype=object))
    counts = get_needs_counts(df)
    assert counts["Water"] == counts["Food"] == int((rich_dataframe["needs"].apply(len) > 0).sum())
    assert get_needs_counts(pd.DataFrame()).empty

def test_urgency_counts_always_list_every_level():
    counts = get_urgency_counts(pd.DataFrame({"urgency_level": ["LOW", "LOW", "HIGH"]}))
    assert list(counts.index) == ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert counts["CRITICAL"] == 0 and counts["LOW"] == 2

def test_situation_summary_mentions_needs_and_unverified_figures(rich_dataframe):
    one_event = rich_dataframe[rich_dataframe["event"] == "hurricane_harvey_2017"].copy()
    one_event["quantities"] = [[{"kind": "killed", "count": 47}]] * 3 + [[{"kind": "killed", "count": 72_000}]] + \
        [[]] * (len(one_event) - 4)
    summary = generate_situation_summary(one_event)
    assert "Most reported needs" in summary["summary_text"]
    assert "unverified" in summary["summary_text"]
    # the consensus value wins over one outlier, and the text says how many tweets cite it
    assert summary["reported_figures"]["killed"] == {"count": 47, "tweets": 3}
    assert "47 killed (3 tweets)" in summary["summary_text"]
    assert "72,000" not in summary["summary_text"]
    assert "Water" in summary["top_needs"]
    empty = generate_situation_summary(pd.DataFrame())
    assert empty["top_needs"] == {} and empty["reported_figures"] == {}

def test_figures_are_not_summarised_across_several_events(rich_dataframe):
    assert rich_dataframe["event"].nunique() > 1
    summary = generate_situation_summary(rich_dataframe)
    assert summary["reported_figures"] == {}
    assert "unverified" not in summary["summary_text"]

def test_most_reported_figures_ties_go_to_the_larger_value():
    from src.analytics import most_reported_figures
    column = [[{"kind": "killed", "count": 10}], [{"kind": "killed", "count": 50}], [{"kind": "injured", "count": 3}]]
    assert most_reported_figures(column) == {"killed": {"count": 50, "tweets": 1}, "injured": {"count": 3, "tweets": 1}}
