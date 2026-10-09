"""
Tests for src/privacy.py (phone / email / street-address redaction).
"""

import pandas as pd
import pytest

from src.privacy import add_redacted_columns, apply_redaction, redact_pii

def test_the_real_help_request_that_exposed_a_home_address_and_phone_number():
    text = "Please help, this lady needs to be rescue Marilyn guidry ,6210 Charles ave, Port Arthur , tx 713 5427751, 80 year old Water waist #harvey"
    out = redact_pii(text)
    assert "6210" not in out and "713" not in out and "5427751" not in out
    assert "[address]" in out and "[phone]" in out
    # the useful operational content survives
    assert "Port Arthur" in out and "needs to be rescue" in out and "#harvey" in out

@pytest.mark.parametrize("text, secret", [
    ("call 713-542-7751 now", "542-7751"),
    ("call (713) 542 7751", "542 7751"),
    ("text +1 713.542.7751", "713.542.7751"),
    ("ring 542-7751 asap", "542-7751"),
    ("contact +44 20 7946 0958 for info", "7946"),
    ("whatsapp +593 99 123 4567", "4567"),
])
def test_phone_formats_are_redacted(text, secret):
    out = redact_pii(text)
    assert "[phone]" in out
    assert secret not in out

def test_email_addresses_are_redacted_but_mentions_are_kept():
    out = redact_pii("write to help.desk+harvey@example.org or ask @RedCross and @realDonaldTrump")
    assert "[email]" in out and "example.org" not in out
    assert "@RedCross" in out and "@realDonaldTrump" in out

@pytest.mark.parametrize("text", [
    "We are at 1420 Main Street and need a boat", "stuck at 22 Oak Ridge Rd", "house 77 elm ave flooded",
])
def test_street_addresses_are_redacted(text):
    assert "[address]" in redact_pii(text)

@pytest.mark.parametrize("text", [
    "Death toll hits 413 in Ecuador quake",              # a count
    "50 people on road to the shelter",                   # a quantity near a street word
    "2,000 homes destroyed, 1.7 million stranded",        # figures with separators
    "2017-08-30 11:16 flooding near Houston",             # a timestamp
    "Category 4 storm, winds 130 mph, 12 buses on way",   # numbers next to street-like words
    "Hurricane Harvey made landfall on 08.25.2017",       # a date
    "Praying for everyone affected",                      # nothing to redact
])
def test_ordinary_numbers_and_text_are_left_alone(text):
    assert redact_pii(text) == text

def test_none_and_empty():
    assert redact_pii(None) == ""
    assert redact_pii("") == ""

def test_view_switches_between_redacted_and_original_without_leaking_helpers():
    df = pd.DataFrame({
        "tweet_id": ["1"],
        "raw_unescaped_text": ["call 713 5427751 at 6210 Charles ave"],
        "tweet_text": ["call 713 5427751 at 6210 Charles ave"],
        "clean_text": ["call 713 5427751 at 6210 charles ave"],
    })
    prepared = add_redacted_columns(df)
    assert "raw_unescaped_text__redacted" in prepared.columns

    hidden = apply_redaction(prepared, redact=True)
    for col in ("raw_unescaped_text", "tweet_text", "clean_text"):
        assert "5427751" not in hidden[col].iloc[0], col
    assert not [c for c in hidden.columns if c.endswith("__redacted")]  # exports cannot carry both versions

    shown = apply_redaction(prepared, redact=False)
    assert "5427751" in shown["raw_unescaped_text"].iloc[0]
    assert not [c for c in shown.columns if c.endswith("__redacted")]

def test_apply_redaction_is_a_no_op_on_frames_without_redacted_columns():
    df = pd.DataFrame({"raw_unescaped_text": ["call 713 5427751"]})
    assert apply_redaction(df, redact=True).equals(df)
