"""
Unit tests for human review feedback storage and safety.
"""

from src.database import init_db, save_feedback, load_all_feedback, get_feedback_stats

def test_save_and_load_feedback(tmp_path):
    test_db = tmp_path / "test_feedback.db"
    init_db(test_db)

    # Insert feedback
    f_id = save_feedback(
        tweet_id="123456",
        original_category="General",
        corrected_category="Emergency",
        original_urgency="LOW",
        corrected_urgency="CRITICAL",
        user_notes="Direct plea for boat evacuation",
        db_path=test_db
    )
    assert f_id > 0

    df = load_all_feedback(test_db)
    assert len(df) == 1
    assert df.iloc[0]["tweet_id"] == "123456"
    assert df.iloc[0]["corrected_category"] == "Emergency"
    assert df.iloc[0]["corrected_urgency"] == "CRITICAL"
    assert df.iloc[0]["user_notes"] == "Direct plea for boat evacuation"

    stats = get_feedback_stats(test_db)
    assert stats["total_feedback_count"] == 1
    assert stats["category_corrections"] == 1
    assert stats["urgency_corrections"] == 1

def test_sql_injection_defense(tmp_path):
    test_db = tmp_path / "test_feedback_sqli.db"
    init_db(test_db)

    # Malicious payload in user_notes and tweet_id
    malicious_note = "'); DROP TABLE feedback; --"
    malicious_id = "100' OR '1'='1"

    f_id = save_feedback(
        tweet_id=malicious_id,
        original_category="General",
        corrected_category="Emergency",
        original_urgency="LOW",
        corrected_urgency="HIGH",
        user_notes=malicious_note,
        db_path=test_db
    )
    assert f_id > 0

    # Ensure table was NOT dropped and row was safely stored as literal text
    df = load_all_feedback(test_db)
    assert len(df) == 1
    assert df.iloc[0]["user_notes"] == malicious_note
    assert df.iloc[0]["tweet_id"] == malicious_id
