"""
Life OS — FeedbackCollector.get_diagnostics() Test Suite

Tests the operational diagnostics method that provides health information
about the feedback pipeline: total counts, breakdowns by type, top dismissed
domains, recency metrics, and cross-DB semantic fact counts.
"""

import json
import uuid

import pytest
from datetime import datetime, timezone, timedelta

from services.feedback_collector.collector import FeedbackCollector


def _insert_feedback(conn, feedback_type, action_type, context=None, timestamp=None):
    """Helper to insert a feedback_log row directly."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO feedback_log
           (id, timestamp, action_id, action_type, feedback_type,
            response_latency_seconds, context, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            ts,
            f"action-{uuid.uuid4().hex[:8]}",
            action_type,
            feedback_type,
            1.0,
            json.dumps(context or {}),
            None,
        ),
    )


def test_diagnostics_empty(db, user_model_store):
    """get_diagnostics() returns expected keys when feedback_log is empty."""
    collector = FeedbackCollector(db, user_model_store)
    diag = collector.get_diagnostics()

    assert diag["total_feedback_entries"] == 0
    assert diag["by_feedback_type"] == {}
    assert diag["by_action_type"] == {}
    assert diag["top_dismissed_domains"] == {}
    assert diag["feedback_last_24h"] == 0
    assert diag["last_feedback_at"] is None
    assert diag["semantic_facts_from_feedback"] == 0
    assert "error" not in diag


def test_diagnostics_counts_correct(db, user_model_store):
    """Counts are correct after inserting 3 dismissed + 2 acted_on entries."""
    collector = FeedbackCollector(db, user_model_store)

    with db.get_connection("preferences") as conn:
        for _ in range(3):
            _insert_feedback(conn, "dismissed", "notification", context={"domain": "email"})
        for _ in range(2):
            _insert_feedback(conn, "engaged", "notification")

    diag = collector.get_diagnostics()

    assert diag["total_feedback_entries"] == 5
    assert diag["by_feedback_type"]["dismissed"] == 3
    assert diag["by_feedback_type"]["engaged"] == 2
    assert diag["by_action_type"]["notification"] == 5


def test_diagnostics_top_dismissed_domains(db, user_model_store):
    """top_dismissed_domains shows correct domains and counts."""
    collector = FeedbackCollector(db, user_model_store)

    with db.get_connection("preferences") as conn:
        for _ in range(4):
            _insert_feedback(conn, "dismissed", "notification", context={"domain": "email"})
        for _ in range(2):
            _insert_feedback(conn, "dismissed", "notification", context={"domain": "calendar"})
        # Engaged entries should NOT appear in dismissed domains
        _insert_feedback(conn, "engaged", "notification", context={"domain": "email"})

    diag = collector.get_diagnostics()

    assert diag["top_dismissed_domains"]["email"] == 4
    assert diag["top_dismissed_domains"]["calendar"] == 2
    assert len(diag["top_dismissed_domains"]) == 2


def test_diagnostics_feedback_last_24h(db, user_model_store):
    """feedback_last_24h counts only recent entries."""
    collector = FeedbackCollector(db, user_model_store)

    old_ts = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    recent_ts = datetime.now(timezone.utc).isoformat()

    with db.get_connection("preferences") as conn:
        # 2 old entries
        _insert_feedback(conn, "dismissed", "notification", timestamp=old_ts)
        _insert_feedback(conn, "dismissed", "notification", timestamp=old_ts)
        # 1 recent entry
        _insert_feedback(conn, "engaged", "notification", timestamp=recent_ts)

    diag = collector.get_diagnostics()

    assert diag["total_feedback_entries"] == 3
    assert diag["feedback_last_24h"] == 1


def test_diagnostics_last_feedback_at(db, user_model_store):
    """last_feedback_at shows the most recent timestamp."""
    collector = FeedbackCollector(db, user_model_store)

    ts1 = "2025-01-01T00:00:00+00:00"
    ts2 = "2025-06-15T12:00:00+00:00"

    with db.get_connection("preferences") as conn:
        _insert_feedback(conn, "dismissed", "notification", timestamp=ts1)
        _insert_feedback(conn, "engaged", "notification", timestamp=ts2)

    diag = collector.get_diagnostics()

    assert diag["last_feedback_at"] == ts2


def test_diagnostics_semantic_facts_from_feedback(db, user_model_store):
    """semantic_facts_from_feedback counts notification_preference facts."""
    collector = FeedbackCollector(db, user_model_store)

    # Insert notification_preference facts via the user model store
    user_model_store.update_semantic_fact(
        key="notification_irrelevant_email",
        category="notification_preference",
        value="User dismisses email notifications",
        confidence=0.4,
    )
    user_model_store.update_semantic_fact(
        key="notification_irrelevant_social",
        category="notification_preference",
        value="User dismisses social notifications",
        confidence=0.3,
    )
    # A fact in a different category should NOT be counted
    user_model_store.update_semantic_fact(
        key="user_likes_coffee",
        category="explicit_preference",
        value="User prefers coffee",
        confidence=0.9,
    )

    diag = collector.get_diagnostics()

    assert diag["semantic_facts_from_feedback"] == 2
