"""
Life OS — FeedbackCollector.get_diagnostics() v2 Test Suite

Covers the per-source, cross-tab, top_engaged_domains, and
top_dismissed_sources additions to get_diagnostics(). Kept in a
separate file from test_feedback_collector_diagnostics.py to avoid
conflicting edits from parallel improvement workers.
"""

import json
import uuid
from datetime import datetime, timezone

from services.feedback_collector.collector import FeedbackCollector


def _insert_feedback(conn, feedback_type, action_type, context=None, timestamp=None):
    """Insert a single feedback_log row with the given attributes."""
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
            json.dumps(context) if context is not None else None,
            None,
        ),
    )


def test_diagnostics_v2_empty(db, user_model_store):
    """New v2 keys are present and empty when feedback_log has no rows."""
    collector = FeedbackCollector(db, user_model_store)
    diag = collector.get_diagnostics()

    assert diag["by_source"] == {}
    assert diag["by_action_and_feedback"] == {}
    assert diag["top_engaged_domains"] == {}
    assert diag["top_dismissed_sources"] == {}
    assert "error" not in diag


def test_diagnostics_by_source_groups_by_context_source(db, user_model_store):
    """by_source aggregates counts grouped by context.source."""
    collector = FeedbackCollector(db, user_model_store)

    with db.get_connection("preferences") as conn:
        for _ in range(3):
            _insert_feedback(conn, "dismissed", "notification",
                             context={"source": "notification_manager"})
        for _ in range(2):
            _insert_feedback(conn, "engaged", "notification",
                             context={"source": "rules_engine"})
        _insert_feedback(conn, "dismissed", "notification",
                         context={"source": "prediction_engine"})

    diag = collector.get_diagnostics()

    assert diag["by_source"] == {
        "notification_manager": 3,
        "rules_engine": 2,
        "prediction_engine": 1,
    }


def test_diagnostics_by_source_excludes_missing_source(db, user_model_store):
    """Rows without context.source (or null context) are silently excluded from by_source."""
    collector = FeedbackCollector(db, user_model_store)

    with db.get_connection("preferences") as conn:
        # Rows with source
        _insert_feedback(conn, "dismissed", "notification",
                         context={"source": "notification_manager"})
        # Row with context but no source key
        _insert_feedback(conn, "dismissed", "notification",
                         context={"domain": "email"})
        # Row with null context entirely
        _insert_feedback(conn, "engaged", "notification", context=None)

    diag = collector.get_diagnostics()

    assert diag["by_source"] == {"notification_manager": 1}
    # Total still reflects all 3 rows
    assert diag["total_feedback_entries"] == 3
    assert "error" not in diag


def test_diagnostics_by_action_and_feedback_cross_tab(db, user_model_store):
    """by_action_and_feedback returns a 2D nested matrix of action × feedback."""
    collector = FeedbackCollector(db, user_model_store)

    with db.get_connection("preferences") as conn:
        for _ in range(5):
            _insert_feedback(conn, "dismissed", "notification")
        for _ in range(2):
            _insert_feedback(conn, "engaged", "notification")
        for _ in range(3):
            _insert_feedback(conn, "dismissed", "suggestion")

    diag = collector.get_diagnostics()
    matrix = diag["by_action_and_feedback"]

    assert matrix["notification"]["dismissed"] == 5
    assert matrix["notification"]["engaged"] == 2
    assert matrix["suggestion"]["dismissed"] == 3
    # Buckets only contain feedback types that actually occurred
    assert "engaged" not in matrix["suggestion"]


def test_diagnostics_top_engaged_domains(db, user_model_store):
    """top_engaged_domains returns ≤5 entries sorted by descending count."""
    collector = FeedbackCollector(db, user_model_store)

    with db.get_connection("preferences") as conn:
        # 7 distinct domains so we can verify the limit of 5
        for _ in range(6):
            _insert_feedback(conn, "engaged", "notification", context={"domain": "email"})
        for _ in range(5):
            _insert_feedback(conn, "engaged", "notification", context={"domain": "calendar"})
        for _ in range(4):
            _insert_feedback(conn, "engaged", "notification", context={"domain": "tasks"})
        for _ in range(3):
            _insert_feedback(conn, "engaged", "notification", context={"domain": "finance"})
        for _ in range(2):
            _insert_feedback(conn, "engaged", "notification", context={"domain": "social"})
        # Two extra domains that should fall outside top 5
        _insert_feedback(conn, "engaged", "notification", context={"domain": "shopping"})
        _insert_feedback(conn, "engaged", "notification", context={"domain": "news"})
        # Dismissed entry should NOT appear in engaged domains
        _insert_feedback(conn, "dismissed", "notification", context={"domain": "email"})

    diag = collector.get_diagnostics()
    engaged = diag["top_engaged_domains"]

    assert len(engaged) <= 5
    # Order: sorted descending by count
    assert list(engaged.keys())[0] == "email"
    assert engaged["email"] == 6
    assert engaged["calendar"] == 5
    # Lowest-count domains should be excluded
    assert "shopping" not in engaged
    assert "news" not in engaged


def test_diagnostics_top_dismissed_sources(db, user_model_store):
    """top_dismissed_sources returns ≤5 sources ranked by dismissal count."""
    collector = FeedbackCollector(db, user_model_store)

    with db.get_connection("preferences") as conn:
        for _ in range(10):
            _insert_feedback(conn, "dismissed", "notification",
                             context={"source": "notification_manager"})
        for _ in range(4):
            _insert_feedback(conn, "dismissed", "notification",
                             context={"source": "prediction_engine"})
        for _ in range(1):
            _insert_feedback(conn, "dismissed", "notification",
                             context={"source": "rules_engine"})
        # Engaged entries must NOT count toward dismissed sources
        for _ in range(20):
            _insert_feedback(conn, "engaged", "notification",
                             context={"source": "notification_manager"})

    diag = collector.get_diagnostics()
    sources = diag["top_dismissed_sources"]

    assert len(sources) <= 5
    assert sources["notification_manager"] == 10
    assert sources["prediction_engine"] == 4
    assert sources["rules_engine"] == 1
    # First key is the highest count
    assert list(sources.keys())[0] == "notification_manager"


def test_diagnostics_v2_preserves_existing_keys(db, user_model_store):
    """Adding v2 fields must not remove or shadow any existing diagnostics keys."""
    collector = FeedbackCollector(db, user_model_store)

    with db.get_connection("preferences") as conn:
        _insert_feedback(conn, "dismissed", "notification",
                         context={"domain": "email", "source": "notification_manager"})

    diag = collector.get_diagnostics()

    # Original keys must still be present
    for key in (
        "total_feedback_entries",
        "by_feedback_type",
        "by_action_type",
        "top_dismissed_domains",
        "feedback_last_24h",
        "last_feedback_at",
        "semantic_facts_from_feedback",
    ):
        assert key in diag, f"existing key {key} disappeared"

    # And the new keys are present too
    for key in (
        "by_source",
        "by_action_and_feedback",
        "top_engaged_domains",
        "top_dismissed_sources",
    ):
        assert key in diag, f"new key {key} missing"
