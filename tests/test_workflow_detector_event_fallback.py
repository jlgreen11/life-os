"""
Tests for WorkflowDetector event-based fallback when episodes table is empty.

When the episodes table has 0 qualifying rows (e.g. after a database rebuild),
_detect_interaction_workflows() must fall back to _detect_interaction_workflows_from_events()
which queries events.db directly for recurring multi-step email thread patterns.

Coverage:
- Fallback triggers when episodes = 0 but email.received events exist
- At least one email communication workflow is detected in fallback mode
- min_occurrences threshold is enforced (senders with too few sessions rejected)
- min_steps threshold is enforced (single-email sessions don't count)
- detect_workflows() end-to-end returns workflows from fallback
- Internal/telemetry event types are excluded from fallback
- Fallback returns empty list when no events match
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from services.workflow_detector.detector import WorkflowDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(dt: datetime) -> str:
    """Format a datetime as ISO 8601 string for SQLite storage."""
    return dt.isoformat()


def _insert_email_received(conn, sender: str, timestamp: datetime) -> str:
    """Insert an email.received event into the events table.

    Returns the generated event ID.
    """
    eid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                            email_from, email_to)
        VALUES (?, 'email.received', 'email', ?, 'normal', '{}', '{}', ?, NULL)
        """,
        (eid, _ts(timestamp), sender),
    )
    return eid


def _verify_no_episodes(db) -> None:
    """Assert that the episodes table is empty (test precondition)."""
    with db.get_connection("user_model") as conn:
        row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
    assert row[0] == 0, f"Expected 0 episodes, found {row[0]}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector(db, user_model_store):
    """A WorkflowDetector wired to temporary, fully-initialized databases."""
    return WorkflowDetector(db, user_model_store)


# ===========================================================================
# 1. Basic fallback detection
# ===========================================================================


class TestFallbackTriggersOnEmptyEpisodes:
    """Verify the fallback path activates when episodes = 0."""

    def test_fallback_detects_thread_workflow(self, detector, db):
        """Sender with 3 multi-email sessions within 12h triggers a workflow.

        Setup: 0 episodes, but one sender who sends 3-email threads on
        3 separate days (each thread within 12h = 1 session each).
        The fallback must detect this as a communication workflow.
        """
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        sender = "boss@company.com"

        with db.get_connection("events") as conn:
            for day_offset in range(3):
                # Three emails from the same sender within a few hours on each day.
                base = now - timedelta(days=10 - day_offset)
                _insert_email_received(conn, sender, base)
                _insert_email_received(conn, sender, base + timedelta(hours=1))
                _insert_email_received(conn, sender, base + timedelta(hours=2))

        workflows = detector._detect_interaction_workflows(lookback_days=30)

        assert len(workflows) >= 1
        names = [w["name"] for w in workflows]
        assert any(sender in name for name in names), (
            f"Expected workflow named after '{sender}', got: {names}"
        )

    def test_fallback_workflow_schema_keys_present(self, detector, db):
        """Fallback workflows contain all required schema keys."""
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        sender = "schema-check@example.com"

        with db.get_connection("events") as conn:
            for day_offset in range(3):
                base = now - timedelta(days=10 - day_offset)
                _insert_email_received(conn, sender, base)
                _insert_email_received(conn, sender, base + timedelta(hours=2))

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        assert len(workflows) >= 1

        wf = next(w for w in workflows if sender in w["name"])
        required_keys = {
            "name", "trigger_conditions", "steps",
            "typical_duration_minutes", "tools_used", "success_rate", "times_observed",
        }
        assert required_keys.issubset(wf.keys()), (
            f"Missing keys: {required_keys - wf.keys()}"
        )
        assert "email" in wf["tools_used"]
        assert len(wf["steps"]) >= 2  # min_steps = 2
        assert wf["success_rate"] == 1.0

    def test_fallback_trigger_conditions_reference_sender(self, detector, db):
        """trigger_conditions contain a reference to the sender address."""
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        sender = "alice@partner.org"

        with db.get_connection("events") as conn:
            for day_offset in range(3):
                base = now - timedelta(days=8 - day_offset)
                _insert_email_received(conn, sender, base)
                _insert_email_received(conn, sender, base + timedelta(hours=3))

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        wf = next((w for w in workflows if sender in w["name"]), None)
        assert wf is not None
        assert any(sender in tc for tc in wf["trigger_conditions"])


# ===========================================================================
# 2. min_occurrences threshold enforcement
# ===========================================================================


class TestMinOccurrencesEnforcement:
    """Verify that senders with fewer than min_occurrences sessions are rejected."""

    def test_sender_with_two_sessions_rejected(self, detector, db):
        """Sender with only 2 multi-email sessions (< min_occurrences=3) is rejected."""
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        sparse_sender = "sparse@example.com"

        with db.get_connection("events") as conn:
            # Only 2 sessions — below the min_occurrences=3 threshold.
            for day_offset in range(2):
                base = now - timedelta(days=5 - day_offset)
                _insert_email_received(conn, sparse_sender, base)
                _insert_email_received(conn, sparse_sender, base + timedelta(hours=2))

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        matching = [w for w in workflows if sparse_sender in w["name"]]
        assert len(matching) == 0, (
            f"Expected 0 workflows for sparse sender, got {len(matching)}"
        )

    def test_sender_with_three_sessions_accepted(self, detector, db):
        """Sender with exactly 3 multi-email sessions (== min_occurrences) is accepted."""
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        exact_sender = "exact@example.com"

        with db.get_connection("events") as conn:
            # Exactly 3 sessions.
            for day_offset in range(3):
                base = now - timedelta(days=6 - day_offset)
                _insert_email_received(conn, exact_sender, base)
                _insert_email_received(conn, exact_sender, base + timedelta(hours=2))

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        matching = [w for w in workflows if exact_sender in w["name"]]
        assert len(matching) == 1, (
            f"Expected 1 workflow for exactly-3-session sender, got {len(matching)}"
        )


# ===========================================================================
# 3. min_steps threshold enforcement
# ===========================================================================


class TestMinStepsEnforcement:
    """Verify that single-email sessions don't contribute toward min_steps."""

    def test_single_email_sessions_do_not_count(self, detector, db):
        """5 sessions of 1 email each (< min_steps=2 per session) are rejected."""
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        single_sender = "single@example.com"

        with db.get_connection("events") as conn:
            # 5 isolated emails spaced 2 days apart — each is its own session
            # of depth 1, so none qualifies as a multi-step session.
            for day_offset in range(5):
                base = now - timedelta(days=15 - day_offset * 2)
                _insert_email_received(conn, single_sender, base)

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        matching = [w for w in workflows if single_sender in w["name"]]
        assert len(matching) == 0, (
            f"Single-email sessions should not produce a workflow, got {len(matching)}"
        )

    def test_mixed_session_depths_only_multistep_count(self, detector, db):
        """Only sessions with >= min_steps emails count toward the threshold.

        Setup: sender has 5 sessions total — 2 single-email (don't count),
        3 two-email (count).  Result: exactly 3 qualifying sessions → accepted.
        """
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        mixed_sender = "mixed@example.com"

        with db.get_connection("events") as conn:
            # 3 two-email sessions (qualifying)
            for day_offset in range(3):
                base = now - timedelta(days=10 - day_offset)
                _insert_email_received(conn, mixed_sender, base)
                _insert_email_received(conn, mixed_sender, base + timedelta(hours=3))

            # 2 single-email sessions (non-qualifying, 2-day gaps)
            _insert_email_received(conn, mixed_sender, now - timedelta(days=20))
            _insert_email_received(conn, mixed_sender, now - timedelta(days=22))

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        matching = [w for w in workflows if mixed_sender in w["name"]]
        assert len(matching) == 1, (
            f"Expected 1 workflow (3 qualifying sessions), got {len(matching)}"
        )


# ===========================================================================
# 4. detect_workflows() end-to-end with empty episodes
# ===========================================================================


class TestDetectWorkflowsEndToEnd:
    """End-to-end: detect_workflows() returns fallback results when episodes=0."""

    def test_detect_workflows_returns_fallback_results(self, detector, db):
        """detect_workflows() with 0 episodes returns email thread workflows."""
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        sender_a = "manager@corp.com"
        sender_b = "colleague@corp.com"
        sender_c = "client@external.com"

        with db.get_connection("events") as conn:
            for sender in (sender_a, sender_b, sender_c):
                for day_offset in range(3):
                    base = now - timedelta(days=10 - day_offset)
                    _insert_email_received(conn, sender, base)
                    _insert_email_received(conn, sender, base + timedelta(hours=1))
                    _insert_email_received(conn, sender, base + timedelta(hours=2))

        workflows = detector.detect_workflows(lookback_days=30)

        # At least one email thread workflow should be detected from fallback.
        thread_workflows = [
            w for w in workflows
            if "Email thread from" in w.get("name", "")
        ]
        assert len(thread_workflows) >= 1, (
            f"Expected >= 1 email thread workflow from fallback, got {len(thread_workflows)}\n"
            f"All workflows: {[w['name'] for w in workflows]}"
        )

    def test_detect_workflows_multiple_senders(self, detector, db):
        """detect_workflows() detects all qualifying senders from the fallback."""
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        qualifying_senders = ["alpha@test.com", "beta@test.com"]
        non_qualifying_sender = "rare@test.com"

        with db.get_connection("events") as conn:
            # Qualifying: 3 two-email sessions each.
            for sender in qualifying_senders:
                for day_offset in range(3):
                    base = now - timedelta(days=8 - day_offset)
                    _insert_email_received(conn, sender, base)
                    _insert_email_received(conn, sender, base + timedelta(hours=2))

            # Non-qualifying: only 2 sessions.
            for day_offset in range(2):
                base = now - timedelta(days=5 - day_offset)
                _insert_email_received(conn, non_qualifying_sender, base)
                _insert_email_received(conn, non_qualifying_sender, base + timedelta(hours=1))

        workflows = detector.detect_workflows(lookback_days=30)
        thread_names = [w["name"] for w in workflows if "Email thread from" in w.get("name", "")]

        for sender in qualifying_senders:
            assert any(sender in name for name in thread_names), (
                f"Expected workflow for qualifying sender '{sender}', got: {thread_names}"
            )

        assert not any(non_qualifying_sender in name for name in thread_names), (
            f"Non-qualifying sender '{non_qualifying_sender}' should not appear in: {thread_names}"
        )


# ===========================================================================
# 5. Internal telemetry type exclusion
# ===========================================================================


class TestInternalTypeExclusion:
    """Verify that internal/telemetry event types are excluded from fallback."""

    def test_usermodel_events_excluded(self, detector, db):
        """Events with type 'usermodel_*' are filtered out by the fallback."""
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        with db.get_connection("events") as conn:
            # Insert 'usermodel_sync' events — should be excluded.
            for i in range(6):
                eid = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                        email_from, email_to)
                    VALUES (?, 'usermodel_sync', 'system', ?, 'normal', '{}', '{}', 'fake@x.com', NULL)
                    """,
                    (eid, _ts(now - timedelta(hours=i))),
                )

        # The fallback specifically queries type='email.received', so this should
        # produce 0 workflows regardless of the internal type filter.
        workflows = detector._detect_interaction_workflows(lookback_days=30)
        assert len(workflows) == 0

    def test_fallback_empty_when_no_qualifying_events(self, detector, db):
        """When events.db has no email.received events, fallback returns empty list."""
        _verify_no_episodes(db)

        # No events inserted at all.
        workflows = detector._detect_interaction_workflows(lookback_days=30)
        assert workflows == []


# ===========================================================================
# 6. Session gap boundary conditions
# ===========================================================================


class TestSessionGapBoundary:
    """Verify that the max_step_gap_hours boundary is respected correctly."""

    def test_emails_within_gap_form_single_session(self, detector, db):
        """3 emails within 12h of each other form a single multi-step session."""
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        sender = "within-gap@example.com"

        with db.get_connection("events") as conn:
            for session_day in range(3):
                base = now - timedelta(days=10 - session_day)
                # 3 emails within 10h — well within the 12h gap threshold.
                _insert_email_received(conn, sender, base)
                _insert_email_received(conn, sender, base + timedelta(hours=4))
                _insert_email_received(conn, sender, base + timedelta(hours=8))

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        matching = [w for w in workflows if sender in w["name"]]
        assert len(matching) == 1
        # Each session has 3 emails → workflow should have 3 steps.
        assert len(matching[0]["steps"]) >= 2

    def test_emails_beyond_gap_form_separate_sessions(self, detector, db):
        """Emails more than 12h apart form separate (single-email) sessions.

        If each "session" is only 1 email deep, it doesn't qualify as multi-step.
        """
        _verify_no_episodes(db)

        now = datetime.now(UTC)
        sender = "beyond-gap@example.com"

        with db.get_connection("events") as conn:
            # 5 emails each 24h apart — each forms its own 1-email session.
            for i in range(5):
                _insert_email_received(conn, sender, now - timedelta(hours=24 * (5 - i)))

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        matching = [w for w in workflows if sender in w["name"]]
        # Single-email sessions should not produce a workflow.
        assert len(matching) == 0
