"""Tests for WorkflowDetector.get_last_run_diagnostics() skip-reason counters.

Validates that per-run diagnostics correctly bucket every candidate evaluated
by the email-sender, task, and calendar detection paths into one of:
considered / skipped_* / emitted.  Without this signal an operator cannot
distinguish "no candidates exist" (e.g. email connector stale) from
"candidates exist but every one was filtered by thresholds".
"""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from services.workflow_detector import WorkflowDetector


@pytest.fixture
def detector(db, user_model_store):
    """A WorkflowDetector wired to the temporary databases."""
    return WorkflowDetector(db, user_model_store)


def _insert_received_email(conn, sender, when, event_id=None):
    """Insert a single email.received event from ``sender`` at time ``when``."""
    conn.execute(
        """
        INSERT INTO events (id, type, source, timestamp, priority,
                            payload, metadata, email_from)
        VALUES (?, 'email.received', 'protonmail', ?, 3, ?, ?, ?)
        """,
        (
            event_id or str(uuid4()),
            when.isoformat(),
            json.dumps({"sender": sender, "subject": "x"}),
            json.dumps({}),
            sender,
        ),
    )


def _insert_sent_email(conn, to_addr, when):
    """Insert a single email.sent event addressed to ``to_addr`` at ``when``."""
    conn.execute(
        """
        INSERT INTO events (id, type, source, timestamp, priority,
                            payload, metadata, email_to)
        VALUES (?, 'email.sent', 'protonmail', ?, 3, ?, ?, ?)
        """,
        (
            str(uuid4()),
            when.isoformat(),
            json.dumps({"to": to_addr}),
            json.dumps({}),
            json.dumps([to_addr]),
        ),
    )


class TestSkipDiagnosticsShape:
    """Structural guarantees: keys present, defaults sane."""

    def test_default_diagnostics_before_any_run(self, detector):
        """Calling get_last_run_diagnostics with no prior run returns zeros."""
        diag = detector.get_last_run_diagnostics()
        assert diag["last_run_at"] is None
        assert diag["total_emitted"] == 0
        for path in ("email_senders", "tasks", "calendar"):
            assert path in diag
            assert diag[path]["considered"] == 0
            assert diag[path]["emitted"] == 0

    def test_zero_data_run_populates_zero_counters(self, detector):
        """A run against an empty events DB still emits zero-valued diagnostics."""
        result = detector.detect_workflows(lookback_days=30)
        assert result == []

        diag = detector.get_last_run_diagnostics()
        assert diag["last_run_at"] is not None
        assert diag["total_emitted"] == 0
        # All path counters must exist and read zero on an empty database.
        for path, expected_keys in (
            (
                "email_senders",
                (
                    "considered",
                    "skipped_below_min_occurrences",
                    "skipped_extreme_volume",
                    "skipped_no_following_actions",
                    "emitted",
                ),
            ),
            (
                "tasks",
                (
                    "considered",
                    "skipped_below_min_occurrences",
                    "skipped_below_min_steps",
                    "emitted",
                ),
            ),
            (
                "calendar",
                (
                    "considered",
                    "skipped_below_min_occurrences",
                    "skipped_below_min_steps",
                    "emitted",
                ),
            ),
        ):
            for key in expected_keys:
                assert diag[path][key] == 0, f"{path}.{key} expected 0"


class TestEmailSenderSkipReasons:
    """Sender candidates routed into the correct skip bucket."""

    def test_below_min_occurrences_increments_counter(self, detector, db):
        """A sender with only 2 received emails is below the 3-occurrence floor."""
        sender = "rare@example.com"
        base = datetime.now(UTC) - timedelta(days=5)
        with db.get_connection("events") as conn:
            for i in range(2):
                _insert_received_email(conn, sender, base + timedelta(hours=i))

        detector.detect_workflows(lookback_days=30)
        diag = detector.get_last_run_diagnostics()["email_senders"]

        assert diag["considered"] >= 1
        assert diag["skipped_below_min_occurrences"] >= 1
        assert diag["emitted"] == 0

    def test_extreme_volume_sender_counted(self, detector, db):
        """A sender with massive volume hits the dynamic max_volume cutoff.

        The dynamic cutoff is ``max(200, total_emails // 5)``.  Seeding a
        single sender with 250 events keeps total_emails small enough that
        the floor of 200 applies, so the sender exceeds it.
        """
        sender = "noisy@example.com"
        base = datetime.now(UTC) - timedelta(days=29)
        with db.get_connection("events") as conn:
            for i in range(250):
                _insert_received_email(conn, sender, base + timedelta(hours=i))

        detector.detect_workflows(lookback_days=30)
        diag = detector.get_last_run_diagnostics()["email_senders"]

        assert diag["skipped_extreme_volume"] >= 1

    def test_successful_emission_updates_counters(self, detector, db):
        """A sender with enough receives + sent replies should emit a workflow."""
        sender = "boss@example.com"
        base = datetime.now(UTC) - timedelta(days=15)
        with db.get_connection("events") as conn:
            # 4 received emails each followed by a reply ~1 hour later.
            for i in range(4):
                t = base + timedelta(days=i)
                _insert_received_email(conn, sender, t)
                _insert_sent_email(conn, sender, t + timedelta(hours=1))

        workflows = detector.detect_workflows(lookback_days=30)
        diag = detector.get_last_run_diagnostics()

        # The detector returned at least one workflow tied to this sender.
        assert any(sender in w.get("name", "") for w in workflows)

        email_diag = diag["email_senders"]
        assert email_diag["considered"] >= 1
        assert email_diag["emitted"] >= 1
        assert diag["total_emitted"] == len(workflows)
        assert diag["last_run_at"] is not None
