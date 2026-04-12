"""
Tests for periodic communication template backfill triggered by the DB health loop.

Verifies that:
(a) _backfill_communication_templates_if_needed is called from _db_health_loop
    when no corruption is detected.
(b) The backfill method skips when templates already exist (count >= 100).
(c) The backfill method triggers when count is 0 and events >= 50.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from storage.manager import DatabaseManager


@pytest.fixture
def db_with_many_events(tmp_path):
    """Fixture providing a fresh DatabaseManager with 50+ communication events."""
    db = DatabaseManager(str(tmp_path))
    db._init_events_db()
    db._init_user_model_db()

    # Insert 55 email events with sufficient body text so the backfill guard passes
    with db.get_connection("events") as conn:
        for i in range(55):
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"email-{i}",
                    "email.sent",
                    "google",
                    f"2026-02-{(i % 28) + 1:02d}T09:00:00.000Z",
                    0.5,
                    json.dumps({
                        "to_addresses": [f"contact{i}@example.com"],
                        "from_address": "user@example.com",
                        "subject": f"Email {i}",
                        "body": f"Hello,\n\nThis is test email number {i}. It has plenty of content.\n\nBest,\nUser",
                        "body_plain": f"Hello,\n\nThis is test email number {i}. It has plenty of content.\n\nBest,\nUser",
                    }),
                    json.dumps({}),
                ),
            )

    return db


@pytest.fixture
def db_with_few_events(tmp_path):
    """Fixture providing a fresh DatabaseManager with fewer than 50 communication events."""
    db = DatabaseManager(str(tmp_path))
    db._init_events_db()
    db._init_user_model_db()

    # Insert only 10 events — below the 50-event threshold
    with db.get_connection("events") as conn:
        for i in range(10):
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"email-{i}",
                    "email.sent",
                    "google",
                    f"2026-02-{i + 1:02d}T09:00:00.000Z",
                    0.5,
                    json.dumps({
                        "to_addresses": [f"contact{i}@example.com"],
                        "from_address": "user@example.com",
                        "body_plain": f"Hello, short email {i} with enough text here.",
                    }),
                    json.dumps({}),
                ),
            )

    return db


@pytest.fixture
def db_with_existing_templates(tmp_path):
    """Fixture providing a DatabaseManager with 100+ pre-existing templates."""
    db = DatabaseManager(str(tmp_path))
    db._init_events_db()
    db._init_user_model_db()

    # Insert 110 communication templates directly using the actual schema columns
    with db.get_connection("user_model") as conn:
        for i in range(110):
            conn.execute(
                """INSERT OR REPLACE INTO communication_templates
                   (id, contact_id, channel, context, greeting, closing, formality,
                    typical_length, uses_emoji, common_phrases, avoids_phrases,
                    tone_notes, example_message_ids, samples_analyzed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"template-{i}",
                    f"contact{i}@example.com",
                    "email",
                    "user_to_contact",
                    "Hi",
                    "Best",
                    0.4,
                    100.0,
                    0,
                    json.dumps([]),
                    json.dumps([]),
                    json.dumps([]),
                    json.dumps([]),
                    3,
                ),
            )

    return db


class TestBackfillMethodSkipsWhenTemplatesExist:
    """Verify _backfill_communication_templates_if_needed skips when count >= 100."""

    def test_skips_when_template_count_at_threshold(self, db_with_existing_templates):
        """Method returns immediately when template_count >= 100 (idempotency guard)."""
        db = db_with_existing_templates

        # Confirm we start with 110 templates
        with db.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM communication_templates"
            ).fetchone()[0]
        assert count == 110, f"Expected 110 templates, got {count}"

        # Build a minimal LifeOS-like object with just the attributes the method needs
        lifeos = MagicMock()
        lifeos.db = db

        # Patch the backfill script import so we can detect if it was called
        with patch("scripts.backfill_communication_templates.backfill_communication_templates") as mock_backfill:
            # Run the real method via asyncio
            from main import LifeOS

            async def _run():
                # Bind the unbound async method to our mock object
                await LifeOS._backfill_communication_templates_if_needed(lifeos)

            asyncio.run(_run())

            # The backfill script must NOT have been called — we already have templates
            mock_backfill.assert_not_called()


class TestBackfillMethodTriggersWhenEmpty:
    """Verify _backfill_communication_templates_if_needed runs when count is 0 and events >= 50."""

    def test_triggers_when_no_templates_and_sufficient_events(self, db_with_many_events):
        """Method calls the backfill script when template_count == 0 and event_count >= 50."""
        db = db_with_many_events

        # Confirm no templates exist
        with db.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM communication_templates"
            ).fetchone()[0]
        assert count == 0, "Should start with no templates"

        lifeos = MagicMock()
        lifeos.db = db

        captured_calls = []

        def fake_backfill(data_dir="data", batch_size=5000, db=None, ums=None, **kwargs):
            """Stub backfill that records it was called and returns plausible stats.

            Updated to accept the ``db`` and ``ums`` kwargs added to avoid WAL lock
            contention when the backfill is called from a running LifeOS instance.
            """
            captured_calls.append({"data_dir": data_dir, "db": db, "ums": ums})
            return {"templates_created": 55, "events_processed": 55, "elapsed_seconds": 1.0}

        with patch("scripts.backfill_communication_templates.backfill_communication_templates", side_effect=fake_backfill):
            from main import LifeOS

            async def _run():
                await LifeOS._backfill_communication_templates_if_needed(lifeos)

            asyncio.run(_run())

        assert len(captured_calls) == 1, "Backfill should have been called exactly once"
        # Verify db and ums are passed so the backfill reuses the server's connections.
        assert captured_calls[0]["db"] is db, "db must be passed to avoid WAL contention"

    def test_skips_when_no_templates_but_insufficient_events(self, db_with_few_events):
        """Method skips the backfill when event_count < 50 (not enough data)."""
        db = db_with_few_events

        lifeos = MagicMock()
        lifeos.db = db

        with patch("scripts.backfill_communication_templates.backfill_communication_templates") as mock_backfill:
            from main import LifeOS

            async def _run():
                await LifeOS._backfill_communication_templates_if_needed(lifeos)

            asyncio.run(_run())

            mock_backfill.assert_not_called()


class TestDbHealthLoopCallsTemplateBackfill:
    """Verify _db_health_loop invokes _backfill_communication_templates_if_needed."""

    def test_health_loop_calls_backfill_when_db_is_healthy(self):
        """_db_health_loop calls the template backfill method when no corruption is detected.

        This is the core regression test for the fix: after a DB rebuild, the
        health loop must periodically re-trigger the template backfill even if no
        restart occurs.
        """
        lifeos = MagicMock()
        lifeos.shutdown_event = MagicMock()

        # Simulate: healthy on first iteration, then set to stop the loop
        iteration_count = 0

        def is_set_side_effect():
            nonlocal iteration_count
            # First call — loop body runs; after first iteration we stop
            if iteration_count == 0:
                iteration_count += 1
                return False
            return True

        lifeos.shutdown_event.is_set.side_effect = is_set_side_effect

        # Mock DB connection to simulate no corruption
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_conn
        mock_conn.fetchone.return_value = [None]
        lifeos.db.get_connection.return_value = mock_conn

        # Track whether the template backfill was awaited
        backfill_called = False

        async def fake_backfill():
            nonlocal backfill_called
            backfill_called = True

        lifeos._backfill_communication_templates_if_needed = AsyncMock(side_effect=fake_backfill)
        lifeos._last_backup_time = 0.0
        lifeos.db.backup_database = MagicMock(return_value=None)

        async def _run():
            # Patch asyncio.sleep so the loop doesn't actually wait 1800 seconds
            with patch("asyncio.sleep", new_callable=AsyncMock):
                from main import LifeOS
                await LifeOS._db_health_loop(lifeos)

        asyncio.run(_run())

        assert backfill_called, (
            "_db_health_loop must call _backfill_communication_templates_if_needed "
            "on each healthy iteration so template loss after DB rebuild is recovered "
            "without requiring a restart."
        )
