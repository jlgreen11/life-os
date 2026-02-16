"""
Tests for automatic episode classification backfill on startup.

This test suite verifies that the system automatically detects and reclassifies
old episodes with generic interaction types on startup, enabling routine and
workflow detection to work correctly after deployments.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone

from main import LifeOS


class TestAutomaticEpisodeBackfill:
    """Test automatic episode classification backfill."""

    @pytest.mark.asyncio
    async def test_backfill_runs_on_startup_with_stale_episodes(self, db, event_bus, event_store, user_model_store):
        """Backfill should automatically run on startup when stale episodes exist."""
        # Create events with different types
        email_received_event = {
            "id": "evt-1",
            "type": "email.received",
            "source": "protonmail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": 1,
            "payload": {"from": "alice@example.com", "subject": "Test"},
            "metadata": {},
        }
        email_sent_event = {
            "id": "evt-2",
            "type": "email.sent",
            "source": "protonmail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": 1,
            "payload": {"to": "bob@example.com", "subject": "Reply"},
            "metadata": {},
        }

        # Store events in events.db
        event_store.store_event(email_received_event)
        event_store.store_event(email_sent_event)

        # Manually create episodes with the OLD generic "communication" type
        with db.get_connection("user_model") as conn:
            conn.execute("""
                INSERT INTO episodes (
                    id, timestamp, event_id, interaction_type,
                    content_summary, content_full
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "ep-1",
                email_received_event["timestamp"],
                "evt-1",
                "communication",  # OLD generic type
                "Email from alice@example.com",
                json.dumps(email_received_event["payload"]),
            ))
            conn.execute("""
                INSERT INTO episodes (
                    id, timestamp, event_id, interaction_type,
                    content_summary, content_full
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "ep-2",
                email_sent_event["timestamp"],
                "evt-2",
                "communication",  # OLD generic type
                "Email to bob@example.com",
                json.dumps(email_sent_event["payload"]),
            ))

        # Verify episodes have generic type before startup
        with db.get_connection("user_model") as conn:
            cursor = conn.execute("SELECT interaction_type FROM episodes ORDER BY id")
            types = [row[0] for row in cursor.fetchall()]
            assert types == ["communication", "communication"]

        # Create LifeOS instance and trigger startup (which runs backfill)
        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "web_port": 8080,
            "ai": {"use_cloud": False},
        }
        app = LifeOS(config=config, db=db, event_bus=event_bus, event_store=event_store, user_model_store=user_model_store)

        # Run the backfill method directly (simulating startup)
        await app._backfill_episode_classification_if_needed()

        # Verify episodes now have granular types
        with db.get_connection("user_model") as conn:
            cursor = conn.execute("SELECT interaction_type FROM episodes ORDER BY id")
            types = [row[0] for row in cursor.fetchall()]
            assert types == ["email_received", "email_sent"]

    @pytest.mark.asyncio
    async def test_backfill_skips_when_no_stale_episodes(self, db, event_bus, event_store, user_model_store):
        """Backfill should be a no-op when all episodes are already classified correctly."""
        # Create episodes that already have granular types
        with db.get_connection("user_model") as conn:
            conn.execute("""
                INSERT INTO episodes (
                    id, timestamp, event_id, interaction_type,
                    content_summary, content_full
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "ep-1",
                datetime.now(timezone.utc).isoformat(),
                "evt-1",
                "email_received",  # Already granular
                "Test",
                "{}",
            ))

        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "web_port": 8080,
            "ai": {"use_cloud": False},
        }
        app = LifeOS(config=config, db=db, event_bus=event_bus, event_store=event_store, user_model_store=user_model_store)

        # Run backfill (should be a no-op)
        await app._backfill_episode_classification_if_needed()

        # Verify episode type unchanged
        with db.get_connection("user_model") as conn:
            cursor = conn.execute("SELECT interaction_type FROM episodes")
            types = [row[0] for row in cursor.fetchall()]
            assert types == ["email_received"]

    @pytest.mark.asyncio
    async def test_backfill_handles_calendar_events(self, db, event_bus, event_store, user_model_store):
        """Backfill should correctly classify calendar events as meetings or blocks."""
        # Create a meeting (has participants)
        meeting_event = {
            "id": "evt-meeting",
            "type": "calendar.event.created",
            "source": "caldav",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": 1,
            "payload": {
                "summary": "Team Standup",
                "participants": ["alice@example.com", "bob@example.com"],
            },
            "metadata": {},
        }

        # Create a personal event (no participants)
        personal_event = {
            "id": "evt-personal",
            "type": "calendar.event.created",
            "source": "caldav",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": 1,
            "payload": {"summary": "Gym time", "participants": []},
            "metadata": {},
        }

        event_store.store_event(meeting_event)
        event_store.store_event(personal_event)

        # Create episodes with generic type
        with db.get_connection("user_model") as conn:
            conn.execute("""
                INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary, content_full)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("ep-1", meeting_event["timestamp"], "evt-meeting", "communication", "Meeting", json.dumps(meeting_event["payload"])))
            conn.execute("""
                INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary, content_full)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("ep-2", personal_event["timestamp"], "evt-personal", "communication", "Personal", json.dumps(personal_event["payload"])))

        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "web_port": 8080,
            "ai": {"use_cloud": False},
        }
        app = LifeOS(config=config, db=db, event_bus=event_bus, event_store=event_store, user_model_store=user_model_store)

        await app._backfill_episode_classification_if_needed()

        # Verify correct classification
        with db.get_connection("user_model") as conn:
            cursor = conn.execute("SELECT event_id, interaction_type FROM episodes ORDER BY id")
            results = [(row[0], row[1]) for row in cursor.fetchall()]
            assert results == [
                ("evt-meeting", "meeting_scheduled"),
                ("evt-personal", "calendar_blocked"),
            ]

    @pytest.mark.asyncio
    async def test_backfill_handles_missing_events(self, db, event_bus, event_store, user_model_store):
        """Backfill should gracefully skip episodes whose events were deleted."""
        # Create an episode pointing to a non-existent event
        with db.get_connection("user_model") as conn:
            conn.execute("""
                INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary, content_full)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "ep-orphan",
                datetime.now(timezone.utc).isoformat(),
                "evt-deleted",  # This event doesn't exist
                "communication",
                "Orphaned episode",
                "{}",
            ))

        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "web_port": 8080,
            "ai": {"use_cloud": False},
        }
        app = LifeOS(config=config, db=db, event_bus=event_bus, event_store=event_store, user_model_store=user_model_store)

        # Backfill should not crash
        await app._backfill_episode_classification_if_needed()

        # Orphaned episode should remain unchanged (graceful degradation)
        with db.get_connection("user_model") as conn:
            cursor = conn.execute("SELECT interaction_type FROM episodes WHERE id = 'ep-orphan'")
            row = cursor.fetchone()
            assert row[0] == "communication"  # Unchanged

    @pytest.mark.asyncio
    async def test_backfill_creates_type_diversity_for_detection(self, db, event_bus, event_store, user_model_store):
        """After backfill, episodes have diverse types enabling pattern detection."""
        # Create a diverse set of morning interactions across multiple days
        # to simulate a morning routine: email check, calendar review, message check
        event_sequence = [
            ("email.received", {"from": "alice@example.com"}),
            ("calendar.event.updated", {"summary": "Review calendar"}),
            ("message.received", {"from": "bob", "channel": "signal"}),
        ]

        event_id = 0
        # Repeat pattern for 4 days to meet min_occurrences threshold
        for day in range(1, 5):
            for hour_offset, (event_type, payload) in enumerate(event_sequence):
                event = {
                    "id": f"evt-{event_id}",
                    "type": event_type,
                    "source": "test",
                    "timestamp": f"2026-02-0{day}T08:{hour_offset*10:02d}:00Z",  # 8:00, 8:10, 8:20
                    "priority": 1,
                    "payload": payload,
                    "metadata": {},
                }
                event_store.store_event(event)

                # Create episode with generic type
                with db.get_connection("user_model") as conn:
                    conn.execute("""
                        INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary, content_full)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (f"ep-{event_id}", event["timestamp"], event["id"], "communication", f"Event {event_id}", json.dumps(payload)))

                event_id += 1

        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "web_port": 8080,
            "ai": {"use_cloud": False},
        }
        app = LifeOS(config=config, db=db, event_bus=event_bus, event_store=event_store, user_model_store=user_model_store)

        # Before backfill: all episodes are "communication" - no type diversity
        with db.get_connection("user_model") as conn:
            cursor = conn.execute("SELECT DISTINCT interaction_type FROM episodes")
            types_before = {row[0] for row in cursor.fetchall()}
            # Only one generic type
            assert types_before == {"communication"}

        # Run backfill
        await app._backfill_episode_classification_if_needed()

        # After backfill: episodes have granular types providing signal for pattern detection
        with db.get_connection("user_model") as conn:
            cursor = conn.execute("SELECT DISTINCT interaction_type FROM episodes")
            types_after = {row[0] for row in cursor.fetchall()}
            # Now have diverse types instead of all "communication"
            assert len(types_after) > 1
            assert "email_received" in types_after
            assert "calendar_reviewed" in types_after
            assert "message_received" in types_after
            # The generic type should be gone
            assert "communication" not in types_after

    @pytest.mark.asyncio
    async def test_backfill_multiple_event_types(self, db, event_bus, event_store, user_model_store):
        """Backfill should handle a mix of different event types correctly."""
        events = [
            ("evt-1", "email.received", {"from": "alice@example.com"}),
            ("evt-2", "email.sent", {"to": "bob@example.com"}),
            ("evt-3", "message.received", {"from": "charlie", "channel": "signal"}),
            ("evt-4", "message.sent", {"to": "dave", "channel": "signal"}),
            ("evt-5", "calendar.event.updated", {"summary": "Review calendar"}),
        ]

        for event_id, event_type, payload in events:
            event = {
                "id": event_id,
                "type": event_type,
                "source": "test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "priority": 1,
                "payload": payload,
                "metadata": {},
            }
            event_store.store_event(event)

            # Create episode with generic type
            with db.get_connection("user_model") as conn:
                conn.execute("""
                    INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary, content_full)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (event_id, event["timestamp"], event_id, "communication", f"Event {event_id}", json.dumps(payload)))

        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "web_port": 8080,
            "ai": {"use_cloud": False},
        }
        app = LifeOS(config=config, db=db, event_bus=event_bus, event_store=event_store, user_model_store=user_model_store)

        await app._backfill_episode_classification_if_needed()

        # Verify all episodes now have correct granular types
        with db.get_connection("user_model") as conn:
            cursor = conn.execute("SELECT event_id, interaction_type FROM episodes ORDER BY event_id")
            results = [(row[0], row[1]) for row in cursor.fetchall()]
            assert results == [
                ("evt-1", "email_received"),
                ("evt-2", "email_sent"),
                ("evt-3", "message_received"),
                ("evt-4", "message_sent"),
                ("evt-5", "calendar_reviewed"),
            ]

    @pytest.mark.asyncio
    async def test_backfill_is_idempotent(self, db, event_bus, event_store, user_model_store):
        """Running backfill multiple times should be safe and produce the same result."""
        event = {
            "id": "evt-1",
            "type": "email.received",
            "source": "protonmail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": 1,
            "payload": {"from": "alice@example.com"},
            "metadata": {},
        }
        event_store.store_event(event)

        with db.get_connection("user_model") as conn:
            conn.execute("""
                INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary, content_full)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("ep-1", event["timestamp"], "evt-1", "communication", "Email", json.dumps(event["payload"])))

        config = {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "web_port": 8080,
            "ai": {"use_cloud": False},
        }
        app = LifeOS(config=config, db=db, event_bus=event_bus, event_store=event_store, user_model_store=user_model_store)

        # Run backfill first time
        await app._backfill_episode_classification_if_needed()

        with db.get_connection("user_model") as conn:
            cursor = conn.execute("SELECT interaction_type FROM episodes WHERE id = 'ep-1'")
            first_result = cursor.fetchone()[0]

        # Run backfill second time
        await app._backfill_episode_classification_if_needed()

        with db.get_connection("user_model") as conn:
            cursor = conn.execute("SELECT interaction_type FROM episodes WHERE id = 'ep-1'")
            second_result = cursor.fetchone()[0]

        # Result should be identical
        assert first_result == second_result == "email_received"
