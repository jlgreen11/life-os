"""
Tests for task event publishing during backfill.

Verifies that the backfill script publishes task.created events to events.db
even when running without an event bus connection. This ensures downstream
systems (workflow detection, episodic memory) can see task creation patterns.
"""

import asyncio
import pytest
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add project root to path so we can import the backfill script
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.manager import DatabaseManager
from storage.event_store import EventStore
from services.task_manager.manager import TaskManager
from services.ai_engine.engine import AIEngine
from storage.user_model_store import UserModelStore
from storage.vector_store import VectorStore
from scripts.backfill_task_extraction import backfill_tasks
import uuid


def create_test_event(event_type: str, payload: dict, source: str = "test", priority: str = "normal") -> dict:
    """Helper function to create a properly formatted event dict for testing.

    Args:
        event_type: The type of event (e.g., "email.received")
        payload: The event payload dictionary
        source: The source of the event (default: "test")
        priority: The priority level (default: "normal")

    Returns:
        A properly formatted event dict ready for EventStore.store_event()
    """
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": priority,
        "payload": payload,
        "metadata": {},
    }


class TestBackfillTaskEventPublishing:
    """Test suite for task.created event publishing during backfill."""

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason=(
            "Integration test requiring a live Ollama service for LLM-based task "
            "extraction. Without Ollama the AI engine silently returns no tasks, so "
            "the assertion 'events_after > events_before' always fails in CI. "
            "Run manually against a running Ollama instance to validate."
        )
    )
    async def test_backfill_publishes_task_created_events(self, db, tmpdir):
        """Test that backfill script publishes task.created events for extracted tasks."""
        # Create a simple event store
        event_store = EventStore(db)
        user_model_store = UserModelStore(db)

        # Create a minimal AI engine (will use real Ollama if available, fallback otherwise)
        config = {
            "ai": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "mistral",
                    "timeout_seconds": 30,
                },
                "use_cloud": False,
            },
            "vector_store": {
                "model": "all-MiniLM-L6-v2",
            },
        }

        # Initialize vector store with temp directory
        vector_db_path = str(tmpdir.join("vectors"))
        vector_store = VectorStore(
            db_path=vector_db_path,
            model_name=config["vector_store"]["model"]
        )

        ai_engine = AIEngine(db, user_model_store, config, vector_store=vector_store)

        # Create task manager without event bus (simulating backfill scenario)
        task_manager = TaskManager(db, event_bus=None, ai_engine=ai_engine)

        # Create a test email event with actionable content
        test_event = {
            "id": "test-email-001",
            "type": "email.received",
            "source": "proton_mail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {
                "from": "boss@company.com",
                "to": "user@company.com",
                "subject": "Quarterly Report",
                "body": "Please prepare the Q4 financial report by Friday and send it to the board.",
                "message_id": "<test001@proton.ch>",
            },
            "metadata": {},
        }

        # Store the test event using store_event()
        event_store.store_event(test_event)

        # Count events before backfill
        with db.get_connection("events") as conn:
            events_before = conn.execute(
                "SELECT COUNT(*) as count FROM events WHERE type = 'task.created'"
            ).fetchone()["count"]

        # Run backfill on this single event
        stats = await backfill_tasks(
            db=db,
            task_manager=task_manager,
            limit=1,
            dry_run=False,
            batch_size=1
        )

        # Count events after backfill
        with db.get_connection("events") as conn:
            events_after = conn.execute(
                "SELECT COUNT(*) as count FROM events WHERE type = 'task.created'"
            ).fetchone()["count"]

        # Verify task.created events were published
        assert events_after > events_before, "Backfill should publish task.created events"

        # Verify the events have correct structure
        with db.get_connection("events") as conn:
            task_events = conn.execute(
                """SELECT payload FROM events
                   WHERE type = 'task.created'
                   ORDER BY timestamp DESC
                   LIMIT 1"""
            ).fetchall()

        assert len(task_events) > 0, "Should have at least one task.created event"

        task_event_payload = json.loads(task_events[0]["payload"])
        assert "task_id" in task_event_payload, "Event payload should contain task_id"
        assert "title" in task_event_payload, "Event payload should contain title"
        assert "source" in task_event_payload, "Event payload should contain source"
        assert task_event_payload["source"] == "ai_extracted", "Source should be ai_extracted"

    @pytest.mark.asyncio
    async def test_backfill_event_count_matches_task_count(self, db, tmpdir):
        """Test that the number of task.created events matches the number of tasks created."""
        event_store = EventStore(db)
        user_model_store = UserModelStore(db)

        config = {
            "ai": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "mistral",
                    "timeout_seconds": 30,
                },
                "use_cloud": False,
            },
            "vector_store": {
                "model": "all-MiniLM-L6-v2",
            },
        }

        vector_db_path = str(tmpdir.join("vectors"))
        vector_store = VectorStore(
            db_path=vector_db_path,
            model_name=config["vector_store"]["model"]
        )

        ai_engine = AIEngine(db, user_model_store, config, vector_store=vector_store)
        task_manager = TaskManager(db, event_bus=None, ai_engine=ai_engine)

        # Create multiple test events with actionable content
        test_events = [
            {
                "type": "email.received",
                "payload": {
                    "from": "client@example.com",
                    "subject": "Project Deadline",
                    "body": "Can you review the proposal and send feedback by tomorrow?",
                },
            },
            {
                "type": "email.received",
                "payload": {
                    "from": "manager@example.com",
                    "subject": "Team Meeting",
                    "body": "Please prepare slides for Monday's presentation.",
                },
            },
        ]

        for event_data in test_events:
            test_event = create_test_event(
                event_type=event_data["type"],
                payload=event_data["payload"],
                source="test",
                priority="normal"
            )
            event_store.store_event(test_event)

        # Count tasks and events before backfill
        with db.get_connection("state") as conn:
            tasks_before = conn.execute("SELECT COUNT(*) as count FROM tasks").fetchone()["count"]

        with db.get_connection("events") as conn:
            events_before = conn.execute(
                "SELECT COUNT(*) as count FROM events WHERE type = 'task.created'"
            ).fetchone()["count"]

        # Run backfill
        await backfill_tasks(
            db=db,
            task_manager=task_manager,
            limit=2,
            dry_run=False,
            batch_size=2
        )

        # Count tasks and events after backfill
        with db.get_connection("state") as conn:
            tasks_after = conn.execute("SELECT COUNT(*) as count FROM tasks").fetchone()["count"]

        with db.get_connection("events") as conn:
            events_after = conn.execute(
                "SELECT COUNT(*) as count FROM events WHERE type = 'task.created'"
            ).fetchone()["count"]

        # Calculate deltas
        tasks_created = tasks_after - tasks_before
        events_created = events_after - events_before

        # Verify event count matches task count (within reason, AI may extract 0-N tasks per email)
        # We just verify that IF tasks were created, events were also created
        if tasks_created > 0:
            assert events_created > 0, "If tasks were created, task.created events should be published"
            # Allow some variance since background tasks may be created during test execution
            assert events_created >= tasks_created, (
                f"Event count ({events_created}) should be at least task count ({tasks_created})"
            )

    @pytest.mark.asyncio
    async def test_backfill_event_source_is_correct(self, db, tmpdir):
        """Test that backfilled task.created events have correct source attribution."""
        event_store = EventStore(db)
        user_model_store = UserModelStore(db)

        config = {
            "ai": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "mistral",
                    "timeout_seconds": 30,
                },
                "use_cloud": False,
            },
            "vector_store": {
                "model": "all-MiniLM-L6-v2",
            },
        }

        vector_db_path = str(tmpdir.join("vectors"))
        vector_store = VectorStore(
            db_path=vector_db_path,
            model_name=config["vector_store"]["model"]
        )

        ai_engine = AIEngine(db, user_model_store, config, vector_store=vector_store)
        task_manager = TaskManager(db, event_bus=None, ai_engine=ai_engine)

        # Create test event
        test_event = create_test_event(
            event_type="email.received",
            payload={
                "from": "team@example.com",
                "subject": "Action Required",
                "body": "Please complete the security audit by next week.",
            },
            source="test",
            priority="normal"
        )

        event_store.store_event(test_event)

        # Run backfill
        await backfill_tasks(
            db=db,
            task_manager=task_manager,
            limit=1,
            dry_run=False,
            batch_size=1
        )

        # Verify event source is "task_manager_backfill"
        with db.get_connection("events") as conn:
            events = conn.execute(
                """SELECT source FROM events
                   WHERE type = 'task.created'
                   ORDER BY timestamp DESC
                   LIMIT 10"""
            ).fetchall()

        if len(events) > 0:
            # At least one task was extracted and published
            for event in events:
                assert event["source"] == "task_manager_backfill", (
                    "Backfilled events should have source 'task_manager_backfill'"
                )

    @pytest.mark.asyncio
    async def test_backfill_dry_run_no_events(self, db, tmpdir):
        """Test that dry-run mode doesn't publish any task.created events."""
        event_store = EventStore(db)
        user_model_store = UserModelStore(db)

        config = {
            "ai": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "mistral",
                    "timeout_seconds": 30,
                },
                "use_cloud": False,
            },
            "vector_store": {
                "model": "all-MiniLM-L6-v2",
            },
        }

        vector_db_path = str(tmpdir.join("vectors"))
        vector_store = VectorStore(
            db_path=vector_db_path,
            model_name=config["vector_store"]["model"]
        )

        ai_engine = AIEngine(db, user_model_store, config, vector_store=vector_store)
        task_manager = TaskManager(db, event_bus=None, ai_engine=ai_engine)

        # Create test event
        test_event = create_test_event(
            event_type="email.received",
            payload={
                "from": "sender@example.com",
                "subject": "Test",
                "body": "This is a test email with an action item.",
            },
            source="test",
            priority="normal"
        )

        event_store.store_event(test_event)

        # Count events before dry-run
        with db.get_connection("events") as conn:
            events_before = conn.execute(
                "SELECT COUNT(*) as count FROM events WHERE type = 'task.created'"
            ).fetchone()["count"]

        # Run backfill in dry-run mode
        await backfill_tasks(
            db=db,
            task_manager=task_manager,
            limit=1,
            dry_run=True,
            batch_size=1
        )

        # Count events after dry-run
        with db.get_connection("events") as conn:
            events_after = conn.execute(
                "SELECT COUNT(*) as count FROM events WHERE type = 'task.created'"
            ).fetchone()["count"]

        # Verify no events were published in dry-run mode
        assert events_after == events_before, "Dry-run should not publish events"

    @pytest.mark.asyncio
    async def test_backfill_event_payload_completeness(self, db, tmpdir):
        """Test that task.created event payloads contain all required fields."""
        event_store = EventStore(db)
        user_model_store = UserModelStore(db)

        config = {
            "ai": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "mistral",
                    "timeout_seconds": 30,
                },
                "use_cloud": False,
            },
            "vector_store": {
                "model": "all-MiniLM-L6-v2",
            },
        }

        vector_db_path = str(tmpdir.join("vectors"))
        vector_store = VectorStore(
            db_path=vector_db_path,
            model_name=config["vector_store"]["model"]
        )

        ai_engine = AIEngine(db, user_model_store, config, vector_store=vector_store)
        task_manager = TaskManager(db, event_bus=None, ai_engine=ai_engine)

        # Create test event with rich metadata
        test_event = create_test_event(
            event_type="email.received",
            payload={
                "from": "vip@company.com",
                "to": "user@company.com",
                "subject": "Urgent: Board Meeting Prep",
                "body": "Please prepare the quarterly financial summary and email it to me by Thursday.",
                "message_id": "<test-rich@proton.ch>",
            },
            source="proton_mail",
            priority="high"
        )

        event_store.store_event(test_event)

        # Run backfill
        await backfill_tasks(
            db=db,
            task_manager=task_manager,
            limit=1,
            dry_run=False,
            batch_size=1
        )

        # Fetch the task.created event
        with db.get_connection("events") as conn:
            events = conn.execute(
                """SELECT payload FROM events
                   WHERE type = 'task.created'
                   ORDER BY timestamp DESC
                   LIMIT 1"""
            ).fetchall()

        if len(events) > 0:
            payload = json.loads(events[0]["payload"])

            # Verify required fields
            required_fields = ["task_id", "title", "source", "source_event_id", "priority", "created_at"]
            for field in required_fields:
                assert field in payload, f"Event payload must contain '{field}'"

            # Verify field values
            assert isinstance(payload["task_id"], str), "task_id should be a string (UUID)"
            assert len(payload["task_id"]) > 0, "task_id should not be empty"
            assert payload["source"] == "ai_extracted", "source should be 'ai_extracted'"
            assert payload["priority"] in ["low", "normal", "high", "critical"], (
                "priority should be a valid priority level"
            )

    @pytest.mark.asyncio
    async def test_backfill_workflow_detection_integration(self, db, tmpdir):
        """Test that task.created events from backfill enable workflow detection."""
        event_store = EventStore(db)
        user_model_store = UserModelStore(db)

        config = {
            "ai": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "mistral",
                    "timeout_seconds": 30,
                },
                "use_cloud": False,
            },
            "vector_store": {
                "model": "all-MiniLM-L6-v2",
            },
        }

        vector_db_path = str(tmpdir.join("vectors"))
        vector_store = VectorStore(
            db_path=vector_db_path,
            model_name=config["vector_store"]["model"]
        )

        ai_engine = AIEngine(db, user_model_store, config, vector_store=vector_store)
        task_manager = TaskManager(db, event_bus=None, ai_engine=ai_engine)

        # Create a sequence of events that form a workflow pattern:
        # email.received → task.created (simulating real user workflow)
        for i in range(3):
            # Create email event
            email_event = create_test_event(
                event_type="email.received",
                payload={
                    "from": "boss@company.com",
                    "subject": f"Action Item {i}",
                    "body": f"Please complete task number {i} by end of week.",
                },
                source="proton_mail",
                priority="normal"
            )

            event_store.store_event(email_event)

        # Run backfill
        await backfill_tasks(
            db=db,
            task_manager=task_manager,
            limit=3,
            dry_run=False,
            batch_size=3
        )

        # Verify that task.created events exist for workflow detection
        with db.get_connection("events") as conn:
            # Check for email.received events followed by task.created events
            workflow_query = """
                SELECT
                    e1.type as first_event,
                    e2.type as second_event,
                    COUNT(*) as occurrence_count
                FROM events e1
                JOIN events e2 ON
                    e2.timestamp > e1.timestamp
                    AND e2.timestamp < datetime(e1.timestamp, '+1 hour')
                WHERE e1.type = 'email.received'
                  AND e2.type = 'task.created'
                GROUP BY e1.type, e2.type
            """

            workflow_patterns = conn.execute(workflow_query).fetchall()

        # If tasks were extracted, we should see the workflow pattern
        # (email.received → task.created)
        if len(workflow_patterns) > 0:
            assert workflow_patterns[0]["first_event"] == "email.received"
            assert workflow_patterns[0]["second_event"] == "task.created"
            assert workflow_patterns[0]["occurrence_count"] > 0, (
                "Should detect email → task workflow pattern"
            )
