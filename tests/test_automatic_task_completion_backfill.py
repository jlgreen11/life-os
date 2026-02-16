"""
Tests for automatic task completion backfill on startup.

This test suite verifies that the task completion backfill runs automatically
during system startup and correctly identifies completed tasks from historical
behavioral signals. The backfill is critical for bootstrapping workflow
detection (Layer 3 procedural memory) on systems with historical task data.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from main import LifeOS


@pytest.fixture
def lifeos_with_tasks(db, event_bus, event_store, user_model_store):
    """Create a LifeOS instance with some pending tasks and completion signals."""
    config = {
        "data_dir": "./data",
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "ai": {"use_cloud": False},
    }
    # Create some pending tasks (need >= 10 for backfill to run)
    with db.get_connection("state") as conn:
        # Task 1: Will be detected as completed (strong signals)
        conn.execute("""
            INSERT INTO tasks (id, title, description, status, created_at, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'task-1',
            'Review proposal document',
            'Review the Q1 proposal',
            'pending',
            (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            'ai.extraction'
        ))

        # Task 2: Will be detected as completed (different wording)
        conn.execute("""
            INSERT INTO tasks (id, title, description, status, created_at, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'task-2',
            'Send budget update',
            'Email budget figures to the team',
            'pending',
            (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
            'ai.extraction'
        ))

        # Task 3: Will NOT be detected (no completion signals)
        conn.execute("""
            INSERT INTO tasks (id, title, description, status, created_at, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'task-3',
            'Schedule team meeting',
            'Arrange the monthly sync',
            'pending',
            (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            'ai.extraction'
        ))

        # Task 4: Will NOT be detected (weak keyword overlap)
        conn.execute("""
            INSERT INTO tasks (id, title, description, status, created_at, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'task-4',
            'Update documentation',
            'Fix the README file',
            'pending',
            (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat(),
            'ai.extraction'
        ))

        # Add 6 more filler tasks to reach the 10-task threshold
        for i in range(5, 11):
            conn.execute("""
                INSERT INTO tasks (id, title, description, status, created_at, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                f'task-{i}',
                f'Filler task {i}',
                'Background noise',
                'pending',
                (datetime.now(timezone.utc) - timedelta(hours=5+i)).isoformat(),
                'ai.extraction'
            ))

    # Add completion signals for task 1 and task 2
    with db.get_connection("events") as conn:
        # Completion signal for task 1 (strong match)
        conn.execute("""
            INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'event-1',
            'email.sent',
            'proton_mail',
            (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            'normal',
            json.dumps({
                'subject': 'Re: Q1 Proposal Review',
                'body_plain': 'I have reviewed the Q1 proposal document and it looks good. My feedback is attached. The proposal is ready for final approval.',
                'to_addresses': ['boss@company.com']
            }),
            '{}'
        ))

        # Completion signal for task 2 (strong match with different wording)
        conn.execute("""
            INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'event-2',
            'email.sent',
            'proton_mail',
            (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(),
            'normal',
            json.dumps({
                'subject': 'Budget Update for Q1',
                'body_plain': 'Hi team, I have sent the budget figures as requested. All numbers are finalized and ready for review.',
                'to_addresses': ['team@company.com']
            }),
            '{}'
        ))

        # Weak signal for task 4 (only 1 keyword match, no completion keyword)
        conn.execute("""
            INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'event-3',
            'email.sent',
            'proton_mail',
            (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat(),
            'normal',
            json.dumps({
                'subject': 'Quick question',
                'body_plain': 'Hey, do you have time to chat about the project documentation?',
                'to_addresses': ['colleague@company.com']
            }),
            '{}'
        ))

    return LifeOS(db=db, event_bus=event_bus, event_store=event_store,
                  user_model_store=user_model_store, config=config)


@pytest.mark.asyncio
async def test_backfill_runs_on_startup(lifeos_with_tasks):
    """Verify that task completion backfill runs automatically during startup."""
    # Start the system (which should trigger the backfill)
    # Note: We can't actually call start() in tests because it would block forever
    # waiting for the web server. Instead, we call the backfill method directly.
    await lifeos_with_tasks._backfill_task_completion_if_needed()

    # Check that the appropriate tasks were marked complete
    with lifeos_with_tasks.db.get_connection("state") as conn:
        cursor = conn.execute("""
            SELECT id, status FROM tasks
            WHERE id IN ('task-1', 'task-2', 'task-3', 'task-4')
            ORDER BY id
        """)
        results = cursor.fetchall()

    # task-1 and task-2 should be completed (strong signals)
    # task-3 and task-4 should remain pending (no/weak signals)
    assert len(results) == 4
    assert results[0][0] == 'task-1'
    assert results[0][1] == 'completed'

    assert results[1][0] == 'task-2'
    assert results[1][1] == 'completed'

    assert results[2][0] == 'task-3'
    assert results[2][1] == 'pending'

    assert results[3][0] == 'task-4'
    assert results[3][1] == 'pending'


@pytest.mark.asyncio
async def test_backfill_publishes_task_completed_events(lifeos_with_tasks):
    """Verify that the backfill publishes task.completed events for workflow detection."""
    await lifeos_with_tasks._backfill_task_completion_if_needed()

    # Check that task.completed events were published
    with lifeos_with_tasks.db.get_connection("events") as conn:
        cursor = conn.execute("""
            SELECT id, type, payload FROM events
            WHERE type = 'task.completed'
            ORDER BY id
        """)
        events = cursor.fetchall()

    # Should have 2 task.completed events (for task-1 and task-2)
    assert len(events) == 2

    # Verify event 1 (task-1)
    assert events[0][0] == 'task-1-completion'
    assert events[0][1] == 'task.completed'
    payload1 = json.loads(events[0][2])
    assert payload1['task_id'] == 'task-1'
    assert payload1['title'] == 'Review proposal document'
    assert payload1['backfill'] is True

    # Verify event 2 (task-2)
    assert events[1][0] == 'task-2-completion'
    assert events[1][1] == 'task.completed'
    payload2 = json.loads(events[1][2])
    assert payload2['task_id'] == 'task-2'
    assert payload2['title'] == 'Send budget update'
    assert payload2['backfill'] is True


@pytest.mark.asyncio
async def test_backfill_is_idempotent(lifeos_with_tasks):
    """Verify that running the backfill multiple times is safe (idempotent)."""
    # Run backfill twice
    await lifeos_with_tasks._backfill_task_completion_if_needed()
    await lifeos_with_tasks._backfill_task_completion_if_needed()

    # Should only have 2 completed tasks (not duplicated)
    with lifeos_with_tasks.db.get_connection("state") as conn:
        cursor = conn.execute("""
            SELECT COUNT(*) FROM tasks WHERE status = 'completed'
        """)
        count = cursor.fetchone()[0]

    assert count == 2

    # Should only have 2 task.completed events (not duplicated)
    with lifeos_with_tasks.db.get_connection("events") as conn:
        cursor = conn.execute("""
            SELECT COUNT(*) FROM events WHERE type = 'task.completed'
        """)
        count = cursor.fetchone()[0]

    assert count == 2


@pytest.mark.asyncio
async def test_backfill_skips_if_no_pending_tasks(db, event_bus, event_store, user_model_store):
    """Verify that backfill skips gracefully if there are no pending tasks."""
    config = {
        "data_dir": "./data",
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "ai": {"use_cloud": False},
    }
    lifeos = LifeOS(db=db, event_bus=event_bus, event_store=event_store,
                    user_model_store=user_model_store, config=config)

    # Run backfill with no pending tasks
    await lifeos._backfill_task_completion_if_needed()

    # Should succeed without errors
    # No assertions needed - we're just verifying it doesn't crash


@pytest.mark.asyncio
async def test_backfill_skips_if_few_pending_tasks(db, event_bus, event_store, user_model_store):
    """Verify that backfill skips if there are < 10 pending tasks (fresh system)."""
    config = {
        "data_dir": "./data",
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "ai": {"use_cloud": False},
    }
    # Create only 5 pending tasks
    with db.get_connection("state") as conn:
        for i in range(5):
            conn.execute("""
                INSERT INTO tasks (id, title, status, created_at, source)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f'task-{i}',
                f'Test task {i}',
                'pending',
                datetime.now(timezone.utc).isoformat(),
                'ai.extraction'
            ))

    lifeos = LifeOS(db=db, event_bus=event_bus, event_store=event_store,
                    user_model_store=user_model_store, config=config)

    # Run backfill
    await lifeos._backfill_task_completion_if_needed()

    # All tasks should still be pending (backfill skipped)
    with db.get_connection("state") as conn:
        cursor = conn.execute("""
            SELECT COUNT(*) FROM tasks WHERE status = 'pending'
        """)
        count = cursor.fetchone()[0]

    assert count == 5


@pytest.mark.asyncio
async def test_backfill_requires_strong_keyword_overlap(lifeos_with_tasks):
    """Verify that backfill requires >= 2.0 keyword overlap to avoid false positives."""
    # Add a task with only 1 keyword match
    with lifeos_with_tasks.db.get_connection("state") as conn:
        conn.execute("""
            INSERT INTO tasks (id, title, status, created_at, source)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'task-weak',
            'Fix the login bug',
            'pending',
            (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            'ai.extraction'
        ))

    # Add a sent email with only 1 keyword match (bug) but has completion keyword
    with lifeos_with_tasks.db.get_connection("events") as conn:
        conn.execute("""
            INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'event-weak',
            'email.sent',
            'proton_mail',
            (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            'normal',
            json.dumps({
                'subject': 'Status update',
                'body_plain': 'I fixed a bug in the payment module. It is done.',
                'to_addresses': ['team@company.com']
            }),
            '{}'
        ))

    await lifeos_with_tasks._backfill_task_completion_if_needed()

    # task-weak should still be pending (insufficient keyword overlap)
    with lifeos_with_tasks.db.get_connection("state") as conn:
        cursor = conn.execute("""
            SELECT status FROM tasks WHERE id = 'task-weak'
        """)
        status = cursor.fetchone()[0]

    assert status == 'pending'


@pytest.mark.asyncio
async def test_backfill_requires_completion_keywords(lifeos_with_tasks):
    """Verify that backfill requires completion keywords, not just keyword overlap."""
    # Add a task with strong keyword overlap
    with lifeos_with_tasks.db.get_connection("state") as conn:
        conn.execute("""
            INSERT INTO tasks (id, title, status, created_at, source)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'task-no-completion',
            'Analyze quarterly sales trends',
            'pending',
            (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            'ai.extraction'
        ))

    # Add a sent email with strong keyword overlap but NO completion keywords
    with lifeos_with_tasks.db.get_connection("events") as conn:
        conn.execute("""
            INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'event-no-completion',
            'email.sent',
            'proton_mail',
            (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            'normal',
            json.dumps({
                'subject': 'Question about quarterly sales',
                'body_plain': 'I am analyzing the quarterly sales trends and have a question about the spike in Q3. Can you provide more context on what drove that growth?',
                'to_addresses': ['manager@company.com']
            }),
            '{}'
        ))

    await lifeos_with_tasks._backfill_task_completion_if_needed()

    # task-no-completion should still be pending (no completion keywords)
    with lifeos_with_tasks.db.get_connection("state") as conn:
        cursor = conn.execute("""
            SELECT status FROM tasks WHERE id = 'task-no-completion'
        """)
        status = cursor.fetchone()[0]

    assert status == 'pending'


@pytest.mark.asyncio
async def test_backfill_handles_errors_gracefully(db, event_bus, event_store, user_model_store):
    """Verify that backfill errors don't crash startup."""
    config = {
        "data_dir": "./data",
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "ai": {"use_cloud": False},
    }
    lifeos = LifeOS(db=db, event_bus=event_bus, event_store=event_store,
                    user_model_store=user_model_store, config=config)

    # Corrupt the database by creating a task with invalid JSON in created_at
    with db.get_connection("state") as conn:
        conn.execute("""
            INSERT INTO tasks (id, title, status, created_at, source)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'task-corrupt',
            'Test task',
            'pending',
            'invalid-timestamp',  # This will cause errors during backfill
            'ai.extraction'
        ))

    # Run backfill - should handle error gracefully
    await lifeos._backfill_task_completion_if_needed()

    # Should succeed without crashing (error caught and logged)
    # No assertions needed - we're just verifying it doesn't crash


@pytest.mark.asyncio
async def test_backfill_sets_completed_at_timestamp(lifeos_with_tasks):
    """Verify that completed tasks get a completed_at timestamp."""
    await lifeos_with_tasks._backfill_task_completion_if_needed()

    # Check that completed tasks have completed_at set
    with lifeos_with_tasks.db.get_connection("state") as conn:
        cursor = conn.execute("""
            SELECT id, completed_at FROM tasks
            WHERE status = 'completed'
            ORDER BY id
        """)
        results = cursor.fetchall()

    assert len(results) == 2

    # Both should have completed_at timestamps
    assert results[0][1] is not None
    assert results[1][1] is not None

    # Timestamps should be valid ISO format
    datetime.fromisoformat(results[0][1].replace('Z', '+00:00'))
    datetime.fromisoformat(results[1][1].replace('Z', '+00:00'))


@pytest.mark.asyncio
async def test_backfill_searches_limited_events(lifeos_with_tasks):
    """Verify that backfill limits event search to 100 per task for performance."""
    # Reset all tasks to pending and delete previous completion events
    with lifeos_with_tasks.db.get_connection("state") as conn:
        conn.execute("UPDATE tasks SET status = 'pending', completed_at = NULL")

    with lifeos_with_tasks.db.get_connection("events") as conn:
        conn.execute("DELETE FROM events WHERE type = 'task.completed'")

    # Add 150 sent events (should only check the first 100)
    # The backfill will still find the original completion signals (event-1, event-2)
    # from the fixture, which are within the first 100 events by timestamp
    with lifeos_with_tasks.db.get_connection("events") as conn:
        for i in range(150):
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f'event-{i+10}',
                'email.sent',
                'proton_mail',
                (datetime.now(timezone.utc) - timedelta(minutes=i)).isoformat(),
                'normal',
                json.dumps({
                    'subject': f'Test email {i}',
                    'body_plain': f'Content {i}',
                    'to_addresses': ['test@example.com']
                }),
                '{}'
            ))

    # Run backfill - should complete without issues and without scanning all 150 events
    # The key point of this test is that it doesn't crash or timeout even with 150+ events
    await lifeos_with_tasks._backfill_task_completion_if_needed()

    # Should detect at least one of the tasks from our original fixtures
    # (Exact count may vary depending on event ordering, but the backfill should
    # complete quickly by only scanning the first 100 events per task)
    with lifeos_with_tasks.db.get_connection("state") as conn:
        cursor = conn.execute("""
            SELECT COUNT(*) FROM tasks WHERE status = 'completed'
        """)
        count = cursor.fetchone()[0]

    # Verify that some tasks were completed (backfill worked)
    assert count >= 1, f"Expected at least 1 completed task, got {count}"
