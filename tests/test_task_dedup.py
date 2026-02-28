"""Tests for AI-extracted task deduplication in TaskManager.

Verifies that ingest_ai_extracted_tasks skips tasks whose normalized title
already exists in the database within the 30-day lookback window.
"""

from __future__ import annotations

import pytest

from services.task_manager.manager import TaskManager


class _FakeAIEngine:
    """Minimal AI engine stub that returns a fixed list of action items."""

    async def extract_action_items(self, text, event_type=None):  # noqa: ARG002
        return []


@pytest.fixture()
def task_manager(db):
    """TaskManager wired to the test database with a stub AI engine."""
    return TaskManager(db, ai_engine=_FakeAIEngine())


@pytest.mark.asyncio
async def test_duplicate_task_is_skipped(task_manager):
    """ingest_ai_extracted_tasks should not create a second task with the same title."""
    tasks = [{"title": "Review quarterly report", "priority": "normal"}]

    # First ingest creates the task
    await task_manager.ingest_ai_extracted_tasks(tasks, source_event_id="evt-1")
    # Second ingest with the same title must be skipped
    await task_manager.ingest_ai_extracted_tasks(tasks, source_event_id="evt-2")

    all_tasks = task_manager.get_tasks(limit=100)
    titles = [t["title"] for t in all_tasks]
    assert titles.count("Review quarterly report") == 1, (
        "Duplicate task was created despite deduplication guard"
    )


@pytest.mark.asyncio
async def test_case_insensitive_dedup(task_manager):
    """Deduplication must be case-insensitive and whitespace-insensitive."""
    await task_manager.ingest_ai_extracted_tasks(
        [{"title": "Send report to Alice", "priority": "normal"}],
        source_event_id="evt-1",
    )
    # Same title with different casing
    await task_manager.ingest_ai_extracted_tasks(
        [{"title": "SEND REPORT TO ALICE", "priority": "normal"}],
        source_event_id="evt-2",
    )
    # Same title with extra whitespace
    await task_manager.ingest_ai_extracted_tasks(
        [{"title": "  send report to alice  ", "priority": "normal"}],
        source_event_id="evt-3",
    )

    all_tasks = task_manager.get_tasks(limit=100)
    matching = [t for t in all_tasks if "send report to alice" in t["title"].lower()]
    assert len(matching) == 1, (
        f"Expected 1 task but got {len(matching)}: {[t['title'] for t in matching]}"
    )


@pytest.mark.asyncio
async def test_distinct_titles_both_created(task_manager):
    """Two tasks with different titles must both be created."""
    await task_manager.ingest_ai_extracted_tasks(
        [
            {"title": "Book flight to Chicago", "priority": "normal"},
            {"title": "Book hotel in Chicago", "priority": "normal"},
        ],
        source_event_id="evt-1",
    )

    all_tasks = task_manager.get_tasks(limit=100)
    titles = {t["title"] for t in all_tasks}
    assert "Book flight to Chicago" in titles
    assert "Book hotel in Chicago" in titles


@pytest.mark.asyncio
async def test_dedup_across_different_source_events(task_manager):
    """Deduplication should apply even when source_event_id differs."""
    await task_manager.ingest_ai_extracted_tasks(
        [{"title": "Update project docs", "priority": "low"}],
        source_event_id="email-abc",
    )
    # Same task extracted from a different email
    await task_manager.ingest_ai_extracted_tasks(
        [{"title": "Update project docs", "priority": "low"}],
        source_event_id="email-def",
    )

    all_tasks = task_manager.get_tasks(limit=100)
    count = sum(1 for t in all_tasks if t["title"] == "Update project docs")
    assert count == 1, f"Expected 1 task but got {count}"


@pytest.mark.asyncio
async def test_is_duplicate_task_returns_false_for_new_title(task_manager):
    """_is_duplicate_task must return False when no matching task exists."""
    assert not task_manager._is_duplicate_task("Completely new task title")


@pytest.mark.asyncio
async def test_is_duplicate_task_returns_true_after_creation(task_manager):
    """_is_duplicate_task must return True after the task is created."""
    title = "Write status update"
    await task_manager.create_task(title, source="user")
    assert task_manager._is_duplicate_task(title)
    assert task_manager._is_duplicate_task(title.upper())
    assert task_manager._is_duplicate_task(f"  {title.lower()}  ")


@pytest.mark.asyncio
async def test_dedup_does_not_block_user_created_tasks(task_manager):
    """User-created tasks bypass deduplication (they use create_task directly)."""
    title = "Call dentist"
    await task_manager.create_task(title, source="user")
    # User can create the same task again manually (create_task has no dedup)
    await task_manager.create_task(title, source="user")

    all_tasks = task_manager.get_tasks(limit=100)
    count = sum(1 for t in all_tasks if t["title"] == title)
    assert count == 2, (
        "User-created tasks should not be subject to AI deduplication"
    )
