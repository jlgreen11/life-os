"""
Life OS — FeedbackCollector Resilience Tests

Verifies that the FeedbackCollector degrades gracefully when user_model.db
is corrupted or unavailable. The primary feedback storage (preferences.db)
must succeed even when secondary writes to user_model.db fail.
"""

import json
import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from models.core import FeedbackType
from services.feedback_collector.collector import FeedbackCollector


@pytest.mark.asyncio
async def test_suggestion_response_succeeds_when_user_model_db_fails(db, user_model_store):
    """process_suggestion_response stores feedback even when user_model.db raises."""
    collector = FeedbackCollector(db, user_model_store)

    suggestion_id = "resilience-suggestion-1"

    # Patch get_connection so that calls to "user_model" raise, but
    # "preferences" (and any other DB) work normally.
    original_get_connection = db.get_connection

    def _failing_user_model(db_name):
        if db_name == "user_model":
            raise sqlite3.OperationalError("database disk image is malformed")
        return original_get_connection(db_name)

    with patch.object(db, "get_connection", side_effect=_failing_user_model):
        # Should NOT raise — the prediction update failure is swallowed
        await collector.process_suggestion_response(
            suggestion_id=suggestion_id,
            accepted=True,
        )

    # Verify the primary feedback record was stored in preferences.db
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (suggestion_id,),
        ).fetchone()

    assert feedback is not None
    assert feedback["feedback_type"] == FeedbackType.ENGAGED.value
    context = json.loads(feedback["context"])
    assert context["accepted"] is True


@pytest.mark.asyncio
async def test_suggestion_response_rejected_when_user_model_db_fails(db, user_model_store):
    """Rejected suggestions also store feedback despite user_model.db failure."""
    collector = FeedbackCollector(db, user_model_store)

    suggestion_id = "resilience-suggestion-2"

    original_get_connection = db.get_connection

    def _failing_user_model(db_name):
        if db_name == "user_model":
            raise sqlite3.OperationalError("database disk image is malformed")
        return original_get_connection(db_name)

    with patch.object(db, "get_connection", side_effect=_failing_user_model):
        await collector.process_suggestion_response(
            suggestion_id=suggestion_id,
            accepted=False,
            user_alternative="Do something else",
        )

    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (suggestion_id,),
        ).fetchone()

    assert feedback is not None
    assert feedback["feedback_type"] == FeedbackType.OVERRIDDEN.value
    context = json.loads(feedback["context"])
    assert context["accepted"] is False
    assert context["user_alternative"] == "Do something else"


@pytest.mark.asyncio
async def test_draft_edit_succeeds_when_template_update_fails(db, user_model_store):
    """process_draft_edit stores feedback even when _update_template_from_edit fails."""
    collector = FeedbackCollector(db, user_model_store)

    original_get_connection = db.get_connection
    call_count = {"user_model": 0}

    def _failing_user_model(db_name):
        """Fail on user_model access during template update."""
        if db_name == "user_model":
            call_count["user_model"] += 1
            raise sqlite3.OperationalError("database disk image is malformed")
        return original_get_connection(db_name)

    # The draft edit adds informal words to trigger a formality shift,
    # which causes _update_template_from_edit to access user_model.db.
    original = "Hello, I wanted to follow up regarding your previous message."
    final = "Hey, just wanted to follow up lol"

    with patch.object(db, "get_connection", side_effect=_failing_user_model):
        # Should NOT raise even though template update will fail
        await collector.process_draft_edit(
            original_draft=original,
            final_message=final,
            contact_id="contact-resilience",
            channel="email",
        )

    # Verify the primary feedback was stored in preferences.db
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_type = 'draft'",
        ).fetchone()

    assert feedback is not None
    assert feedback["feedback_type"] == FeedbackType.OVERRIDDEN.value
    context = json.loads(feedback["context"])
    assert context["contact_id"] == "contact-resilience"
    assert context["formality_shift"] == "more_informal"

    # Confirm user_model was attempted (the try/except caught it)
    assert call_count["user_model"] >= 1


def test_update_template_from_edit_catches_db_error(db, user_model_store):
    """_update_template_from_edit returns without raising when user_model.db is corrupted."""
    collector = FeedbackCollector(db, user_model_store)

    original_get_connection = db.get_connection

    def _failing_user_model(db_name):
        if db_name == "user_model":
            raise sqlite3.OperationalError("database disk image is malformed")
        return original_get_connection(db_name)

    edit_context = {
        "formality_shift": "more_informal",
        "length_change_pct": -0.3,
        "words_added": 2,
        "words_removed": 5,
    }

    with patch.object(db, "get_connection", side_effect=_failing_user_model):
        # Should NOT raise — the exception is caught internally
        collector._update_template_from_edit(
            contact_id="contact-x",
            channel="email",
            original="Hello there",
            final="Hey lol",
            edit_context=edit_context,
        )
    # If we reach here without an exception, the test passes
