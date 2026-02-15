"""
Life OS — FeedbackCollector Test Suite

Comprehensive test coverage for the feedback collection and learning loop.
Tests all feedback types, learning mechanisms, and semantic fact updates.
"""

import json
import pytest
from datetime import datetime, timezone
from services.feedback_collector.collector import FeedbackCollector
from models.core import FeedbackType, Priority


# -----------------------------------------------------------------------------
# Notification Response Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notification_dismissed_quick(db, user_model_store):
    """Test that quick dismissals (<2 sec) create strong negative signal."""
    collector = FeedbackCollector(db, user_model_store)

    # Create a test notification
    with db.get_connection("state") as conn:
        notif_id = "test-notif-1"
        conn.execute(
            """INSERT INTO notifications
               (id, priority, domain, title, body, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, Priority.NORMAL.value, "email", "Test title", "Test message",
             "delivered", datetime.now(timezone.utc).isoformat())
        )

    # Process quick dismissal (1 second)
    await collector.process_notification_response(
        notification_id=notif_id,
        response_type=FeedbackType.DISMISSED.value,
        response_time_seconds=1.0
    )

    # Verify feedback was stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (notif_id,)
        ).fetchone()

        assert feedback is not None
        assert feedback["feedback_type"] == FeedbackType.DISMISSED.value
        assert feedback["response_latency_seconds"] == 1.0

        context = json.loads(feedback["context"])
        assert context["priority"] == Priority.NORMAL.value
        assert context["domain"] == "email"

    # Verify semantic fact was created (irrelevant signal)
    facts = user_model_store.get_semantic_facts(category="notification_preference")
    email_fact = next((f for f in facts if "email" in f["key"]), None)
    assert email_fact is not None
    assert email_fact["confidence"] == 0.4
    assert "quickly dismisses" in email_fact["value"]


@pytest.mark.asyncio
async def test_notification_dismissed_slow(db, user_model_store):
    """Test that slow dismissals (>10 sec) are recorded but don't create strong signal."""
    collector = FeedbackCollector(db, user_model_store)

    # Create a test notification
    with db.get_connection("state") as conn:
        notif_id = "test-notif-2"
        conn.execute(
            """INSERT INTO notifications
               (id, priority, domain, title, body, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, Priority.LOW.value, "calendar", "Test title", "Test reminder",
             "delivered", datetime.now(timezone.utc).isoformat())
        )

    # Process slow dismissal (15 seconds)
    await collector.process_notification_response(
        notification_id=notif_id,
        response_type=FeedbackType.DISMISSED.value,
        response_time_seconds=15.0
    )

    # Verify feedback was stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (notif_id,)
        ).fetchone()

        assert feedback is not None
        assert feedback["response_latency_seconds"] == 15.0

    # Verify NO semantic fact was created (slow dismissal doesn't trigger learning)
    facts = user_model_store.get_semantic_facts(category="notification_preference")
    calendar_fact = next((f for f in facts if "calendar" in f["key"]), None)
    assert calendar_fact is None


@pytest.mark.asyncio
async def test_notification_engaged_fast(db, user_model_store):
    """Test that fast engagement (<30 sec) creates positive signal."""
    collector = FeedbackCollector(db, user_model_store)

    # Create a test notification
    with db.get_connection("state") as conn:
        notif_id = "test-notif-3"
        conn.execute(
            """INSERT INTO notifications
               (id, priority, domain, title, body, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, Priority.HIGH.value, "task", "Deadline", "Deadline approaching",
             "delivered", datetime.now(timezone.utc).isoformat())
        )

    # Process fast engagement (10 seconds)
    await collector.process_notification_response(
        notification_id=notif_id,
        response_type=FeedbackType.ENGAGED.value,
        response_time_seconds=10.0
    )

    # Verify feedback was stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (notif_id,)
        ).fetchone()

        assert feedback is not None
        assert feedback["feedback_type"] == FeedbackType.ENGAGED.value

    # Verify semantic fact was created (relevant signal)
    facts = user_model_store.get_semantic_facts(category="notification_preference")
    task_fact = next((f for f in facts if "task" in f["key"]), None)
    assert task_fact is not None
    assert task_fact["confidence"] == 0.6
    assert "quickly acts on" in task_fact["value"]


@pytest.mark.asyncio
async def test_notification_engaged_slow(db, user_model_store):
    """Test that slow engagement (>30 sec) is recorded but doesn't create strong signal."""
    collector = FeedbackCollector(db, user_model_store)

    # Create a test notification
    with db.get_connection("state") as conn:
        notif_id = "test-notif-4"
        conn.execute(
            """INSERT INTO notifications
               (id, priority, domain, title, body, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, Priority.NORMAL.value, "finance", "Transaction", "Transaction alert",
             "delivered", datetime.now(timezone.utc).isoformat())
        )

    # Process slow engagement (60 seconds)
    await collector.process_notification_response(
        notification_id=notif_id,
        response_type=FeedbackType.ENGAGED.value,
        response_time_seconds=60.0
    )

    # Verify feedback was stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (notif_id,)
        ).fetchone()

        assert feedback is not None
        assert feedback["response_latency_seconds"] == 60.0

    # Verify NO semantic fact was created (slow engagement doesn't trigger learning)
    facts = user_model_store.get_semantic_facts(category="notification_preference")
    finance_fact = next((f for f in facts if "finance" in f["key"]), None)
    assert finance_fact is None


@pytest.mark.asyncio
async def test_notification_ignored(db, user_model_store):
    """Test that ignored notifications create strongest negative signal."""
    collector = FeedbackCollector(db, user_model_store)

    # Create a test notification
    with db.get_connection("state") as conn:
        notif_id = "test-notif-5"
        conn.execute(
            """INSERT INTO notifications
               (id, priority, domain, title, body, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, Priority.LOW.value, "social", "Social", "New follower",
             "delivered", datetime.now(timezone.utc).isoformat())
        )

    # Process ignore event
    await collector.process_notification_response(
        notification_id=notif_id,
        response_type=FeedbackType.IGNORED.value,
        response_time_seconds=0.0
    )

    # Verify feedback was stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (notif_id,)
        ).fetchone()

        assert feedback is not None
        assert feedback["feedback_type"] == FeedbackType.IGNORED.value

    # Verify semantic fact was created (unwanted signal)
    facts = user_model_store.get_semantic_facts(category="notification_preference")
    social_fact = next((f for f in facts if "social" in f["key"]), None)
    assert social_fact is not None
    assert social_fact["confidence"] == 0.5
    assert "ignores notifications about" in social_fact["value"]


@pytest.mark.asyncio
async def test_notification_response_with_context(db, user_model_store):
    """Test that additional context is preserved in feedback."""
    collector = FeedbackCollector(db, user_model_store)

    # Create a test notification
    with db.get_connection("state") as conn:
        notif_id = "test-notif-6"
        conn.execute(
            """INSERT INTO notifications
               (id, priority, domain, title, body, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, Priority.HIGH.value, "email", "Important", "Important message",
             "delivered", datetime.now(timezone.utc).isoformat())
        )

    # Process with custom context
    await collector.process_notification_response(
        notification_id=notif_id,
        response_type=FeedbackType.ENGAGED.value,
        response_time_seconds=5.0,
        context={"device": "iphone", "location": "home"}
    )

    # Verify context was preserved
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (notif_id,)
        ).fetchone()

        context = json.loads(feedback["context"])
        assert context["device"] == "iphone"
        assert context["location"] == "home"
        assert "hour_of_day" in context


@pytest.mark.asyncio
async def test_notification_response_nonexistent(db, user_model_store):
    """Test that processing feedback for nonexistent notification fails gracefully."""
    collector = FeedbackCollector(db, user_model_store)

    # Process response for notification that doesn't exist
    await collector.process_notification_response(
        notification_id="nonexistent",
        response_type=FeedbackType.ENGAGED.value,
        response_time_seconds=10.0
    )

    # Should not create feedback record
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            ("nonexistent",)
        ).fetchone()

        assert feedback is None


# -----------------------------------------------------------------------------
# Draft Edit Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_draft_accepted_as_is(db, user_model_store):
    """Test that accepting draft without changes creates strong positive signal."""
    collector = FeedbackCollector(db, user_model_store)

    original = "Hey, just wanted to check in on the project status."
    final = original  # Accepted as-is

    await collector.process_draft_edit(
        original_draft=original,
        final_message=final,
        contact_id="contact-1",
        channel="imessage"
    )

    # Verify feedback was stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_type = 'draft'"
        ).fetchone()

        assert feedback is not None
        assert feedback["feedback_type"] == FeedbackType.ENGAGED.value

        context = json.loads(feedback["context"])
        assert context["accepted_as_is"] is True
        assert context["contact_id"] == "contact-1"
        assert context["channel"] == "imessage"


@pytest.mark.asyncio
async def test_draft_made_more_informal(db, user_model_store):
    """Test detection of user making draft more casual."""
    collector = FeedbackCollector(db, user_model_store)

    # First, create a communication template
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO communication_templates
               (id, context, contact_id, channel, formality, greeting, closing, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("template-1", "work", "contact-2", "email", 0.8, "Hello", "Best regards",
             datetime.now(timezone.utc).isoformat())
        )

    original = "Hello, I wanted to follow up regarding your previous message."
    final = "Hey, just wanted to follow up lol"

    await collector.process_draft_edit(
        original_draft=original,
        final_message=final,
        contact_id="contact-2",
        channel="email"
    )

    # Verify feedback was stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_type = 'draft'"
        ).fetchone()

        assert feedback is not None
        assert feedback["feedback_type"] == FeedbackType.OVERRIDDEN.value

        context = json.loads(feedback["context"])
        assert context["formality_shift"] == "more_informal"
        assert context["words_added"] > 0
        assert context["words_removed"] > 0

    # Verify template formality was reduced
    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT * FROM communication_templates WHERE id = ?",
            ("template-1",)
        ).fetchone()

        # Original was 0.8, should now be 0.75 (reduced by 0.05)
        assert template["formality"] == 0.75


@pytest.mark.asyncio
async def test_draft_made_more_formal(db, user_model_store):
    """Test detection of user making draft more formal."""
    collector = FeedbackCollector(db, user_model_store)

    # Create a communication template
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO communication_templates
               (id, context, contact_id, channel, formality, greeting, closing, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("template-2", "casual", "contact-3", "email", 0.3, "Hey", "Thanks",
             datetime.now(timezone.utc).isoformat())
        )

    original = "Hey, wanna grab coffee sometime?"
    final = "Good morning, I would like to schedule a meeting regarding the project."

    await collector.process_draft_edit(
        original_draft=original,
        final_message=final,
        contact_id="contact-3",
        channel="email"
    )

    # Verify feedback context
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_type = 'draft'"
        ).fetchone()

        context = json.loads(feedback["context"])
        assert context["formality_shift"] == "more_formal"

    # Verify template formality was increased
    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT * FROM communication_templates WHERE id = ?",
            ("template-2",)
        ).fetchone()

        # Original was 0.3, should now be 0.35 (increased by 0.05)
        assert template["formality"] == 0.35


@pytest.mark.asyncio
async def test_draft_length_change(db, user_model_store):
    """Test tracking of draft length changes."""
    collector = FeedbackCollector(db, user_model_store)

    original = "Quick update."
    final = "Quick update on the project status. Everything is on track and we're making good progress toward the deadline."

    await collector.process_draft_edit(
        original_draft=original,
        final_message=final,
        contact_id="contact-4"
    )

    # Verify length change was captured
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_type = 'draft'"
        ).fetchone()

        context = json.loads(feedback["context"])
        # User made it much longer
        assert context["length_change_pct"] > 5.0
        assert context["words_added"] > context["words_removed"]


@pytest.mark.asyncio
async def test_draft_template_formality_clamp_low(db, user_model_store):
    """Test that formality doesn't go below 0.0."""
    collector = FeedbackCollector(db, user_model_store)

    # Create template with very low formality
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO communication_templates
               (id, context, contact_id, channel, formality, greeting, closing, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("template-3", "very_casual", "contact-5", "sms", 0.02, "yo", "cya",
             datetime.now(timezone.utc).isoformat())
        )

    original = "Hello there"
    final = "hey lol"

    await collector.process_draft_edit(
        original_draft=original,
        final_message=final,
        contact_id="contact-5",
        channel="sms"
    )

    # Verify formality clamped at 0.0
    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT * FROM communication_templates WHERE id = ?",
            ("template-3",)
        ).fetchone()

        assert template["formality"] == 0.0


@pytest.mark.asyncio
async def test_draft_template_formality_clamp_high(db, user_model_store):
    """Test that formality doesn't go above 1.0."""
    collector = FeedbackCollector(db, user_model_store)

    # Create template with very high formality
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO communication_templates
               (id, context, contact_id, channel, formality, greeting, closing, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("template-4", "formal", "contact-6", "email", 0.98, "Dear", "Sincerely",
             datetime.now(timezone.utc).isoformat())
        )

    original = "hey"
    final = "Dear Sir, regarding your previous correspondence, please be advised..."

    await collector.process_draft_edit(
        original_draft=original,
        final_message=final,
        contact_id="contact-6",
        channel="email"
    )

    # Verify formality clamped at 1.0
    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT * FROM communication_templates WHERE id = ?",
            ("template-4",)
        ).fetchone()

        assert template["formality"] == 1.0


@pytest.mark.asyncio
async def test_draft_no_template_exists(db, user_model_store):
    """Test that draft edit without existing template still records feedback."""
    collector = FeedbackCollector(db, user_model_store)

    original = "Hello"
    final = "Hey there lol"

    await collector.process_draft_edit(
        original_draft=original,
        final_message=final,
        contact_id="contact-999",  # No template exists
        channel="unknown"
    )

    # Verify feedback was still stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_type = 'draft'"
        ).fetchone()

        assert feedback is not None
        context = json.loads(feedback["context"])
        assert context["contact_id"] == "contact-999"


# -----------------------------------------------------------------------------
# Suggestion Response Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_suggestion_accepted(db, user_model_store):
    """Test processing accepted suggestion."""
    collector = FeedbackCollector(db, user_model_store)

    # Create a prediction record
    with db.get_connection("user_model") as conn:
        suggestion_id = "suggestion-1"
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (suggestion_id, "next_action", "Send email to John", 0.75, "default",
             datetime.now(timezone.utc).isoformat())
        )

    await collector.process_suggestion_response(
        suggestion_id=suggestion_id,
        accepted=True
    )

    # Verify feedback was stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (suggestion_id,)
        ).fetchone()

        assert feedback is not None
        assert feedback["feedback_type"] == FeedbackType.ENGAGED.value

        context = json.loads(feedback["context"])
        assert context["accepted"] is True

    # Verify prediction was marked as accurate
    with db.get_connection("user_model") as conn:
        prediction = conn.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (suggestion_id,)
        ).fetchone()

        assert prediction["user_response"] == "accepted"
        assert prediction["was_accurate"] == 1
        assert prediction["resolved_at"] is not None


@pytest.mark.asyncio
async def test_suggestion_rejected_with_alternative(db, user_model_store):
    """Test processing rejected suggestion with user alternative."""
    collector = FeedbackCollector(db, user_model_store)

    # Create a prediction record
    with db.get_connection("user_model") as conn:
        suggestion_id = "suggestion-2"
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (suggestion_id, "next_action", "Call mom", 0.65, "suggest",
             datetime.now(timezone.utc).isoformat())
        )

    await collector.process_suggestion_response(
        suggestion_id=suggestion_id,
        accepted=False,
        user_alternative="Send her a text instead"
    )

    # Verify feedback was stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (suggestion_id,)
        ).fetchone()

        assert feedback is not None
        assert feedback["feedback_type"] == FeedbackType.OVERRIDDEN.value

        context = json.loads(feedback["context"])
        assert context["accepted"] is False
        assert context["user_alternative"] == "Send her a text instead"

    # Verify prediction was marked as inaccurate
    with db.get_connection("user_model") as conn:
        prediction = conn.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (suggestion_id,)
        ).fetchone()

        assert prediction["user_response"] == "rejected"
        assert prediction["was_accurate"] == 0


# -----------------------------------------------------------------------------
# Explicit Feedback Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explicit_positive_feedback(db, user_model_store):
    """Test processing explicit positive feedback."""
    collector = FeedbackCollector(db, user_model_store)

    message = "That was really helpful, thanks!"

    await collector.process_explicit_feedback(message)

    # Verify feedback was stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_type = 'explicit'"
        ).fetchone()

        assert feedback is not None
        assert feedback["feedback_type"] == FeedbackType.EXPLICIT_POSITIVE.value
        assert feedback["notes"] == message


@pytest.mark.asyncio
async def test_explicit_negative_feedback(db, user_model_store):
    """Test processing explicit negative feedback."""
    collector = FeedbackCollector(db, user_model_store)

    message = "Stop sending me notifications about this, it's annoying"

    await collector.process_explicit_feedback(message)

    # Verify feedback was stored
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_type = 'explicit'"
        ).fetchone()

        assert feedback is not None
        assert feedback["feedback_type"] == FeedbackType.EXPLICIT_NEGATIVE.value


@pytest.mark.asyncio
async def test_explicit_preference_stored_as_fact(db, user_model_store):
    """Test that explicit preferences become semantic facts."""
    collector = FeedbackCollector(db, user_model_store)

    message = "I prefer email over text messages for work communication"

    await collector.process_explicit_feedback(message)

    # Verify semantic fact was created
    with db.get_connection("user_model") as conn:
        facts = conn.execute(
            """SELECT * FROM semantic_facts
               WHERE category = 'explicit_preference'
               ORDER BY first_observed DESC
               LIMIT 1"""
        ).fetchone()

        assert facts is not None
        # Value is stored as JSON string
        assert json.loads(facts["value"]) == message
        assert facts["confidence"] == 0.95  # High confidence for explicit statements


@pytest.mark.asyncio
async def test_explicit_feedback_neutral_classification(db, user_model_store):
    """Test that ambiguous feedback is classified as neutral."""
    collector = FeedbackCollector(db, user_model_store)

    message = "I'm not sure about this"

    await collector.process_explicit_feedback(message)

    # Verify feedback was stored as neutral (ENGAGED)
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_type = 'explicit'"
        ).fetchone()

        assert feedback is not None
        # No clear positive or negative words, defaults to ENGAGED
        assert feedback["feedback_type"] == FeedbackType.ENGAGED.value


# -----------------------------------------------------------------------------
# Feedback Summary Tests
# -----------------------------------------------------------------------------

def test_feedback_summary_empty(db, user_model_store):
    """Test feedback summary with no data."""
    collector = FeedbackCollector(db, user_model_store)

    summary = collector.get_feedback_summary()

    assert summary == {}


@pytest.mark.asyncio
async def test_feedback_summary_aggregated(db, user_model_store):
    """Test that feedback summary aggregates by type and action."""
    collector = FeedbackCollector(db, user_model_store)

    # Create multiple notifications
    with db.get_connection("state") as conn:
        for i in range(5):
            conn.execute(
                """INSERT INTO notifications
                   (id, priority, domain, title, body, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (f"notif-{i}", Priority.NORMAL.value, "email", f"Title {i}", f"Message {i}",
                 "delivered", datetime.now(timezone.utc).isoformat())
            )

    # Process 3 dismissals and 2 engagements
    for i in range(3):
        await collector.process_notification_response(
            notification_id=f"notif-{i}",
            response_type=FeedbackType.DISMISSED.value,
            response_time_seconds=1.0
        )

    for i in range(3, 5):
        await collector.process_notification_response(
            notification_id=f"notif-{i}",
            response_type=FeedbackType.ENGAGED.value,
            response_time_seconds=10.0
        )

    summary = collector.get_feedback_summary()

    assert summary["notification:dismissed"] == 3
    assert summary["notification:engaged"] == 2


# -----------------------------------------------------------------------------
# Edge Cases and Error Handling
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_feedback_on_same_action(db, user_model_store):
    """Test that multiple feedback entries for same action are all recorded."""
    collector = FeedbackCollector(db, user_model_store)

    # Create a notification
    with db.get_connection("state") as conn:
        notif_id = "notif-multi"
        conn.execute(
            """INSERT INTO notifications
               (id, priority, domain, title, body, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, Priority.HIGH.value, "task", "Test", "Test notification",
             "delivered", datetime.now(timezone.utc).isoformat())
        )

    # Process multiple responses (edge case: user dismisses, then engages later)
    await collector.process_notification_response(
        notification_id=notif_id,
        response_type=FeedbackType.DISMISSED.value,
        response_time_seconds=1.0
    )

    await collector.process_notification_response(
        notification_id=notif_id,
        response_type=FeedbackType.ENGAGED.value,
        response_time_seconds=300.0
    )

    # Both should be recorded
    with db.get_connection("preferences") as conn:
        feedback_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM feedback_log WHERE action_id = ?",
            (notif_id,)
        ).fetchone()

        assert feedback_count["cnt"] == 2


@pytest.mark.asyncio
async def test_feedback_with_empty_context(db, user_model_store):
    """Test that feedback works with minimal/empty context."""
    collector = FeedbackCollector(db, user_model_store)

    message = "okay"

    await collector.process_explicit_feedback(message)

    # Should still create feedback entry
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_type = 'explicit'"
        ).fetchone()

        assert feedback is not None
        context = json.loads(feedback["context"])
        assert "raw_message" in context


def test_classify_explicit_feedback_positive_keywords(db, user_model_store):
    """Test positive keyword classification."""
    collector = FeedbackCollector(db, user_model_store)

    result = collector._classify_explicit_feedback("This is great and awesome!")
    assert result == FeedbackType.EXPLICIT_POSITIVE.value


def test_classify_explicit_feedback_negative_keywords(db, user_model_store):
    """Test negative keyword classification."""
    collector = FeedbackCollector(db, user_model_store)

    result = collector._classify_explicit_feedback("This is terrible and annoying")
    assert result == FeedbackType.EXPLICIT_NEGATIVE.value


def test_classify_explicit_feedback_mixed_keywords(db, user_model_store):
    """Test mixed sentiment defaults to majority."""
    collector = FeedbackCollector(db, user_model_store)

    # More positive than negative
    result = collector._classify_explicit_feedback("good great but slightly bad")
    assert result == FeedbackType.EXPLICIT_POSITIVE.value


@pytest.mark.asyncio
async def test_draft_edit_no_contact_or_channel(db, user_model_store):
    """Test draft edit without contact or channel still records feedback."""
    collector = FeedbackCollector(db, user_model_store)

    original = "Original text"
    final = "Edited text"

    await collector.process_draft_edit(
        original_draft=original,
        final_message=final
    )

    # Should still create feedback entry
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_type = 'draft'"
        ).fetchone()

        assert feedback is not None
        assert feedback["feedback_type"] == FeedbackType.OVERRIDDEN.value
