"""
Tests for episodic memory creation.

Verifies that the master event handler correctly converts events into
episodic memory entries in the user_model.db episodes table.

NOTE on interaction_type values:
  These tests assert the granular interaction types introduced when the
  routine detector was improved.  The old generic types ("communication",
  "calendar", "task", "financial", "context") were replaced by specific
  values so the detector can distinguish inbox-checking from correspondence,
  planning from participation, etc.  The mapping is:

    email.received          → "email_received"
    email.sent              → "email_sent"
    message.received        → "message_received"
    message.sent            → "message_sent"
    calendar.event.created  → "meeting_scheduled" (with attendees)
                              "calendar_blocked"  (without attendees)
    task.completed          → "task_completed"
    finance.transaction.new → "income"    (amount >= 0)
                              "spending"  (amount < 0)
    location.arrived        → "location_arrived"
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from models.core import EventType


@pytest.mark.asyncio
async def test_episodic_memory_email_received(db, event_store, user_model_store, event_bus):
    """Test that email.received events create episodic memories."""
    from main import LifeOS

    # Create a test email event
    event = {
        "id": str(uuid.uuid4()),
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "gmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "from_address": "alice@example.com",
            "to_addresses": ["me@example.com"],
            "subject": "Quarterly review meeting",
            "body_plain": "Let's discuss the Q1 results next Tuesday.",
            "message_id": "test-message-1",
        },
        "metadata": {
            "domain": "work",
        },
    }

    # Initialize LifeOS with minimal config
    config = {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }

    lifeos = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )

    # Manually call the episode creation method
    await lifeos._create_episode(event)

    # Verify episode was created
    with db.get_connection("user_model") as conn:
        episodes = conn.execute("SELECT * FROM episodes WHERE event_id = ?", (event["id"],)).fetchall()
        assert len(episodes) == 1

        episode = dict(episodes[0])
        assert episode["event_id"] == event["id"]
        # Granular type introduced so the routine detector can distinguish
        # inbox-checking ("email_received") from correspondence ("email_sent").
        assert episode["interaction_type"] == "email_received"
        assert episode["active_domain"] == "work"

        # Verify contacts were extracted
        contacts = json.loads(episode["contacts_involved"])
        assert "alice@example.com" in contacts

        # Verify summary is generated
        assert "alice@example.com" in episode["content_summary"]
        assert "Quarterly review" in episode["content_summary"]

        # Verify metadata fields are preserved in compact content_full
        # (body fields are stripped to prevent DB bloat — see iteration 43)
        content_full = json.loads(episode["content_full"])
        assert content_full["subject"] == "Quarterly review meeting"


@pytest.mark.asyncio
async def test_episodic_memory_email_sent(db, event_store, user_model_store, event_bus):
    """Test that email.sent events create episodic memories."""
    from main import LifeOS

    event = {
        "id": str(uuid.uuid4()),
        "type": EventType.EMAIL_SENT.value,
        "source": "gmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "from_address": "me@example.com",
            "to_addresses": ["bob@company.com", "charlie@company.com"],
            "subject": "Project update",
            "body_plain": "The feature is now complete and ready for review.",
            "message_id": "test-message-2",
        },
        "metadata": {
            "domain": "work",
        },
    }

    config = {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }

    lifeos = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )

    await lifeos._create_episode(event)

    with db.get_connection("user_model") as conn:
        episodes = conn.execute("SELECT * FROM episodes WHERE event_id = ?", (event["id"],)).fetchall()
        assert len(episodes) == 1

        episode = dict(episodes[0])
        # Granular type distinguishes outbound correspondence from inbound inbox-checking.
        assert episode["interaction_type"] == "email_sent"

        # Verify multiple recipients are captured
        contacts = json.loads(episode["contacts_involved"])
        assert "bob@company.com" in contacts
        assert "charlie@company.com" in contacts

        # Verify summary shows "to" direction
        assert "Email to" in episode["content_summary"]
        assert "Project update" in episode["content_summary"]


@pytest.mark.asyncio
async def test_episodic_memory_calendar_event(db, event_store, user_model_store, event_bus):
    """Test that calendar events create episodic memories."""
    from main import LifeOS

    event = {
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Team standup",
            "start_time": "2026-02-16T09:00:00Z",
            "end_time": "2026-02-16T09:30:00Z",
            "location": "Conference Room A",
            "attendees": ["alice@example.com", "bob@example.com"],
        },
        "metadata": {
            "domain": "work",
        },
    }

    config = {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }

    lifeos = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )

    await lifeos._create_episode(event)

    with db.get_connection("user_model") as conn:
        episodes = conn.execute("SELECT * FROM episodes WHERE event_id = ?", (event["id"],)).fetchall()
        assert len(episodes) == 1

        episode = dict(episodes[0])
        # Calendar events with attendees are classified as meetings so the
        # routine detector can distinguish participation from solo planning.
        assert episode["interaction_type"] == "meeting_scheduled"
        assert episode["location"] == "Conference Room A"

        # Verify summary contains event details
        assert "Meeting:" in episode["content_summary"]
        assert "Team standup" in episode["content_summary"]


@pytest.mark.asyncio
async def test_episodic_memory_task_completed(db, event_store, user_model_store, event_bus):
    """Test that task completion creates episodic memories."""
    from main import LifeOS

    event = {
        "id": str(uuid.uuid4()),
        "type": EventType.TASK_COMPLETED.value,
        "source": "lifeos",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Fix authentication bug",
            "task_id": "task-123",
        },
        "metadata": {
            "domain": "work",
        },
    }

    config = {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }

    lifeos = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )

    await lifeos._create_episode(event)

    with db.get_connection("user_model") as conn:
        episodes = conn.execute("SELECT * FROM episodes WHERE event_id = ?", (event["id"],)).fetchall()
        assert len(episodes) == 1

        episode = dict(episodes[0])
        # Granular type distinguishes task creation (work-planning) from
        # task completion (execution) for routine pattern detection.
        assert episode["interaction_type"] == "task_completed"

        # Verify summary shows task completion
        assert "Task completed:" in episode["content_summary"]
        assert "Fix authentication bug" in episode["content_summary"]


@pytest.mark.asyncio
async def test_episodic_memory_financial_transaction(db, event_store, user_model_store, event_bus):
    """Test that financial transactions create episodic memories."""
    from main import LifeOS

    event = {
        "id": str(uuid.uuid4()),
        "type": EventType.TRANSACTION_NEW.value,
        "source": "plaid",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "amount": 45.23,
            "merchant": "Whole Foods",
            "category": "groceries",
            "account_id": "checking-123",
        },
        "metadata": {
            "domain": "personal",
        },
    }

    config = {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }

    lifeos = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )

    await lifeos._create_episode(event)

    with db.get_connection("user_model") as conn:
        episodes = conn.execute("SELECT * FROM episodes WHERE event_id = ?", (event["id"],)).fetchall()
        assert len(episodes) == 1

        episode = dict(episodes[0])
        # Positive amounts are classified as "income" so the routine detector
        # can distinguish spending patterns from income events.
        # The test amount is 45.23 (positive), so the expected type is "income".
        assert episode["interaction_type"] == "income"

        # Verify summary shows transaction details
        assert "Transaction:" in episode["content_summary"]
        assert "$45.23" in episode["content_summary"]
        assert "Whole Foods" in episode["content_summary"]


@pytest.mark.asyncio
async def test_episodic_memory_skips_system_events(db, event_store, user_model_store, event_bus):
    """Test that system events do NOT create episodic memories."""
    from main import LifeOS

    # System events that should be skipped
    system_events = [
        {
            "id": str(uuid.uuid4()),
            "type": EventType.CONNECTOR_SYNC_COMPLETE.value,
            "source": "gmail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "low",
            "payload": {},
            "metadata": {},
        },
        {
            "id": str(uuid.uuid4()),
            "type": EventType.RULE_TRIGGERED.value,
            "source": "lifeos",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "low",
            "payload": {},
            "metadata": {},
        },
        {
            "id": str(uuid.uuid4()),
            "type": "usermodel.prediction.generated",
            "source": "lifeos",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "low",
            "payload": {},
            "metadata": {},
        },
    ]

    config = {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }

    lifeos = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )

    for event in system_events:
        await lifeos._create_episode(event)

    # Verify NO episodes were created
    with db.get_connection("user_model") as conn:
        for event in system_events:
            episodes = conn.execute("SELECT * FROM episodes WHERE event_id = ?", (event["id"],)).fetchall()
            assert len(episodes) == 0, f"System event {event['type']} should not create an episode"


@pytest.mark.asyncio
async def test_episodic_memory_with_mood_context(db, event_store, user_model_store, event_bus):
    """Test that episodes capture mood context when available."""
    from main import LifeOS

    # Store mood signals in the format the MoodExtractor writes:
    # a list of typed signal dicts under "recent_signals".  The
    # compute_current_mood() method reads this key to build a MoodState.
    # MoodSignal requires: signal_type, value, delta_from_baseline, weight, source, timestamp.
    # These are the fields the MoodExtractor stores when writing to the profile.
    now_iso = datetime.now(timezone.utc).isoformat()
    mood_profile = {
        "recent_signals": [
            {
                "signal_type": "energy_proxy",
                "value": 0.7,
                "delta_from_baseline": 0.2,
                "weight": 1.0,
                "source": "calendar",
                "timestamp": now_iso,
            },
            {
                "signal_type": "positive_language",
                "value": 0.6,
                "delta_from_baseline": 0.1,
                "weight": 0.8,
                "source": "email",
                "timestamp": now_iso,
            },
        ]
    }
    user_model_store.update_signal_profile("mood_signals", mood_profile)

    event = {
        "id": str(uuid.uuid4()),
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "gmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "from_address": "urgent@company.com",
            "to_addresses": ["me@example.com"],
            "subject": "URGENT: Production down",
            "body_plain": "The production server is offline.",
        },
        "metadata": {
            "domain": "work",
        },
    }

    config = {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }

    lifeos = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )

    await lifeos._create_episode(event)

    with db.get_connection("user_model") as conn:
        episodes = conn.execute("SELECT * FROM episodes WHERE event_id = ?", (event["id"],)).fetchall()
        assert len(episodes) == 1

        episode = dict(episodes[0])

        # Verify mood context was captured from the stored signals.
        # The mood signals we stored should result in a non-null inferred_mood.
        # We check that mood fields are present rather than exact values, because
        # compute_current_mood() derives them from signal types (energy_proxy,
        # positive_language, etc.) rather than storing them verbatim.
        assert episode["inferred_mood"] is not None
        inferred_mood = json.loads(episode["inferred_mood"])
        # MoodState always has these keys (defaulted to 0.5 / 0.3)
        assert "energy_level" in inferred_mood
        assert "stress_level" in inferred_mood
        assert "emotional_valence" in inferred_mood
        # episode.energy_level is populated from the mood state
        assert episode["energy_level"] is not None


@pytest.mark.asyncio
async def test_episodic_memory_message_events(db, event_store, user_model_store, event_bus):
    """Test that message events (SMS, Slack, iMessage) create episodic memories."""
    from main import LifeOS

    message_events = [
        {
            "id": str(uuid.uuid4()),
            "type": EventType.MESSAGE_RECEIVED.value,
            "source": "slack",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {
                "from_address": "@alice",
                "channel": "slack",
                "body_plain": "Can you review the PR?",
                "snippet": "Can you review the PR?",
            },
            "metadata": {"domain": "work"},
        },
        {
            "id": str(uuid.uuid4()),
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {
                "to_addresses": ["+1234567890"],
                "channel": "imessage",
                "body_plain": "Running 10 minutes late",
            },
            "metadata": {"domain": "personal"},
        },
    ]

    config = {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }

    lifeos = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )

    for event in message_events:
        await lifeos._create_episode(event)

    with db.get_connection("user_model") as conn:
        episodes = conn.execute("SELECT * FROM episodes ORDER BY timestamp").fetchall()
        assert len(episodes) == 2

        # Verify first message (received)
        episode1 = dict(episodes[0])
        # Granular types distinguish inbound ("message_received") from
        # outbound ("message_sent") for routine and cadence detection.
        assert episode1["interaction_type"] == "message_received"
        assert "Message from @alice" in episode1["content_summary"]

        # Verify second message (sent)
        episode2 = dict(episodes[1])
        assert episode2["interaction_type"] == "message_sent"
        assert "Message to" in episode2["content_summary"]


@pytest.mark.asyncio
async def test_episodic_memory_location_events(db, event_store, user_model_store, event_bus):
    """Test that location change events create episodic memories."""
    from main import LifeOS

    event = {
        "id": str(uuid.uuid4()),
        "type": EventType.LOCATION_ARRIVED.value,
        "source": "ios",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "low",
        "payload": {
            "location": "Office",
            "latitude": 37.7749,
            "longitude": -122.4194,
        },
        "metadata": {
            "domain": "work",
        },
    }

    config = {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }

    lifeos = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )

    await lifeos._create_episode(event)

    with db.get_connection("user_model") as conn:
        episodes = conn.execute("SELECT * FROM episodes WHERE event_id = ?", (event["id"],)).fetchall()
        assert len(episodes) == 1

        episode = dict(episodes[0])
        # Granular type distinguishes arrivals ("location_arrived") from
        # departures ("location_departed") for spatial routine detection.
        assert episode["interaction_type"] == "location_arrived"
        assert episode["location"] == "Office"
        assert "Location arrived at Office" in episode["content_summary"]


@pytest.mark.asyncio
async def test_episode_summary_truncation(db, event_store, user_model_store, event_bus):
    """Test that episode summaries are truncated to 200 characters."""
    from main import LifeOS

    # Create an event with a very long subject line
    long_subject = "A" * 250  # 250 characters

    event = {
        "id": str(uuid.uuid4()),
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "gmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "from_address": "test@example.com",
            "to_addresses": ["me@example.com"],
            "subject": long_subject,
            "body_plain": "Body text",
        },
        "metadata": {
            "domain": "work",
        },
    }

    config = {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }

    lifeos = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )

    await lifeos._create_episode(event)

    with db.get_connection("user_model") as conn:
        episodes = conn.execute("SELECT * FROM episodes WHERE event_id = ?", (event["id"],)).fetchall()
        assert len(episodes) == 1

        episode = dict(episodes[0])
        # Summary should be truncated to 200 chars
        assert len(episode["content_summary"]) <= 200


@pytest.mark.asyncio
async def test_episode_content_full_strips_large_body(db, event_store, user_model_store, event_bus):
    """Episode content_full must strip large body fields to prevent DB bloat.

    Root cause (iteration 43): storing the raw event payload caused
    user_model.db to reach 7.4 GB because email bodies (often 50–200 KB of
    HTML) were stored verbatim in every episode row.  The fix strips fields
    named ``body``, ``html_body``, ``raw``, etc., keeping only metadata and
    a short snippet.  The full payload is always recoverable from events.db
    via event_id.

    This test verifies:
    - Large ``body`` field is stripped to ≤ 500 chars (with "…" suffix)
    - Metadata fields (subject, from_address) are preserved intact
    - ``html_body`` is stripped similarly
    - Total content_full JSON stays under 4 000 chars even for huge payloads
    """
    from main import LifeOS

    large_body = "A" * 10_000  # 10 KB body — typical HTML email
    large_html = "<html>" + "B" * 10_000 + "</html>"

    event = {
        "id": str(uuid.uuid4()),
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "gmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "from_address": "sender@example.com",
            "to_addresses": ["me@example.com"],
            "subject": "Important newsletter",
            "message_id": "test-large-body",
            "body": large_body,
            "html_body": large_html,
        },
        "metadata": {"domain": "personal"},
    }

    config = {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }

    lifeos = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )

    await lifeos._create_episode(event)

    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
        assert len(episodes) == 1, "Episode should be created"

        episode = dict(episodes[0])
        content_full_raw = episode["content_full"]

        # Total JSON must be under the 4 000-char hard cap
        assert len(content_full_raw) <= 4_000, (
            f"content_full exceeds 4000 chars: {len(content_full_raw)}"
        )

        content_full = json.loads(content_full_raw)

        # Metadata fields must survive compaction
        assert content_full["subject"] == "Important newsletter"
        assert content_full["from_address"] == "sender@example.com"

        # Large body fields must be stripped to a snippet (≤ 500 chars + "…")
        assert "body" in content_full, "body field should still exist as a snippet"
        assert len(content_full["body"]) <= 502, (  # 500 + len("…") ≤ 502
            f"body not stripped: {len(content_full['body'])} chars"
        )
        assert content_full["body"].endswith("…"), "Stripped body must end with ellipsis"

        assert "html_body" in content_full, "html_body field should still exist as a snippet"
        assert len(content_full["html_body"]) <= 502
        assert content_full["html_body"].endswith("…")
