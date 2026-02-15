"""
Test suite for ContextAssembler.

ContextAssembler is responsible for building optimal context windows for AI queries
by aggregating data from multiple sources (preferences, calendar, tasks, semantic
facts, mood signals, communication templates, relationships). This module tests all
public methods and their integration with DatabaseManager and UserModelStore.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.ai_engine.context import ContextAssembler


def create_test_event(
    event_type: str = "test.event",
    source: str = "test-source",
    priority: str = "normal",
    payload: dict = None,
    timestamp: datetime = None,
) -> dict:
    """Helper to create a well-formed test event."""
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": source,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "priority": priority,
        "payload": payload or {},
        "metadata": {},
    }


def insert_communication_template(
    conn,
    contact_id: str = None,
    channel: str = "email",
    context: str = "general",
    formality: float = 0.5,
    greeting: str = None,
    closing: str = None,
    typical_length: int = 100,
    uses_emoji: int = 0,
    common_phrases: list = None,
    samples_analyzed: int = 10,
):
    """Helper to insert a communication template with all required fields."""
    conn.execute(
        """INSERT INTO communication_templates (
            id, context, contact_id, channel, formality, greeting, closing,
            typical_length, uses_emoji, common_phrases, samples_analyzed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            context,
            contact_id,
            channel,
            formality,
            greeting,
            closing,
            typical_length,
            uses_emoji,
            json.dumps(common_phrases or []),
            samples_analyzed,
        ),
    )


class TestContextAssemblerInit:
    """Test ContextAssembler initialization."""

    def test_init_stores_dependencies(self, db, user_model_store):
        """ContextAssembler should store DatabaseManager and UserModelStore references."""
        assembler = ContextAssembler(db, user_model_store)
        assert assembler.db is db
        assert assembler.ums is user_model_store


class TestAssembleBriefingContext:
    """Test briefing context assembly with all data sources."""

    def test_minimal_briefing_with_no_data(self, db, user_model_store):
        """Should return valid briefing context even with empty database."""
        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # All briefings must include timestamp
        assert "Current time:" in context
        # Should include all section delimiters
        assert "---" in context
        # Should have fallback messages for empty data
        assert "not yet configured" in context or "none" in context.lower()

    def test_briefing_includes_preferences(self, db, user_model_store):
        """Should include user preferences in briefing context."""
        # Seed preferences
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES (?, ?)",
                ("preferred_name", "Alex"),
            )
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES (?, ?)",
                ("verbosity", "concise"),
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        assert "User preferences:" in context
        assert "Alex" in context
        assert "concise" in context
        # Preferences should be formatted as JSON
        assert "preferred_name" in context

    def test_briefing_includes_current_time(self, db, user_model_store):
        """Should include formatted timestamp in briefing."""
        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        assert "Current time:" in context
        # Should include day of week
        now = datetime.now(timezone.utc)
        day_name = now.strftime("%A")
        assert day_name in context

    def test_briefing_includes_pending_tasks(self, db, user_model_store):
        """Should include pending tasks sorted by priority and due date."""
        # Seed tasks with different priorities
        with db.get_connection("state") as conn:
            # Critical task (should appear first)
            conn.execute(
                """INSERT INTO tasks (id, title, priority, status, due_date, domain)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("t1", "Deploy hotfix", "critical", "pending", "2026-02-16", "work"),
            )
            # Low priority task (should appear last)
            conn.execute(
                """INSERT INTO tasks (id, title, priority, status, due_date, domain)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("t2", "Clean garage", "low", "pending", "2026-02-20", "personal"),
            )
            # High priority task (should be second)
            conn.execute(
                """INSERT INTO tasks (id, title, priority, status, due_date, domain)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("t3", "Finish proposal", "high", "pending", "2026-02-17", "work"),
            )
            # Completed task (should NOT appear)
            conn.execute(
                """INSERT INTO tasks (id, title, priority, status, due_date, domain)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("t4", "Old task", "critical", "completed", "2026-02-15", "work"),
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        assert "Pending tasks:" in context
        assert "Deploy hotfix" in context
        assert "Finish proposal" in context
        assert "Clean garage" in context
        # Completed task should not appear
        assert "Old task" not in context
        # Should show priority tags
        assert "[critical]" in context
        assert "[high]" in context
        assert "[low]" in context

    def test_briefing_task_priority_ordering(self, db, user_model_store):
        """Tasks should be ordered: critical > high > normal > low, then by due date."""
        with db.get_connection("state") as conn:
            # Seed tasks with same due date but different priorities
            for i, priority in enumerate(["low", "critical", "normal", "high"]):
                conn.execute(
                    """INSERT INTO tasks (id, title, priority, status, due_date, domain)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (f"t{i}", f"Task {priority}", priority, "pending", "2026-02-20", "test"),
                )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # Extract task section
        task_section = context[context.find("Pending tasks:"):]
        # Critical should appear before high
        assert task_section.find("Task critical") < task_section.find("Task high")
        # High should appear before normal
        assert task_section.find("Task high") < task_section.find("Task normal")
        # Normal should appear before low
        assert task_section.find("Task normal") < task_section.find("Task low")

    def test_briefing_task_limit_20(self, db, user_model_store):
        """Should limit tasks to 20 to respect token budget."""
        with db.get_connection("state") as conn:
            # Seed 25 tasks
            for i in range(25):
                conn.execute(
                    """INSERT INTO tasks (id, title, priority, status, due_date, domain)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (f"t{i}", f"Task {i}", "normal", "pending", "2026-02-20", "test"),
                )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # Count task lines (each starts with "- [")
        task_lines = [line for line in context.split("\n") if line.strip().startswith("- [")]
        assert len(task_lines) <= 20

    def test_briefing_includes_unread_count(self, db, user_model_store, event_store):
        """Should count recent messages from last 12 hours."""
        now = datetime.now(timezone.utc)

        # Recent emails (within 12 hours)
        for i in range(5):
            event_store.store_event(
                create_test_event(
                    event_type="email.received",
                    source="gmail",
                    priority="normal",
                    payload={"subject": f"Email {i}"},
                    timestamp=now,
                )
            )

        # Recent messages (within 12 hours)
        for i in range(3):
            event_store.store_event(
                create_test_event(
                    event_type="message.received",
                    source="signal",
                    priority="normal",
                    payload={"body": f"Message {i}"},
                    timestamp=now,
                )
            )

        # Old email (should not be counted)
        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="gmail",
                priority="normal",
                payload={"subject": "Old email"},
                timestamp=now - timedelta(hours=24),
            )
        )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        assert "Messages in last 12 hours: 8" in context

    def test_briefing_includes_semantic_facts(self, db, user_model_store):
        """Should include high-confidence semantic facts."""
        # Seed semantic facts with different confidence levels
        user_model_store.update_semantic_fact(
            key="preferred_language",
            category="preference",
            value="English",
            confidence=0.9,
        )
        user_model_store.update_semantic_fact(
            key="works_at",
            category="fact",
            value="Acme Corp",
            confidence=0.8,
        )
        # Low confidence fact (should not appear - below 0.6 threshold)
        user_model_store.update_semantic_fact(
            key="hobby",
            category="preference",
            value="painting",
            confidence=0.4,
        )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        assert "Known facts about user:" in context
        # Values are stored as JSON, so they appear with quotes
        assert 'preferred_language: "English"' in context or "preferred_language: English" in context
        assert 'works_at: "Acme Corp"' in context or "works_at: Acme Corp" in context
        # Low confidence fact should be filtered out
        assert "painting" not in context

    def test_briefing_semantic_facts_limit_20(self, db, user_model_store):
        """Should limit semantic facts to 20."""
        # Seed 25 high-confidence facts
        for i in range(25):
            user_model_store.update_semantic_fact(
                key=f"fact_{i}",
                category="test",
                value=f"value_{i}",
                confidence=0.9,
            )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # Count fact lines (each starts with "- ")
        facts_section = context[context.find("Known facts"):]
        fact_lines = [line for line in facts_section.split("\n") if line.strip().startswith("- fact_")]
        assert len(fact_lines) <= 20

    def test_briefing_includes_mood_signals(self, db, user_model_store):
        """Should include recent mood context."""
        # Create mood signal profile with recent signals
        user_model_store.update_signal_profile(
            profile_type="mood_signals",
            data={
                "recent_signals": [
                    {"timestamp": "2026-02-15T08:00:00Z", "valence": 0.7, "arousal": 0.5},
                    {"timestamp": "2026-02-15T12:00:00Z", "valence": 0.6, "arousal": 0.4},
                    {"timestamp": "2026-02-15T18:00:00Z", "valence": 0.8, "arousal": 0.6},
                ]
            },
        )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        assert "User mood context:" in context
        # Should only include last 3 signals (all of them in this case)
        assert "valence" in context
        assert "0.8" in context  # Most recent valence

    def test_briefing_mood_signals_limit_last_3(self, db, user_model_store):
        """Should only include last 3 mood signals."""
        # Create profile with 10 signals
        signals = [
            {"timestamp": f"2026-02-15T{i:02d}:00:00Z", "valence": 0.5 + i * 0.05}
            for i in range(10)
        ]
        user_model_store.update_signal_profile(
            profile_type="mood_signals",
            data={"recent_signals": signals},
        )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # Parse the mood context section
        mood_section = context[context.find("User mood context:"):]
        mood_data = json.loads(mood_section.split("\n")[0].split("User mood context: ")[1])
        assert len(mood_data) == 3
        # Should be the last 3 signals (indices 7, 8, 9)
        assert mood_data[0]["timestamp"] == "2026-02-15T07:00:00Z"

    def test_briefing_with_no_tasks(self, db, user_model_store):
        """Should handle empty task list gracefully."""
        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        assert "Pending tasks: none" in context

    def test_briefing_section_separation(self, db, user_model_store):
        """All sections should be separated by '---' delimiters."""
        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # Count section separators
        separator_count = context.count("\n\n---\n\n")
        # Should have multiple sections separated
        assert separator_count >= 3


class TestAssembleDraftContext:
    """Test draft reply context assembly."""

    def test_draft_context_minimal(self, db, user_model_store):
        """Should return valid draft context even with no templates or profiles."""
        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="alice@example.com",
            channel="email",
            incoming_message="Hey, how are you?",
        )

        # Should always include the incoming message
        assert "Incoming message to reply to:" in context
        assert "Hey, how are you?" in context

    def test_draft_includes_communication_template(self, db, user_model_store):
        """Should include contact-specific communication template."""
        # Seed a communication template
        with db.get_connection("user_model") as conn:
            insert_communication_template(
                conn,
                contact_id="bob@work.com",
                channel="email",
                context="work",
                formality=0.8,
                greeting="Hi Bob",
                closing="Best regards",
                typical_length=120,
                uses_emoji=0,
                common_phrases=["Let me know", "Thanks for", "I appreciate"],
                samples_analyzed=50,
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="bob@work.com",
            channel="email",
            incoming_message="Can you review this?",
        )

        assert "Communication style for this context:" in context
        assert "Formality: 0.8" in context
        assert "Greeting: Hi Bob" in context
        assert "Closing: Best regards" in context
        # Typical length is stored as float, so check for both formats
        assert "Typical length: 120" in context
        assert "words" in context
        assert "Uses emoji: no" in context
        assert "Let me know" in context
        assert "Thanks for" in context

    def test_draft_template_contact_priority_over_channel(self, db, user_model_store):
        """Contact-specific template should take priority over channel default."""
        with db.get_connection("user_model") as conn:
            # Channel-wide default
            insert_communication_template(
                conn,
                contact_id=None,
                channel="email",
                context="casual",
                formality=0.5,
                greeting="Hey",
                typical_length=80,
                uses_emoji=1,
                common_phrases=["Cool"],
                samples_analyzed=100,
            )
            # Contact-specific template (more samples)
            insert_communication_template(
                conn,
                contact_id="carol@friend.com",
                channel="email",
                context="friend",
                formality=0.3,
                greeting="Yo",
                typical_length=50,
                uses_emoji=1,
                common_phrases=["lol", "omg"],
                samples_analyzed=200,
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="carol@friend.com",
            channel="email",
            incoming_message="Wanna hang?",
        )

        # Should use contact-specific template (more samples_analyzed)
        assert "Formality: 0.3" in context
        assert "Greeting: Yo" in context
        assert "lol" in context
        # Should NOT use channel default
        assert "Cool" not in context

    def test_draft_includes_relationship_context(self, db, user_model_store):
        """Should include interaction count from relationship profile."""
        # Create relationship profile
        user_model_store.update_signal_profile(
            profile_type="relationships",
            data={
                "contacts": {
                    "dave@client.com": {"interaction_count": 150, "last_contact": "2026-02-14"}
                }
            },
        )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="dave@client.com",
            channel="email",
            incoming_message="Project update?",
        )

        assert "Relationship: 150 total interactions" in context

    def test_draft_includes_linguistic_profile(self, db, user_model_store):
        """Should include general linguistic formality as fallback."""
        # Create linguistic profile
        user_model_store.update_signal_profile(
            profile_type="linguistic",
            data={"averages": {"formality": 0.65, "complexity": 0.7}},
        )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="unknown@example.com",
            channel="email",
            incoming_message="Hello there",
        )

        assert "User's general style: formality=0.7" in context or "formality=0.6" in context

    def test_draft_common_phrases_limit_5(self, db, user_model_store):
        """Should limit common phrases to 5."""
        with db.get_connection("user_model") as conn:
            # Template with 10 common phrases
            phrases = [f"phrase_{i}" for i in range(10)]
            insert_communication_template(
                conn,
                contact_id="test@example.com",
                channel="email",
                context="test",
                formality=0.5,
                common_phrases=phrases,
                samples_analyzed=50,
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="test@example.com",
            channel="email",
            incoming_message="Test",
        )

        # Count how many phrases appear
        phrase_count = sum(1 for i in range(10) if f"phrase_{i}" in context)
        assert phrase_count <= 5

    def test_draft_handles_null_template_fields(self, db, user_model_store):
        """Should handle NULL greeting/closing/common_phrases gracefully."""
        with db.get_connection("user_model") as conn:
            insert_communication_template(
                conn,
                contact_id="sparse@example.com",
                channel="sms",
                context="minimal",
                formality=0.5,
                greeting=None,
                closing=None,
                typical_length=30,
                uses_emoji=0,
                common_phrases=None,
                samples_analyzed=10,
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="sparse@example.com",
            channel="sms",
            incoming_message="Quick question",
        )

        # Should show "none" for null fields
        assert "Greeting: none" in context
        assert "Closing: none" in context
        # Should not crash on null common_phrases
        assert "Communication style" in context

    def test_draft_emoji_flag(self, db, user_model_store):
        """Should correctly represent emoji usage."""
        with db.get_connection("user_model") as conn:
            # Template with emoji usage
            insert_communication_template(
                conn,
                contact_id="emoji@example.com",
                channel="sms",
                context="casual",
                formality=0.3,
                typical_length=40,
                uses_emoji=1,
                common_phrases=[],
                samples_analyzed=20,
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="emoji@example.com",
            channel="sms",
            incoming_message="😊 Hey!",
        )

        assert "Uses emoji: yes" in context


class TestAssembleSearchContext:
    """Test life-search context assembly."""

    def test_search_context_includes_query(self, db, user_model_store):
        """Should include the search query in context."""
        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_search_context("meetings with Alice last week")

        assert "User is searching" in context
        assert "meetings with Alice last week" in context

    def test_search_context_minimal(self, db, user_model_store):
        """Search context should be minimal (hook for future enrichment)."""
        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_search_context("test query")

        # Should be a simple one-line context for now
        assert context.count("\n") == 0


class TestPrivateHelpers:
    """Test private helper methods."""

    def test_get_preference_context_empty(self, db, user_model_store):
        """Should return fallback when no preferences are set."""
        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_preference_context()

        assert "User preferences: not yet configured" in result

    def test_get_preference_context_with_data(self, db, user_model_store):
        """Should format preferences as JSON."""
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES (?, ?)",
                ("theme", "dark"),
            )
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES (?, ?)",
                ("timezone", "America/New_York"),
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_preference_context()

        assert "User preferences:" in result
        # Should be valid JSON
        prefs = json.loads(result.split("User preferences: ")[1])
        assert prefs["theme"] == "dark"
        assert prefs["timezone"] == "America/New_York"

    def test_get_calendar_context_placeholder(self, db, user_model_store):
        """Calendar context is a placeholder for now."""
        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_calendar_context()

        assert "Calendar:" in result
        assert "CalDAV" in result

    def test_get_task_context_empty(self, db, user_model_store):
        """Should return 'none' when no pending tasks exist."""
        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_task_context()

        assert "Pending tasks: none" in result

    def test_get_task_context_with_data(self, db, user_model_store):
        """Should format tasks with priority and due date."""
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO tasks (id, title, priority, status, due_date, domain)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("t1", "Important task", "high", "pending", "2026-02-18", "work"),
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_task_context()

        assert "Pending tasks:" in result
        assert "Important task" in result
        assert "[high]" in result
        assert "2026-02-18" in result

    def test_get_task_context_no_due_date(self, db, user_model_store):
        """Should handle tasks without due dates."""
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO tasks (id, title, priority, status, due_date, domain)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("t1", "Someday task", "normal", "pending", None, "personal"),
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_task_context()

        assert "Someday task" in result
        assert "no date" in result

    def test_get_unread_context_zero(self, db, user_model_store):
        """Should handle zero unread messages."""
        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        assert "Messages in last 12 hours: 0" in result

    def test_get_unread_context_counts_both_types(self, db, user_model_store, event_store):
        """Should count both email.received and message.received."""
        now = datetime.now(timezone.utc)

        # Add emails
        for i in range(3):
            event_store.store_event(
                create_test_event(
                    event_type="email.received",
                    source="gmail",
                    priority="normal",
                    payload={"subject": f"Email {i}"},
                    timestamp=now,
                )
            )

        # Add messages
        for i in range(2):
            event_store.store_event(
                create_test_event(
                    event_type="message.received",
                    source="signal",
                    priority="normal",
                    payload={"body": f"Message {i}"},
                    timestamp=now,
                )
            )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        assert "Messages in last 12 hours: 5" in result

    def test_get_unread_context_excludes_old_messages(self, db, user_model_store, event_store):
        """Should only count messages from last 12 hours."""
        now = datetime.now(timezone.utc)

        # Recent message (counts)
        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="gmail",
                priority="normal",
                payload={"subject": "Recent"},
                timestamp=now - timedelta(hours=6),
            )
        )

        # Old message (should not count - 25 hours old to be safe)
        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="gmail",
                priority="normal",
                payload={"subject": "Old"},
                timestamp=now - timedelta(hours=25),
            )
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        # Should only count the recent message
        assert "Messages in last 12 hours:" in result
        # Extract the count and verify it's 1
        count = int(result.split("Messages in last 12 hours: ")[1])
        assert count == 1


class TestIntegration:
    """Integration tests combining multiple data sources."""

    def test_full_briefing_integration(self, db, user_model_store, event_store):
        """Test full briefing context with all data sources populated."""
        # Seed preferences
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES (?, ?)",
                ("preferred_name", "Jordan"),
            )
            conn.commit()

        # Seed tasks
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO tasks (id, title, priority, status, due_date, domain)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("t1", "Review PR", "high", "pending", "2026-02-16", "work"),
            )
            conn.commit()

        # Seed events
        now = datetime.now(timezone.utc)
        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="gmail",
                priority="normal",
                payload={"subject": "Test"},
                timestamp=now,
            )
        )

        # Seed semantic facts
        user_model_store.update_semantic_fact(
            key="role",
            category="fact",
            value="software_engineer",
            confidence=0.9,
        )

        # Seed mood signals
        user_model_store.update_signal_profile(
            profile_type="mood_signals",
            data={"recent_signals": [{"valence": 0.7, "arousal": 0.5}]},
        )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # Verify all sections present
        assert "Jordan" in context  # Preferences
        assert "Current time:" in context  # Timestamp
        assert "Review PR" in context  # Tasks
        assert "Messages in last 12 hours: 1" in context  # Unread count
        # Semantic facts values are stored as JSON
        assert 'role: "software_engineer"' in context or "role: software_engineer" in context
        assert "valence" in context  # Mood signals

    def test_full_draft_integration(self, db, user_model_store):
        """Test full draft context with all layers populated."""
        # Seed communication template
        with db.get_connection("user_model") as conn:
            insert_communication_template(
                conn,
                contact_id="boss@company.com",
                channel="email",
                context="formal_work",
                formality=0.9,
                greeting="Good morning",
                closing="Regards",
                typical_length=150,
                uses_emoji=0,
                common_phrases=["Please advise", "Thank you"],
                samples_analyzed=100,
            )
            conn.commit()

        # Seed relationship profile
        user_model_store.update_signal_profile(
            profile_type="relationships",
            data={"contacts": {"boss@company.com": {"interaction_count": 300}}},
        )

        # Seed linguistic profile
        user_model_store.update_signal_profile(
            profile_type="linguistic",
            data={"averages": {"formality": 0.7}},
        )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="boss@company.com",
            channel="email",
            incoming_message="Can you send the report?",
        )

        # Verify all layers present
        assert "Formality: 0.9" in context  # Template
        assert "Good morning" in context  # Template greeting
        assert "300 total interactions" in context  # Relationship
        assert "formality=" in context  # Linguistic fallback
        assert "Can you send the report?" in context  # Incoming message
