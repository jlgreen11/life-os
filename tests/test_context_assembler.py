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
        """Should include high-confidence semantic facts with new categorized format."""
        # Seed semantic facts with different confidence levels.
        # The new _get_semantic_facts_context() groups facts by category and
        # formats them with human-readable labels and confidence scores.
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
        # Low confidence fact (should not appear — below 0.6 threshold)
        user_model_store.update_semantic_fact(
            key="hobby",
            category="preference",
            value="painting",
            confidence=0.4,
        )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # New header from _get_semantic_facts_context()
        assert "Layer 2 Semantic Memory" in context
        # Values appear with human-readable label and confidence
        assert "English" in context
        assert "Acme" in context
        # Low confidence fact should be filtered out
        assert "painting" not in context

    def test_briefing_semantic_facts_limit_20(self, db, user_model_store):
        """Should limit semantic facts to at most 18 (values≤5 + behavioral≤5 + prefs≤8)."""
        # Seed 25 high-confidence preference facts.  The per-group caps are
        # values ≤ 5, behavioral ≤ 5, preferences ≤ 8 — so at most 18 total.
        for i in range(25):
            user_model_store.update_semantic_fact(
                key=f"fact_{i}",
                category="test",
                value=f"value_{i}",
                confidence=0.9,
            )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # Count fact lines in the semantic section (each starts with "  - ")
        facts_section = context[context.find("Layer 2 Semantic Memory"):]
        fact_lines = [
            line for line in facts_section.split("\n")
            if line.strip().startswith("- fact_")
        ]
        assert len(fact_lines) <= 18

    def test_briefing_includes_mood_signals(self, db, user_model_store):
        """Should include computed mood context from mood_history (not raw signals).

        Updated for PR #251 (iteration 251): _get_mood_context() now queries the
        mood_history table instead of reading raw recent_signals JSON, so the
        LLM receives labelled dimension values (energy, stress, valence, trend)
        rather than uninterpreted signal dicts.
        """
        from datetime import datetime, timezone

        # Insert a mood_history snapshot directly (as the pipeline does every 15 min).
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO mood_history
                   (timestamp, energy_level, stress_level, emotional_valence,
                    social_battery, cognitive_load, confidence, trend)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    0.80,  # high energy
                    0.25,  # low stress
                    0.70,  # positive valence
                    0.60,  # social battery
                    0.30,  # cognitive load
                    0.80,  # confidence
                    "stable",
                ),
            )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        assert "User mood context:" in context
        # New format: labelled dimension values, not raw JSON
        assert "energy_level=0.80" in context
        assert "stress_level=0.25" in context
        assert "emotional_valence=0.70" in context
        assert "trend=stable" in context
        # Old raw format should NOT appear
        assert '"signal_type"' not in context

    def test_briefing_mood_signals_limit_last_3(self, db, user_model_store):
        """Mood context uses the single most recent mood_history snapshot.

        Updated for PR #251 (iteration 251): the old test checked that only the
        last 3 raw signal dicts were included in a JSON array.  The new
        implementation queries mood_history for the most recent row, so the
        test now inserts two mood_history rows (old and new) and verifies that
        only the newer row's values appear in the briefing.
        """
        from datetime import datetime, timedelta, timezone

        older_ts = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        newer_ts = datetime.now(timezone.utc).isoformat()

        with db.get_connection("user_model") as conn:
            # Older snapshot: low energy
            conn.execute(
                """INSERT INTO mood_history
                   (timestamp, energy_level, stress_level, emotional_valence,
                    social_battery, cognitive_load, confidence, trend)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (older_ts, 0.20, 0.80, 0.30, 0.40, 0.70, 0.50, "declining"),
            )
            # Newer snapshot: high energy
            conn.execute(
                """INSERT INTO mood_history
                   (timestamp, energy_level, stress_level, emotional_valence,
                    social_battery, cognitive_load, confidence, trend)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (newer_ts, 0.90, 0.15, 0.85, 0.75, 0.20, 0.90, "improving"),
            )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # Must reflect the newer row
        assert "energy_level=0.90" in context
        assert "trend=improving" in context
        # Older row values must NOT appear
        assert "energy_level=0.20" not in context
        assert "trend=declining" not in context

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

    def test_search_context_enriched(self, db, user_model_store):
        """Search context now includes timestamp and preferences for disambiguation.

        The method was previously a one-liner stub.  It now assembles a
        multi-section context (intent, timestamp, preferences) separated by
        '---' delimiters — consistent with assemble_briefing_context().
        """
        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_search_context("test query")

        # Must span multiple sections (no longer a bare one-liner).
        assert "---" in context
        # The search intent section is always first.
        assert context.startswith("User is searching")
        # Current timestamp anchors relative time expressions.
        assert "Current time:" in context
        # User preferences are included for output calibration.
        assert "User preferences:" in context


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

    def test_get_calendar_context_empty(self, db, user_model_store):
        """Calendar context returns a 'none' message when no events are in the DB.

        The method was updated (PR #195) from a hardcoded placeholder to a real
        query of the events database; this test verifies the empty-DB code path.
        """
        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_calendar_context()

        assert "Upcoming calendar events" in result
        assert "none" in result

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
        # Extract the count from the first line (may have additional breakdown lines)
        first_line = result.split("\n")[0]
        count = int(first_line.split("Messages in last 12 hours: ")[1])
        assert count == 1

    def test_unread_context_shows_priority_contact_breakdown(
        self, db, user_model_store, event_store
    ):
        """Should show priority contact breakdown when relationships profile exists.

        A priority contact is one with outbound_count >= 3, meaning the user
        regularly writes back to them. Their messages should be called out
        explicitly in the unread context so the LLM can highlight them in
        the morning briefing.
        """
        now = datetime.now(timezone.utc)

        # Seed relationship profile with one priority contact (outbound >= 3)
        user_model_store.update_signal_profile(
            profile_type="relationships",
            data={
                "contacts": {
                    "alice@example.com": {
                        "interaction_count": 20,
                        "inbound_count": 12,
                        "outbound_count": 8,  # Priority contact: user writes back often
                    },
                    "noreply@automated.com": {
                        "interaction_count": 50,
                        "inbound_count": 50,
                        "outbound_count": 0,  # Not a priority contact: never replied to
                    },
                }
            },
        )

        # 3 emails from priority contact
        for i in range(3):
            event_store.store_event(
                create_test_event(
                    event_type="email.received",
                    source="gmail",
                    priority="normal",
                    payload={"subject": f"Hi {i}", "from_address": "alice@example.com"},
                    timestamp=now,
                )
            )

        # 5 emails from non-priority sender
        for i in range(5):
            event_store.store_event(
                create_test_event(
                    event_type="email.received",
                    source="gmail",
                    priority="normal",
                    payload={"subject": f"Promo {i}", "from_address": "noreply@automated.com"},
                    timestamp=now,
                )
            )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        # First line must always show the total
        assert "Messages in last 12 hours: 8" in result
        # Should call out the priority contact with message count
        assert "alice@example.com" in result
        assert "3 messages" in result
        # Should include "From priority contacts:" label
        assert "From priority contacts:" in result
        # Should show remaining "other senders" count
        assert "From other senders: 5" in result
        # Non-priority automated sender should NOT be called out as priority
        assert "noreply@automated.com" not in result.split("From priority contacts:")[1]

    def test_unread_context_no_breakdown_for_inbound_only_contacts(
        self, db, user_model_store, event_store
    ):
        """Contacts with outbound_count < 3 should not appear in priority breakdown.

        If the user has never written back to a contact (or has done so fewer
        than 3 times), that contact is treated as non-priority and suppressed
        from the breakdown to reduce noise.
        """
        now = datetime.now(timezone.utc)

        user_model_store.update_signal_profile(
            profile_type="relationships",
            data={
                "contacts": {
                    "sender@example.com": {
                        "interaction_count": 100,
                        "inbound_count": 100,
                        "outbound_count": 1,  # Below threshold: not a priority contact
                    }
                }
            },
        )

        for i in range(4):
            event_store.store_event(
                create_test_event(
                    event_type="email.received",
                    source="gmail",
                    priority="normal",
                    payload={"subject": f"Email {i}", "from_address": "sender@example.com"},
                    timestamp=now,
                )
            )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        # Total count must still be shown
        assert "Messages in last 12 hours: 4" in result
        # No priority breakdown should appear for below-threshold contacts
        assert "From priority contacts:" not in result

    def test_unread_context_multiple_priority_contacts_sorted_by_count(
        self, db, user_model_store, event_store
    ):
        """Multiple priority contacts should appear sorted by message count (desc)."""
        now = datetime.now(timezone.utc)

        user_model_store.update_signal_profile(
            profile_type="relationships",
            data={
                "contacts": {
                    "bob@work.com": {
                        "interaction_count": 30,
                        "inbound_count": 15,
                        "outbound_count": 15,
                    },
                    "alice@example.com": {
                        "interaction_count": 20,
                        "inbound_count": 10,
                        "outbound_count": 10,
                    },
                }
            },
        )

        # 1 message from alice (lower count)
        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="gmail",
                priority="normal",
                payload={"subject": "Hi", "from_address": "alice@example.com"},
                timestamp=now,
            )
        )

        # 3 messages from bob (higher count — should appear first)
        for i in range(3):
            event_store.store_event(
                create_test_event(
                    event_type="email.received",
                    source="gmail",
                    priority="normal",
                    payload={"subject": f"Update {i}", "from_address": "bob@work.com"},
                    timestamp=now,
                )
            )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        assert "Messages in last 12 hours: 4" in result
        assert "bob@work.com" in result
        assert "alice@example.com" in result
        # Bob (3 messages) must appear before Alice (1 message) in the output
        assert result.index("bob@work.com") < result.index("alice@example.com")
        # No "other senders" line when all messages are from priority contacts
        assert "From other senders:" not in result

    def test_unread_context_no_breakdown_without_relationships_profile(
        self, db, user_model_store, event_store
    ):
        """Should degrade gracefully to count-only when no relationships profile exists."""
        now = datetime.now(timezone.utc)

        for i in range(3):
            event_store.store_event(
                create_test_event(
                    event_type="email.received",
                    source="gmail",
                    priority="normal",
                    payload={"subject": f"Email {i}", "from_address": f"sender{i}@test.com"},
                    timestamp=now,
                )
            )

        # No relationships profile seeded
        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        assert "Messages in last 12 hours: 3" in result
        # Without a relationships profile, no priority breakdown
        assert "From priority contacts:" not in result

    def test_unread_context_single_message_grammar(
        self, db, user_model_store, event_store
    ):
        """'1 message' should use singular grammar, not '1 messages'."""
        now = datetime.now(timezone.utc)

        user_model_store.update_signal_profile(
            profile_type="relationships",
            data={
                "contacts": {
                    "alice@example.com": {
                        "interaction_count": 10,
                        "inbound_count": 5,
                        "outbound_count": 5,
                    }
                }
            },
        )

        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="gmail",
                priority="normal",
                payload={"subject": "Hello", "from_address": "alice@example.com"},
                timestamp=now,
            )
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        assert "1 message)" in result
        assert "1 messages)" not in result

    def test_unread_context_subjects_shown_for_priority_sender(
        self, db, user_model_store, event_store
    ):
        """Subject lines should appear alongside count for priority senders.

        When a priority contact sends emails with subject lines, those subjects
        are surfaced in the unread context so the LLM can reference them
        directly in the morning briefing.
        """
        now = datetime.now(timezone.utc)

        user_model_store.update_signal_profile(
            profile_type="relationships",
            data={
                "contacts": {
                    "alice@example.com": {
                        "interaction_count": 10,
                        "inbound_count": 4,
                        "outbound_count": 6,  # Priority: outbound >= 3
                    }
                }
            },
        )

        subjects = ["Project deadline moved", "Can we meet Friday?", "Quick question"]
        for subject in subjects:
            event_store.store_event(
                create_test_event(
                    event_type="email.received",
                    source="proton",
                    priority="normal",
                    payload={"from_address": "alice@example.com", "subject": subject},
                    timestamp=now,
                )
            )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        # All three subjects should appear in the output
        assert '"Project deadline moved"' in result
        assert '"Can we meet Friday?"' in result
        assert '"Quick question"' in result
        # Priority contact line should use the per-line dash format
        assert "- alice@example.com (3 messages):" in result

    def test_unread_context_subjects_capped_at_three(
        self, db, user_model_store, event_store
    ):
        """No more than 3 subjects should be shown per priority sender.

        Capping at 3 prevents the context window from being overwhelmed by
        a high-volume sender.  Only the 3 most recent subjects are shown.
        """
        now = datetime.now(timezone.utc)

        user_model_store.update_signal_profile(
            profile_type="relationships",
            data={
                "contacts": {
                    "alice@example.com": {
                        "interaction_count": 10,
                        "inbound_count": 4,
                        "outbound_count": 6,
                    }
                }
            },
        )

        # Store 5 emails — only the 3 most recent subjects should appear.
        subjects_newest_first = [
            "Email 5 — newest",
            "Email 4",
            "Email 3",
            "Email 2",
            "Email 1 — oldest",
        ]
        for i, subject in enumerate(reversed(subjects_newest_first)):
            event_store.store_event(
                create_test_event(
                    event_type="email.received",
                    source="proton",
                    priority="normal",
                    payload={"from_address": "alice@example.com", "subject": subject},
                    # Increment by 1 second so ORDER BY timestamp DESC is deterministic
                    timestamp=now + timedelta(seconds=i),
                )
            )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        assert "5 messages" in result
        # 3 most recent subjects should appear
        assert '"Email 5 — newest"' in result
        assert '"Email 4"' in result
        assert '"Email 3"' in result
        # Older subjects (4th and 5th most recent) should NOT appear
        assert '"Email 1 — oldest"' not in result
        assert '"Email 2"' not in result

    def test_unread_context_empty_subjects_excluded(
        self, db, user_model_store, event_store
    ):
        """Empty or whitespace-only subjects should not appear in output.

        Some email clients send messages with blank subjects.  These should
        not pollute the subject list with empty quoted strings.
        """
        now = datetime.now(timezone.utc)

        user_model_store.update_signal_profile(
            profile_type="relationships",
            data={
                "contacts": {
                    "alice@example.com": {
                        "interaction_count": 10,
                        "inbound_count": 4,
                        "outbound_count": 6,
                    }
                }
            },
        )

        # One email with a real subject, two with empty/whitespace subjects
        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="proton",
                payload={"from_address": "alice@example.com", "subject": "Real subject"},
                timestamp=now,
            )
        )
        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="proton",
                payload={"from_address": "alice@example.com", "subject": ""},
                timestamp=now + timedelta(seconds=1),
            )
        )
        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="proton",
                payload={"from_address": "alice@example.com", "subject": "   "},
                timestamp=now + timedelta(seconds=2),
            )
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        assert "3 messages" in result
        # Real subject should appear
        assert '"Real subject"' in result
        # Empty/whitespace subjects should not produce empty quoted strings
        assert '""' not in result
        assert '"   "' not in result

    def test_unread_context_no_subjects_if_field_missing(
        self, db, user_model_store, event_store
    ):
        """When emails have no subject field, count is shown without subjects section.

        Graceful fallback: if no subjects are available, the format degrades
        to just showing the count (no colon or subject list).
        """
        now = datetime.now(timezone.utc)

        user_model_store.update_signal_profile(
            profile_type="relationships",
            data={
                "contacts": {
                    "alice@example.com": {
                        "interaction_count": 5,
                        "inbound_count": 2,
                        "outbound_count": 3,
                    }
                }
            },
        )

        # Event payload without subject field
        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="proton",
                payload={"from_address": "alice@example.com"},
                timestamp=now,
            )
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        assert "alice@example.com" in result
        assert "1 message)" in result
        # No colon+subjects section when there are no subjects
        assert "alice@example.com (1 message):" not in result

    def test_unread_context_non_priority_subjects_not_shown(
        self, db, user_model_store, event_store
    ):
        """Non-priority sender subjects must not appear in the output.

        Only priority senders (outbound_count >= 3) should have their subjects
        surfaced.  Subjects from marketing or low-interaction senders would
        dilute the signal and increase context noise.
        """
        now = datetime.now(timezone.utc)

        user_model_store.update_signal_profile(
            profile_type="relationships",
            data={
                "contacts": {
                    "alice@example.com": {
                        "interaction_count": 10,
                        "inbound_count": 4,
                        "outbound_count": 6,  # Priority
                    },
                    "marketing@shop.com": {
                        "interaction_count": 50,
                        "inbound_count": 50,
                        "outbound_count": 0,  # Not priority
                    },
                }
            },
        )

        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="proton",
                payload={"from_address": "alice@example.com", "subject": "Priority email"},
                timestamp=now,
            )
        )
        event_store.store_event(
            create_test_event(
                event_type="email.received",
                source="proton",
                payload={"from_address": "marketing@shop.com", "subject": "50% off everything"},
                timestamp=now,
            )
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_unread_context()

        assert "Messages in last 12 hours: 2" in result
        # Priority sender subject should appear
        assert '"Priority email"' in result
        # Non-priority sender and its subject should not appear in breakdown
        assert '"50% off everything"' not in result
        assert "marketing@shop.com" not in result


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

        # Seed mood history snapshot (mood_history, not raw signal profile).
        # Updated for PR #251 (iteration 251): mood context now comes from the
        # mood_history table rather than the mood_signals signal profile.
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO mood_history
                   (timestamp, energy_level, stress_level, emotional_valence,
                    social_battery, cognitive_load, confidence, trend)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    0.75, 0.30, 0.70, 0.60, 0.35, 0.80, "stable",
                ),
            )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # Verify all sections present
        assert "Jordan" in context  # Preferences
        assert "Current time:" in context  # Timestamp
        assert "Review PR" in context  # Tasks
        assert "Messages in last 12 hours: 1" in context  # Unread count
        # Semantic facts: new categorized format shows the value without JSON quotes
        # and with a capitalized label and confidence score (PR #265).
        assert "software_engineer" in context  # value present regardless of format
        assert "Layer 2 Semantic Memory" in context  # new section header
        assert "emotional_valence" in context  # Mood context (computed dimensions)

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


class TestRecentCompletionsContext:
    """Tests for _get_recent_completions_context().

    The recent-completions section surfaces tasks finished in the last 24 hours
    so the LLM can acknowledge wins and avoid treating completed work as pending.
    """

    def _insert_task(self, conn, title: str, priority: str, domain: str,
                     status: str, completed_at: str | None = None):
        """Helper: insert a task row into the state DB."""
        conn.execute(
            """INSERT INTO tasks (id, title, priority, status, domain, completed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), title, priority, status, domain, completed_at),
        )

    def test_returns_empty_when_no_completed_tasks(self, db, user_model_store):
        """Should return empty string when no tasks have been completed."""
        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_completions_context()

        assert result == ""

    def test_returns_empty_when_completions_older_than_24h(self, db, user_model_store):
        """Should exclude tasks completed more than 24 hours ago."""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        with db.get_connection("state") as conn:
            self._insert_task(conn, "Old task", "normal", "work",
                              "completed", completed_at=old_time)
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_completions_context()

        assert result == ""

    def test_includes_recently_completed_tasks(self, db, user_model_store):
        """Should include tasks completed within the last 24 hours."""
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with db.get_connection("state") as conn:
            self._insert_task(conn, "Submit expense report", "high", "work",
                              "completed", completed_at=recent_time)
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_completions_context()

        assert "Recently completed tasks (last 24h):" in result
        assert "Submit expense report" in result
        assert "[high]" in result
        assert "(work)" in result

    def test_excludes_pending_tasks(self, db, user_model_store):
        """Should not show tasks that are still pending."""
        with db.get_connection("state") as conn:
            self._insert_task(conn, "Pending task", "normal", "work", "pending")
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_completions_context()

        assert result == ""

    def test_uses_general_when_domain_is_null(self, db, user_model_store):
        """Should display 'general' when a completed task has no domain."""
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with db.get_connection("state") as conn:
            self._insert_task(conn, "No domain task", "low", None,
                              "completed", completed_at=recent_time)
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_completions_context()

        assert "(general)" in result

    def test_orders_by_most_recently_completed(self, db, user_model_store):
        """Most recent completions should appear before older ones."""
        t1 = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        t2 = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with db.get_connection("state") as conn:
            self._insert_task(conn, "Older task", "normal", "work",
                              "completed", completed_at=t1)
            self._insert_task(conn, "Newer task", "high", "work",
                              "completed", completed_at=t2)
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_completions_context()

        newer_pos = result.index("Newer task")
        older_pos = result.index("Older task")
        assert newer_pos < older_pos, "Most recent completion should appear first"

    def test_caps_at_10_tasks(self, db, user_model_store):
        """Should cap at 10 tasks to respect token budget."""
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with db.get_connection("state") as conn:
            for i in range(15):
                self._insert_task(conn, f"Task {i}", "normal", "work",
                                  "completed", completed_at=recent_time)
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_completions_context()

        # Count bullet points — each task produces exactly one "- [" line
        bullet_count = result.count("- [")
        assert bullet_count == 10, f"Expected 10 tasks, got {bullet_count}"

    def test_briefing_includes_completions_section(self, db, user_model_store):
        """assemble_briefing_context() should include recent completions when present."""
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        with db.get_connection("state") as conn:
            self._insert_task(conn, "Shipped the feature", "high", "work",
                              "completed", completed_at=recent_time)
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        briefing = assembler.assemble_briefing_context()

        assert "Recently completed tasks (last 24h):" in briefing
        assert "Shipped the feature" in briefing

    def test_briefing_omits_completions_when_none(self, db, user_model_store):
        """assemble_briefing_context() should not add the section when no completions exist."""
        assembler = ContextAssembler(db, user_model_store)
        briefing = assembler.assemble_briefing_context()

        assert "Recently completed tasks" not in briefing


class TestRecentEpisodesContext:
    """Tests for _get_recent_episodes_context().

    The recent-episodes section surfaces Layer 1 episodic memories from the
    last 24 hours so the LLM has concrete narrative context about the user's
    recent digital activity — emails exchanged, calendar events attended, etc.
    """

    def _insert_episode(
        self,
        conn,
        interaction_type: str,
        content_summary: str,
        timestamp: str,
        contacts_involved: list = None,
        topics: list = None,
        active_domain: str = None,
    ):
        """Helper: insert an episode row into the user_model DB."""
        import uuid as _uuid
        conn.execute(
            """INSERT INTO episodes
               (id, timestamp, event_id, interaction_type, content_summary,
                contacts_involved, topics, active_domain)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(_uuid.uuid4()),
                timestamp,
                str(_uuid.uuid4()),
                interaction_type,
                content_summary,
                json.dumps(contacts_involved or []),
                json.dumps(topics or []),
                active_domain,
            ),
        )

    def test_returns_empty_when_no_recent_episodes(self, db, user_model_store):
        """Should return empty string when no episodes exist in the last 24 hours."""
        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_episodes_context()

        assert result == ""

    def test_returns_empty_when_episodes_older_than_24h(self, db, user_model_store):
        """Should exclude episodes from more than 24 hours ago."""
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        with db.get_connection("user_model") as conn:
            self._insert_episode(conn, "email_received", "Old email summary", old_ts)
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_episodes_context()

        assert result == ""

    def test_includes_recent_episodes(self, db, user_model_store):
        """Should include episodes from the last 24 hours."""
        recent_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with db.get_connection("user_model") as conn:
            self._insert_episode(
                conn, "email_sent", "Replied to Alice about Q1 budget", recent_ts,
                contacts_involved=["alice@example.com"],
                topics=["finance", "planning"],
                active_domain="work",
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_episodes_context()

        assert "Recent activity (last 24h):" in result
        assert "[email_sent]" in result
        assert "Replied to Alice about Q1 budget" in result

    def test_shows_domain_when_present(self, db, user_model_store):
        """Should annotate episodes with their active domain."""
        ts = datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            self._insert_episode(
                conn, "calendar_event", "Attended team standup", ts,
                active_domain="work",
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_episodes_context()

        assert "(work)" in result

    def test_shows_contacts_when_present(self, db, user_model_store):
        """Should annotate episodes with contacts involved (up to 2)."""
        ts = datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            self._insert_episode(
                conn, "email_received", "Email about the project",
                ts,
                contacts_involved=["bob@work.com", "carol@work.com", "dave@work.com"],
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_episodes_context()

        # Should show first 2 contacts but not the 3rd (truncated at 2)
        assert "bob@work.com" in result
        assert "carol@work.com" in result
        assert "dave@work.com" not in result
        assert "[contacts:" in result

    def test_shows_topics_when_present(self, db, user_model_store):
        """Should annotate episodes with topics (up to 3)."""
        ts = datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            self._insert_episode(
                conn, "email_sent", "Sent update",
                ts,
                topics=["budget", "timeline", "risk", "stakeholders"],
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_episodes_context()

        # Should show first 3 topics, not the 4th
        assert "budget" in result
        assert "timeline" in result
        assert "risk" in result
        assert "stakeholders" not in result
        assert "[topics:" in result

    def test_excludes_empty_summary_episodes(self, db, user_model_store):
        """Episodes with empty content_summary should not appear."""
        ts = datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            # Episode with empty summary (should be excluded)
            self._insert_episode(conn, "email_received", "", ts)
            # Episode with real summary (should be included)
            self._insert_episode(conn, "email_sent", "Sent reply to Bob", ts)
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_episodes_context()

        assert "Sent reply to Bob" in result
        # The empty-summary episode should be filtered out entirely, so only
        # one bullet should appear (the email_sent episode).
        bullet_count = result.count("- [")
        assert bullet_count == 1

    def test_orders_by_most_recent_first(self, db, user_model_store):
        """Episodes should appear newest first."""
        t1 = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        t2 = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with db.get_connection("user_model") as conn:
            self._insert_episode(conn, "email_received", "Older email", t1)
            self._insert_episode(conn, "email_sent", "Newer reply", t2)
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_episodes_context()

        assert result.index("Newer reply") < result.index("Older email")

    def test_caps_at_8_episodes(self, db, user_model_store):
        """Should limit output to 8 episodes to respect token budget."""
        ts = datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            for i in range(12):
                self._insert_episode(
                    conn, "email_received", f"Email {i} summary", ts
                )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_episodes_context()

        # Count bullet points
        bullet_count = result.count("- [")
        assert bullet_count == 8

    def test_handles_null_contacts_gracefully(self, db, user_model_store):
        """Should not crash when contacts_involved is NULL or empty JSON."""
        ts = datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            self._insert_episode(
                conn, "email_received", "Email with no contacts", ts,
                contacts_involved=[],  # Empty list
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_episodes_context()

        assert "Email with no contacts" in result
        # No contacts annotation when list is empty
        assert "[contacts:" not in result

    def test_handles_null_topics_gracefully(self, db, user_model_store):
        """Should not crash when topics is NULL or empty JSON."""
        ts = datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            self._insert_episode(
                conn, "calendar_event", "Meeting with no topics", ts,
                topics=[],
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_recent_episodes_context()

        assert "Meeting with no topics" in result
        assert "[topics:" not in result

    def test_briefing_includes_episodes_section(self, db, user_model_store):
        """assemble_briefing_context() should include recent episodes when present."""
        ts = datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            self._insert_episode(
                conn, "email_sent", "Sent weekly report to manager", ts,
                active_domain="work",
            )
            conn.commit()

        assembler = ContextAssembler(db, user_model_store)
        briefing = assembler.assemble_briefing_context()

        assert "Recent activity (last 24h):" in briefing
        assert "Sent weekly report to manager" in briefing

    def test_briefing_omits_episodes_when_none(self, db, user_model_store):
        """assemble_briefing_context() should not add the section when no recent episodes exist."""
        assembler = ContextAssembler(db, user_model_store)
        briefing = assembler.assemble_briefing_context()

        assert "Recent activity (last 24h):" not in briefing

    def test_episodes_section_precedes_semantic_facts(self, db, user_model_store):
        """Episodes section should appear before semantic facts in the briefing.

        The ordering is: recent activity (Layer 1 episodic) before abstract
        knowledge (Layer 2 semantic). This keeps the context window structured
        from most-recent-and-concrete to most-general-and-abstract.
        """
        ts = datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            self._insert_episode(conn, "email_sent", "Sent a message", ts)
            conn.commit()

        user_model_store.update_semantic_fact(
            key="works_at",
            category="fact",
            value="Acme Corp",
            confidence=0.9,
        )

        assembler = ContextAssembler(db, user_model_store)
        briefing = assembler.assemble_briefing_context()

        episodes_pos = briefing.find("Recent activity (last 24h):")
        # New header from _get_semantic_facts_context() (PR #265)
        facts_pos = briefing.find("Layer 2 Semantic Memory")

        assert episodes_pos != -1, "Episodes section must be present"
        assert facts_pos != -1, "Semantic facts section must be present"
        assert episodes_pos < facts_pos, (
            "Episodes (Layer 1) should appear before semantic facts (Layer 2)"
        )

