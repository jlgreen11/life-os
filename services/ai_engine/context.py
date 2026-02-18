"""
Life OS — Context Assembler

Assembles the optimal context window for a given query type.

Pulls from:
    - Current state (time, location, calendar, mood)
    - Relevant semantic memories
    - Recent episodes
    - User preferences
    - Relevant communication templates
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


class ContextAssembler:
    """
    Assembles the optimal context window for a given query.
    """

    def __init__(self, db: DatabaseManager, ums: UserModelStore):
        # DatabaseManager provides access to multiple SQLite databases (events,
        # state, preferences, user_model) via named connections.
        self.db = db
        # UserModelStore holds the user's behavioral profiles (linguistic style,
        # mood signals, relationship data) and semantic facts learned over time.
        self.ums = ums

    def assemble_briefing_context(self) -> str:
        """Build context for the morning briefing.

        Assembles a multi-section context string from several data sources.
        Each section is separated by "---" delimiters so the LLM can easily
        distinguish between context categories. The ordering is intentional:
        preferences first (sets the tone), then temporal context, then
        actionable items (calendar, tasks, messages), then background knowledge
        (semantic facts, mood). This prioritization ensures the most important
        information appears early in the context window.
        """
        parts = []

        # Section 1: User preferences (verbosity, preferred name, etc.)
        # Loaded first so the LLM knows how to calibrate its output style.
        parts.append(self._get_preference_context())

        # Section 2: Current timestamp -- gives the LLM temporal awareness
        # so it can reference "today", "this morning", day of week, etc.
        now = datetime.now(timezone.utc)
        parts.append(f"Current time: {now.strftime('%A, %B %d, %Y at %H:%M UTC')}")

        # Section 3: Calendar events for today (meetings, deadlines, etc.)
        parts.append(self._get_calendar_context())

        # Section 4: Pending tasks, sorted by priority then due date.
        # Capped at 20 items to avoid overwhelming the context window.
        parts.append(self._get_task_context())

        # Section 5: Count of unread messages in the last 12 hours.
        # Gives the user a sense of their inbox backlog.
        parts.append(self._get_unread_context())

        # Section 6: Semantic facts the system has learned about the user
        # (e.g., "preferred_language: English", "works_at: Acme Corp").
        # Only facts with confidence >= 0.6 are included to avoid noise.
        # Capped at 20 facts to respect the token budget.
        facts = self.ums.get_semantic_facts(min_confidence=0.6)
        if facts:
            fact_lines = [f"- {f['key']}: {f['value']}" for f in facts[:20]]
            parts.append("Known facts about user:\n" + "\n".join(fact_lines))

        # Section 7: Recent mood signals (last 3 data points). This allows
        # the LLM to adjust its tone -- e.g., more encouraging if the user
        # has been stressed, more energetic if mood is positive.
        mood_profile = self.ums.get_signal_profile("mood_signals")
        if mood_profile:
            parts.append(f"User mood context: {json.dumps(mood_profile['data'].get('recent_signals', [])[-3:])}")

        # Join all sections with delimiter lines for clear visual separation
        # in the prompt. The LLM can parse these as distinct context blocks.
        return "\n\n---\n\n".join(parts)

    def assemble_draft_context(self, contact_id: str, channel: str,
                               incoming_message: str) -> str:
        """Build context for drafting a reply.

        Combines three layers of style information to help the LLM match the
        user's voice: (1) contact/channel-specific communication templates,
        (2) relationship history with this contact, and (3) general linguistic
        profile. The most specific data (templates) takes priority.
        """
        parts = []

        # --- Layer 1: Communication template (most specific) ---
        # Look up how the user typically writes to this specific contact or on
        # this channel. Templates are ranked by samples_analyzed so the most
        # data-rich template wins. The OR query allows fallback from a
        # contact-specific template to a channel-wide default.
        with self.db.get_connection("user_model") as conn:
            template = conn.execute(
                """SELECT * FROM communication_templates
                   WHERE contact_id = ? OR channel = ?
                   ORDER BY samples_analyzed DESC LIMIT 1""",
                (contact_id, channel),
            ).fetchone()

            # Surface all style dimensions the LLM needs to replicate:
            # formality level, greeting/closing phrases, message length,
            # emoji usage, and commonly used phrases.
            if template:
                parts.append(f"Communication style for this context:")
                parts.append(f"  Formality: {template['formality']}")
                parts.append(f"  Greeting: {template['greeting'] or 'none'}")
                parts.append(f"  Closing: {template['closing'] or 'none'}")
                parts.append(f"  Typical length: {template['typical_length']} words")
                parts.append(f"  Uses emoji: {'yes' if template['uses_emoji'] else 'no'}")
                # Show up to 5 common phrases the user frequently uses with
                # this contact/channel, so the LLM can naturally incorporate them.
                common = json.loads(template['common_phrases'] or '[]')
                if common:
                    parts.append(f"  Common phrases: {', '.join(common[:5])}")

        # --- Layer 2: Relationship context ---
        # Provides interaction history depth. A contact with 200 interactions
        # implies a close relationship (casual tone), while 3 interactions
        # suggests a newer contact (more formal tone).
        rel_profile = self.ums.get_signal_profile("relationships")
        if rel_profile and contact_id in rel_profile["data"].get("contacts", {}):
            contact_data = rel_profile["data"]["contacts"][contact_id]
            parts.append(f"Relationship: {contact_data.get('interaction_count', 0)} total interactions")

        # --- Layer 3: General linguistic profile (least specific, broadest) ---
        # Falls back to the user's overall writing style if no contact-specific
        # template exists. Provides a baseline formality score (0.0 = very
        # casual, 1.0 = very formal).
        ling_profile = self.ums.get_signal_profile("linguistic")
        if ling_profile and "averages" in ling_profile["data"]:
            avg = ling_profile["data"]["averages"]
            parts.append(f"User's general style: formality={avg.get('formality', 0.5):.1f}")

        # Finally, append the incoming message that the user needs to reply to.
        # This is the last section so all style context is available first.
        parts.append(f"\nIncoming message to reply to:\n{incoming_message}")

        return "\n".join(parts)

    def assemble_search_context(self, query: str) -> str:
        """Build context for a life-search query.

        Currently returns a minimal context string. The heavy lifting (actual
        search results) is appended by AIEngine.search_life() after querying
        the events database. This method exists as a hook for future enrichment
        (e.g., adding user preferences, recent context, or search history).
        """
        return f"User is searching across their entire digital life for: {query}"

    def _get_preference_context(self) -> str:
        """Load all user preferences as a JSON object for the context window.

        Preferences include settings like verbosity level, preferred name,
        notification preferences, etc. These are key-value pairs stored in the
        preferences database. Returns a fallback string if none are configured.
        """
        with self.db.get_connection("preferences") as conn:
            rows = conn.execute("SELECT key, value FROM user_preferences").fetchall()
            if rows:
                # Flatten rows into a dict for compact JSON representation.
                prefs = {row["key"]: row["value"] for row in rows}
                return f"User preferences: {json.dumps(prefs)}"
            return "User preferences: not yet configured"

    def _get_calendar_context(self) -> str:
        """Fetch upcoming calendar events from the events database.

        Queries the ``events`` table for ``calendar.event.created`` entries
        whose ``start_time`` falls within the next 7 days.  All-day events
        (birthdays, holidays) are included but formatted differently from
        timed events so the LLM can distinguish between scheduled meetings
        and informational date markers.

        Deduplicates by (title, start_time) so recurring syncs that store
        multiple copies of the same occurrence don't inflate the event list.

        Returns a human-readable multi-line string capped at 20 events, or a
        "no events" message if the calendar is empty for the period.
        """
        with self.db.get_connection("events") as conn:
            rows = conn.execute(
                """SELECT DISTINCT
                       json_extract(payload, '$.title')       AS title,
                       json_extract(payload, '$.start_time')  AS start_time,
                       json_extract(payload, '$.end_time')    AS end_time,
                       json_extract(payload, '$.is_all_day')  AS is_all_day,
                       json_extract(payload, '$.location')    AS location
                   FROM events
                   WHERE type = 'calendar.event.created'
                     AND date(json_extract(payload, '$.start_time'))
                         BETWEEN date('now') AND date('now', '+7 days')
                   ORDER BY json_extract(payload, '$.start_time') ASC
                   LIMIT 20"""
            ).fetchall()

        if not rows:
            return "Upcoming calendar events (next 7 days): none"

        lines: list[str] = []
        for row in rows:
            title = row["title"] or "(untitled)"
            start = row["start_time"] or ""
            location = row["location"]

            if row["is_all_day"]:
                # All-day entries: show date and title only; no time noise.
                entry = f"- [all-day] {start}: {title}"
            else:
                # Timed entries: include location when available.
                entry = f"- {start}: {title}"
                if location:
                    entry += f" @ {location}"

            lines.append(entry)

        return "Upcoming calendar events (next 7 days):\n" + "\n".join(lines)

    def _get_task_context(self) -> str:
        """Fetch pending tasks, sorted by priority then due date.

        The CASE expression maps priority labels to sort-order integers so that
        critical tasks float to the top, followed by high, normal, and low.
        The secondary sort (due_date ASC) ensures earlier deadlines appear first
        within the same priority tier. Capped at 20 to respect token budget.
        """
        with self.db.get_connection("state") as conn:
            tasks = conn.execute(
                """SELECT title, priority, due_date, domain
                   FROM tasks WHERE status = 'pending'
                   ORDER BY
                       CASE priority
                           WHEN 'critical' THEN 1
                           WHEN 'high' THEN 2
                           WHEN 'normal' THEN 3
                           ELSE 4
                       END,
                       due_date ASC
                   LIMIT 20"""
            ).fetchall()

            if tasks:
                # Format each task as a concise bullet with priority tag and due date.
                lines = [f"- [{t['priority']}] {t['title']} (due: {t['due_date'] or 'no date'})"
                         for t in tasks]
                return "Pending tasks:\n" + "\n".join(lines)
            return "Pending tasks: none"

    def _get_unread_context(self) -> str:
        """Count recent inbound messages (emails + chat) from the last 12 hours.

        This provides a simple "inbox pressure" signal for the briefing.
        It counts both email.received and message.received event types.
        A future enhancement could break this down by source/sender priority.
        """
        with self.db.get_connection("events") as conn:
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM events
                   WHERE type IN ('email.received', 'message.received')
                   AND timestamp > datetime('now', '-12 hours')"""
            ).fetchone()
            count = row["cnt"] if row else 0
            return f"Messages in last 12 hours: {count}"
