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
        self.db = db
        self.ums = ums

    def assemble_briefing_context(self) -> str:
        """Build context for the morning briefing."""
        parts = []

        # User preferences
        parts.append(self._get_preference_context())

        # Current time context
        now = datetime.now(timezone.utc)
        parts.append(f"Current time: {now.strftime('%A, %B %d, %Y at %H:%M UTC')}")

        # Today's calendar (from state DB)
        parts.append(self._get_calendar_context())

        # Pending tasks
        parts.append(self._get_task_context())

        # Unread message summary
        parts.append(self._get_unread_context())

        # Recent high-confidence semantic facts
        facts = self.ums.get_semantic_facts(min_confidence=0.6)
        if facts:
            fact_lines = [f"- {f['key']}: {f['value']}" for f in facts[:20]]
            parts.append("Known facts about user:\n" + "\n".join(fact_lines))

        # Current mood (for tone adjustment)
        mood_profile = self.ums.get_signal_profile("mood_signals")
        if mood_profile:
            parts.append(f"User mood context: {json.dumps(mood_profile['data'].get('recent_signals', [])[-3:])}")

        return "\n\n---\n\n".join(parts)

    def assemble_draft_context(self, contact_id: str, channel: str,
                               incoming_message: str) -> str:
        """Build context for drafting a reply."""
        parts = []

        # Communication template for this contact/channel
        with self.db.get_connection("user_model") as conn:
            template = conn.execute(
                """SELECT * FROM communication_templates 
                   WHERE contact_id = ? OR channel = ?
                   ORDER BY samples_analyzed DESC LIMIT 1""",
                (contact_id, channel),
            ).fetchone()

            if template:
                parts.append(f"Communication style for this context:")
                parts.append(f"  Formality: {template['formality']}")
                parts.append(f"  Greeting: {template['greeting'] or 'none'}")
                parts.append(f"  Closing: {template['closing'] or 'none'}")
                parts.append(f"  Typical length: {template['typical_length']} words")
                parts.append(f"  Uses emoji: {'yes' if template['uses_emoji'] else 'no'}")
                common = json.loads(template['common_phrases'] or '[]')
                if common:
                    parts.append(f"  Common phrases: {', '.join(common[:5])}")

        # Relationship context
        rel_profile = self.ums.get_signal_profile("relationships")
        if rel_profile and contact_id in rel_profile["data"].get("contacts", {}):
            contact_data = rel_profile["data"]["contacts"][contact_id]
            parts.append(f"Relationship: {contact_data.get('interaction_count', 0)} total interactions")

        # Linguistic profile (general writing style)
        ling_profile = self.ums.get_signal_profile("linguistic")
        if ling_profile and "averages" in ling_profile["data"]:
            avg = ling_profile["data"]["averages"]
            parts.append(f"User's general style: formality={avg.get('formality', 0.5):.1f}")

        parts.append(f"\nIncoming message to reply to:\n{incoming_message}")

        return "\n".join(parts)

    def assemble_search_context(self, query: str) -> str:
        """Build context for a life-search query."""
        return f"User is searching across their entire digital life for: {query}"

    def _get_preference_context(self) -> str:
        with self.db.get_connection("preferences") as conn:
            rows = conn.execute("SELECT key, value FROM user_preferences").fetchall()
            if rows:
                prefs = {row["key"]: row["value"] for row in rows}
                return f"User preferences: {json.dumps(prefs)}"
            return "User preferences: not yet configured"

    def _get_calendar_context(self) -> str:
        # Placeholder — in production this reads from calendar connector state
        return "Calendar: (connect CalDAV to populate)"

    def _get_task_context(self) -> str:
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
                lines = [f"- [{t['priority']}] {t['title']} (due: {t['due_date'] or 'no date'})"
                         for t in tasks]
                return "Pending tasks:\n" + "\n".join(lines)
            return "Pending tasks: none"

    def _get_unread_context(self) -> str:
        # Placeholder — count recent unprocessed inbound messages
        with self.db.get_connection("events") as conn:
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM events 
                   WHERE type IN ('email.received', 'message.received')
                   AND timestamp > datetime('now', '-12 hours')"""
            ).fetchone()
            count = row["cnt"] if row else 0
            return f"Messages in last 12 hours: {count}"
