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
from datetime import datetime, timedelta, timezone

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
        actionable items (calendar, tasks, messages, predictions), then
        background knowledge (semantic facts, mood). This prioritization
        ensures the most important information appears early in the context
        window.

        Predictions are included so the LLM can surface relationship
        maintenance reminders, upcoming preparation needs, calendar conflicts,
        and other high-confidence signals that the prediction engine has
        identified — without the user needing to ask explicitly.
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

        # Section 6: Active predictions from the prediction engine.
        # These are high-confidence, unresolved predictions that have already
        # passed all confidence gates and been surfaced to the user. Including
        # them here lets the LLM weave prediction-based insights ("you haven't
        # replied to Alice in 10 days") naturally into the briefing narrative
        # rather than emitting them only as separate push notifications.
        predictions_context = self._get_predictions_context()
        if predictions_context:
            parts.append(predictions_context)

        # Section 7: Cross-signal insights discovered by the InsightEngine.
        # Insights are pattern discoveries surfaced by correlating multiple
        # signal profiles (cadence, mood, spatial, linguistic, topics, etc.).
        # Unlike predictions (which say "do X soon"), insights say "you
        # consistently do Y on Fridays" — they provide behavioral narrative that
        # helps the LLM generate a richer, more personalised briefing. Ordered
        # by confidence and capped at 10 to keep the context window focused.
        insights_context = self._get_insights_context()
        if insights_context:
            parts.append(insights_context)

        # Section 8: Tasks completed in the last 24 hours.
        # Surfaces recent wins so the LLM can acknowledge accomplishments and
        # distinguish work-in-progress from already-finished items. Without
        # this, the LLM has no visibility into what the user finished yesterday
        # and may incorrectly imply those items are still outstanding.
        completions_context = self._get_recent_completions_context()
        if completions_context:
            parts.append(completions_context)

        # Section 9: Semantic facts the system has learned about the user
        # (e.g., "preferred_language: English", "works_at: Acme Corp").
        # Only facts with confidence >= 0.6 are included to avoid noise.
        # Capped at 20 facts to respect the token budget.
        facts = self.ums.get_semantic_facts(min_confidence=0.6)
        if facts:
            fact_lines = [f"- {f['key']}: {f['value']}" for f in facts[:20]]
            parts.append("Known facts about user:\n" + "\n".join(fact_lines))

        # Section 10: Recent mood signals (last 3 data points). This allows
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

        # --- Layer 3: Full outbound linguistic profile ---
        # Surfaces all key averages from the user's linguistic fingerprint so
        # the LLM can match not just formality but also question-asking tendency,
        # hedging, emoji use, and vocabulary richness.  PR #261 added these
        # metrics to LinguisticExtractor but assemble_draft_context() previously
        # exposed only ``formality``, wasting the other nine computed dimensions.
        ling_profile = self.ums.get_signal_profile("linguistic")
        if ling_profile and "averages" in ling_profile["data"]:
            avg = ling_profile["data"]["averages"]
            style_parts = [f"formality={avg.get('formality', 0.5):.2f}"]

            # Append non-trivial signal dimensions only when they carry useful
            # signal (i.e. above a noise threshold).  This avoids cluttering the
            # context with zeros for metrics the user simply never exhibits.
            question_rate = avg.get("question_rate", 0.0)
            if question_rate > 0.05:
                # > 0.05 questions-per-sentence means the user regularly asks
                # questions — the LLM should mirror this inquisitive tone.
                style_parts.append(f"question_rate={question_rate:.2f}")

            hedge_rate = avg.get("hedge_rate", 0.0)
            if hedge_rate > 0.05:
                # High hedge_rate means the user typically softens statements
                # ("I think", "maybe") — avoid overly assertive drafts.
                style_parts.append(f"hedge_rate={hedge_rate:.2f}")

            emoji_rate = avg.get("emoji_rate", 0.0)
            if emoji_rate > 0.01:
                # User frequently uses emoji — include them in the draft.
                style_parts.append(f"emoji_rate={emoji_rate:.3f}")

            vocab_diversity = avg.get("unique_word_ratio", 0.0)
            if vocab_diversity > 0:
                # High ratio → rich vocabulary; low ratio → simpler repetitive
                # word choice.  Helps the LLM calibrate lexical complexity.
                style_parts.append(f"vocabulary_diversity={vocab_diversity:.2f}")

            avg_sentence_length = avg.get("avg_sentence_length", 0.0)
            if avg_sentence_length > 0:
                # Typical sentence length guides draft verbosity.
                style_parts.append(f"avg_sentence_length={avg_sentence_length:.0f}w")

            parts.append("User's general style: " + ", ".join(style_parts))

            # Surface common greetings/closings as a fallback when no
            # contact-specific template was found (template section is Layer 1).
            # These are the user's most-used openers and sign-offs, extracted
            # across all outbound messages.
            common_greetings = ling_profile["data"].get("common_greetings", [])
            common_closings = ling_profile["data"].get("common_closings", [])
            if common_greetings:
                parts.append(
                    "User's typical greetings: " + ", ".join(common_greetings[:3])
                )
            if common_closings:
                parts.append(
                    "User's typical closings: " + ", ".join(common_closings[:3])
                )

        # --- Layer 4: Contact's inbound writing style ---
        # Shows how *this specific contact* writes to the user (formality, hedge
        # rate, emoji usage, sentence length).  Knowing the contact's style lets
        # the LLM craft a response that naturally mirrors or acknowledges their
        # register — e.g., if they write casually with many questions, the draft
        # can be warmer and more directly responsive.
        # Data source: ``linguistic_inbound`` signal profile, which stores
        # per-contact averages built from 104K+ inbound message samples.
        try:
            inbound_profile = self.ums.get_signal_profile("linguistic_inbound")
            if inbound_profile:
                contact_avg = (
                    inbound_profile["data"]
                    .get("per_contact_averages", {})
                    .get(contact_id)
                )
                if contact_avg:
                    contact_parts = [
                        f"formality={contact_avg.get('formality', 0.5):.2f}"
                    ]
                    c_question = contact_avg.get("question_rate", 0.0)
                    if c_question > 0.05:
                        contact_parts.append(f"question_rate={c_question:.2f}")
                    c_hedge = contact_avg.get("hedge_rate", 0.0)
                    if c_hedge > 0.05:
                        contact_parts.append(f"hedge_rate={c_hedge:.2f}")
                    c_emoji = contact_avg.get("emoji_rate", 0.0)
                    if c_emoji > 0.01:
                        contact_parts.append(f"emoji_rate={c_emoji:.3f}")
                    c_sentence = contact_avg.get("avg_sentence_length", 0.0)
                    if c_sentence > 0:
                        contact_parts.append(
                            f"avg_sentence_length={c_sentence:.0f}w"
                        )
                    parts.append(
                        f"Contact's writing style ({contact_id}): "
                        + ", ".join(contact_parts)
                    )
        except Exception:
            # Fail-open: inbound style is a nice-to-have.  If the profile is
            # missing or malformed, the draft still has all outbound style data.
            pass

        # Finally, append the incoming message that the user needs to reply to.
        # This is the last section so all style context is available first.
        parts.append(f"\nIncoming message to reply to:\n{incoming_message}")

        return "\n".join(parts)

    def assemble_search_context(self, query: str) -> str:
        """Build context for a life-search query.

        Assembles a richer context string that helps the LLM:
          1. Understand *who* the user is (preferences, known facts) so it can
             disambiguate references like "my project" or "Mike".
          2. Anchor relative time expressions ("last month", "yesterday") to
             the actual current date.
          3. Calibrate output style (verbosity, preferred name) from preferences.

        The search results themselves are appended by ``AIEngine.search_life()``
        after this method returns; this context appears *before* those results so
        the LLM can interpret them with full user context.

        Args:
            query: The natural-language search string entered by the user.

        Returns:
            A multi-section context string separated by "---" delimiters.

        Example usage::

            ctx = assembler.assemble_search_context("What did Mike say about the Denver project?")
            # Returns a string with preferences, timestamp, known facts, and mood context
            # that the LLM can use to disambiguate "Mike" and "Denver project".
        """
        parts = []

        # Section 1: The search intent — always first so the LLM knows what
        # question to answer when it reads the subsequent context sections.
        parts.append(f"User is searching across their entire digital life for: {query}")

        # Section 2: Current timestamp — anchors relative time expressions
        # ("last month", "yesterday", "this week") to a concrete date so the
        # LLM can compute the correct time window rather than guessing.
        now = datetime.now(timezone.utc)
        parts.append(f"Current time: {now.strftime('%A, %B %d, %Y at %H:%M UTC')}")

        # Section 3: User preferences — verbosity level, preferred name, etc.
        # This lets the LLM calibrate its answer length and form of address.
        parts.append(self._get_preference_context())

        # Section 4: High-confidence semantic facts about the user.
        # These help disambiguate references: e.g., "my boss" maps to a known
        # name, "the project" could map to a known employer or project fact.
        # Only facts with confidence >= 0.6 are included to keep the context
        # signal-rich and avoid injecting speculative noise.
        try:
            facts = self.ums.get_semantic_facts(min_confidence=0.6)
            if facts:
                fact_lines = [f"- {f['key']}: {f['value']}" for f in facts[:15]]
                parts.append("Known facts about user (use for disambiguation):\n"
                             + "\n".join(fact_lines))
        except Exception:
            # Fail-open: missing facts degrade search quality slightly but
            # should never prevent the search from returning a result.
            pass

        # Section 5: Recent mood signals (last 3 entries).
        # Gives the LLM soft context about the user's recent emotional state
        # so it can frame results empathetically (e.g., a stressed user asking
        # about overdue tasks deserves an encouraging framing).
        try:
            mood_profile = self.ums.get_signal_profile("mood_signals")
            if mood_profile:
                recent = mood_profile["data"].get("recent_signals", [])[-3:]
                if recent:
                    parts.append(f"Recent mood context: {recent}")
        except Exception:
            # Fail-open: mood context is a nice-to-have; omit it silently.
            pass

        return "\n\n---\n\n".join(parts)

    def _get_predictions_context(self) -> str:
        """Fetch active, surfaced predictions to surface in the briefing.

        Queries the predictions table for entries that:
          - Have been surfaced (was_surfaced = 1), meaning they passed all
            confidence gates and were shown to the user as notifications.
          - Are still unresolved (resolved_at IS NULL), meaning the predicted
            condition hasn't been confirmed or dismissed yet.
          - Were created within the last 7 days to avoid surfacing stale
            predictions from long-completed situations.
          - Have no filter_reason, confirming they passed all quality gates.

        Results are sorted by confidence descending so the most reliable
        predictions appear first, capped at 10 to respect token budget.

        Returns an empty string when there are no active predictions, which
        causes the caller to skip this section entirely (no "none" noise).

        Example output::

            Active predictions from the system:
            - [opportunity] You haven't replied to alice@example.com in 10 days — consider following up (confidence: 0.82)
            - [reminder] Prepare for "Q1 Planning" starting in 2 hours (confidence: 0.75)
            - [conflict] Calendar conflict: "Standup" overlaps "Team Lunch" on 2026-02-19 (confidence: 0.91)
        """
        with self.db.get_connection("user_model") as conn:
            rows = conn.execute(
                """SELECT prediction_type, description, confidence, suggested_action
                   FROM predictions
                   WHERE was_surfaced = 1
                     AND resolved_at IS NULL
                     AND filter_reason IS NULL
                     AND datetime(created_at) > datetime('now', '-7 days')
                   ORDER BY confidence DESC
                   LIMIT 10"""
            ).fetchall()

        if not rows:
            return ""

        lines = []
        for row in rows:
            # Format each prediction as a concise bullet: type, description,
            # and confidence rounded to 2 decimal places.
            line = f"- [{row['prediction_type']}] {row['description']} (confidence: {row['confidence']:.2f})"
            # Append suggested action when the prediction engine provides one.
            # This gives the LLM concrete, actionable language to include.
            if row["suggested_action"]:
                line += f" — suggested: {row['suggested_action']}"
            lines.append(line)

        return "Active predictions from the system:\n" + "\n".join(lines)

    def _get_insights_context(self) -> str:
        """Fetch recent, high-confidence insights to surface in the briefing.

        Queries the insights table for entries that are still within their
        staleness TTL window.  The TTL is stored per-insight in
        ``staleness_ttl_hours`` (default 168 h = 7 days) so each correlator
        can tune its own freshness window without a shared constant.

        Results are sorted by confidence descending so the most reliable
        pattern discoveries appear first, capped at 10 to respect the token
        budget.  Insights with explicit negative feedback (thumbs-down) are
        excluded so the LLM does not repeat dismissed observations.

        Returns an empty string when there are no active insights, which causes
        the caller to skip this section entirely (no "none" noise).

        Example output::

            Behavioral patterns and insights:
            - [relationship_intelligence] You haven't contacted Alice in 14 days — usual gap is 3 days (confidence: 0.87)
            - [behavioral_pattern] You typically send emails on Tuesday and Thursday mornings (confidence: 0.81)
            - [communication_style] Your writing to the marketing team is 30% more casual than average (confidence: 0.72)
        """
        with self.db.get_connection("user_model") as conn:
            rows = conn.execute(
                """SELECT type, summary, confidence, category, entity
                   FROM insights
                   WHERE feedback IS NOT 'negative'
                     AND datetime(created_at) >
                         datetime('now', '-' || staleness_ttl_hours || ' hours')
                   ORDER BY confidence DESC
                   LIMIT 10"""
            ).fetchall()

        if not rows:
            return ""

        lines = []
        for row in rows:
            # Use category as the display label when set; fall back to type.
            # category is more human-readable (e.g. "behavioral_pattern"
            # vs the generic "pattern" type string).
            label = row["category"] if row["category"] else row["type"]
            line = f"- [{label}] {row['summary']} (confidence: {row['confidence']:.2f})"
            lines.append(line)

        return "Behavioral patterns and insights:\n" + "\n".join(lines)

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

    def _get_recent_completions_context(self) -> str:
        """Fetch tasks completed in the last 24 hours.

        Surfaces recent accomplishments so the LLM can:
          - Acknowledge the user's wins when generating the morning briefing
          - Distinguish completed work from still-pending items
          - Reference finished tasks when discussing related follow-ups

        Without this section the LLM only sees pending work, making it
        impossible to say "Yesterday you finished X — here's what remains."

        Results are ordered most-recently-completed first and capped at 10 to
        keep the context window focused. Domains are included so the LLM can
        group completions by life area ("you wrapped up three work tasks").

        Returns an empty string when no tasks were completed in the window,
        which causes the caller to skip this section entirely.

        Example output::

            Recently completed tasks (last 24h):
            - [high] Submit Q1 expense report (work)
            - [normal] Call dentist for appointment (health)
            - [low] Update reading list (personal)
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

        with self.db.get_connection("state") as conn:
            tasks = conn.execute(
                """SELECT title, priority, domain, completed_at
                   FROM tasks
                   WHERE status = 'completed'
                     AND completed_at >= ?
                   ORDER BY completed_at DESC
                   LIMIT 10""",
                (cutoff,),
            ).fetchall()

        if not tasks:
            return ""

        lines = []
        for t in tasks:
            domain = t["domain"] or "general"
            lines.append(f"- [{t['priority']}] {t['title']} ({domain})")

        return "Recently completed tasks (last 24h):\n" + "\n".join(lines)

    def _get_unread_context(self) -> str:
        """Count recent inbound messages (emails + chat) from the last 12 hours.

        Breaks down messages by priority sender — contacts the user regularly
        writes back to (outbound_count >= 3 in the relationships signal profile).
        For priority senders, also surfaces up to 3 recent email subject lines
        so the LLM can reference specific messages in the morning briefing.

        Priority threshold: outbound_count >= 3 means the user has replied to
        this contact at least three times, indicating a genuine human relationship
        rather than a marketing or automated sender.

        Output format (when priority contacts exist with subjects):
            Messages in last 12 hours: 8
              From priority contacts:
                - alice@example.com (3 messages): "Project deadline", "Can we meet?"
                - bob@work.com (1 message): "Lunch tomorrow?"
              From other senders: 4

        Output format (when priority contacts have no subjects):
            Messages in last 12 hours: 8
              From priority contacts:
                - alice@example.com (3 messages)
              From other senders: 5

        Output format (when no priority contacts have sent messages):
            Messages in last 12 hours: 8
        """
        # Use a parameterized Python timestamp rather than SQLite's datetime('now')
        # function.  SQLite's datetime() returns space-separated strings like
        # "2026-02-18 12:04:12", but events are stored in ISO-8601 format with a
        # 'T' separator ("2026-02-18T00:04:12+00:00").  SQLite compares TEXT
        # fields lexicographically, so 'T' (ASCII 84) > ' ' (ASCII 32), meaning
        # any T-format timestamp would incorrectly compare as GREATER THAN the
        # space-format cutoff — effectively disabling the 12-hour filter entirely.
        # Passing a Python isoformat() string as a parameter keeps both sides in
        # the same format so string comparison is semantically correct.
        cutoff_12h = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        with self.db.get_connection("events") as conn:
            # Fetch all recent messages with subjects, sorted newest-first.
            # We group per-sender and collect subjects in Python rather than
            # using a GROUP BY + LIMIT per-sender SQL pattern, which would
            # require complex correlated subqueries.  The row count is bounded
            # by the 12-hour window, so loading all rows into Python is safe.
            rows = conn.execute(
                """SELECT
                       json_extract(payload, '$.from_address') AS from_address,
                       json_extract(payload, '$.subject') AS subject
                   FROM events
                   WHERE type IN ('email.received', 'message.received')
                     AND timestamp > ?
                   ORDER BY timestamp DESC""",
                (cutoff_12h,),
            ).fetchall()

        if not rows:
            return "Messages in last 12 hours: 0"

        # Group by sender: tally message counts and collect up to 3 recent subjects.
        # Rows are newest-first so the first 3 subjects collected per sender are
        # the most recent 3, which are most actionable in the briefing.
        # Rows with a NULL from_address still contribute to the total count but
        # are excluded from the priority-contact breakdown.
        sender_data: dict[str, dict] = {}
        total = 0
        for row in rows:
            total += 1
            addr = row["from_address"]
            if addr:
                if addr not in sender_data:
                    sender_data[addr] = {"count": 0, "subjects": []}
                sender_data[addr]["count"] += 1
                # Collect up to 3 non-empty subjects per sender.
                subject = (row["subject"] or "").strip()
                if subject and len(sender_data[addr]["subjects"]) < 3:
                    sender_data[addr]["subjects"].append(subject)

        lines = [f"Messages in last 12 hours: {total}"]

        # Identify priority contacts — senders the user actively writes back to.
        # Cross-reference sender addresses against the relationships signal profile.
        # Only contacts with outbound_count >= 3 qualify; lower counts suggest
        # automated mailers or one-time contacts that don't need to be called out.
        try:
            rel_profile = self.ums.get_signal_profile("relationships")
            if rel_profile and sender_data:
                contacts = rel_profile["data"].get("contacts", {})
                priority_senders: dict[str, dict] = {
                    addr: sender_data[addr]
                    for addr, data in contacts.items()
                    if data.get("outbound_count", 0) >= 3 and addr in sender_data
                }
                if priority_senders:
                    # Sort descending by message count so the most active sender
                    # appears first, making the most urgent sender immediately visible.
                    sorted_priority = sorted(
                        priority_senders.items(), key=lambda x: -x[1]["count"]
                    )
                    lines.append("  From priority contacts:")
                    for addr, info in sorted_priority:
                        count = info["count"]
                        subjects = info["subjects"]
                        cnt_str = f"{count} message{'s' if count != 1 else ''}"
                        if subjects:
                            # Quote each subject so the LLM can reference them
                            # directly in the briefing (e.g. "Alice sent 'Project
                            # deadline' — you may want to prioritise that reply").
                            subjects_str = ", ".join(f'"{s}"' for s in subjects)
                            lines.append(f"    - {addr} ({cnt_str}): {subjects_str}")
                        else:
                            lines.append(f"    - {addr} ({cnt_str})")
                    other_count = total - sum(
                        info["count"] for info in priority_senders.values()
                    )
                    if other_count > 0:
                        lines.append(f"  From other senders: {other_count}")
        except Exception:
            # Fail-open: priority breakdown is a nice-to-have.
            # If the relationships profile is missing or malformed, the basic
            # count is still returned on the first line.
            pass

        return "\n".join(lines)
