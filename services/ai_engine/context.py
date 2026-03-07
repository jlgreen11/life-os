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
import logging
from datetime import datetime, timedelta, timezone

from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore

logger = logging.getLogger(__name__)


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

        # Section 7: Relationship highlights from the contacts table.
        # Surfaces overdue and priority contacts so the LLM can weave
        # relationship maintenance reminders into the briefing narrative
        # without depending on the prediction pipeline.
        relationship_context = self._get_relationship_highlights_context()
        if relationship_context:
            parts.append(relationship_context)

        # Section 8: Cross-signal insights discovered by the InsightEngine.
        # Insights are pattern discoveries surfaced by correlating multiple
        # signal profiles (cadence, mood, spatial, linguistic, topics, etc.).
        # Unlike predictions (which say "do X soon"), insights say "you
        # consistently do Y on Fridays" — they provide behavioral narrative that
        # helps the LLM generate a richer, more personalised briefing. Ordered
        # by confidence and capped at 10 to keep the context window focused.
        insights_context = self._get_insights_context()
        if insights_context:
            parts.append(insights_context)

        # Section 9: Tasks completed in the last 24 hours.
        # Surfaces recent wins so the LLM can acknowledge accomplishments and
        # distinguish work-in-progress from already-finished items. Without
        # this, the LLM has no visibility into what the user finished yesterday
        # and may incorrectly imply those items are still outstanding.
        completions_context = self._get_recent_completions_context()
        if completions_context:
            parts.append(completions_context)

        # Section 10: Recent episodic memories (Layer 1 of the user model).
        # Each episode records a specific interaction (email read, message sent,
        # calendar event attended) with a content summary, contacts involved,
        # topics discussed, and inferred domain. Including recent episodes gives
        # the LLM concrete narrative material: "Yesterday you exchanged emails
        # with Alice about the Q1 budget and attended a planning meeting." This
        # is distinct from the unread count (section 5, which counts INBOUND
        # messages) and completed tasks (section 8, which lists task completions)
        # — episodes cover ALL recent interactions, giving the LLM awareness of
        # what the user was actually doing in their digital life.
        episodes_context = self._get_recent_episodes_context()
        if episodes_context:
            parts.append(episodes_context)
        else:
            # Fallback: when episodes are unavailable (user_model.db corrupted
            # or empty), produce an activity summary from events.db instead.
            # This gives the LLM concrete material even without episodic memory.
            activity_summary = self._get_recent_activity_summary()
            if activity_summary:
                parts.append(activity_summary)

        # Section 11: Semantic facts (Layer 2 — Semantic Memory) learned about
        # the user, organized into human-readable categories: values, behavioral
        # patterns, and preferences.  The helper method filters out low-signal
        # noise (generic topic words, per-contact relationship entries already
        # covered by the relationships profile) and groups remaining facts so
        # the LLM receives a coherent portrait of *who the user is* rather than
        # a raw key-value dump.
        semantic_context = self._get_semantic_facts_context()
        if semantic_context:
            parts.append(semantic_context)
        else:
            # Fallback: when semantic facts are unavailable (user_model.db
            # corrupted or empty), produce a contact activity summary from
            # events.db to give the LLM a mini people-radar.
            contact_summary = self._get_contact_activity_summary()
            if contact_summary:
                parts.append(contact_summary)

        # Section 12: Habitual behavioral routines (Layer 3 Procedural Memory).
        # Routines are time- or location-triggered behavioral patterns detected
        # by the RoutineDetector (e.g., "morning email review", "evening wind-down").
        # Including them here lets the LLM give time-appropriate advice, reference
        # familiar patterns, and anticipate what the user typically does next.
        # Only routines with consistency_score >= 0.5 are surfaced to avoid
        # misleading the LLM with poorly-observed one-off patterns.
        routines_context = self._get_routines_context()
        if routines_context:
            parts.append(routines_context)

        # Section 13: Computed mood snapshot from mood_history.  Using the
        # pre-aggregated row from mood_history (written every 15 min by
        # SignalExtractorPipeline.get_current_mood) is far more useful than
        # the raw recent_signals array: the LLM receives human-readable
        # dimension labels (energy, stress, valence, trend) rather than
        # typed signal dicts it must interpret itself.
        mood_context = self._get_mood_context()
        if mood_context:
            parts.append(mood_context)

        # Join all sections with delimiter lines for clear visual separation
        # in the prompt. The LLM can parse these as distinct context blocks.
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Mood context helper
    # ------------------------------------------------------------------

    def _get_mood_context(self) -> str:
        """Return a human-readable mood summary from the most recent mood_history snapshot.

        Queries the ``mood_history`` table (written every 15 minutes by
        ``SignalExtractorPipeline.get_current_mood()``) for the most recent
        row within the last 24 hours.  Each row already contains pre-aggregated
        mood dimensions (energy_level, stress_level, emotional_valence, trend,
        social_battery, cognitive_load) so the LLM receives clean, named values
        rather than raw signal dicts it has to interpret.

        Why this is better than the previous approach (raw recent_signals JSON):

        - **Old**: ``[{"signal_type": "circadian_energy", "value": 0.8, ...}, ...]``
          The LLM had to know that ``circadian_energy`` maps to energy and that
          ``calendar_density`` maps to stress — an implicit, brittle mapping.
        - **New**: ``energy_level=0.82 (high), stress_level=0.25 (low), ...``
          The LLM reads the dimension name directly, making tone calibration
          reliable even when signal types change or new extractors are added.

        Label thresholds used for each dimension:

        - energy_level:     ≥0.65 → "high", 0.35–0.65 → "moderate", <0.35 → "low"
        - stress_level:     ≥0.65 → "high", 0.35–0.65 → "moderate", <0.35 → "low"
        - emotional_valence:≥0.65 → "positive", 0.35–0.65 → "neutral", <0.35 → "negative"

        Falls back gracefully to an empty string (which causes the caller to
        skip this section) when:
        - The mood_history table does not exist (pre-migration instances).
        - No snapshots exist within the last 24 hours (system just started).

        Example output::

            User mood context: energy_level=0.82 (high), stress_level=0.25 (low),
            emotional_valence=0.73 (positive), social_battery=0.60, trend=stable
            (confidence: 0.90)
        """
        def _label(value: float, dim: str) -> str:
            """Convert a 0-1 scalar to a human-readable label.

            Thresholds are intentionally coarse (3 buckets) so the LLM does
            not over-index on small numerical differences.  The dimension name
            is provided so context-sensitive labels can be returned (e.g.,
            'positive' instead of 'high' for emotional_valence).
            """
            if dim == "emotional_valence":
                if value >= 0.65:
                    return "positive"
                if value >= 0.35:
                    return "neutral"
                return "negative"
            # energy and stress use the same generic scale
            if value >= 0.65:
                return "high"
            if value >= 0.35:
                return "moderate"
            return "low"

        try:
            with self.db.get_connection("user_model") as conn:
                row = conn.execute(
                    """SELECT energy_level, stress_level, emotional_valence,
                              social_battery, cognitive_load, confidence, trend
                       FROM mood_history
                       WHERE datetime(timestamp) > datetime('now', '-24 hours')
                       ORDER BY timestamp DESC
                       LIMIT 1"""
                ).fetchone()
        except Exception:
            # mood_history table may not exist on older deployments — fail open
            # and skip the section rather than crashing the briefing assembly.
            return ""

        if not row:
            return ""

        energy = row["energy_level"]
        stress = row["stress_level"]
        valence = row["emotional_valence"]
        social = row["social_battery"]
        confidence = row["confidence"]
        trend = row["trend"] or "stable"

        parts = [
            f"energy_level={energy:.2f} ({_label(energy, 'energy_level')})",
            f"stress_level={stress:.2f} ({_label(stress, 'stress_level')})",
            f"emotional_valence={valence:.2f} ({_label(valence, 'emotional_valence')})",
            f"social_battery={social:.2f}",
            f"trend={trend}",
        ]
        # Only include confidence when it is non-trivial (≥0.3) — low-confidence
        # mood readings add noise rather than signal to the tone calibration.
        if confidence >= 0.3:
            parts.append(f"confidence={confidence:.2f}")

        return "User mood context: " + ", ".join(parts)

    def assemble_draft_context(self, contact_id: str, channel: str,
                               incoming_message: str) -> str:
        """Build context for drafting a reply.

        Combines five layers of information to help the LLM write a reply that
        sounds like the user and references the real relationship:

        1. Communication template — how the user typically writes to this
           contact/channel (formality, greeting, closing, common phrases).
        2. Relationship depth — total interaction count for tone calibration.
        3. Outbound linguistic profile — per-contact (or global) style metrics.
        4. Contact's inbound style — how *they* write, so the draft can mirror
           their register naturally.
        5. Recent conversation history — the last N episodes shared with this
           contact, so the LLM can reference prior topics, avoid repetition, and
           continue ongoing threads coherently.

        The incoming message is appended last so all context precedes it.
        """
        parts = []

        # --- Layer 1: Communication template (most specific) ---
        # Look up how the user typically writes to this specific contact or on
        # this channel. Templates are ranked by samples_analyzed so the most
        # data-rich template wins. The OR query allows fallback from a
        # contact-specific template to a channel-wide default.
        template = self.ums.get_communication_template(
            contact_id=contact_id, channel=channel
        )

        # Surface all style dimensions the LLM needs to replicate:
        # formality level, greeting/closing phrases, message length,
        # emoji usage, and commonly used phrases.
        if template:
            parts.append("Communication style for this context:")
            parts.append(f"  Formality: {template['formality']}")
            parts.append(f"  Greeting: {template['greeting'] or 'none'}")
            parts.append(f"  Closing: {template['closing'] or 'none'}")
            parts.append(f"  Typical length: {template['typical_length']} words")
            parts.append(f"  Uses emoji: {'yes' if template['uses_emoji'] else 'no'}")
            # Show up to 5 common phrases the user frequently uses with
            # this contact/channel, so the LLM can naturally incorporate them.
            # JSON fields are already deserialized by the store method.
            common = template.get("common_phrases") or []
            if common:
                parts.append(f"  Common phrases: {', '.join(common[:5])}")
            # Surface phrases the user avoids so the LLM doesn't accidentally
            # use language the user has deliberately stopped using.
            avoids = template.get("avoids_phrases") or []
            if avoids:
                parts.append(f"  Avoids phrases: {', '.join(avoids[:5])}")
            # Tone notes capture stylistic observations like "always leads
            # with conclusion" or "uses bullet points" — free-text guidance
            # that can't be expressed as numeric metrics.
            tone_notes = template.get("tone_notes") or []
            if tone_notes:
                parts.append(f"  Tone notes: {'; '.join(tone_notes[:5])}")

        # --- Layer 2: Relationship context ---
        # Provides interaction history depth. A contact with 200 interactions
        # implies a close relationship (casual tone), while 3 interactions
        # suggests a newer contact (more formal tone).
        rel_profile = self.ums.get_signal_profile("relationships")
        if rel_profile and contact_id in rel_profile["data"].get("contacts", {}):
            contact_data = rel_profile["data"]["contacts"][contact_id]
            parts.append(f"Relationship: {contact_data.get('interaction_count', 0)} total interactions")

        # --- Layer 3: Outbound linguistic profile (per-contact when available) ---
        # Surfaces style averages from the user's linguistic fingerprint so the
        # LLM can match formality, question-asking tendency, hedging, emoji use,
        # and vocabulary richness.
        #
        # Since PR #281 the ``linguistic`` profile stores ``per_contact_averages``
        # — running style summaries derived from the per-contact ring buffer —
        # alongside the global ``averages``.  We prefer per-contact data when it
        # exists and has enough samples (>= 3) because "how you write to Alice"
        # is more useful for drafting a reply to Alice than "how you write in
        # general".  When per-contact data is absent (new/rare contact) we fall
        # back to the global averages so the draft is never left with no style
        # guidance.
        ling_profile = self.ums.get_signal_profile("linguistic")
        if ling_profile and "averages" in ling_profile["data"]:
            global_avg = ling_profile["data"]["averages"]

            # Prefer per-contact averages when the contact has enough samples.
            per_contact_avgs = ling_profile["data"].get("per_contact_averages", {})
            contact_avg = per_contact_avgs.get(contact_id)
            if contact_avg and contact_avg.get("samples_count", 0) >= 3:
                # Use per-contact style — more specific than global average.
                avg = contact_avg
                style_label = (
                    f"User's style with this contact "
                    f"({contact_avg['samples_count']} msgs)"
                )
            else:
                # No per-contact data yet — fall back to global baseline.
                avg = global_avg
                style_label = "User's general style"

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

            # Extended style signals — these feed from the linguistic extractor's
            # pattern banks (humor, greeting, closing, Oxford comma, capitalization)
            # and give the LLM concrete stylistic cues beyond basic metrics.

            assertion_rate = avg.get("assertion_rate", 0.0)
            if assertion_rate > 0.05:
                # High assertion_rate means the user states things confidently
                # — the LLM should mirror this direct tone.
                style_parts.append(f"assertion_rate={assertion_rate:.2f}")

            avg_word_length = avg.get("avg_word_length", 0.0)
            if avg_word_length > 0:
                # Guides vocabulary sophistication — longer words suggest
                # a more academic/professional register.
                style_parts.append(f"avg_word_length={avg_word_length:.1f}")

            parts.append(style_label + ": " + ", ".join(style_parts))

            # Greeting and closing conventions — the LLM should mirror these.
            greeting = avg.get("top_greeting") or avg.get("greeting_detected")
            if greeting:
                parts.append(f'Preferred greeting: "{greeting}"')

            closing = avg.get("top_closing") or avg.get("closing_detected")
            if closing:
                parts.append(f'Preferred closing: "{closing}"')

            # Humor markers — if the user frequently uses humor words, guide
            # the LLM to maintain a light, conversational tone.
            humor_rate = avg.get("humor_rate", 0.0)
            if humor_rate > 0.02:
                humor_type = avg.get("top_humor_marker", "")
                humor_note = f"Uses humor frequently (rate={humor_rate:.2f})"
                if humor_type:
                    humor_note += f' — common marker: "{humor_type}"'
                parts.append(humor_note)

            # Oxford comma preference — guide list formatting.
            oxford = avg.get("oxford_comma_preference")
            if oxford is not None:
                parts.append(f'Oxford comma: {"yes" if oxford else "no"}')

            # Capitalization style — guide casing conventions.
            cap_style = avg.get("capitalization_style")
            if cap_style and cap_style != "unknown":
                parts.append(f"Capitalization style: {cap_style}")

            # If per-contact style is notably different from the global baseline
            # (formality delta > 0.15), surface the comparison so the LLM knows
            # this contact gets a distinctly different register.
            if contact_avg and contact_avg.get("samples_count", 0) >= 3:
                global_formality = global_avg.get("formality", 0.5)
                contact_formality = contact_avg.get("formality", 0.5)
                delta = contact_formality - global_formality
                if abs(delta) > 0.15:
                    direction = "more formal" if delta > 0 else "more casual"
                    parts.append(
                        f"Note: you write {direction} with this contact than usual "
                        f"(Δ{abs(delta):.2f} vs global avg {global_formality:.2f})."
                    )

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
            logger.warning(
                "Draft context: failed to load inbound writing style for %s",
                contact_id,
                exc_info=True,
            )

        # --- Layer 5: Recent conversation history with this contact ---
        # Query the last 5 episodes where the contact appears in
        # contacts_involved.  This gives the LLM concrete knowledge of what was
        # recently discussed so it can:
        #   • Continue ongoing threads naturally (e.g. "Following up on the Q3
        #     roadmap we discussed last week…")
        #   • Avoid re-introducing topics already covered
        #   • Reference the correct relationship context
        # The LIKE-based JSON search is safe for email addresses: they contain
        # no SQL special characters and contacts_involved is always valid JSON
        # written by our own code.  A full-text or JSON-path query would be more
        # precise but SQLite JSON functions are not available in all deployments,
        # so LIKE is the portable fallback.
        try:
            with self.db.get_connection("user_model") as conn:
                recent_episodes = conn.execute(
                    """SELECT timestamp, interaction_type, content_summary, topics
                       FROM episodes
                       WHERE contacts_involved LIKE ?
                       ORDER BY timestamp DESC
                       LIMIT 5""",
                    (f"%{contact_id}%",),
                ).fetchall()

            if recent_episodes:
                history_lines = ["Recent conversation history with this contact:"]
                for ep in recent_episodes:
                    # Format date as YYYY-MM-DD for brevity.
                    ts_raw = ep["timestamp"] or ""
                    ts_short = ts_raw[:10] if len(ts_raw) >= 10 else ts_raw

                    # Include topics when present (non-empty JSON array).
                    try:
                        topic_list = json.loads(ep["topics"] or "[]")
                    except (ValueError, TypeError):
                        topic_list = []
                    topic_str = (
                        " [topics: " + ", ".join(topic_list[:3]) + "]"
                        if topic_list else ""
                    )

                    interaction = ep["interaction_type"] or "unknown"
                    summary = ep["content_summary"] or ""
                    history_lines.append(
                        f"  {ts_short} ({interaction}): {summary}{topic_str}"
                    )
                parts.append("\n".join(history_lines))
        except Exception:
            # Fail-open: conversation history is enrichment, not critical path.
            # If the user_model DB is unavailable the draft still has style data.
            logger.warning(
                "Draft context: failed to load conversation history for %s",
                contact_id,
                exc_info=True,
            )

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
        facts_found = False
        try:
            facts = self.ums.get_semantic_facts(min_confidence=0.6)
            if facts:
                fact_lines = [f"- {f['key']}: {f['value']}" for f in facts[:15]]
                parts.append("Known facts about user (use for disambiguation):\n"
                             + "\n".join(fact_lines))
                facts_found = True
        except Exception:
            # Fail-open: missing facts degrade search quality slightly but
            # should never prevent the search from returning a result.
            logger.warning("Search context: failed to load semantic facts", exc_info=True)

        # Fallback: when semantic facts are unavailable, provide contact activity
        # from events.db so the LLM can still disambiguate people references.
        if not facts_found:
            contact_summary = self._get_contact_activity_summary()
            if contact_summary:
                parts.append(contact_summary)

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
            logger.warning("Search context: failed to load mood profile", exc_info=True)

        return "\n\n---\n\n".join(parts)

    def _get_relationship_highlights_context(self) -> str:
        """Fetch relationship highlights from the contacts table in entities.db.

        Queries contacts that have ``contact_frequency_days`` and
        ``last_contact`` set, then computes how overdue or on-track each
        relationship is.  A contact is considered "overdue" when the days
        since last contact exceeds 1.5x their typical contact frequency.

        Results are sorted by priority first, then by the ratio of
        days-since-contact to expected frequency (most overdue first),
        capped at 8 contacts to respect the token budget.

        Returns an empty string when no contacts have relationship metrics,
        causing the caller to skip this section entirely.

        Example output::

            Relationship highlights:
            - Alice Smith: last contact 15 days ago (typical: every 5 days) -- overdue
            - Bob Jones: last contact 3 days ago (typical: every 7 days) -- on track
        """
        try:
            with self.db.get_connection("entities") as conn:
                rows = conn.execute(
                    """SELECT name, last_contact, contact_frequency_days, is_priority
                       FROM contacts
                       WHERE contact_frequency_days IS NOT NULL
                         AND last_contact IS NOT NULL
                       ORDER BY is_priority DESC,
                                (julianday('now') - julianday(last_contact))
                                    / NULLIF(contact_frequency_days, 0) DESC
                       LIMIT 8"""
                ).fetchall()
        except Exception as e:
            logger.debug("context: _get_relationship_highlights_context unavailable, skipping: %s", e)
            return ""

        if not rows:
            return ""

        now = datetime.now(timezone.utc)
        lines = []
        for row in rows:
            try:
                last_dt = datetime.fromisoformat(row["last_contact"])
                # Ensure timezone-aware comparison
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                days_since = (now - last_dt).days
            except (ValueError, TypeError):
                continue

            freq = row["contact_frequency_days"]
            status = "overdue" if days_since > freq * 1.5 else "on track"
            freq_int = int(freq)
            lines.append(
                f"- {row['name']}: last contact {days_since} days ago "
                f"(typical: every {freq_int} days) -- {status}"
            )

        if not lines:
            return ""

        return "Relationship highlights:\n" + "\n".join(lines)

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
        try:
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
        except Exception as e:
            logger.debug("context: _get_predictions_context unavailable, skipping: %s", e)
            return ""

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
        try:
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
        except Exception as e:
            logger.debug("context: _get_insights_context unavailable, skipping: %s", e)
            return ""

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
        try:
            with self.db.get_connection("preferences") as conn:
                rows = conn.execute("SELECT key, value FROM user_preferences").fetchall()
                if rows:
                    # Flatten rows into a dict for compact JSON representation.
                    prefs = {row["key"]: row["value"] for row in rows}
                    return f"User preferences: {json.dumps(prefs)}"
                return "User preferences: not yet configured"
        except Exception as e:
            logger.debug("context: _get_preference_context unavailable, skipping: %s", e)
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
        try:
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
        except Exception as e:
            logger.warning("_get_calendar_context failed: %s", e)
            return "Upcoming calendar events (next 7 days): unavailable (data error)"

    def _get_task_context(self) -> str:
        """Fetch pending tasks, sorted by priority then due date.

        The CASE expression maps priority labels to sort-order integers so that
        critical tasks float to the top, followed by high, normal, and low.
        The secondary sort (due_date ASC) ensures earlier deadlines appear first
        within the same priority tier. Capped at 20 to respect token budget.
        """
        try:
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
        except Exception as e:
            logger.debug("context: _get_task_context unavailable, skipping: %s", e)
            return ""

    def _get_recent_episodes_context(self) -> str:
        """Fetch recent episodic memories to surface in the morning briefing.

        Queries the episodes table for entries from the last 24 hours.
        Episodes are Layer 1 of the user model — individual interactions with
        full context. Each episode records a specific interaction (email read,
        message sent, calendar event attended, etc.) with a text summary,
        the contacts involved, and the topics discussed.

        Including recent episodes in the briefing gives the LLM concrete
        narrative material about what the user was actually doing in their
        digital life — distinct from the unread message count (which only
        counts INBOUND messages) and completed tasks (which only list task
        completions). Episodic context enables briefing lines like:
        "Yesterday you exchanged emails with Alice about Q1 planning and
        attended a 1-hour standup."

        Results are sorted by timestamp descending (most recent first) and
        capped at 8 to respect the token budget. Only episodes with a
        non-empty content_summary are returned — summaryless episodes are
        noise. JSON fields (contacts_involved, topics) are parsed and
        truncated to keep each line concise.

        Returns an empty string when no recent episodes exist, which causes
        the caller to skip this section entirely (no "none" noise).

        Example output::

            Recent activity (last 24h):
            - [email_sent] Replied to alice@example.com about Q1 budget (work) [topics: finance, planning]
            - [email_received] Message from bob@work.com: "Re: Project deadline" (work) [topics: project]
            - [calendar_event] Attended "Team Standup" for 60 min (work)
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT interaction_type, content_summary, contacts_involved,
                              topics, active_domain
                       FROM episodes
                       WHERE timestamp >= ?
                         AND content_summary IS NOT NULL
                         AND content_summary != ''
                       ORDER BY timestamp DESC
                       LIMIT 8""",
                    (cutoff,),
                ).fetchall()
        except Exception:
            # Fail-open: if the episodes table is missing or malformed, skip
            # the section entirely rather than crashing the briefing.
            return ""

        if not rows:
            return ""

        lines = []
        for row in rows:
            interaction_type = row["interaction_type"] or "interaction"
            summary = row["content_summary"]
            domain = row["active_domain"]

            # Build the primary line: [type] summary (domain)
            line = f"- [{interaction_type}] {summary}"
            if domain:
                line += f" ({domain})"

            # Parse contacts from JSON array and annotate with up to 2 names.
            # Contacts are email addresses or display names; we show at most 2
            # to keep the line readable while still conveying social context.
            try:
                contacts = json.loads(row["contacts_involved"] or "[]")
                if contacts:
                    contact_str = ", ".join(str(c) for c in contacts[:2])
                    line += f" [contacts: {contact_str}]"
            except (json.JSONDecodeError, TypeError):
                pass  # Missing contacts field is not an error

            # Parse topics and annotate with up to 3 tags.
            # Topics are single words or short phrases extracted by the signal
            # pipeline; they provide fast keyword context to the LLM.
            try:
                topics = json.loads(row["topics"] or "[]")
                if topics:
                    topic_str = ", ".join(str(t) for t in topics[:3])
                    line += f" [topics: {topic_str}]"
            except (json.JSONDecodeError, TypeError):
                pass  # Missing topics field is not an error

            lines.append(line)

        return "Recent activity (last 24h):\n" + "\n".join(lines)

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

        try:
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
        except Exception as e:
            logger.debug("context: _get_recent_completions_context unavailable, skipping: %s", e)
            return ""

        if not tasks:
            return ""

        lines = []
        for t in tasks:
            domain = t["domain"] or "general"
            lines.append(f"- [{t['priority']}] {t['title']} ({domain})")

        return "Recently completed tasks (last 24h):\n" + "\n".join(lines)

    def _get_routines_context(self) -> str:
        """Fetch the user's top behavioral routines from Layer 3 Procedural Memory.

        Routines are habitual, time- or location-triggered behavioral patterns
        detected by the RoutineDetector (e.g., "morning_email_review",
        "evening_wind_down", "arrive_home"). Including them in the morning
        briefing allows the LLM to:

          - Give time-appropriate advice ("This is your typical focused-work window")
          - Reference familiar patterns ("It's your usual morning email review time")
          - Anticipate next steps ("You typically move to deep work after 9 am")

        Only routines with consistency_score >= 0.5 are surfaced.  Poorly-
        observed one-off patterns (low consistency) would mislead the LLM into
        describing uncertain habits as reliable facts.

        Results are sorted by consistency_score descending (most reliable first)
        and capped at 5 to respect the token budget.  Returns empty string when
        no qualifying routines have been detected yet (new user or sparse data).

        Example output::

            Observed behavioral routines (Layer 3 Procedural Memory):
            - morning_email_review (8 steps, ~35 min, seen 28x, consistency: 0.88)
            - evening_wind_down (4 steps, ~20 min, seen 19x, consistency: 0.74)
            - arrive_home (3 steps, ~10 min, seen 31x, consistency: 0.91)
        """
        try:
            routines = self.ums.get_routines()
            if not routines:
                return ""

            # Filter to reliable patterns only.  The consistency_score is a
            # fraction of how often the routine fires in the expected window;
            # 0.5 means it triggers at least half the time, which is enough
            # to describe it as a genuine habit.
            quality_routines = [
                r for r in routines if (r.get("consistency_score") or 0.0) >= 0.5
            ][:5]

            if not quality_routines:
                return ""

            lines = []
            for r in quality_routines:
                step_count = len(r.get("steps") or [])
                duration = r.get("typical_duration_minutes")
                times = r.get("times_observed") or 0
                score = r.get("consistency_score") or 0.0
                # Use trigger_condition as the label; fall back to name when absent.
                label = r.get("trigger") or r.get("name") or "unknown"

                # Build a concise one-line summary per routine so the LLM has
                # both the label and key statistics without verbose JSON.
                parts_inner = []
                if step_count:
                    parts_inner.append(f"{step_count} steps")
                if duration:
                    parts_inner.append(f"~{duration:.0f} min")
                parts_inner.append(f"seen {times}x")
                parts_inner.append(f"consistency: {score:.2f}")

                line = f"- {label} ({', '.join(parts_inner)})"
                lines.append(line)

            return (
                "Observed behavioral routines (Layer 3 Procedural Memory):\n"
                + "\n".join(lines)
            )
        except Exception:
            # Fail-open: missing routines degrade the briefing slightly but
            # must never prevent it from generating a response.
            return ""

    def _get_semantic_facts_context(self) -> str:
        """Build a categorized Layer 2 (Semantic Memory) section for the briefing.

        The raw semantic_facts table stores hundreds of inferred key-value pairs
        that span multiple conceptual categories: explicitly-stated values
        ("work_life_boundaries"), behavioral observations ("most_productive_day"),
        inferred preferences ("communication_style_directness"), and lower-signal
        interest tokens ("interest_lspace").  Dumping them all flat makes it
        impossible for the LLM to weight them correctly.

        This method organises high-confidence facts (>= 0.6) into three
        human-readable groups:

          values        — Explicitly-inferred value judgements (e.g. "weekday_only_work")
          behavioral    — Observable patterns keyed on known prefixes
                          (communication_style_*, peak_*, stress_*, most_productive_*)
          preferences   — Remaining implicit preferences, minus per-contact
                          relationship entries (already covered by the relationships
                          signal profile) and generic noise tokens (interest_* facts
                          whose value is a single common word).

        Relationship-specific facts (relationship_balance_*, relationship_priority_*)
        are excluded because 53+ such entries would dominate the section while
        adding little new information beyond what the unread/priority-contact
        breakdown already shows.

        Generic interest noise tokens (interest_lspace, interest_here, interest_more,
        interest_please, etc.) are excluded because they are HTML artefacts or
        stop-words that do not represent real interests.  A token is classified
        as noise when the ``interest_*`` key's value is a single word of six
        characters or fewer — short strings that match the stop-word profile.

        Results are capped per group (values ≤ 5, behavioral ≤ 5, preferences ≤ 8)
        and sorted by confidence descending within each group so the most reliable
        signals appear first.

        Returns an empty string when no qualifying facts exist, which causes the
        caller to omit this section entirely (no "none" noise in the briefing).

        Example output::

            What the system knows about you (Layer 2 Semantic Memory):
              Values:
              - Work-life boundaries: weekday_only_work (confidence: 0.95)
              - Flexible work boundaries: flexible_boundaries (confidence: 0.95)
              Behavioral patterns:
              - Communication directness: direct (confidence: 1.00)
              - Peak communication hour: 5 (confidence: 1.00)
              - Stress baseline: low_stress (confidence: 1.00)
              - Most productive day: tuesday (confidence: 0.85)
              Preferences:
              - Work location type: home_office (confidence: 1.00)
              - Primary work location: Residence Inn Clayton (confidence: 1.00)
        """
        try:
            facts = self.ums.get_semantic_facts(min_confidence=0.6)
        except Exception:
            # Fail-open: missing semantic facts degrade context quality slightly
            # but must never prevent the briefing from generating.
            return ""

        if not facts:
            return ""

        # ------------------------------------------------------------------ #
        # Known-noise prefixes for relationship-specific keys.  These 53+     #
        # facts are per-contact signals that belong in the relationships       #
        # profile view, not the semantic identity section.                     #
        # ------------------------------------------------------------------ #
        _RELATIONSHIP_PREFIXES = ("relationship_balance_", "relationship_priority_")

        # ------------------------------------------------------------------ #
        # Behavioral-pattern key prefixes that deserve their own named group. #
        # Sorted from most specific to least so the correct prefix matches.   #
        # ------------------------------------------------------------------ #
        _BEHAVIORAL_PREFIXES = (
            "communication_style_",
            "peak_communication_",
            "stress_baseline",
            "most_productive_",
            "incoming_pressure_",
            "chronotype",
        )

        values_facts: list[dict] = []
        behavioral_facts: list[dict] = []
        preference_facts: list[dict] = []

        for fact in facts:
            key: str = fact["key"]
            category: str = fact.get("category") or ""
            value = fact["value"]
            confidence: float = fact.get("confidence", 0.0)

            # Skip per-contact relationship entries — too numerous and already
            # represented in the unread/priority-contact breakdown.
            if any(key.startswith(p) for p in _RELATIONSHIP_PREFIXES):
                continue

            # Skip repetitive location entries that duplicate primary_work_location.
            # The "frequent_location_*" and "location_domain_*" keys repeat the
            # same place name as the primary_work_location fact.
            if key.startswith("frequent_location_") or key.startswith("location_domain_"):
                continue

            # Skip generic interest noise tokens.  An ``interest_*`` fact whose
            # value is a single word of ≤ 6 characters is almost certainly an
            # HTML artefact or stop-word (lspace, rspace, here, please, more,
            # free, shop, line) rather than a genuine topic of interest.
            if key.startswith("interest_"):
                val_str = str(value).strip('"').strip()
                if len(val_str) <= 6:
                    continue

            # Route to the correct bucket.
            if category == "values":
                values_facts.append(fact)
            elif any(key.startswith(p) for p in _BEHAVIORAL_PREFIXES) or key == "stress_baseline":
                behavioral_facts.append(fact)
            else:
                preference_facts.append(fact)

        # Sort each group by confidence descending and apply per-group caps.
        values_facts = sorted(values_facts, key=lambda f: -f.get("confidence", 0))[:5]
        behavioral_facts = sorted(behavioral_facts, key=lambda f: -f.get("confidence", 0))[:5]
        preference_facts = sorted(preference_facts, key=lambda f: -f.get("confidence", 0))[:8]

        if not any([values_facts, behavioral_facts, preference_facts]):
            return ""

        lines: list[str] = ["What the system knows about you (Layer 2 Semantic Memory):"]

        def _label(key: str) -> str:
            """Convert a snake_case key to a human-readable label.

            Example::
                _label("most_productive_day")  → "Most productive day"
                _label("communication_style_directness")  → "Communication style directness"
            """
            return key.replace("_", " ").capitalize()

        def _fmt_value(value: object) -> str:
            """Strip JSON-encoding quotes from string values for clean display."""
            s = str(value).strip('"').strip()
            return s

        if values_facts:
            lines.append("  Values:")
            for f in values_facts:
                lines.append(
                    f"  - {_label(f['key'])}: {_fmt_value(f['value'])}"
                    f" (confidence: {f['confidence']:.2f})"
                )

        if behavioral_facts:
            lines.append("  Behavioral patterns:")
            for f in behavioral_facts:
                lines.append(
                    f"  - {_label(f['key'])}: {_fmt_value(f['value'])}"
                    f" (confidence: {f['confidence']:.2f})"
                )

        if preference_facts:
            lines.append("  Preferences:")
            for f in preference_facts:
                lines.append(
                    f"  - {_label(f['key'])}: {_fmt_value(f['value'])}"
                    f" (confidence: {f['confidence']:.2f})"
                )

        return "\n".join(lines)

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
        try:
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
        except Exception as e:
            logger.warning("_get_unread_context failed: %s", e)
            return "Messages in last 12 hours: unavailable (data error)"

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

    # ------------------------------------------------------------------
    # Degraded-mode fallback helpers (events.db-based)
    # ------------------------------------------------------------------

    def _get_recent_activity_summary(self) -> str:
        """Produce an activity summary from events.db for the last 24 hours.

        Used as a fallback when episodic memory (user_model.db) is unavailable
        due to corruption or simply having no data.  Queries the healthy
        events.db to give the LLM concrete material about what happened
        recently, grouped by event type with top email senders/subjects.

        Returns an empty string when events.db has no recent data or on error,
        which causes the caller to skip this section entirely.

        Example output::

            Recent activity summary (last 24h, from event log):
            - email.received: 15 events
            - calendar.event.created: 2 events
            - message.received: 4 events
            Top senders: alice@example.com (5), bob@work.com (3)
            Recent subjects: "Q1 Budget Review", "Team Lunch Tomorrow"
        """
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

            with self.db.get_connection("events") as conn:
                # Group events by type for the last 24 hours
                type_rows = conn.execute(
                    """SELECT type, COUNT(*) AS cnt
                       FROM events
                       WHERE timestamp > ?
                       GROUP BY type
                       ORDER BY cnt DESC
                       LIMIT 10""",
                    (cutoff,),
                ).fetchall()

                if not type_rows:
                    return ""

                lines = ["Recent activity summary (last 24h, from event log):"]
                for row in type_rows:
                    lines.append(f"- {row['type']}: {row['cnt']} events")

                # Extract top email senders and recent subjects for personalization
                email_rows = conn.execute(
                    """SELECT
                           json_extract(payload, '$.from_address') AS sender,
                           json_extract(payload, '$.subject') AS subject
                       FROM events
                       WHERE type IN ('email.received', 'message.received')
                         AND timestamp > ?
                       ORDER BY timestamp DESC
                       LIMIT 50""",
                    (cutoff,),
                ).fetchall()

                if email_rows:
                    # Count interactions per sender
                    sender_counts: dict[str, int] = {}
                    subjects: list[str] = []
                    for row in email_rows:
                        sender = row["sender"]
                        if sender:
                            sender_counts[sender] = sender_counts.get(sender, 0) + 1
                        subj = row["subject"]
                        if subj and len(subjects) < 3:
                            subjects.append(subj)

                    # Top 3 senders by count
                    if sender_counts:
                        top_senders = sorted(
                            sender_counts.items(), key=lambda x: -x[1]
                        )[:3]
                        sender_strs = [f"{addr} ({cnt})" for addr, cnt in top_senders]
                        lines.append("Top senders: " + ", ".join(sender_strs))

                    if subjects:
                        subj_strs = [f'"{s}"' for s in subjects]
                        lines.append("Recent subjects: " + ", ".join(subj_strs))

            return "\n".join(lines)
        except Exception:
            logger.warning(
                "context: _get_recent_activity_summary failed, skipping",
                exc_info=True,
            )
            return ""

    def _get_contact_activity_summary(self) -> str:
        """Produce a mini people-radar from events.db for the last 7 days.

        Used as a fallback when semantic facts / relationship signal profiles
        from user_model.db are unavailable.  Queries events.db for email and
        message events, groups by contact address, and returns the top 5
        most-interacted contacts.

        Returns an empty string when no contact activity exists or on error,
        which causes the caller to skip this section entirely.

        Example output::

            Active contacts this week (from event log):
            - alice@example.com: 8 interactions (5 received, 3 sent)
            - bob@work.com: 3 interactions (2 received, 1 sent)
        """
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

            with self.db.get_connection("events") as conn:
                # Gather inbound contacts (from_address) and outbound (to_address)
                rows = conn.execute(
                    """SELECT type,
                              json_extract(payload, '$.from_address') AS from_addr,
                              json_extract(payload, '$.to_address') AS to_addr,
                              json_extract(payload, '$.sender') AS sender
                       FROM events
                       WHERE type IN (
                           'email.received', 'email.sent',
                           'message.received', 'message.sent'
                       )
                         AND timestamp > ?
                       ORDER BY timestamp DESC
                       LIMIT 500""",
                    (cutoff,),
                ).fetchall()

            if not rows:
                return ""

            # Tally interactions per contact
            contact_stats: dict[str, dict[str, int]] = {}
            for row in rows:
                event_type = row["type"]
                is_inbound = event_type in ("email.received", "message.received")

                # Determine the contact address
                if is_inbound:
                    addr = row["from_addr"] or row["sender"]
                else:
                    addr = row["to_addr"]

                if not addr:
                    continue

                if addr not in contact_stats:
                    contact_stats[addr] = {"received": 0, "sent": 0}

                if is_inbound:
                    contact_stats[addr]["received"] += 1
                else:
                    contact_stats[addr]["sent"] += 1

            if not contact_stats:
                return ""

            # Sort by total interactions descending, take top 5
            sorted_contacts = sorted(
                contact_stats.items(),
                key=lambda x: -(x[1]["received"] + x[1]["sent"]),
            )[:5]

            lines = ["Active contacts this week (from event log):"]
            for addr, stats in sorted_contacts:
                total = stats["received"] + stats["sent"]
                lines.append(
                    f"- {addr}: {total} interactions "
                    f"({stats['received']} received, {stats['sent']} sent)"
                )

            return "\n".join(lines)
        except Exception:
            logger.warning(
                "context: _get_contact_activity_summary failed, skipping",
                exc_info=True,
            )
            return ""
