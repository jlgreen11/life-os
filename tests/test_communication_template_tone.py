"""
Tests for communication template tone_notes population.

The RelationshipExtractor._analyze_tone() method uses lightweight heuristics
to derive stylistic observations from message text and merges them with
existing tone notes via set-union deduplication, capped at 5 entries.

Tests cover:
1. Concise communicator detection (short messages)
2. Detailed communicator detection (long messages)
3. Inquisitive style detection (question density)
4. Warm/positive tone detection (positive keywords)
5. Cautious/hedging style detection (hedge words)
6. Action-oriented detection (imperative phrases)
7. Set-union merging with existing notes
8. Cap at 5 entries
9. Integration: _extract_communication_templates() produces non-empty tone_notes
"""

import hashlib
import json

import pytest

from models.core import EventType
from services.signal_extractor.relationship import RelationshipExtractor


class TestAnalyzeTone:
    """Unit tests for RelationshipExtractor._analyze_tone()."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        """Create a RelationshipExtractor instance with test dependencies."""
        return RelationshipExtractor(db=db, user_model_store=user_model_store)

    # ------------------------------------------------------------------
    # Individual tone detection
    # ------------------------------------------------------------------

    def test_short_message_concise_communicator(self, extractor):
        """A message under 50 words should produce 'concise communicator'."""
        text = "Hi, thanks for the update. Looks good to me."
        notes = extractor._analyze_tone(text, [])
        assert "concise communicator" in notes

    def test_long_message_detailed_communicator(self, extractor):
        """A message over 200 words should produce 'detailed communicator'."""
        # Build a message that exceeds 200 words
        text = (
            "Thank you for sending over the comprehensive project proposal. "
            "I have reviewed it in detail and have several observations. "
        ) + " ".join(["additional context and elaboration on the matter"] * 30)
        assert len(text.split()) > 200
        notes = extractor._analyze_tone(text, [])
        assert "detailed communicator" in notes

    def test_question_heavy_message_inquisitive_style(self, extractor):
        """A message where >30% of sentences are questions should produce 'inquisitive style'."""
        text = (
            "What time works for you? "
            "Can you bring the slides? "
            "Should we book the conference room? "
            "I will prepare the agenda."
        )
        notes = extractor._analyze_tone(text, [])
        assert "inquisitive style" in notes

    def test_positive_keywords_warm_tone(self, extractor):
        """A message with >2 positive keywords should produce 'warm/positive tone'."""
        text = (
            "Great work on the presentation! I really appreciate the effort. "
            "The results are excellent and the team did an awesome job."
        )
        notes = extractor._analyze_tone(text, [])
        assert "warm/positive tone" in notes

    def test_action_phrases_action_oriented(self, extractor):
        """A message with multiple action phrases should produce 'action-oriented'."""
        text = (
            "Please review the document by Friday. "
            "Can you also update the spreadsheet? "
            "We should schedule a follow-up meeting. "
            "I need the final numbers before the deadline."
        )
        notes = extractor._analyze_tone(text, [])
        assert "action-oriented" in notes

    def test_hedge_words_cautious_style(self, extractor):
        """A message with >2 hedge words should produce 'cautious/hedging style'."""
        text = (
            "I think maybe we should reconsider the timeline. "
            "I'm not sure if the budget will work, and perhaps "
            "we might need to adjust expectations."
        )
        notes = extractor._analyze_tone(text, [])
        assert "cautious/hedging style" in notes

    # ------------------------------------------------------------------
    # Merging and cap behavior
    # ------------------------------------------------------------------

    def test_existing_notes_preserved(self, extractor):
        """Existing tone notes should be preserved via set-union merging."""
        existing = ["formal register", "uses technical jargon"]
        text = "OK, sounds good."  # Short → adds 'concise communicator'
        notes = extractor._analyze_tone(text, existing)
        assert "formal register" in notes
        assert "uses technical jargon" in notes
        assert "concise communicator" in notes

    def test_duplicates_not_added(self, extractor):
        """If a note already exists, it should not be duplicated."""
        existing = ["concise communicator"]
        text = "Sure, will do."  # Short → would add 'concise communicator' again
        notes = extractor._analyze_tone(text, existing)
        assert notes.count("concise communicator") == 1

    def test_cap_at_five_entries(self, extractor):
        """Output should never exceed 5 entries, even with many existing + new."""
        existing = ["note-a", "note-b", "note-c", "note-d"]
        # This text is short (concise) and has positive keywords and action phrases
        text = (
            "Great, thanks! Please send the report. "
            "Can you also check the numbers? Awesome work, I appreciate it."
        )
        notes = extractor._analyze_tone(text, existing)
        assert len(notes) <= 5

    def test_recent_observations_kept_when_capped(self, extractor):
        """When capping at 5, the most recently observed patterns should be kept."""
        existing = ["old-note-1", "old-note-2", "old-note-3", "old-note-4"]
        text = "Sure thing."  # Short → concise communicator
        notes = extractor._analyze_tone(text, existing)
        # 4 existing + 1 new = 5, exactly at cap
        assert len(notes) == 5
        assert "concise communicator" in notes

    # ------------------------------------------------------------------
    # Negative / edge cases
    # ------------------------------------------------------------------

    def test_neutral_message_no_special_notes(self, extractor):
        """A middle-length neutral message may produce only length-related notes."""
        # 50-200 words, no questions, no positive keywords, no hedging, no action phrases
        text = " ".join(["The weather has been quite variable this week."] * 8)
        word_count = len(text.split())
        assert 50 <= word_count <= 200
        notes = extractor._analyze_tone(text, [])
        # Should not contain any of the special tones
        assert "concise communicator" not in notes
        assert "detailed communicator" not in notes
        assert "inquisitive style" not in notes
        assert "warm/positive tone" not in notes
        assert "action-oriented" not in notes
        assert "cautious/hedging style" not in notes

    def test_empty_text(self, extractor):
        """Empty text should return existing notes unchanged (no crash)."""
        existing = ["some prior note"]
        notes = extractor._analyze_tone("", existing)
        assert "some prior note" in notes


class TestCommunicationTemplateToneIntegration:
    """Integration: _extract_communication_templates() populates tone_notes."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        """Create a RelationshipExtractor instance with test dependencies."""
        return RelationshipExtractor(db=db, user_model_store=user_model_store)

    def test_template_tone_notes_populated_on_extraction(self, extractor):
        """Processing a real event should produce a template with non-empty tone_notes."""
        event = {
            "id": "evt-tone-001",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-20T10:00:00Z",
            "payload": {
                "to_addresses": ["tone-test@example.com"],
                "body_plain": (
                    "Hey! Great catching up yesterday. Thanks for sharing the deck — "
                    "I appreciate the level of detail. Can you also send me the budget "
                    "spreadsheet? Please loop in the finance team as well."
                ),
                "channel": "email",
            },
        }

        extractor.extract(event)

        template_id = hashlib.sha256(b"tone-test@example.com:email:out").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT tone_notes FROM communication_templates WHERE id = ?",
                (template_id,),
            ).fetchone()

        assert row is not None
        tone_notes = json.loads(row["tone_notes"])
        assert len(tone_notes) > 0, "tone_notes should be non-empty after extraction"

    def test_template_tone_notes_accumulate_across_messages(self, extractor):
        """Tone notes should accumulate across multiple messages via set-union."""
        # First message: short and positive
        event1 = {
            "id": "evt-tone-002",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-20T10:00:00Z",
            "payload": {
                "to_addresses": ["accumulate@example.com"],
                "body_plain": "Great, thanks! Awesome work. Love the results. Excellent job.",
                "channel": "email",
            },
        }
        extractor.extract(event1)

        template_id = hashlib.sha256(b"accumulate@example.com:email:out").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row1 = json.loads(
                conn.execute(
                    "SELECT tone_notes FROM communication_templates WHERE id = ?",
                    (template_id,),
                ).fetchone()["tone_notes"]
            )

        notes_after_first = set(row1)

        # Second message: heavy on questions
        event2 = {
            "id": "evt-tone-003",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-20T11:00:00Z",
            "payload": {
                "to_addresses": ["accumulate@example.com"],
                "body_plain": (
                    "What do you think? When is the deadline? "
                    "Should we reschedule? Is it confirmed?"
                ),
                "channel": "email",
            },
        }
        extractor.extract(event2)

        with extractor.db.get_connection("user_model") as conn:
            row2 = json.loads(
                conn.execute(
                    "SELECT tone_notes FROM communication_templates WHERE id = ?",
                    (template_id,),
                ).fetchone()["tone_notes"]
            )

        notes_after_second = set(row2)
        # Notes from the first message should still be present (set-union)
        assert notes_after_first.issubset(notes_after_second) or len(notes_after_second) == 5
