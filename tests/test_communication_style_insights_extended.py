"""
Tests for extended _communication_style_insights and linguistic averages fix.

Covers two changes shipped together:

1. ``LinguisticExtractor._update_profile`` now includes ``question_rate``,
   ``ellipsis_rate``, and ``unique_word_ratio`` in the computed averages dict
   (previously these were calculated per-sample but never aggregated).

2. ``InsightEngine._communication_style_insights`` now generates up to five
   complementary insights (formality, question rate, hedge rate, emoji rate,
   vocabulary diversity) instead of only the single formality insight.
"""

from __future__ import annotations

import pytest

from services.signal_extractor.linguistic import LinguisticExtractor
from services.insight_engine.engine import InsightEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prime_ums_with_linguistic_profile(ums, averages: dict, samples_count: int = 10):
    """Store a synthetic linguistic profile so InsightEngine can read it.

    Calls update_signal_profile ``samples_count`` times to ensure the stored
    ``samples_count`` column reaches the desired value (the storage layer
    increments it by 1 on every call).
    """
    profile_data = {"averages": averages, "samples": [], "per_contact": {}}
    for _ in range(samples_count):
        ums.update_signal_profile("linguistic", profile_data)


def _make_engine(db, ums, averages: dict, samples_count: int = 10) -> InsightEngine:
    """Inject a synthetic linguistic profile and return a wired InsightEngine."""
    _prime_ums_with_linguistic_profile(ums, averages, samples_count=samples_count)
    return InsightEngine(db, ums)


def _make_outbound_event(body: str) -> dict:
    """Minimal email.sent event with the given body text."""
    return {
        "type": "email.sent",
        "source": "test",
        "timestamp": "2026-02-19T10:00:00Z",
        "priority": "normal",
        "payload": {
            "body": body,
            "to_addresses": ["alice@example.com"],
            "channel": "email",
        },
    }


# ===========================================================================
# Part 1 — LinguisticExtractor averages completeness
# ===========================================================================

class TestLinguisticAveragesCompleteness:
    """Verify that _update_profile now includes question_rate, ellipsis_rate,
    and unique_word_ratio in the aggregated averages dict."""

    def test_question_rate_in_averages(self, db, user_model_store):
        """question_rate must appear in the averages dict after processing a message."""
        extractor = LinguisticExtractor(db, user_model_store)
        body = "Do you agree? What do you think? Should we proceed?"
        extractor.extract(_make_outbound_event(body))

        profile = user_model_store.get_signal_profile("linguistic")
        assert profile is not None
        averages = profile["data"].get("averages", {})
        assert "question_rate" in averages, (
            "question_rate should be aggregated into averages — "
            "it was computed per-sample but missing from the averages dict before this fix"
        )
        assert averages["question_rate"] >= 0.0

    def test_ellipsis_rate_in_averages(self, db, user_model_store):
        """ellipsis_rate must appear in the averages dict after processing a message."""
        extractor = LinguisticExtractor(db, user_model_store)
        body = "I was thinking... maybe we should reconsider... not sure though..."
        extractor.extract(_make_outbound_event(body))

        profile = user_model_store.get_signal_profile("linguistic")
        averages = profile["data"].get("averages", {})
        assert "ellipsis_rate" in averages, (
            "ellipsis_rate should be aggregated into averages"
        )
        assert averages["ellipsis_rate"] >= 0.0

    def test_unique_word_ratio_in_averages(self, db, user_model_store):
        """unique_word_ratio (vocabulary diversity) must appear in averages."""
        extractor = LinguisticExtractor(db, user_model_store)
        body = "The quick brown fox jumped over the lazy dog near the river bank."
        extractor.extract(_make_outbound_event(body))

        profile = user_model_store.get_signal_profile("linguistic")
        averages = profile["data"].get("averages", {})
        assert "unique_word_ratio" in averages, (
            "unique_word_ratio should be aggregated into averages"
        )
        assert 0.0 < averages["unique_word_ratio"] <= 1.0

    def test_question_rate_value_reflects_actual_questions(self, db, user_model_store):
        """question_rate average should be higher for question-heavy messages."""
        extractor = LinguisticExtractor(db, user_model_store)
        # Message with many question marks
        body = "Did you get my email? When can we meet? Is Thursday OK? What time works?"
        extractor.extract(_make_outbound_event(body))

        profile = user_model_store.get_signal_profile("linguistic")
        averages = profile["data"]["averages"]
        assert averages["question_rate"] > 0.5, (
            "A message consisting almost entirely of questions should produce "
            "a question_rate well above 0.5 questions/sentence"
        )

    def test_existing_averages_still_present(self, db, user_model_store):
        """Adding new fields must not remove existing averages (non-regression)."""
        extractor = LinguisticExtractor(db, user_model_store)
        body = "Let us proceed with the plan. I think it is the best course of action."
        extractor.extract(_make_outbound_event(body))

        profile = user_model_store.get_signal_profile("linguistic")
        averages = profile["data"]["averages"]
        for key in ("avg_sentence_length", "formality", "hedge_rate",
                    "assertion_rate", "exclamation_rate", "emoji_rate", "profanity_rate"):
            assert key in averages, f"Existing average '{key}' should still be present"

    def test_averages_recomputed_from_multiple_samples(self, db, user_model_store):
        """Averages must be computed across the full sample window."""
        extractor = LinguisticExtractor(db, user_model_store)
        # First message: no questions
        extractor.extract(_make_outbound_event(
            "This is a declarative statement. It contains no questions at all."
        ))
        # Second message: all questions
        extractor.extract(_make_outbound_event(
            "Are you ready? Is the report done? Did you review the draft?"
        ))

        profile = user_model_store.get_signal_profile("linguistic")
        averages = profile["data"]["averages"]
        # Average of 0 questions + ~1 question/sentence should be somewhere in-between
        assert 0.0 < averages["question_rate"] < 1.5, (
            "question_rate average should reflect the mean across both samples"
        )


# ===========================================================================
# Part 2 — InsightEngine._communication_style_insights extensions
# ===========================================================================

class TestCommunicationStyleInsightsExtended:
    """Verify the five insight types that _communication_style_insights now generates."""

    def test_formality_insight_formal(self, db, user_model_store):
        """A high formality score should produce a 'formal' insight."""
        engine = _make_engine(db, user_model_store, {"formality": 0.8}, 10)
        insights = engine._communication_style_insights()
        formality_insights = [i for i in insights if "formal" in i.summary.lower()]
        assert formality_insights, "Should produce a formality insight for high formality"
        assert formality_insights[0].entity == "formal"

    def test_formality_insight_casual(self, db, user_model_store):
        """A low formality score should produce a 'casual' insight."""
        engine = _make_engine(db, user_model_store, {"formality": 0.2}, 10)
        insights = engine._communication_style_insights()
        casual = [i for i in insights if "casual" in i.summary.lower()]
        assert casual, "Should produce a casual insight for low formality"

    def test_inquisitive_insight_high_question_rate(self, db, user_model_store):
        """A question_rate >= 0.35 should surface an inquisitive-style insight."""
        engine = _make_engine(
            db, user_model_store,
            {"formality": 0.5, "question_rate": 0.40},
            samples_count=10,
        )
        insights = engine._communication_style_insights()
        inquisitive = [
            i for i in insights
            if "questions" in i.summary.lower() or "inquisitive" in i.summary.lower()
        ]
        assert inquisitive, (
            "question_rate=0.40 (>= 0.35 threshold) should produce an inquisitive insight"
        )
        assert inquisitive[0].entity == "inquisitive"

    def test_declarative_insight_low_question_rate(self, db, user_model_store):
        """A question_rate <= 0.05 should surface a declarative-style insight."""
        engine = _make_engine(
            db, user_model_store,
            {"formality": 0.5, "question_rate": 0.02},
            samples_count=10,
        )
        insights = engine._communication_style_insights()
        declarative = [i for i in insights if "declarative" in i.summary.lower()]
        assert declarative, (
            "question_rate=0.02 (<= 0.05 threshold) should produce a declarative insight"
        )

    def test_no_question_insight_mid_range(self, db, user_model_store):
        """A mid-range question_rate (0.20) should NOT fire either question insight."""
        engine = _make_engine(
            db, user_model_store,
            {"formality": 0.5, "question_rate": 0.20},
            samples_count=10,
        )
        insights = engine._communication_style_insights()
        mid_insights = [i for i in insights if i.entity in ("inquisitive", "declarative")]
        assert not mid_insights, (
            "question_rate=0.20 is between thresholds and should not produce a question insight"
        )

    def test_tentative_insight_high_hedge_rate(self, db, user_model_store):
        """A hedge_rate >= 0.5 should surface a tentative-phrasing insight."""
        engine = _make_engine(
            db, user_model_store,
            {"formality": 0.5, "hedge_rate": 0.6},
            samples_count=10,
        )
        insights = engine._communication_style_insights()
        tentative = [i for i in insights if i.entity == "tentative"]
        assert tentative, "hedge_rate=0.6 should produce a tentative insight"
        assert (
            "hedge" in tentative[0].summary.lower()
            or "maybe" in tentative[0].summary.lower()
        )

    def test_confident_insight_low_hedge_rate(self, db, user_model_store):
        """A hedge_rate <= 0.05 should surface a confident-style insight."""
        engine = _make_engine(
            db, user_model_store,
            {"formality": 0.5, "hedge_rate": 0.02},
            samples_count=10,
        )
        insights = engine._communication_style_insights()
        confident = [i for i in insights if i.entity == "confident"]
        assert confident, "hedge_rate=0.02 should produce a confident insight"

    def test_emoji_insight_high_emoji_rate(self, db, user_model_store):
        """An emoji_rate >= 0.05 should surface an expressive-emoji insight."""
        engine = _make_engine(
            db, user_model_store,
            {"formality": 0.3, "emoji_rate": 0.08},
            samples_count=10,
        )
        insights = engine._communication_style_insights()
        emoji_insights = [i for i in insights if i.entity == "expressive"]
        assert emoji_insights, "emoji_rate=0.08 should produce an expressive insight"

    def test_no_emoji_insight_low_rate(self, db, user_model_store):
        """A low emoji_rate (< 0.05) must not produce an emoji insight."""
        engine = _make_engine(
            db, user_model_store,
            {"formality": 0.5, "emoji_rate": 0.01},
            samples_count=10,
        )
        insights = engine._communication_style_insights()
        expressive = [i for i in insights if i.entity == "expressive"]
        assert not expressive, "emoji_rate=0.01 should not produce an expressive insight"

    def test_rich_vocabulary_insight(self, db, user_model_store):
        """A unique_word_ratio >= 0.75 should surface a rich-vocabulary insight."""
        engine = _make_engine(
            db, user_model_store,
            {"formality": 0.5, "unique_word_ratio": 0.80},
            samples_count=10,
        )
        insights = engine._communication_style_insights()
        rich = [i for i in insights if i.entity == "rich_vocabulary"]
        assert rich, "unique_word_ratio=0.80 should produce a rich_vocabulary insight"

    def test_focused_vocabulary_insight(self, db, user_model_store):
        """A unique_word_ratio <= 0.40 should surface a focused-vocabulary insight."""
        engine = _make_engine(
            db, user_model_store,
            {"formality": 0.5, "unique_word_ratio": 0.35},
            samples_count=10,
        )
        insights = engine._communication_style_insights()
        focused = [i for i in insights if i.entity == "focused_vocabulary"]
        assert focused, "unique_word_ratio=0.35 should produce a focused_vocabulary insight"

    def test_minimum_samples_gate(self, db, user_model_store):
        """With fewer than 3 samples, no insights should be generated."""
        _prime_ums_with_linguistic_profile(
            user_model_store,
            {"formality": 0.8},
            samples_count=2,
        )
        engine = InsightEngine(db, user_model_store)
        insights = engine._communication_style_insights()
        assert not insights, "Fewer than 3 samples should produce no insights"

    def test_additional_insights_require_5_samples(self, db, user_model_store):
        """With exactly 3-4 samples, only the formality insight should fire."""
        _prime_ums_with_linguistic_profile(
            user_model_store,
            {
                "formality": 0.8,
                "question_rate": 0.5,
                "hedge_rate": 0.6,
                "emoji_rate": 0.1,
                "unique_word_ratio": 0.8,
            },
            samples_count=4,
        )
        engine = InsightEngine(db, user_model_store)
        insights = engine._communication_style_insights()
        # Only the formality insight fires; all additional ones require >= 5 samples
        assert len(insights) == 1, (
            "With 4 samples only the formality insight should fire; "
            "additional insights require >= 5 samples"
        )
        assert "formal" in insights[0].summary.lower()

    def test_multiple_insights_generated_simultaneously(self, db, user_model_store):
        """All five insight dimensions should fire when all thresholds are met."""
        engine = _make_engine(
            db, user_model_store,
            {
                "formality": 0.8,           # -> formal
                "question_rate": 0.40,       # -> inquisitive
                "hedge_rate": 0.6,           # -> tentative
                "emoji_rate": 0.08,          # -> expressive
                "unique_word_ratio": 0.80,   # -> rich_vocabulary
            },
            samples_count=20,
        )
        insights = engine._communication_style_insights()
        entities = {i.entity for i in insights}
        assert "formal" in entities
        assert "inquisitive" in entities
        assert "tentative" in entities
        assert "expressive" in entities
        assert "rich_vocabulary" in entities
        assert len(insights) == 5, "All five style dimensions should fire"

    def test_all_insights_have_category_communication_style(self, db, user_model_store):
        """Every generated insight must have category='communication_style'."""
        engine = _make_engine(
            db, user_model_store,
            {
                "formality": 0.5,
                "question_rate": 0.40,
                "hedge_rate": 0.6,
            },
            samples_count=10,
        )
        insights = engine._communication_style_insights()
        for insight in insights:
            assert insight.category == "communication_style", (
                f"Insight '{insight.entity}' has wrong category: {insight.category}"
            )

    def test_all_insights_have_dedup_key(self, db, user_model_store):
        """Every generated insight must have a non-empty dedup_key."""
        engine = _make_engine(
            db, user_model_store,
            {
                "formality": 0.2,
                "question_rate": 0.40,
                "emoji_rate": 0.10,
            },
            samples_count=10,
        )
        insights = engine._communication_style_insights()
        for insight in insights:
            assert insight.dedup_key, f"Insight '{insight.entity}' is missing a dedup_key"
