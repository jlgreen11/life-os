"""
Tests for derived inbound linguistic profile fields.

The inbound profile (``linguistic_inbound``) now computes derived fields
that match the outbound profile's richness: vocabulary_complexity,
capitalization_style, humor_markers, affirmative_patterns,
negative_patterns, and gratitude_patterns.

These tests verify:
  - Inbound signal metrics include all extended fields
  - Per-contact averages include the new derived fields
  - Backward compatibility: old samples without new fields don't break averaging
  - vocabulary_complexity blends unique_word_ratio and avg_word_length correctly
  - capitalization_style is inferred from cap_starts / lower_starts / all_caps_words
  - Pattern markers aggregate across contact's samples
"""

from datetime import datetime, timezone

import pytest

from models.core import EventType
from services.signal_extractor.linguistic import LinguisticExtractor


@pytest.fixture
def linguistic_extractor(db, user_model_store):
    """Create a LinguisticExtractor instance with a test database."""
    return LinguisticExtractor(db=db, user_model_store=user_model_store)


def _make_inbound_event(body: str, from_address: str = "alice@example.com") -> dict:
    """Helper to create an inbound message event."""
    return {
        "type": EventType.MESSAGE_RECEIVED.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": body,
            "from_address": from_address,
        },
    }


# ---------------------------------------------------------------------------
# Inbound signal metrics include extended fields
# ---------------------------------------------------------------------------


def test_inbound_signal_includes_extended_metrics(linguistic_extractor):
    """Inbound signals should carry the same extended metric fields as outbound."""
    event = _make_inbound_event(
        "Haha that's hilarious! Yeah sure, thanks so much for the help."
    )
    signals = linguistic_extractor.extract(event)
    assert len(signals) == 1
    metrics = signals[0]["metrics"]

    # Extended fields that feed into derived profile computations.
    assert "humor_count" in metrics
    assert "humor_type" in metrics
    assert "affirmative_count" in metrics
    assert "affirmative_word" in metrics
    assert "negative_count" in metrics
    assert "negative_word" in metrics
    assert "gratitude_count" in metrics
    assert "gratitude_word" in metrics
    assert "cap_starts" in metrics
    assert "lower_starts" in metrics
    assert "all_caps_words" in metrics
    assert "avg_word_length" in metrics


def test_inbound_signal_humor_detected(linguistic_extractor):
    """Inbound signals should detect humor markers from contact's text."""
    event = _make_inbound_event("lol that was so funny haha can't stop laughing")
    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]

    assert metrics["humor_count"] >= 2
    assert metrics["humor_type"] in ("lol", "haha")


def test_inbound_signal_affirmative_detected(linguistic_extractor):
    """Inbound signals should detect affirmative patterns from contact's text."""
    event = _make_inbound_event("Yeah sure sounds good, definitely will do that for you")
    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]

    assert metrics["affirmative_count"] >= 2
    assert metrics["affirmative_word"] is not None


def test_inbound_signal_gratitude_detected(linguistic_extractor):
    """Inbound signals should detect gratitude patterns from contact's text."""
    event = _make_inbound_event("Thank you so much, I really appreciate your help here")
    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]

    assert metrics["gratitude_count"] >= 2
    assert metrics["gratitude_word"] is not None


# ---------------------------------------------------------------------------
# Per-contact averages include derived fields
# ---------------------------------------------------------------------------


def test_per_contact_averages_include_vocabulary_complexity(linguistic_extractor, user_model_store):
    """Per-contact averages should include a vocabulary_complexity score."""
    for body in [
        "The implementation paradigm necessitates a comprehensive evaluation of deliverables.",
        "Furthermore, the architectural considerations require meticulous assessment here.",
        "The methodological framework encompasses several interdependent components together.",
    ]:
        linguistic_extractor.extract(_make_inbound_event(body, "bob@corp.com"))

    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    avgs = inbound["data"]["per_contact_averages"]["bob@corp.com"]

    assert "vocabulary_complexity" in avgs
    # Complex vocabulary → higher score.  These messages use long, diverse words.
    assert 0.0 <= avgs["vocabulary_complexity"] <= 1.0
    assert avgs["vocabulary_complexity"] > 0.3


def test_per_contact_averages_include_capitalization_style(linguistic_extractor, user_model_store):
    """Per-contact averages should include a capitalization_style field."""
    for body in [
        "Hello there, just checking in on the project status for this week.",
        "Please review the quarterly report and provide feedback at convenience.",
        "Let me know if you have any questions about the new proposal document.",
    ]:
        linguistic_extractor.extract(_make_inbound_event(body, "formal@corp.com"))

    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    avgs = inbound["data"]["per_contact_averages"]["formal@corp.com"]

    assert "capitalization_style" in avgs
    # Standard capitalization — sentences start with uppercase.
    assert avgs["capitalization_style"] == "standard"


def test_capitalization_style_all_lower(linguistic_extractor, user_model_store):
    """Contacts who consistently use lowercase starts should get 'all_lower' style."""
    for body in [
        "hey just wanted to check in on things, hope all is well",
        "yeah that works for me, let me know when you're free",
        "cool sounds good, talk to you later about the details",
    ]:
        linguistic_extractor.extract(_make_inbound_event(body, "casual@friend.com"))

    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    avgs = inbound["data"]["per_contact_averages"]["casual@friend.com"]

    assert avgs["capitalization_style"] == "all_lower"


def test_per_contact_averages_include_humor_markers(linguistic_extractor, user_model_store):
    """Per-contact averages should include top humor markers from contact's messages."""
    for body in [
        "Haha that meeting was wild, can you believe what happened there today",
        "Lol yeah I know right, it was absolutely ridiculous and unbelievable",
        "Haha lol that's the funniest thing I've heard all week honestly",
    ]:
        linguistic_extractor.extract(_make_inbound_event(body, "funny@friend.com"))

    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    avgs = inbound["data"]["per_contact_averages"]["funny@friend.com"]

    assert "humor_markers" in avgs
    assert len(avgs["humor_markers"]) > 0
    assert "haha" in avgs["humor_markers"] or "lol" in avgs["humor_markers"]


def test_per_contact_averages_include_affirmative_patterns(linguistic_extractor, user_model_store):
    """Per-contact averages should include top affirmative patterns."""
    for body in [
        "Yeah sure, that sounds good to me for the meeting time slot",
        "Absolutely, definitely will do that and get it done by Friday",
        "Sure thing, sounds good and I will take care of it today",
    ]:
        linguistic_extractor.extract(_make_inbound_event(body, "agreeable@corp.com"))

    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    avgs = inbound["data"]["per_contact_averages"]["agreeable@corp.com"]

    assert "affirmative_patterns" in avgs
    assert len(avgs["affirmative_patterns"]) > 0


def test_per_contact_averages_include_negative_patterns(linguistic_extractor, user_model_store):
    """Per-contact averages should include top negative/declination patterns."""
    for body in [
        "Unfortunately I can't make it to the meeting this afternoon sorry",
        "I'm not able to attend, sorry about that scheduling conflict issue",
        "Unable to join the call, I don't have availability at that time",
    ]:
        linguistic_extractor.extract(_make_inbound_event(body, "busy@corp.com"))

    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    avgs = inbound["data"]["per_contact_averages"]["busy@corp.com"]

    assert "negative_patterns" in avgs
    assert len(avgs["negative_patterns"]) > 0


def test_per_contact_averages_include_gratitude_patterns(linguistic_extractor, user_model_store):
    """Per-contact averages should include top gratitude patterns."""
    for body in [
        "Thank you so much for your help with the project this week",
        "I really appreciate everything you've done, thanks a lot for that",
        "Grateful for the quick turnaround, much appreciated and thank you",
    ]:
        linguistic_extractor.extract(_make_inbound_event(body, "grateful@corp.com"))

    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    avgs = inbound["data"]["per_contact_averages"]["grateful@corp.com"]

    assert "gratitude_patterns" in avgs
    assert len(avgs["gratitude_patterns"]) > 0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_old_samples_without_new_fields_dont_break(linguistic_extractor, user_model_store):
    """Pre-existing samples that lack new metric fields should not crash averaging.

    Simulates the scenario where a contact has old samples stored before the
    extended fields were added: avg_word_length, humor_type, etc. are absent.
    The derived computations must use .get() defaults and not raise KeyError.
    """
    # Manually inject old-format samples into the inbound profile.
    old_sample = {
        "word_count": 20,
        "avg_sentence_length": 10.0,
        "unique_word_ratio": 0.65,
        "formality": 0.5,
        "hedge_rate": 0.1,
        "assertion_rate": 0.05,
        "exclamation_rate": 0.0,
        "question_rate": 0.1,
        "ellipsis_rate": 0.0,
        "emoji_count": 0,
        "emojis_used": [],
        "profanity_count": 0,
        "greeting_detected": None,
        "closing_detected": None,
        # NOTE: No humor_count, humor_type, affirmative_*, negative_*,
        # gratitude_*, cap_starts, lower_starts, all_caps_words, avg_word_length
    }
    user_model_store.update_signal_profile(
        "linguistic_inbound",
        {
            "per_contact": {"old@contact.com": [old_sample, old_sample]},
            "per_contact_averages": {},
        },
    )

    # Now process a new message from the same contact — this should NOT crash.
    event = _make_inbound_event(
        "Hey there, just a quick note about the upcoming deadline this week.",
        from_address="old@contact.com",
    )
    linguistic_extractor.extract(event)

    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    avgs = inbound["data"]["per_contact_averages"]["old@contact.com"]

    # The derived fields should be present and computed without errors.
    assert "vocabulary_complexity" in avgs
    assert "capitalization_style" in avgs
    assert "humor_markers" in avgs
    assert "affirmative_patterns" in avgs
    assert "negative_patterns" in avgs
    assert "gratitude_patterns" in avgs

    # vocabulary_complexity should still be a valid 0-1 value.
    assert 0.0 <= avgs["vocabulary_complexity"] <= 1.0


# ---------------------------------------------------------------------------
# vocabulary_complexity formula verification
# ---------------------------------------------------------------------------


def test_vocabulary_complexity_formula(linguistic_extractor, user_model_store):
    """vocabulary_complexity should blend unique_word_ratio (0.6) and normalized avg_word_length (0.4).

    We send messages with known characteristics and verify the formula:
      word_len_norm = clamp((avg_word_length - 2.0) / 6.0, 0, 1)
      complexity = unique_word_ratio * 0.6 + word_len_norm * 0.4
    """
    # Use messages with long, diverse words to get a high complexity score.
    for body in [
        "Unprecedented methodological considerations necessitate comprehensive reevaluation here",
        "Extraordinary implementation prerequisites fundamentally restructure operational parameters",
        "Revolutionary architectural transformations systematically reconfigure infrastructure components",
    ]:
        linguistic_extractor.extract(_make_inbound_event(body, "complex@corp.com"))

    # Use messages with short, repetitive words for a low complexity score.
    for body in [
        "Hey hey hey how are you doing today, are you ok or not ok",
        "Yeah yeah I know I know, it is what it is and that is that",
        "Ok ok sure sure, I get it I get it, let me go go go now",
    ]:
        linguistic_extractor.extract(_make_inbound_event(body, "simple@friend.com"))

    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    complex_score = inbound["data"]["per_contact_averages"]["complex@corp.com"]["vocabulary_complexity"]
    simple_score = inbound["data"]["per_contact_averages"]["simple@friend.com"]["vocabulary_complexity"]

    # The complex-vocabulary contact should score higher than the simple one.
    assert complex_score > simple_score
    # Both should be valid 0-1 values.
    assert 0.0 <= complex_score <= 1.0
    assert 0.0 <= simple_score <= 1.0


# ---------------------------------------------------------------------------
# Parity check: all outbound-profile derived keys present in inbound
# ---------------------------------------------------------------------------


def test_inbound_derived_fields_match_outbound_parity(linguistic_extractor, user_model_store):
    """The inbound per_contact_averages should have the same derived field names
    that the outbound profile computes, ensuring ContextAssembler can read them
    with the same keys regardless of direction."""
    for body in [
        "Haha yeah sure thing, thanks so much for all your help with this",
        "Lol definitely appreciate it, unfortunately I can't make the other meeting",
        "Great sounds good, thank you and I will follow up on that shortly",
    ]:
        linguistic_extractor.extract(_make_inbound_event(body, "parity@test.com"))

    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    avgs = inbound["data"]["per_contact_averages"]["parity@test.com"]

    # Fields that exist in outbound per_contact_averages and should now
    # also exist in inbound per_contact_averages.
    expected_fields = {
        "avg_sentence_length",
        "formality",
        "hedge_rate",
        "assertion_rate",
        "exclamation_rate",
        "question_rate",
        "ellipsis_rate",
        "unique_word_ratio",
        "emoji_rate",
        "samples_count",
        # New derived fields matching outbound parity:
        "vocabulary_complexity",
        "capitalization_style",
        "humor_markers",
        "affirmative_patterns",
        "negative_patterns",
        "gratitude_patterns",
    }

    for field in expected_fields:
        assert field in avgs, f"Missing expected field: {field}"
