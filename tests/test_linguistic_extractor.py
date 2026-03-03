"""
Tests for the LinguisticExtractor service.

The LinguisticExtractor builds the user's linguistic profile by analyzing
outbound messages (emails, texts, voice commands) and extracting patterns
in vocabulary, formality, emotional tone, and communication conventions.

These tests verify:
  - Correct event filtering (only outbound messages)
  - Accurate metric calculation (sentence length, vocabulary richness, etc.)
  - Pattern detection (hedge words, assertions, profanity, emojis)
  - Greeting/closing extraction
  - Profile persistence and incremental updates
  - Per-contact style tracking
"""

from datetime import datetime, timezone

import pytest

from models.core import EventType
from services.signal_extractor.linguistic import LinguisticExtractor


@pytest.fixture
def linguistic_extractor(db, user_model_store):
    """Create a LinguisticExtractor instance with a test database."""
    return LinguisticExtractor(db=db, user_model_store=user_model_store)


# ---------------------------------------------------------------------------
# Event Filtering Tests
# ---------------------------------------------------------------------------


def test_can_process_outbound_email(linguistic_extractor):
    """Verify that sent emails are processed."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "payload": {"body": "Hello world"},
    }
    assert linguistic_extractor.can_process(event) is True


def test_can_process_outbound_message(linguistic_extractor):
    """Verify that sent messages are processed."""
    event = {
        "type": EventType.MESSAGE_SENT.value,
        "payload": {"body": "Hello world"},
    }
    assert linguistic_extractor.can_process(event) is True


def test_can_process_voice_command(linguistic_extractor):
    """Verify that voice commands are processed."""
    event = {
        "type": "system.user.command",
        "payload": {"body": "Hello world"},
    }
    assert linguistic_extractor.can_process(event) is True


def test_accepts_inbound_email(linguistic_extractor):
    """Verify that received emails are processed for per-contact incoming style profiling."""
    event = {
        "type": EventType.EMAIL_RECEIVED.value,
        "payload": {"body": "Hello world"},
    }
    assert linguistic_extractor.can_process(event) is True


def test_accepts_inbound_message(linguistic_extractor):
    """Verify that received messages are processed for per-contact incoming style profiling."""
    event = {
        "type": EventType.MESSAGE_RECEIVED.value,
        "payload": {"body": "Hello world"},
    }
    assert linguistic_extractor.can_process(event) is True


# ---------------------------------------------------------------------------
# Basic Metric Calculation Tests
# ---------------------------------------------------------------------------


def test_extract_basic_metrics(linguistic_extractor):
    """Verify word count, sentence count, and vocabulary richness calculations."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Hello there. How are you doing? I hope everything is going well!",
            "to_addresses": ["test@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    assert len(signals) == 1

    metrics = signals[0]["metrics"]
    assert metrics["word_count"] == 12
    # 3 sentences -> avg length ~4 words
    assert 3.5 <= metrics["avg_sentence_length"] <= 4.5
    # All unique words -> high ratio
    assert metrics["unique_word_ratio"] >= 0.9


def test_skips_very_short_messages(linguistic_extractor):
    """Verify that messages with fewer than 3 words are skipped."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"body": "OK"},
    }

    signals = linguistic_extractor.extract(event)
    assert len(signals) == 0


def test_skips_empty_body(linguistic_extractor):
    """Verify that messages with no text are skipped."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"body": ""},
    }

    signals = linguistic_extractor.extract(event)
    assert len(signals) == 0


def test_falls_back_to_body_plain(linguistic_extractor):
    """Verify that body_plain is used when body is missing."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body_plain": "This is plain text content for testing purposes.",
            "to_addresses": ["test@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    assert len(signals) == 1
    assert signals[0]["metrics"]["word_count"] == 8


# ---------------------------------------------------------------------------
# Formality Estimation Tests
# ---------------------------------------------------------------------------


def test_detects_informal_language(linguistic_extractor):
    """Verify that informal markers reduce formality score."""
    event = {
        "type": EventType.MESSAGE_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Hey yeah gonna check that out lol haha ok cool btw idk",
            "to_addresses": ["friend@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]
    # All informal markers -> low formality
    assert metrics["formality"] < 0.3


def test_detects_formal_language(linguistic_extractor):
    """Verify that formal markers increase formality score."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Regarding your request, furthermore I would like to add that "
            "therefore we should proceed accordingly. Sincerely respectfully submitted.",
            "to_addresses": ["boss@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]
    # All formal markers -> high formality
    assert metrics["formality"] > 0.7


def test_neutral_formality_default(linguistic_extractor):
    """Verify that messages without formal/informal markers default to 0.5."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "This is a neutral message without special markers for testing.",
            "to_addresses": ["test@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]
    assert metrics["formality"] == 0.5


# ---------------------------------------------------------------------------
# Pattern Detection Tests
# ---------------------------------------------------------------------------


def test_detects_hedge_words(linguistic_extractor):
    """Verify that hedge patterns (tentativeness) are counted correctly."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Maybe we could try this. I think it might work. Perhaps not sure though.",
            "to_addresses": ["test@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]
    # Contains: maybe, I think, might, perhaps, not sure
    # 3 sentences -> rate should be ~1.67 per sentence
    assert metrics["hedge_rate"] >= 1.5


def test_detects_assertions(linguistic_extractor):
    """Verify that assertion patterns (confidence) are counted correctly."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "We need to act now. This must happen. Definitely the right approach. "
            "Clearly obvious without a doubt.",
            "to_addresses": ["test@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]
    # Contains: we need to, must, definitely, clearly, without a doubt
    # 4 sentences -> rate should be at least 1.0 per sentence
    assert metrics["assertion_rate"] >= 1.0


def test_detects_profanity(linguistic_extractor):
    """Verify that profanity is counted (for emotional intensity tracking, not censorship)."""
    event = {
        "type": EventType.MESSAGE_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "This is damn frustrating. What the hell happened. Shit.",
            "to_addresses": ["friend@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]
    # Contains: damn, hell, shit
    assert metrics["profanity_count"] == 3


def test_detects_punctuation_patterns(linguistic_extractor):
    """Verify that exclamation marks, questions, and ellipses are counted."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Really! Amazing! What? Why? Not sure... Maybe...",
            "to_addresses": ["test@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]
    # 2 exclamations, 2 questions, 2 ellipses across 6 sentence fragments
    assert metrics["exclamation_rate"] > 0
    assert metrics["question_rate"] > 0
    assert metrics["ellipsis_rate"] > 0


def test_detects_emojis(linguistic_extractor):
    """Verify that emojis are counted and extracted."""
    event = {
        "type": EventType.MESSAGE_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Hey! 😊 Looking forward to this 🎉🎉",
            "to_addresses": ["friend@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]
    # The emoji pattern matches sequences, so consecutive emojis may be grouped
    assert metrics["emoji_count"] >= 2
    assert len(metrics["emojis_used"]) >= 2


# ---------------------------------------------------------------------------
# Greeting and Closing Detection Tests
# ---------------------------------------------------------------------------


def test_detects_greeting(linguistic_extractor):
    """Verify that common greetings are detected at the start of messages."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Hey there! How are you doing today? Hope all is well.",
            "to_addresses": ["test@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]
    assert metrics["greeting_detected"] == "hey"


def test_detects_closing(linguistic_extractor):
    """Verify that common closings are detected at the end of messages."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Please let me know if you have any questions. Thanks for your help!",
            "to_addresses": ["test@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]
    assert metrics["closing_detected"] == "thanks"


def test_no_greeting_when_absent(linguistic_extractor):
    """Verify that None is returned when no greeting is present."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Just wanted to follow up on our previous conversation about the project.",
            "to_addresses": ["test@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    metrics = signals[0]["metrics"]
    assert metrics["greeting_detected"] is None


# ---------------------------------------------------------------------------
# Profile Persistence and Update Tests
# ---------------------------------------------------------------------------


def test_creates_initial_profile(linguistic_extractor, user_model_store):
    """Verify that the first message creates a new linguistic profile."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Hello there. This is my first message for testing purposes.",
            "to_addresses": ["test@example.com"],
        },
    }

    linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    assert profile is not None
    assert len(profile["data"]["samples"]) == 1
    assert "averages" in profile["data"]


def test_updates_profile_incrementally(linguistic_extractor, user_model_store):
    """Verify that subsequent messages append to the profile."""
    events = [
        {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": f"Message number {i} with various words for testing.",
                "to_addresses": ["test@example.com"],
            },
        }
        for i in range(3)
    ]

    for event in events:
        linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    assert len(profile["data"]["samples"]) == 3


def test_caps_samples_at_500(linguistic_extractor, user_model_store):
    """Verify that the sample buffer is capped at 500 to prevent unbounded growth."""
    # Create 510 messages
    for i in range(510):
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": f"Sample message {i} for buffer overflow testing purposes here.",
                "to_addresses": ["test@example.com"],
            },
        }
        linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    # Should keep only the most recent 500
    assert len(profile["data"]["samples"]) == 500


def test_tracks_per_contact_style(linguistic_extractor, user_model_store):
    """Verify that style is tracked separately per recipient."""
    # Send formal message to boss
    event_formal = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Regarding your request, I will proceed accordingly. Respectfully submitted.",
            "to_addresses": ["boss@company.com"],
        },
    }

    # Send informal message to friend
    event_informal = {
        "type": EventType.MESSAGE_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Hey yeah gonna hang out later lol cool btw",
            "to_addresses": ["friend@example.com"],
        },
    }

    linguistic_extractor.extract(event_formal)
    linguistic_extractor.extract(event_informal)

    profile = user_model_store.get_signal_profile("linguistic")
    assert "boss@company.com" in profile["data"]["per_contact"]
    assert "friend@example.com" in profile["data"]["per_contact"]
    assert len(profile["data"]["per_contact"]["boss@company.com"]) == 1
    assert len(profile["data"]["per_contact"]["friend@example.com"]) == 1


def test_caps_per_contact_samples_at_100(linguistic_extractor, user_model_store):
    """Verify that per-contact sample buffers are capped at 100."""
    for i in range(110):
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": f"Message {i} to same contact for buffer testing.",
                "to_addresses": ["test@example.com"],
            },
        }
        linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    # Should keep only the most recent 100 per contact
    assert len(profile["data"]["per_contact"]["test@example.com"]) == 100


def test_computes_running_averages(linguistic_extractor, user_model_store):
    """Verify that averages are recomputed with each update."""
    # Send 3 messages with varying formality
    events = [
        {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "Hey yeah gonna do this cool btw lol",  # Very informal
                "to_addresses": ["test@example.com"],
            },
        },
        {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "Regarding your request, I will proceed accordingly.",  # Very formal
                "to_addresses": ["test@example.com"],
            },
        },
        {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "This is a neutral message without markers.",  # Neutral
                "to_addresses": ["test@example.com"],
            },
        },
    ]

    for event in events:
        linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    averages = profile["data"]["averages"]

    # Verify that all expected average fields are present
    assert "formality" in averages
    assert "hedge_rate" in averages
    assert "assertion_rate" in averages
    assert "emoji_rate" in averages
    # Average formality should be somewhere between 0 and 1
    assert 0.0 <= averages["formality"] <= 1.0


def test_tracks_common_greetings(linguistic_extractor, user_model_store):
    """Verify that the most common greetings are surfaced in the profile."""
    # Send messages with various greetings
    greetings = ["hey", "hey", "hey", "hello", "hello", "hi"]
    for greeting in greetings:
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": f"{greeting} there! How are you doing today? All good here.",
                "to_addresses": ["test@example.com"],
            },
        }
        linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    common_greetings = profile["data"]["common_greetings"]

    # Top 3 should be: hey (3x), hello (2x), hi (1x)
    assert "hey" in common_greetings
    assert len(common_greetings) <= 3


def test_tracks_common_closings(linguistic_extractor, user_model_store):
    """Verify that the most common closings are surfaced in the profile."""
    # Send messages with various closings
    closings = ["thanks", "thanks", "best", "best", "cheers"]
    for closing in closings:
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": f"Let me know what you think. {closing} for all your help!",
                "to_addresses": ["test@example.com"],
            },
        }
        linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    common_closings = profile["data"]["common_closings"]

    # Top 3 should include thanks and best
    assert "thanks" in common_closings or "best" in common_closings
    assert len(common_closings) <= 3


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


def test_handles_missing_to_addresses(linguistic_extractor):
    """Verify that messages without recipients are still processed."""
    event = {
        "type": "system.user.command",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "This is a voice command without a recipient.",
        },
    }

    signals = linguistic_extractor.extract(event)
    assert len(signals) == 1
    assert signals[0]["contact_id"] is None


def test_handles_single_word_sentences(linguistic_extractor):
    """Verify that single-word sentences don't break calculations."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Hello. Yes. Absolutely. Done.",
            "to_addresses": ["test@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    assert len(signals) == 1
    metrics = signals[0]["metrics"]
    # 4 sentences, 4 words -> avg = 1.0
    assert metrics["avg_sentence_length"] == 1.0


def test_handles_no_sentences(linguistic_extractor):
    """Verify that text without sentence-ending punctuation is handled gracefully."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "just some words without punctuation marks",
            "to_addresses": ["test@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    # Should still process even without explicit sentence boundaries
    assert len(signals) == 1


# ---------------------------------------------------------------------------
# Inbound Message Analysis Tests
# ---------------------------------------------------------------------------


def test_inbound_email_stores_to_inbound_profile(linguistic_extractor, user_model_store):
    """Inbound messages should populate the linguistic_inbound profile, not the outbound one."""
    event = {
        "type": EventType.EMAIL_RECEIVED.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Dear colleague, regarding the quarterly report, I need your input.",
            "from_address": "bob@clientcorp.com",
            "to_addresses": ["me@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    assert len(signals) == 1
    assert signals[0]["direction"] == "inbound"
    assert signals[0]["contact_id"] == "bob@clientcorp.com"

    # Inbound profile should be populated
    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    assert inbound is not None
    assert "bob@clientcorp.com" in inbound["data"]["per_contact"]
    assert len(inbound["data"]["per_contact"]["bob@clientcorp.com"]) == 1

    # Outbound profile should NOT be affected
    outbound = user_model_store.get_signal_profile("linguistic")
    assert outbound is None


def test_inbound_message_builds_per_contact_averages(linguistic_extractor, user_model_store):
    """Multiple inbound messages from the same contact should produce running averages."""
    for body in [
        "Dear team, regarding the budget, please review accordingly.",
        "Furthermore, the quarterly projections require your attention sincerely.",
    ]:
        event = {
            "type": EventType.MESSAGE_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": body,
                "from_address": "+15551234567",
            },
        }
        linguistic_extractor.extract(event)

    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    averages = inbound["data"]["per_contact_averages"]["+15551234567"]
    assert averages["samples_count"] == 2
    # Both messages are formal → formality average should be > 0.5
    assert averages["formality"] > 0.5


def test_outbound_still_uses_original_profile(linguistic_extractor, user_model_store):
    """Outbound messages should still go to the 'linguistic' profile, unchanged."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Hey yeah this is totally casual btw lol haha",
            "to_addresses": ["friend@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    assert signals[0]["direction"] == "outbound"

    outbound = user_model_store.get_signal_profile("linguistic")
    assert outbound is not None
    assert len(outbound["data"]["samples"]) == 1

    # Inbound profile should NOT be affected
    inbound = user_model_store.get_signal_profile("linguistic_inbound")
    assert inbound is None


def test_inbound_contact_resolved_from_from_address(linguistic_extractor):
    """Inbound signals should use from_address as contact_id, not to_addresses."""
    event = {
        "type": EventType.EMAIL_RECEIVED.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "Hello there, just checking in on the project status update.",
            "from_address": "alice@example.com",
            "to_addresses": ["me@example.com"],
        },
    }

    signals = linguistic_extractor.extract(event)
    assert signals[0]["contact_id"] == "alice@example.com"


# ---------------------------------------------------------------------------
# style_by_contact / samples_analyzed / last_updated Tests
# ---------------------------------------------------------------------------


def test_style_by_contact_populated(linguistic_extractor, user_model_store):
    """Per-contact averages should be wired to style_by_contact after enough samples."""
    contact = "alice@example.com"
    bodies = [
        "Hey yeah gonna check that out lol haha ok cool btw idk stuff",
        "Hey nah gonna skip that one lol haha ok cool btw idk things",
        "Hey yep gonna do it now lol haha ok cool btw idk more things",
    ]
    for body in bodies:
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": body,
                "to_addresses": [contact],
            },
        }
        linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    style = profile["data"]["style_by_contact"]

    # The contact should appear in style_by_contact since we sent 3 messages
    # (meeting the _MIN_PER_CONTACT_SAMPLES threshold).
    assert contact in style
    # Verify expected metric keys are present.
    assert "formality" in style[contact]
    assert "hedge_rate" in style[contact]
    assert "emoji_rate" in style[contact]
    assert "samples_count" in style[contact]
    assert style[contact]["samples_count"] == 3


def test_samples_analyzed_tracks_count(linguistic_extractor, user_model_store):
    """samples_analyzed should reflect the current number of samples in the ring buffer."""
    n_messages = 5
    for i in range(n_messages):
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": f"Message number {i} with enough words for processing.",
                "to_addresses": ["test@example.com"],
            },
        }
        linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    assert profile["data"]["samples_analyzed"] == n_messages


def test_last_updated_set(linguistic_extractor, user_model_store):
    """last_updated should be a valid ISO datetime string set within the last few seconds."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": "A message to trigger the profile update and set last_updated.",
            "to_addresses": ["test@example.com"],
        },
    }
    before = datetime.now(timezone.utc)
    linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    last_updated_str = profile["data"]["last_updated"]

    # Must be a valid ISO datetime.
    parsed = datetime.fromisoformat(last_updated_str)
    # Must be within a few seconds of "now".
    assert (parsed - before).total_seconds() < 5
    assert (parsed - before).total_seconds() >= 0


# ---------------------------------------------------------------------------
# Canonical Field Aliasing Tests
# ---------------------------------------------------------------------------


def test_canonical_pattern_fields_aliased(linguistic_extractor, user_model_store):
    """Profile data must contain both top_* keys and canonical LinguisticProfile field names.

    The LinguisticProfile model expects humor_markers, affirmative_patterns,
    negative_patterns, and gratitude_patterns (without a 'top_' prefix).
    The extractor must write both the top_* keys (for backward compat) and
    the canonical keys (for correct deserialization into LinguisticProfile).
    """
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": (
                "Haha that's hilarious! Yeah sure sounds good, thanks so much. "
                "Unfortunately I can't make it to the other thing. "
                "Lol appreciate it, definitely will do!"
            ),
            "to_addresses": ["test@example.com"],
        },
    }

    linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    data = profile["data"]

    # Both the top_* keys and canonical keys must be present.
    assert "top_humor_markers" in data
    assert "top_affirmative_patterns" in data
    assert "top_negative_patterns" in data
    assert "top_gratitude_patterns" in data

    assert "humor_markers" in data
    assert "affirmative_patterns" in data
    assert "negative_patterns" in data
    assert "gratitude_patterns" in data

    # Canonical keys must have the same values as their top_* counterparts.
    assert data["humor_markers"] == data["top_humor_markers"]
    assert data["affirmative_patterns"] == data["top_affirmative_patterns"]
    assert data["negative_patterns"] == data["top_negative_patterns"]
    assert data["gratitude_patterns"] == data["top_gratitude_patterns"]

    # Since the message contains humor/affirmative/negative/gratitude words,
    # the lists should be non-empty.
    assert len(data["humor_markers"]) > 0
    assert len(data["affirmative_patterns"]) > 0
    assert len(data["negative_patterns"]) > 0
    assert len(data["gratitude_patterns"]) > 0


def test_avg_sentence_length_promoted_to_top_level(linguistic_extractor, user_model_store):
    """avg_sentence_length must be promoted from data['averages'] to data's top level.

    The LinguisticProfile model defines avg_sentence_length with a default of 12.0.
    The extractor computes the real average in data['averages']['avg_sentence_length']
    but must also copy it to data['avg_sentence_length'] so downstream consumers
    (user model summaries, AI context assembly, /api/user-model endpoint) see the
    actual computed value instead of the hardcoded default.
    """
    # Process 3 email events with known text content of varying sentence lengths.
    bodies = [
        # Short sentences: ~3 words per sentence (3 sentences, 9 words)
        "Go now please. Do it fast. Run away quickly.",
        # Medium sentences: ~7 words per sentence (2 sentences, 14 words)
        "The weather today is really quite pleasant. I think we should go outside soon.",
        # Longer sentences: ~12 words per sentence (2 sentences, 24 words)
        "I have been working on this complicated project for several weeks now. "
        "The results are finally starting to look promising after all the effort.",
    ]
    for body in bodies:
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": body,
                "to_addresses": ["test@example.com"],
            },
        }
        linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    data = profile["data"]

    # 1. avg_sentence_length must exist at the top level of data.
    assert "avg_sentence_length" in data, (
        "avg_sentence_length missing from top-level data dict — "
        "it should be promoted from data['averages']"
    )

    # 2. The value must NOT be the LinguisticProfile default of 12.0 —
    #    our test messages have varying but known sentence lengths.
    assert data["avg_sentence_length"] != 12.0, (
        f"avg_sentence_length is {data['avg_sentence_length']} (the default); "
        "the computed average should differ from the hardcoded default"
    )

    # 3. The top-level value must equal the value stored in averages.
    assert data["avg_sentence_length"] == data["averages"]["avg_sentence_length"], (
        "Top-level avg_sentence_length does not match data['averages']['avg_sentence_length']"
    )


def test_profanity_contexts_populated(linguistic_extractor, user_model_store):
    """profanity_contexts should list channels where the user has used profanity."""
    # Send a message with profanity via "signal" source
    event_profane = {
        "type": EventType.MESSAGE_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "signal",
        "payload": {
            "body": "This is so damn frustrating, what the hell happened here.",
            "to_addresses": ["friend@example.com"],
        },
    }

    # Send a clean message via "email" source
    event_clean = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "email",
        "payload": {
            "body": "Please review the attached proposal at your earliest convenience.",
            "to_addresses": ["boss@example.com"],
        },
    }

    linguistic_extractor.extract(event_profane)
    linguistic_extractor.extract(event_clean)

    profile = user_model_store.get_signal_profile("linguistic")
    data = profile["data"]

    assert "profanity_contexts" in data
    # The profane message came from "signal", so that should be a context.
    assert "signal" in data["profanity_contexts"]
    # "email" had no profanity, so it should NOT appear.
    assert "email" not in data["profanity_contexts"]


def test_profanity_contexts_empty_when_no_profanity(linguistic_extractor, user_model_store):
    """profanity_contexts should be empty when the user never swears."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "email",
        "payload": {
            "body": "This is a perfectly polite and professional message for testing.",
            "to_addresses": ["colleague@example.com"],
        },
    }

    linguistic_extractor.extract(event)

    profile = user_model_store.get_signal_profile("linguistic")
    assert profile["data"]["profanity_contexts"] == []
