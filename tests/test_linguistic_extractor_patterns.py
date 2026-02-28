"""
Tests for the extended linguistic pattern detection fields added to LinguisticExtractor.

Covers the LinguisticProfile fields that were previously left at their hardcoded
defaults (humor_markers, affirmative_patterns, negative_patterns, gratitude_patterns,
uses_oxford_comma, capitalization_style, vocabulary_complexity).

Architecture note:
    The LinguisticExtractor stores per-message metrics in a rolling sample buffer
    and recomputes running averages on every update.  The new fields are derived
    from those same samples, so tests verify both the per-message extraction (in
    the ``metrics`` dict) and the derived profile-level aggregates (in ``data``).
"""

from models.core import EventType
from services.signal_extractor.linguistic import LinguisticExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_extractor(db, user_model_store):
    """Return a LinguisticExtractor wired to the test databases."""
    return LinguisticExtractor(db=db, user_model_store=user_model_store)


def _sent_event(body: str) -> dict:
    """Return a minimal email.sent event wrapping *body*."""
    return {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": "2026-02-28T12:00:00Z",
        "payload": {
            "body": body,
            "to_addresses": ["alice@example.com"],
        },
    }


# ---------------------------------------------------------------------------
# _detect_first helper
# ---------------------------------------------------------------------------


def test_detect_first_returns_matched_word(db, user_model_store):
    """_detect_first should return the lowercased matched text, not the regex."""
    ex = make_extractor(db, user_model_store)
    result = ex._detect_first("That's hilarious lol right?", ex.HUMOR_PATTERNS)
    assert result == "lol"


def test_detect_first_case_insensitive(db, user_model_store):
    """_detect_first should match regardless of case."""
    ex = make_extractor(db, user_model_store)
    result = ex._detect_first("LOL I can't believe it", ex.HUMOR_PATTERNS)
    assert result == "lol"


def test_detect_first_returns_none_on_no_match(db, user_model_store):
    """_detect_first should return None when no pattern fires."""
    ex = make_extractor(db, user_model_store)
    result = ex._detect_first("This is a perfectly serious message.", ex.HUMOR_PATTERNS)
    assert result is None


# ---------------------------------------------------------------------------
# Per-message metrics — humor
# ---------------------------------------------------------------------------


def test_extract_detects_humor_lol(db, user_model_store):
    """Humor count and type should be populated when 'lol' is present."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event("I can't believe that worked lol it's amazing!"))
    assert len(signals) == 1
    m = signals[0]["metrics"]
    assert m["humor_count"] >= 1
    assert m["humor_type"] == "lol"


def test_extract_humor_count_zero_on_serious_message(db, user_model_store):
    """Humor count should be zero for a plainly serious message."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event(
        "Please find attached the revised quarterly report for your review."
    ))
    assert signals[0]["metrics"]["humor_count"] == 0
    assert signals[0]["metrics"]["humor_type"] is None


def test_extract_humor_just_kidding(db, user_model_store):
    """'just kidding' should match the humor bank."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event("I'm quitting, just kidding, I love it here."))
    m = signals[0]["metrics"]
    assert m["humor_count"] >= 1
    assert m["humor_type"] == "just kidding"


# ---------------------------------------------------------------------------
# Per-message metrics — affirmative
# ---------------------------------------------------------------------------


def test_extract_affirmative_sounds_good(db, user_model_store):
    """'sounds good' should register as an affirmative pattern."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event("Sounds good, I'll be there at noon."))
    m = signals[0]["metrics"]
    assert m["affirmative_count"] >= 1
    assert m["affirmative_word"] == "sounds good"


def test_extract_affirmative_will_do(db, user_model_store):
    """'will do' should register as an affirmative pattern."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event("Will do, I'll have it ready by Friday."))
    m = signals[0]["metrics"]
    assert m["affirmative_count"] >= 1


def test_extract_no_affirmative_in_neutral_message(db, user_model_store):
    """Affirmative count should be zero when no affirmative phrases are present."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event("The meeting has been rescheduled to Thursday."))
    assert signals[0]["metrics"]["affirmative_count"] == 0


# ---------------------------------------------------------------------------
# Per-message metrics — gratitude
# ---------------------------------------------------------------------------


def test_extract_gratitude_thank_you(db, user_model_store):
    """'thank you' should register as a gratitude pattern."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event("Thank you so much for your help with this."))
    m = signals[0]["metrics"]
    assert m["gratitude_count"] >= 1
    assert m["gratitude_word"] == "thank you"


def test_extract_gratitude_appreciate(db, user_model_store):
    """'appreciate' should register as a gratitude pattern."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event("I really appreciate your prompt response."))
    m = signals[0]["metrics"]
    assert m["gratitude_count"] >= 1


# ---------------------------------------------------------------------------
# Per-message metrics — negative / declination
# ---------------------------------------------------------------------------


def test_extract_negative_unfortunately(db, user_model_store):
    """'unfortunately' should register as a negative/declination pattern."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event(
        "Unfortunately, I won't be able to attend the meeting on Friday."
    ))
    m = signals[0]["metrics"]
    assert m["negative_count"] >= 1
    assert m["negative_word"] == "unfortunately"


# ---------------------------------------------------------------------------
# Per-message metrics — Oxford comma
# ---------------------------------------------------------------------------


def test_extract_oxford_comma_detected(db, user_model_store):
    """Oxford comma pattern fires for 'a, b, and c' constructions."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event(
        "Please bring eggs, bacon, and coffee to the meeting."
    ))
    m = signals[0]["metrics"]
    assert m["oxford_comma_count"] >= 1
    assert m["no_oxford_count"] == 0


def test_extract_no_oxford_comma_detected(db, user_model_store):
    """No-oxford pattern fires for 'a, b and c' constructions (no serial comma)."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event(
        "Please bring eggs, bacon and coffee to the meeting."
    ))
    m = signals[0]["metrics"]
    assert m["no_oxford_count"] >= 1
    assert m["oxford_comma_count"] == 0


def test_extract_no_comma_fire_on_simple_and(db, user_model_store):
    """'and' without a preceding comma list should not match either pattern."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event(
        "I went to the store and bought milk for the recipe."
    ))
    m = signals[0]["metrics"]
    assert m["oxford_comma_count"] == 0
    assert m["no_oxford_count"] == 0


# ---------------------------------------------------------------------------
# Per-message metrics — capitalization
# ---------------------------------------------------------------------------


def test_extract_cap_starts_counted(db, user_model_store):
    """Sentence starts should be counted correctly."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event(
        "This is the first sentence. This is the second sentence."
    ))
    m = signals[0]["metrics"]
    assert m["cap_starts"] == 2
    assert m["lower_starts"] == 0


def test_extract_lower_starts_counted(db, user_model_store):
    """Lowercase sentence starts should be counted correctly."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event(
        "this is a message. another sentence here."
    ))
    m = signals[0]["metrics"]
    assert m["lower_starts"] >= 1


def test_extract_avg_word_length_present(db, user_model_store):
    """avg_word_length should be a positive float in the metrics."""
    ex = make_extractor(db, user_model_store)
    signals = ex.extract(_sent_event("The cat sat on the mat by the flat."))
    m = signals[0]["metrics"]
    assert isinstance(m["avg_word_length"], float)
    assert m["avg_word_length"] > 0


# ---------------------------------------------------------------------------
# Profile-level derived fields
# ---------------------------------------------------------------------------


def test_profile_vocabulary_complexity_populated(db, user_model_store):
    """After processing a message, vocabulary_complexity should be in the profile."""
    ex = make_extractor(db, user_model_store)
    ex.extract(_sent_event(
        "I fundamentally believe that collaborative teamwork accelerates productivity "
        "and ensures comprehensive outcomes across multifaceted organizational challenges."
    ))
    profile = user_model_store.get_signal_profile("linguistic")
    assert profile is not None
    assert "vocabulary_complexity" in profile["data"]
    assert 0.0 <= profile["data"]["vocabulary_complexity"] <= 1.0


def test_profile_capitalization_style_populated(db, user_model_store):
    """After processing a message, capitalization_style should be in the profile."""
    ex = make_extractor(db, user_model_store)
    ex.extract(_sent_event("Hello there. This is a standard message. How are you doing?"))
    profile = user_model_store.get_signal_profile("linguistic")
    assert "capitalization_style" in profile["data"]
    assert profile["data"]["capitalization_style"] in {"standard", "all_lower", "all_caps_emphasis"}


def test_profile_capitalization_style_all_lower(db, user_model_store):
    """Profile should detect 'all_lower' style when most sentence starts are lowercase."""
    ex = make_extractor(db, user_model_store)
    # Send several lowercase-starting messages to push the ratio above 0.7
    for _ in range(5):
        ex.extract(_sent_event("hey this is cool. i wanted to let you know. sounds great."))
    profile = user_model_store.get_signal_profile("linguistic")
    assert profile["data"]["capitalization_style"] == "all_lower"


def test_profile_uses_oxford_comma_none_before_threshold(db, user_model_store):
    """uses_oxford_comma should be None when fewer than 5 list instances have been seen."""
    ex = make_extractor(db, user_model_store)
    # Only 1 list instance — below the 5-instance threshold
    ex.extract(_sent_event("I need eggs, milk, and bread from the store."))
    profile = user_model_store.get_signal_profile("linguistic")
    assert profile["data"]["uses_oxford_comma"] is None


def test_profile_uses_oxford_comma_true_after_threshold(db, user_model_store):
    """uses_oxford_comma should become True after ≥5 oxford-comma list instances."""
    ex = make_extractor(db, user_model_store)
    # Send 6 messages each containing an oxford-comma list
    oxford_messages = [
        "I bought eggs, milk, and bread.",
        "The team includes Alice, Bob, and Carol.",
        "Please review slides one, two, and three.",
        "We need coffee, sugar, and cream.",
        "I like hiking, cycling, and swimming.",
        "The report covers sales, marketing, and operations.",
    ]
    for msg in oxford_messages:
        ex.extract(_sent_event(msg))
    profile = user_model_store.get_signal_profile("linguistic")
    assert profile["data"]["uses_oxford_comma"] is True


def test_profile_uses_oxford_comma_false_after_threshold(db, user_model_store):
    """uses_oxford_comma should become False after ≥5 no-serial-comma list instances."""
    ex = make_extractor(db, user_model_store)
    no_oxford_messages = [
        "I bought eggs, milk and bread.",
        "The team includes Alice, Bob and Carol.",
        "Please review slides one, two and three.",
        "We need coffee, sugar and cream.",
        "I like hiking, cycling and swimming.",
    ]
    for msg in no_oxford_messages:
        ex.extract(_sent_event(msg))
    profile = user_model_store.get_signal_profile("linguistic")
    assert profile["data"]["uses_oxford_comma"] is False


def test_profile_top_humor_markers_populated(db, user_model_store):
    """top_humor_markers should list the most common humor words used."""
    ex = make_extractor(db, user_model_store)
    for _ in range(3):
        ex.extract(_sent_event("That is hilarious lol I can't stop laughing."))
    ex.extract(_sent_event("haha that was unexpected."))
    profile = user_model_store.get_signal_profile("linguistic")
    humor = profile["data"].get("top_humor_markers", [])
    assert "lol" in humor


def test_profile_top_affirmative_patterns_populated(db, user_model_store):
    """top_affirmative_patterns should list the user's most common yes-phrases."""
    ex = make_extractor(db, user_model_store)
    for _ in range(3):
        ex.extract(_sent_event("Sounds good, I will take care of it right away."))
    profile = user_model_store.get_signal_profile("linguistic")
    affirmative = profile["data"].get("top_affirmative_patterns", [])
    assert "sounds good" in affirmative


def test_profile_top_gratitude_patterns_populated(db, user_model_store):
    """top_gratitude_patterns should list the user's most common thanks-phrases."""
    ex = make_extractor(db, user_model_store)
    for _ in range(3):
        ex.extract(_sent_event("Thank you so much for the help, I really appreciate it."))
    profile = user_model_store.get_signal_profile("linguistic")
    gratitude = profile["data"].get("top_gratitude_patterns", [])
    assert any(w in gratitude for w in ["thank you", "appreciate"])


def test_profile_averages_include_new_rates(db, user_model_store):
    """The averages sub-dict should include humor_rate, affirmative_rate, etc."""
    ex = make_extractor(db, user_model_store)
    ex.extract(_sent_event("Sounds good! Thanks! lol. Please bring eggs, milk, and bread."))
    profile = user_model_store.get_signal_profile("linguistic")
    avgs = profile["data"].get("averages", {})
    for key in ("humor_rate", "affirmative_rate", "negative_rate", "gratitude_rate"):
        assert key in avgs, f"Missing average key: {key}"
        assert avgs[key] >= 0.0


def test_existing_samples_without_new_keys_do_not_crash(db, user_model_store):
    """Old samples missing the new keys should not cause KeyError in _update_profile."""
    ex = make_extractor(db, user_model_store)
    # Manually inject a legacy sample with only the old keys
    existing = {
        "samples": [{
            "word_count": 10,
            "avg_sentence_length": 10.0,
            "unique_word_ratio": 0.5,
            "formality": 0.5,
            "hedge_rate": 0.0,
            "assertion_rate": 0.0,
            "exclamation_rate": 0.0,
            "question_rate": 0.0,
            "ellipsis_rate": 0.0,
            "emoji_count": 0,
            "emojis_used": [],
            "profanity_count": 0,
            "greeting_detected": None,
            "closing_detected": None,
            # Note: no humor_count, affirmative_count, etc.
        }],
        "per_contact": {},
    }
    user_model_store.update_signal_profile("linguistic", existing)
    # Processing a new message should not crash even with legacy samples present
    ex.extract(_sent_event("Sounds good, thank you for sending this over."))
    profile = user_model_store.get_signal_profile("linguistic")
    assert profile is not None
    assert "vocabulary_complexity" in profile["data"]
