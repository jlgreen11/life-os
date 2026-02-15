"""
Tests for the MoodInferenceEngine signal extractor.

Covers:
- Event type filtering (can_process)
- Signal extraction from communication events
- Sleep quality and duration signal generation
- Calendar density and spending spike detection
- Weighted average computation
- Current mood state computation
- Baseline fallback logic
- Mood signal accumulation and ring buffer management
"""

import pytest
from datetime import datetime, timezone

from models.core import EventType
from models.user_model import MoodState
from services.signal_extractor.mood import MoodInferenceEngine


class TestMoodInferenceEngine:
    """Test suite for MoodInferenceEngine."""

    @pytest.fixture
    def engine(self, db, user_model_store):
        """Create a MoodInferenceEngine instance with test database."""
        return MoodInferenceEngine(db, user_model_store)

    # --- Event Type Filtering Tests ---

    def test_can_process_accepts_email_sent(self, engine):
        """MoodInferenceEngine should accept email.sent events."""
        event = {"type": EventType.EMAIL_SENT.value}
        assert engine.can_process(event) is True

    def test_can_process_accepts_message_sent(self, engine):
        """MoodInferenceEngine should accept message.sent events."""
        event = {"type": EventType.MESSAGE_SENT.value}
        assert engine.can_process(event) is True

    def test_can_process_accepts_health_metrics(self, engine):
        """MoodInferenceEngine should accept health metric events."""
        event = {"type": EventType.HEALTH_METRIC_UPDATED.value}
        assert engine.can_process(event) is True

    def test_can_process_accepts_sleep_events(self, engine):
        """MoodInferenceEngine should accept sleep.recorded events."""
        event = {"type": EventType.SLEEP_RECORDED.value}
        assert engine.can_process(event) is True

    def test_can_process_accepts_calendar_events(self, engine):
        """MoodInferenceEngine should accept calendar events."""
        event = {"type": EventType.CALENDAR_EVENT_CREATED.value}
        assert engine.can_process(event) is True

    def test_can_process_accepts_transactions(self, engine):
        """MoodInferenceEngine should accept financial transactions."""
        event = {"type": EventType.TRANSACTION_NEW.value}
        assert engine.can_process(event) is True

    def test_can_process_accepts_location_changes(self, engine):
        """MoodInferenceEngine should accept location change events."""
        event = {"type": EventType.LOCATION_CHANGED.value}
        assert engine.can_process(event) is True

    def test_can_process_accepts_user_commands(self, engine):
        """MoodInferenceEngine should accept system.user.command events."""
        event = {"type": "system.user.command"}
        assert engine.can_process(event) is True

    def test_can_process_rejects_email_received(self, engine):
        """MoodInferenceEngine should reject email.received (inbound) events."""
        event = {"type": EventType.EMAIL_RECEIVED.value}
        assert engine.can_process(event) is False

    def test_can_process_rejects_unrelated_events(self, engine):
        """MoodInferenceEngine should reject unrelated event types."""
        event = {"type": "system.connector.sync_complete"}
        assert engine.can_process(event) is False

    # --- Communication Signal Extraction Tests ---

    def test_extract_message_length_signal_from_email(self, engine):
        """Should extract message_length signal from email.sent event."""
        event = {
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "This is a test message with some content.",
            },
        }

        signals = engine.extract(event)

        assert len(signals) == 1
        assert signals[0]["type"] == "mood_signal"
        mood_signals = signals[0]["signals"]

        # Should contain message_length signal
        length_signal = next(s for s in mood_signals if s["signal_type"] == "message_length")
        assert length_signal["value"] == 8  # 8 words
        assert length_signal["weight"] == 0.3
        assert length_signal["source"] == "gmail"

    def test_extract_negative_language_signal(self, engine):
        """Should detect negative language in messages and create high-weight signal."""
        event = {
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "I'm frustrated and exhausted with this problem. Unfortunately, it's difficult.",
            },
        }

        signals = engine.extract(event)

        assert len(signals) == 1
        mood_signals = signals[0]["signals"]

        # Should contain both message_length and negative_language signals
        negative_signal = next(
            s for s in mood_signals if s["signal_type"] == "negative_language"
        )
        assert negative_signal["weight"] == 0.6  # High weight for negative language
        assert negative_signal["value"] > 0  # Non-zero negative word density
        assert negative_signal["source"] == "imessage"

    def test_extract_computes_delta_from_baseline(self, engine):
        """Message length delta should be computed relative to baseline."""
        # Engine should use DEFAULT_BASELINES (25.0 words) when no custom baseline exists
        event = {
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": " ".join(["word"] * 50),  # 50 words = 2x baseline
            },
        }

        signals = engine.extract(event)
        mood_signals = signals[0]["signals"]
        length_signal = next(s for s in mood_signals if s["signal_type"] == "message_length")

        # Delta should be +1.0 (100% above baseline of 25 words)
        assert length_signal["delta_from_baseline"] == pytest.approx(1.0, abs=0.01)

    def test_extract_skips_empty_messages(self, engine):
        """Should return empty list for events with no text content."""
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {},
        }

        signals = engine.extract(event)
        assert signals == []

    # --- Sleep Signal Extraction Tests ---

    def test_extract_sleep_quality_signal(self, engine):
        """Should extract sleep_quality signal with high weight."""
        event = {
            "type": EventType.SLEEP_RECORDED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "duration_hours": 8.0,
                "quality_score": 0.85,
            },
        }

        signals = engine.extract(event)
        mood_signals = signals[0]["signals"]

        quality_signal = next(s for s in mood_signals if s["signal_type"] == "sleep_quality")
        assert quality_signal["value"] == 0.85
        assert quality_signal["weight"] == 0.8  # High weight
        # Delta from baseline of 0.7
        assert quality_signal["delta_from_baseline"] == pytest.approx(0.15, abs=0.01)
        assert quality_signal["source"] == "health"

    def test_extract_sleep_duration_signal(self, engine):
        """Should extract sleep_duration signal with fractional delta."""
        event = {
            "type": EventType.SLEEP_RECORDED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "duration_hours": 6.0,
                "quality_score": 0.5,
            },
        }

        signals = engine.extract(event)
        mood_signals = signals[0]["signals"]

        duration_signal = next(s for s in mood_signals if s["signal_type"] == "sleep_duration")
        assert duration_signal["value"] == 6.0
        assert duration_signal["weight"] == 0.5
        # Delta from baseline of 7.5: (6.0 - 7.5) / 7.5 = -0.2
        assert duration_signal["delta_from_baseline"] == pytest.approx(-0.2, abs=0.01)

    def test_extract_sleep_uses_defaults_when_missing(self, engine):
        """Should use default values (7 hours, 0.5 quality) when payload fields missing."""
        event = {
            "type": EventType.SLEEP_RECORDED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {},
        }

        signals = engine.extract(event)
        mood_signals = signals[0]["signals"]

        duration_signal = next(s for s in mood_signals if s["signal_type"] == "sleep_duration")
        assert duration_signal["value"] == 7  # Default

        quality_signal = next(s for s in mood_signals if s["signal_type"] == "sleep_quality")
        assert quality_signal["value"] == 0.5  # Default

    # --- Calendar Density Signal Tests ---

    def test_extract_calendar_density_signal(self, engine):
        """Should create calendar_density signal for new events."""
        event = {
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "summary": "Team meeting",
            },
        }

        signals = engine.extract(event)
        mood_signals = signals[0]["signals"]

        density_signal = next(s for s in mood_signals if s["signal_type"] == "calendar_density")
        assert density_signal["value"] == 1.0
        assert density_signal["delta_from_baseline"] == 0.0
        assert density_signal["weight"] == 0.2
        assert density_signal["source"] == "calendar"

    # --- Spending Spike Signal Tests ---

    def test_extract_spending_spike_for_large_transaction(self, engine):
        """Should create spending_spike signal for transactions > $100."""
        event = {
            "type": EventType.TRANSACTION_NEW.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "amount": -250.00,
                "merchant": "Electronics Store",
            },
        }

        signals = engine.extract(event)
        mood_signals = signals[0]["signals"]

        spending_signal = next(s for s in mood_signals if s["signal_type"] == "spending_spike")
        assert spending_signal["value"] == 250.00  # Absolute value
        assert spending_signal["delta_from_baseline"] == 0.5
        assert spending_signal["weight"] == 0.3
        assert spending_signal["source"] == "finance"

    def test_extract_ignores_small_transactions(self, engine):
        """Should not create spending_spike signal for transactions <= $100."""
        event = {
            "type": EventType.TRANSACTION_NEW.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "amount": -45.00,
                "merchant": "Coffee Shop",
            },
        }

        signals = engine.extract(event)
        # Should return empty list since amount <= $100
        assert signals == []

    # --- Mood Signal Persistence Tests ---

    def test_extract_persists_signals_to_profile(self, engine, user_model_store):
        """Extracted signals should be persisted to mood_signals profile."""
        event = {
            "type": EventType.SLEEP_RECORDED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "duration_hours": 8.5,
                "quality_score": 0.9,
            },
        }

        engine.extract(event)

        # Verify signals were stored
        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        assert "recent_signals" in profile["data"]
        assert len(profile["data"]["recent_signals"]) == 2  # quality + duration

    def test_extract_caps_signal_buffer_at_200(self, engine, user_model_store):
        """Signal buffer should be capped at 200 entries to prevent unbounded growth."""
        # Pre-populate with 199 signals
        existing_signals = [
            {
                "signal_type": "test_signal",
                "value": i,
                "delta_from_baseline": 0,
                "weight": 0.5,
                "source": "test",
            }
            for i in range(199)
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": existing_signals})

        # Add 2 more signals via extraction (should exceed cap)
        event = {
            "type": EventType.SLEEP_RECORDED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "duration_hours": 7.0,
                "quality_score": 0.8,
            },
        }
        engine.extract(event)

        # Buffer should be capped at 200 (oldest entry should be dropped)
        profile = user_model_store.get_signal_profile("mood_signals")
        assert len(profile["data"]["recent_signals"]) == 200

        # First signal should be value=1 (value=0 was dropped)
        assert profile["data"]["recent_signals"][0]["value"] == 1

    # --- Weighted Average Computation Tests ---

    def test_weighted_average_with_valid_signals(self, engine):
        """Should compute correct weight-normalized average."""
        signals = [
            {"delta_from_baseline": 0.5, "weight": 0.8},
            {"delta_from_baseline": 0.3, "weight": 0.2},
        ]

        result = engine._weighted_average(signals)

        # Expected: (0.5*0.8 + 0.3*0.2) / (0.8 + 0.2) = 0.46
        assert result == pytest.approx(0.46, abs=0.01)

    def test_weighted_average_uses_abs_values(self, engine):
        """Weighted average uses absolute values of delta_from_baseline."""
        signals = [
            {"delta_from_baseline": -2.0, "weight": 1.0},
        ]

        result = engine._weighted_average(signals)
        # abs(-2.0) * 1.0 / 1.0 = 2.0, clamped to 1.0
        assert result == 1.0

    def test_weighted_average_clamps_to_one(self, engine):
        """Weighted averages > 1.0 should be clamped to 1.0."""
        signals = [
            {"delta_from_baseline": 5.0, "weight": 1.0},
        ]

        result = engine._weighted_average(signals)
        assert result == 1.0

    def test_weighted_average_returns_default_for_empty_list(self, engine):
        """Should return default value when no signals provided."""
        result = engine._weighted_average([], default=0.7)
        assert result == 0.7

    def test_weighted_average_returns_default_for_zero_weight(self, engine):
        """Should return default value when total weight is zero."""
        signals = [
            {"delta_from_baseline": 0.5, "weight": 0.0},
        ]

        result = engine._weighted_average(signals, default=0.5)
        assert result == 0.5

    # --- Current Mood State Computation Tests ---

    def test_compute_current_mood_returns_neutral_when_no_data(self, engine):
        """Should return neutral MoodState when no signals exist."""
        mood = engine.compute_current_mood()

        assert isinstance(mood, MoodState)
        assert mood.energy_level == 0.5
        assert mood.stress_level == 0.3
        assert mood.social_battery == 0.5
        assert mood.cognitive_load == 0.3  # Default from MoodState
        assert mood.emotional_valence == 0.5
        assert mood.confidence == 0.0

    def test_compute_current_mood_from_sleep_signals(self, engine, user_model_store):
        """Should compute energy_level from sleep signals."""
        # Inject sleep signals
        signals = [
            {
                "signal_type": "sleep_quality",
                "value": 0.9,
                "delta_from_baseline": 0.2,
                "weight": 0.8,
                "source": "health",
            },
            {
                "signal_type": "sleep_duration",
                "value": 8.0,
                "delta_from_baseline": 0.067,
                "weight": 0.5,
                "source": "health",
            },
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": signals})

        mood = engine.compute_current_mood()

        # Energy should be derived from sleep signals
        assert mood.energy_level > 0.0
        assert mood.confidence == 0.2  # 2 signals * 0.1

    def test_compute_current_mood_from_stress_signals(self, engine, user_model_store):
        """Should compute stress_level from negative language and calendar density."""
        signals = [
            {
                "signal_type": "negative_language",
                "value": 0.3,
                "delta_from_baseline": 0.3,
                "weight": 0.6,
                "source": "imessage",
            },
            {
                "signal_type": "calendar_density",
                "value": 1.0,
                "delta_from_baseline": 0.0,
                "weight": 0.2,
                "source": "calendar",
            },
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": signals})

        mood = engine.compute_current_mood()

        # Stress should be elevated
        assert mood.stress_level > 0.0
        # Cognitive load should increase with stress signal count
        assert mood.cognitive_load == pytest.approx(0.3, abs=0.01)  # 2 signals * 0.15

    def test_compute_current_mood_inverts_valence(self, engine, user_model_store):
        """Emotional valence should be inverted: high negative signals = low valence."""
        signals = [
            {
                "signal_type": "negative_language",
                "value": 0.5,
                "delta_from_baseline": 0.5,
                "weight": 0.6,
                "source": "email",
            },
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": signals})

        mood = engine.compute_current_mood()

        # Valence should be inverted (1.0 - weighted_average)
        assert mood.emotional_valence < 0.7  # Should be reduced by negative signal

    def test_compute_current_mood_caps_cognitive_load(self, engine, user_model_store):
        """Cognitive load should be capped at 1.0 even with many stress signals."""
        # Create 10 stress signals (would compute to 1.5 without cap)
        signals = [
            {
                "signal_type": "calendar_density",
                "value": 1.0,
                "delta_from_baseline": 0.0,
                "weight": 0.2,
                "source": "calendar",
            }
            for _ in range(10)
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": signals})

        mood = engine.compute_current_mood()

        assert mood.cognitive_load == 1.0  # Capped

    def test_compute_current_mood_includes_contributing_signals(self, engine, user_model_store):
        """MoodState should include the 10 most recent contributing signals."""
        signals = [
            {
                "signal_type": f"test_{i}",
                "value": float(i),
                "delta_from_baseline": 0.0,
                "weight": 0.5,
                "source": "test",
            }
            for i in range(15)
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": signals})

        mood = engine.compute_current_mood()

        # Should include only the last 10 signals
        assert len(mood.contributing_signals) == 10
        # Signals should be from the end of the list (most recent)
        assert mood.contributing_signals[0].signal_type == "test_5"
        assert mood.contributing_signals[-1].signal_type == "test_14"

    def test_compute_current_mood_sets_stable_trend(self, engine, user_model_store):
        """Trend should default to 'stable' in current implementation."""
        signals = [
            {
                "signal_type": "sleep_quality",
                "value": 0.8,
                "delta_from_baseline": 0.1,
                "weight": 0.8,
                "source": "health",
            },
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": signals})

        mood = engine.compute_current_mood()

        assert mood.trend == "stable"

    def test_compute_current_mood_scales_confidence_with_signal_count(self, engine, user_model_store):
        """Confidence should scale linearly with signal count, capped at 1.0."""
        # 5 signals = 0.5 confidence
        signals = [
            {
                "signal_type": "test",
                "value": 1.0,
                "delta_from_baseline": 0.0,
                "weight": 0.5,
                "source": "test",
            }
            for _ in range(5)
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": signals})

        mood = engine.compute_current_mood()
        assert mood.confidence == 0.5

        # 15 signals should cap at 1.0
        signals_15 = signals * 3
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": signals_15})

        mood = engine.compute_current_mood()
        assert mood.confidence == 1.0

    # --- Baseline Retrieval Tests ---

    def test_get_baseline_uses_custom_baseline_when_available(self, engine, user_model_store):
        """Should use custom baseline from baselines profile when available."""
        user_model_store.update_signal_profile("baselines", {
            "message_length_words": 50.0,
        })

        baseline = engine._get_baseline("message_length_words")
        assert baseline == 50.0

    def test_get_baseline_falls_back_to_default(self, engine):
        """Should fall back to DEFAULT_BASELINES when custom baseline not found."""
        baseline = engine._get_baseline("typing_speed_wpm")
        assert baseline == 40.0  # From DEFAULT_BASELINES

    def test_get_baseline_returns_default_for_unknown_metric(self, engine):
        """Should return 0.5 for metrics not in DEFAULT_BASELINES."""
        baseline = engine._get_baseline("unknown_metric")
        assert baseline == 0.5

    # --- Integration Tests ---

    def test_full_pipeline_email_with_negative_language(self, engine, user_model_store):
        """End-to-end test: email with negative language should create signals and update profile."""
        event = {
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "I'm really frustrated with this problem. It's been difficult and exhausting.",
            },
        }

        # Extract signals
        signals = engine.extract(event)

        # Verify extraction
        assert len(signals) == 1
        mood_signals = signals[0]["signals"]
        assert len(mood_signals) == 2  # message_length + negative_language

        # Verify profile was updated
        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        assert len(profile["data"]["recent_signals"]) == 2

        # Compute mood and verify stress/valence reflect negative content
        mood = engine.compute_current_mood()
        # Stress comes from negative_language signal
        assert mood.stress_level > 0.0  # Should have stress signal
        assert mood.emotional_valence < 0.75  # Should be reduced by negative content
        assert mood.cognitive_load == 0.15  # 1 stress signal * 0.15

    def test_full_pipeline_sleep_to_mood_computation(self, engine):
        """End-to-end test: good sleep should result in positive energy signal."""
        event = {
            "type": EventType.SLEEP_RECORDED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "duration_hours": 8.5,
                "quality_score": 0.9,
            },
        }

        # Extract and compute mood
        engine.extract(event)
        mood = engine.compute_current_mood()

        # Energy should be computed from sleep signals (positive deltas from baseline)
        # The weighted average computation uses abs() so even though deltas are positive,
        # the result is bounded by the weights and values
        assert mood.energy_level > 0.0  # Should have energy signals
        assert mood.energy_level <= 1.0  # Should be clamped
        # No stress signals, so stress should be at default
        assert mood.stress_level == 0.3  # Default when no stress signals
        # Confidence should be low with only 2 signals
        assert mood.confidence == 0.2
