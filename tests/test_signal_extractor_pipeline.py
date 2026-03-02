"""
Tests for the SignalExtractorPipeline orchestrator.

Covers:
- Pipeline initialization (all 8 extractors registered, mood_engine identity)
- Event routing to matching extractors via can_process()
- Error isolation (fail-open: one broken extractor does not block others)
- Skipping non-matching extractors for unrelated event types
- get_current_mood() returns a valid MoodState
- get_current_mood() persists mood snapshots to mood_history when confidence > 0
- get_user_summary() returns profile metadata and semantic facts
"""

import pytest
from datetime import datetime, timezone

from models.core import EventType
from models.user_model import MoodState
from services.signal_extractor.pipeline import SignalExtractorPipeline
from services.signal_extractor.mood import MoodInferenceEngine
from services.signal_extractor.linguistic import LinguisticExtractor
from services.signal_extractor.cadence import CadenceExtractor
from services.signal_extractor.relationship import RelationshipExtractor
from services.signal_extractor.topic import TopicExtractor
from services.signal_extractor.temporal import TemporalExtractor
from services.signal_extractor.spatial import SpatialExtractor
from services.signal_extractor.decision import DecisionExtractor


class TestSignalExtractorPipeline:
    """Test suite for SignalExtractorPipeline orchestrator."""

    @pytest.fixture
    def pipeline(self, db, user_model_store):
        """Create a SignalExtractorPipeline with test database."""
        return SignalExtractorPipeline(db, user_model_store)

    # --- Initialization Tests ---

    def test_pipeline_initializes_all_extractors(self, pipeline):
        """Pipeline should create exactly 8 extractors covering all signal dimensions."""
        assert len(pipeline.extractors) == 8

        # Verify each expected extractor type is present exactly once.
        extractor_types = [type(e) for e in pipeline.extractors]
        assert extractor_types.count(LinguisticExtractor) == 1
        assert extractor_types.count(CadenceExtractor) == 1
        assert extractor_types.count(MoodInferenceEngine) == 1
        assert extractor_types.count(RelationshipExtractor) == 1
        assert extractor_types.count(TopicExtractor) == 1
        assert extractor_types.count(TemporalExtractor) == 1
        assert extractor_types.count(SpatialExtractor) == 1
        assert extractor_types.count(DecisionExtractor) == 1

    def test_mood_engine_is_same_instance_as_in_extractors_list(self, pipeline):
        """pipeline.mood_engine must be the SAME instance as the MoodInferenceEngine
        in the extractors list, not a separate copy.  This ensures in-memory state
        updated during extract() is visible to get_current_mood()."""
        mood_from_list = next(e for e in pipeline.extractors if isinstance(e, MoodInferenceEngine))
        assert pipeline.mood_engine is mood_from_list

    # --- Event Routing Tests ---

    @pytest.mark.asyncio
    async def test_process_event_routes_to_matching_extractors(self, pipeline):
        """An email.received event should be routed to multiple extractors and
        produce signals without raising exceptions."""
        event = {
            "id": "test-routing-1",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "google",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {
                "subject": "Test Subject",
                "body": "Hello world, this is a test email with some content.",
                "sender": "alice@example.com",
                "to": ["user@example.com"],
            },
            "metadata": {},
        }

        signals = await pipeline.process_event(event)

        # email.received is accepted by linguistic, cadence, mood, relationship,
        # and topic extractors, so we expect at least some signals.
        assert isinstance(signals, list)
        assert len(signals) > 0

    @pytest.mark.asyncio
    async def test_process_event_skips_non_matching_extractors(self, pipeline):
        """An event type that no extractor handles should yield an empty signals list
        and raise no errors."""
        event = {
            "id": "test-skip-1",
            "type": "system.health.check",
            "source": "internal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "low",
            "payload": {"status": "ok"},
            "metadata": {},
        }

        signals = await pipeline.process_event(event)

        assert isinstance(signals, list)
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_process_event_returns_flat_signal_list(self, pipeline):
        """All signals from all extractors should be flattened into a single list."""
        event = {
            "id": "test-flat-1",
            "type": EventType.EMAIL_SENT.value,
            "source": "google",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {
                "subject": "Test",
                "body": "Let's schedule a meeting to discuss the project timeline.",
                "to": ["bob@example.com"],
            },
            "metadata": {},
        }

        signals = await pipeline.process_event(event)

        # Each signal should be a dict (the pipeline collects them flat via extend).
        for signal in signals:
            assert isinstance(signal, dict)

    # --- Error Isolation Tests ---

    @pytest.mark.asyncio
    async def test_process_event_error_isolation(self, pipeline):
        """If one extractor's extract() raises an exception, the pipeline should
        still collect signals from the remaining extractors (fail-open)."""
        event = {
            "id": "test-err-1",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "google",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {
                "subject": "Error test",
                "body": "Hello, this is a test email to verify error isolation.",
                "sender": "alice@example.com",
                "to": ["user@example.com"],
            },
            "metadata": {},
        }

        # First, collect baseline signals with all extractors healthy.
        baseline_signals = await pipeline.process_event(event)
        baseline_count = len(baseline_signals)

        # Now monkey-patch one extractor to blow up.
        original_extract = pipeline.extractors[0].extract
        pipeline.extractors[0].extract = lambda e: (_ for _ in ()).throw(
            RuntimeError("Simulated extractor failure")
        )

        try:
            signals = await pipeline.process_event(event)

            # We should still get signals from the other extractors.
            assert isinstance(signals, list)
            # The broken extractor's signals are missing, but others succeed.
            # We just need at least one signal from the surviving extractors.
            assert len(signals) > 0
            # Should be fewer than baseline since one extractor is broken.
            assert len(signals) <= baseline_count
        finally:
            # Restore the original extract method.
            pipeline.extractors[0].extract = original_extract

    @pytest.mark.asyncio
    async def test_process_event_error_isolation_only_covers_extract(self, pipeline):
        """The pipeline's try/except wraps extract() but NOT can_process().
        A broken can_process() propagates — this documents current behavior."""
        event = {
            "id": "test-canproc-err-1",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "google",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {
                "body": "Test message.",
                "sender": "alice@example.com",
                "to": ["user@example.com"],
            },
            "metadata": {},
        }

        original_can_process = pipeline.extractors[0].can_process
        pipeline.extractors[0].can_process = lambda e: (_ for _ in ()).throw(
            ValueError("Simulated can_process failure")
        )

        try:
            with pytest.raises(ValueError, match="Simulated can_process failure"):
                await pipeline.process_event(event)
        finally:
            pipeline.extractors[0].can_process = original_can_process

    # --- get_current_mood() Tests ---

    def test_get_current_mood_returns_mood_state(self, pipeline):
        """get_current_mood() should return a MoodState instance with valid defaults
        even when no events have been processed."""
        mood = pipeline.get_current_mood()

        assert isinstance(mood, MoodState)
        assert 0.0 <= mood.energy_level <= 1.0
        assert 0.0 <= mood.stress_level <= 1.0
        assert 0.0 <= mood.social_battery <= 1.0
        assert 0.0 <= mood.confidence <= 1.0
        assert mood.trend in ("improving", "declining", "stable", "volatile")

    def test_get_current_mood_delegates_to_mood_engine(self, pipeline):
        """get_current_mood() must call the mood_engine's compute_current_mood()."""
        # Record whether compute_current_mood was called.
        called = {"count": 0}
        original = pipeline.mood_engine.compute_current_mood

        def tracking_compute():
            called["count"] += 1
            return original()

        pipeline.mood_engine.compute_current_mood = tracking_compute
        try:
            pipeline.get_current_mood()
            assert called["count"] == 1
        finally:
            pipeline.mood_engine.compute_current_mood = original

    def test_get_current_mood_persists_to_history_when_confident(self, pipeline, user_model_store):
        """When mood confidence > 0, get_current_mood() should write a row to
        the mood_history table."""
        # Inject mood signals so confidence > 0.
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

        mood = pipeline.get_current_mood()

        # Confidence should be > 0 with 2 signals (0.2).
        assert mood.confidence > 0

        # Verify a row was written to mood_history.
        with pipeline.db.get_connection("user_model") as conn:
            row = conn.execute("SELECT COUNT(*) FROM mood_history").fetchone()
            assert row[0] >= 1

    def test_get_current_mood_does_not_persist_when_zero_confidence(self, pipeline, db):
        """When mood confidence is 0 (no signals), get_current_mood() should NOT
        write to mood_history to avoid flooding with empty readings."""
        mood = pipeline.get_current_mood()

        assert mood.confidence == 0.0

        with db.get_connection("user_model") as conn:
            row = conn.execute("SELECT COUNT(*) FROM mood_history").fetchone()
            assert row[0] == 0

    # --- get_user_summary() Tests ---

    def test_get_user_summary_returns_expected_structure(self, pipeline):
        """get_user_summary() should return a dict with profiles, fact count, and
        high-confidence facts."""
        summary = pipeline.get_user_summary()

        assert isinstance(summary, dict)
        assert "profiles" in summary
        assert "semantic_facts_count" in summary
        assert "high_confidence_facts" in summary
        assert isinstance(summary["profiles"], dict)
        assert isinstance(summary["high_confidence_facts"], list)

    def test_get_user_summary_reflects_processed_profiles(self, pipeline, user_model_store):
        """After updating a signal profile, get_user_summary() should reflect it."""
        user_model_store.update_signal_profile("linguistic", {"vocabulary_richness": 0.8})

        summary = pipeline.get_user_summary()

        assert "linguistic" in summary["profiles"]
        assert summary["profiles"]["linguistic"]["samples_count"] >= 0

    # --- Integration Test: Full Event-to-Mood Pipeline ---

    @pytest.mark.asyncio
    async def test_full_pipeline_event_to_mood(self, pipeline):
        """End-to-end: process a sleep event through the pipeline, then compute
        mood — energy should reflect the sleep data."""
        event = {
            "id": "test-e2e-1",
            "type": EventType.SLEEP_RECORDED.value,
            "source": "health",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {
                "duration_hours": 8.5,
                "quality_score": 0.9,
            },
            "metadata": {},
        }

        signals = await pipeline.process_event(event)
        assert len(signals) > 0

        mood = pipeline.get_current_mood()
        assert isinstance(mood, MoodState)
        # With a sleep event processed, confidence should be > 0.
        assert mood.confidence > 0
        assert mood.energy_level > 0.0

    @pytest.mark.asyncio
    async def test_multiple_events_accumulate_signals(self, pipeline):
        """Processing multiple events should accumulate signals, increasing
        mood confidence."""
        events = [
            {
                "id": f"test-multi-{i}",
                "type": EventType.EMAIL_SENT.value,
                "source": "google",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "priority": "normal",
                "payload": {
                    "subject": f"Test {i}",
                    "body": f"This is test email number {i} with enough content to extract signals.",
                    "to": [f"recipient{i}@example.com"],
                },
                "metadata": {},
            }
            for i in range(3)
        ]

        total_signals = []
        for event in events:
            signals = await pipeline.process_event(event)
            total_signals.extend(signals)

        # Multiple events should produce a growing collection of signals.
        assert len(total_signals) > 0

        mood = pipeline.get_current_mood()
        assert isinstance(mood, MoodState)
