"""
Tests for post-execution verification in semantic inference and routine
detection background loops.

The loops in main.py (`_semantic_inference_loop`, `_routine_detection_loop`)
snapshot DB row counts before and after each cycle so we can distinguish
"nothing changed" from a silent failure (loop ran but wrote nothing).
These tests verify:

  1. The consecutive-zero streak counter increments when the loop yields no
     new rows.
  2. The streak resets to 0 when the loop yields at least one new row.
  3. A WARNING is logged once the streak reaches the configured threshold.
  4. The cycle id increments monotonically.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_lifeos_stub(**overrides):
    """Build a minimal stub with the attributes the two loops read from.

    We avoid instantiating the real ``LifeOS`` (which requires NATS, Ollama,
    settings.yaml, etc.) by attaching just the dependencies the methods
    touch and then binding the unbound coroutine to the stub.
    """
    stub = MagicMock()
    stub.shutdown_event = asyncio.Event()

    # Streak counters and threshold — the production __init__ sets these,
    # so tests must seed them on the stub.
    stub._semantic_inference_cycle = 0
    stub._semantic_inference_zero_streak = 0
    stub._routine_detection_cycle = 0
    stub._routine_detection_zero_streak = 0
    stub._zero_streak_warn_threshold = 3

    stub.semantic_fact_inferrer = MagicMock()
    stub.semantic_fact_inferrer.run_all_inference = MagicMock()
    stub.semantic_fact_inferrer._total_facts_written_last_cycle = 0

    stub.routine_detector = MagicMock()
    stub.routine_detector.detect_routines = MagicMock(return_value=[])
    stub.routine_detector.store_routines = MagicMock(return_value=0)

    stub.workflow_detector = MagicMock()
    stub.workflow_detector.detect_workflows = MagicMock(return_value=[])
    stub.workflow_detector.store_workflows = MagicMock(return_value=0)

    # Event bus disabled so the routine loop doesn't attempt to publish.
    stub.event_bus = MagicMock()
    stub.event_bus.is_connected = False

    # Apply overrides last so callers can replace any default.
    for key, value in overrides.items():
        setattr(stub, key, value)

    return stub


def _bind(method_name: str, stub) -> asyncio.Future:
    """Bind one of the LifeOS loop coroutines to a stub.

    Returns the coroutine object — the caller is responsible for awaiting
    it (typically wrapped in ``asyncio.wait_for``).
    """
    from main import LifeOS

    method = getattr(LifeOS, method_name)
    return method(stub)


async def _run_loop_for_n_cycles(loop_name: str, stub, n: int, inter_cycle_sleep: int):
    """Drive a loop through ``n`` cycles, then set shutdown.

    We monkey-patch ``asyncio.sleep`` so the warmup and inter-cycle sleeps
    don't actually block. After the nth inter-cycle sleep we signal the
    shutdown event so the loop exits.
    """
    original_sleep = asyncio.sleep
    inter_cycle_seen = {"count": 0}

    async def fake_sleep(duration):
        # Match the inter-cycle sleep — either the fixed cadence for the
        # semantic loop or any of the adaptive retry values used by the
        # routine loop (3600 / 10800 / 43200).
        if duration in inter_cycle_sleep:
            inter_cycle_seen["count"] += 1
            if inter_cycle_seen["count"] >= n:
                stub.shutdown_event.set()
        await original_sleep(0)

    with patch("asyncio.sleep", side_effect=fake_sleep):
        await asyncio.wait_for(_bind(loop_name, stub), timeout=5)


# ---------------------------------------------------------------------------
# Semantic inference loop
# ---------------------------------------------------------------------------


class TestSemanticInferenceLoopVerification:
    """Validate streak tracking + warn threshold for the semantic loop."""

    @pytest.mark.asyncio
    async def test_zero_streak_increments_when_no_facts_added(self):
        """When the inferrer writes nothing, the streak grows each cycle."""
        stub = _make_lifeos_stub()
        # Count stays constant at 5 — no new facts are produced.
        with patch.object(type(stub), "_count_semantic_facts", create=True, return_value=5):
            stub._count_semantic_facts = lambda: 5
            await _run_loop_for_n_cycles(
                "_semantic_inference_loop", stub, n=2, inter_cycle_sleep={3600}
            )

        assert stub._semantic_inference_cycle == 2
        assert stub._semantic_inference_zero_streak == 2

    @pytest.mark.asyncio
    async def test_zero_streak_resets_when_facts_added(self):
        """A cycle that produces a new fact resets the streak to 0."""
        stub = _make_lifeos_stub()
        # First call returns 5 (before cycle 1), second 5 (after cycle 1 — no growth),
        # third 5 (before cycle 2), fourth 7 (after cycle 2 — 2 new facts).
        counts = iter([5, 5, 5, 7])
        stub._count_semantic_facts = lambda: next(counts)

        await _run_loop_for_n_cycles(
            "_semantic_inference_loop", stub, n=2, inter_cycle_sleep={3600}
        )

        assert stub._semantic_inference_cycle == 2
        # After cycle 1 streak was 1; cycle 2 added 2 facts → reset to 0.
        assert stub._semantic_inference_zero_streak == 0

    @pytest.mark.asyncio
    async def test_warning_logged_at_threshold(self, caplog):
        """Once the streak hits the threshold, log level escalates to WARNING."""
        stub = _make_lifeos_stub()
        stub._count_semantic_facts = lambda: 0  # Always zero growth.

        caplog.set_level(logging.WARNING, logger="main")
        await _run_loop_for_n_cycles(
            "_semantic_inference_loop", stub, n=3, inter_cycle_sleep={3600}
        )

        # By the 3rd cycle (threshold=3) the loop should emit a WARNING.
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "SemanticInferenceLoop" in r.getMessage()
        ]
        assert warning_records, "expected a WARNING after threshold streak"
        assert stub._semantic_inference_zero_streak == 3


# ---------------------------------------------------------------------------
# Routine detection loop
# ---------------------------------------------------------------------------


class TestRoutineDetectionLoopVerification:
    """Validate streak tracking + warn threshold for the routine loop."""

    @pytest.mark.asyncio
    async def test_zero_streak_increments_when_no_routines_added(self):
        """When detection produces no DB delta, streak grows."""
        stub = _make_lifeos_stub()
        stub._count_routines = lambda: 3  # Steady — no growth.

        await _run_loop_for_n_cycles(
            "_routine_detection_loop",
            stub,
            n=2,
            # The loop chooses retry from {3600, 10800, 43200} based on
            # total_patterns. With 0 routines + 0 workflows, retry=3600.
            inter_cycle_sleep={3600, 10800, 43200},
        )

        assert stub._routine_detection_cycle == 2
        assert stub._routine_detection_zero_streak == 2

    @pytest.mark.asyncio
    async def test_zero_streak_resets_when_routines_added(self):
        """A cycle that grows the routines table resets the streak."""
        stub = _make_lifeos_stub()
        # Cycle 1: before=3, after=3 (no growth) → streak=1
        # Cycle 2: before=3, after=5 (2 new) → streak=0
        counts = iter([3, 3, 3, 5])
        stub._count_routines = lambda: next(counts)

        await _run_loop_for_n_cycles(
            "_routine_detection_loop",
            stub,
            n=2,
            inter_cycle_sleep={3600, 10800, 43200},
        )

        assert stub._routine_detection_cycle == 2
        assert stub._routine_detection_zero_streak == 0

    @pytest.mark.asyncio
    async def test_warning_logged_at_threshold(self, caplog):
        """Streak at threshold triggers a WARNING log line."""
        stub = _make_lifeos_stub()
        stub._count_routines = lambda: 0

        caplog.set_level(logging.WARNING, logger="main")
        await _run_loop_for_n_cycles(
            "_routine_detection_loop",
            stub,
            n=3,
            inter_cycle_sleep={3600, 10800, 43200},
        )

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "RoutineDetectionLoop" in r.getMessage()
        ]
        assert warning_records, "expected a WARNING after threshold streak"
        assert stub._routine_detection_zero_streak == 3

    @pytest.mark.asyncio
    async def test_error_in_detection_still_records_cycle(self):
        """If detect_routines raises, the loop still finishes the cycle and
        advances counters — silent failures are exactly what we want to see.
        """
        stub = _make_lifeos_stub()
        stub.routine_detector.detect_routines = MagicMock(side_effect=RuntimeError("boom"))
        stub._count_routines = lambda: 0

        await _run_loop_for_n_cycles(
            "_routine_detection_loop",
            stub,
            n=1,
            # On exception the loop sets retry_seconds=3600.
            inter_cycle_sleep={3600},
        )

        assert stub._routine_detection_cycle == 1
        # Zero new routines → streak should reflect that.
        assert stub._routine_detection_zero_streak == 1


# ---------------------------------------------------------------------------
# Count helpers
# ---------------------------------------------------------------------------


class TestCountHelpers:
    """The two count helpers must be resilient to DB errors (fail-open)."""

    def test_count_semantic_facts_returns_zero_on_error(self):
        """A failing DB connection should yield 0, not propagate."""
        from main import LifeOS

        stub = MagicMock()
        stub.user_model_store.db.get_connection.side_effect = RuntimeError("db down")
        assert LifeOS._count_semantic_facts(stub) == 0

    def test_count_routines_returns_zero_on_error(self):
        """Same fail-open guarantee for the routines count helper."""
        from main import LifeOS

        stub = MagicMock()
        stub.user_model_store.db.get_connection.side_effect = RuntimeError("db down")
        assert LifeOS._count_routines(stub) == 0
