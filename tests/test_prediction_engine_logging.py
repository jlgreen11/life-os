"""
Tests for PredictionEngine logging behaviour.

Verifies that:
1. PredictionEngine uses the standard logging module (not print()) for all
   diagnostic output.
2. Low-accuracy multiplier warnings are emitted at WARNING level so they
   surface even when DEBUG logging is disabled.
3. Diagnostic per-run stats are emitted at DEBUG level so they can be
   silenced in production without hiding warnings.
4. Exception handlers that previously used bare ``pass`` now emit debug-level
   log messages so developers can diagnose missing predictions.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.prediction_engine.engine import PredictionEngine


@pytest.fixture
def engine(db, user_model_store):
    """Prediction engine with empty temporary databases."""
    return PredictionEngine(db, user_model_store)


# ---------------------------------------------------------------------------
# Logger presence
# ---------------------------------------------------------------------------

def test_engine_uses_logging_module():
    """PredictionEngine module must define a module-level logger via logging.getLogger.

    This is the canonical pattern used by every other service in the codebase
    (InsightEngine, SemanticFactInferrer, AIEngine, etc.).
    """
    import services.prediction_engine.engine as eng_module
    assert hasattr(eng_module, "logger"), (
        "services.prediction_engine.engine must expose a module-level 'logger' "
        "created with logging.getLogger(__name__)"
    )
    assert isinstance(eng_module.logger, logging.Logger)


def test_logger_name_is_module_path():
    """The logger must be named after the module (__name__) for correct hierarchy."""
    import services.prediction_engine.engine as eng_module
    assert eng_module.logger.name == "services.prediction_engine.engine"


# ---------------------------------------------------------------------------
# Accuracy-multiplier WARNING
# ---------------------------------------------------------------------------

def test_accuracy_multiplier_warning_emitted(db, user_model_store, engine, caplog):
    """_get_accuracy_multiplier() emits a WARNING when accuracy < 20% with ≥10 samples.

    The warning must appear at WARNING level (not DEBUG) because operators need
    to see accuracy degradation in production logs even when debug output is off.
    """
    # Seed 10 resolved predictions for type 'opportunity' — all inaccurate
    with db.get_connection("user_model") as conn:
        for i in range(10):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at, resolution_reason)
                   VALUES (?, 'opportunity', 'test', 0.5, 'SUGGEST', 1, 0, datetime('now'), NULL)""",
                (f"opp-{i}",),
            )

    with caplog.at_level(logging.WARNING, logger="services.prediction_engine.engine"):
        multiplier = engine._get_accuracy_multiplier("opportunity")

    assert multiplier == 0.3, "Floor of 0.3 should be returned for <20% accuracy with ≥10 samples"

    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "accuracy_multiplier" in r.getMessage()
    ]
    assert len(warning_records) == 1, (
        "Exactly one WARNING record about accuracy_multiplier should be emitted"
    )
    msg = warning_records[0].getMessage()
    assert "opportunity" in msg
    assert "0.0%" in msg or "0%" in msg, "Message should include the actual accuracy rate"
    assert "0.3" in msg, "Message should mention the floor value"


def test_accuracy_multiplier_no_warning_above_threshold(db, user_model_store, engine, caplog):
    """No WARNING should be emitted when accuracy is above the 20% threshold."""
    # Seed 10 predictions — 5 accurate (50% accuracy, well above the 20% floor)
    with db.get_connection("user_model") as conn:
        for i in range(10):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at, resolution_reason)
                   VALUES (?, 'reminder', 'test', 0.5, 'SUGGEST', 1, ?, datetime('now'), NULL)""",
                (f"rem-{i}", 1 if i < 5 else 0),
            )

    with caplog.at_level(logging.WARNING, logger="services.prediction_engine.engine"):
        multiplier = engine._get_accuracy_multiplier("reminder")

    assert multiplier > 0.3, "No floor penalty should be applied at 50% accuracy"
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "accuracy_multiplier" in r.getMessage()
    ]
    assert len(warning_records) == 0, "No WARNING should be emitted above the 20% threshold"


def test_contact_accuracy_multiplier_warning_emitted(db, user_model_store, engine, caplog):
    """_get_contact_accuracy_multiplier() emits a WARNING for contacts with <20% accuracy."""
    email = "cold@example.com"
    # Seed 3 inaccurate resolved opportunity predictions for this contact
    with db.get_connection("user_model") as conn:
        for i in range(3):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at, resolution_reason,
                    supporting_signals)
                   VALUES (?, 'opportunity', 'test', 0.5, 'SUGGEST', 1, 0, datetime('now'), NULL, ?)""",
                (f"cold-{i}", f'{{"contact_email": "{email}"}}'),
            )

    with caplog.at_level(logging.WARNING, logger="services.prediction_engine.engine"):
        multiplier = engine._get_contact_accuracy_multiplier(email)

    assert multiplier == 0.5, "Per-contact floor of 0.5 expected for <20% accuracy with ≥3 samples"

    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "contact_accuracy_multiplier" in r.getMessage()
    ]
    assert len(warning_records) == 1
    msg = warning_records[0].getMessage()
    assert email in msg


# ---------------------------------------------------------------------------
# No print() leakage
# ---------------------------------------------------------------------------

def test_no_print_calls_in_engine_source():
    """PredictionEngine source must not contain any bare print() calls.

    All diagnostic output must go through the logging module so it respects
    the configured log level and is captured by structured log aggregators.
    """
    import inspect
    import services.prediction_engine.engine as eng_module
    source = inspect.getsource(eng_module)

    # Find lines that look like standalone print() calls (not in comments/docstrings)
    import re
    print_lines = [
        (i + 1, line)
        for i, line in enumerate(source.splitlines())
        if re.search(r"^\s*print\(", line) and not line.lstrip().startswith("#")
    ]
    assert print_lines == [], (
        f"Found bare print() calls in engine.py — migrate to logger.*(): {print_lines}"
    )


# ---------------------------------------------------------------------------
# Debug logging in exception handlers (bare-pass → debug log)
# ---------------------------------------------------------------------------


def test_quiet_hours_logs_on_malformed_json(db, user_model_store):
    """_is_quiet_hours logs a debug message when the quiet_hours preference contains invalid JSON."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # Insert malformed JSON into user_preferences
    with db.get_connection("preferences") as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_preferences (key, value) VALUES (?, ?)",
            ("quiet_hours", "{not valid json"),
        )

    now = datetime.now(timezone.utc)

    with patch("services.prediction_engine.engine.logger") as mock_logger:
        result = engine._is_quiet_hours(now)

    # Fail-open: should return False when config is malformed
    assert result is False
    # Should have logged a debug message about the parse failure
    mock_logger.debug.assert_called()
    log_msg = mock_logger.debug.call_args[0][0]
    assert "quiet_hours" in log_msg


def test_quiet_hours_returns_false_with_no_config(db, user_model_store):
    """_is_quiet_hours returns False when no quiet_hours preference exists."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")
    now = datetime.now(timezone.utc)

    result = engine._is_quiet_hours(now)
    assert result is False


@pytest.mark.asyncio
async def test_calendar_reminders_logs_on_corrupt_dedup_entry(db, event_store, user_model_store):
    """_check_calendar_event_reminders logs when dedup entries have corrupt supporting_signals."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")
    now = datetime.now(timezone.utc)

    # Insert a calendar.event.created event in the 2-24h reminder window so
    # the method reaches the dedup section instead of returning early.
    start_time = now + timedelta(hours=6)
    end_time = start_time + timedelta(hours=1)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "google",
        "timestamp": now.isoformat(),
        "payload": {
            "event_id": "cal-evt-123",
            "title": "Team Meeting",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "is_all_day": False,
            "location": "",
        },
        "metadata": {},
    })

    # Insert a prediction with corrupt supporting_signals directly into the DB.
    # The predictions table has no JSON trigger, so raw text is accepted.
    # The dedup query filters on created_at > (now - 48h).
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions (id, prediction_type, description, confidence,
               confidence_gate, supporting_signals, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "reminder",
                "Some old reminder",
                0.7,
                "suggest",
                "{bad json here",  # Corrupt JSON — triggers the handler
                now.isoformat(),
            ),
        )

    with patch("services.prediction_engine.engine.logger") as mock_logger:
        # Should complete without error despite corrupt dedup entry
        await engine._check_calendar_event_reminders({})

    # Should have logged a debug message about the malformed entry
    debug_calls = mock_logger.debug.call_args_list
    dedup_log_found = any(
        "calendar_reminders" in str(call) and "malformed dedup" in str(call)
        for call in debug_calls
    )
    assert dedup_log_found, (
        f"Expected a 'calendar_reminders: skipping malformed dedup entry' debug log. "
        f"Got debug calls: {debug_calls}"
    )


def test_count_calendar_event_types_logs_on_malformed_payload(db, user_model_store):
    """_count_calendar_event_types logs when event payloads contain invalid JSON.

    The events table has a functional index (json_extract on payload) that
    prevents inserting truly malformed JSON, so we mock the DB query to
    simulate corrupt data that could result from manual edits or migration
    issues.
    """
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # Create mock rows: one corrupt, one valid
    corrupt_row = {"payload": "{not valid json"}
    valid_row = {"payload": json.dumps({"is_all_day": False, "title": "Valid Event"})}

    with patch.object(engine.db, "get_connection") as mock_conn:
        # Set up the context manager mock to return our rows
        mock_ctx = mock_conn.return_value.__enter__.return_value
        mock_ctx.execute.return_value.fetchall.return_value = [corrupt_row, valid_row]

        with patch("services.prediction_engine.engine.logger") as mock_logger:
            all_day, timed = engine._count_calendar_event_types()

    # Valid event should still be counted despite the corrupt one
    assert timed == 1
    assert all_day == 0

    # Should have logged a debug message about the malformed payload
    debug_calls = mock_logger.debug.call_args_list
    payload_log_found = any(
        "calendar_stats" in str(call) and "malformed event payload" in str(call)
        for call in debug_calls
    )
    assert payload_log_found, (
        f"Expected a 'calendar_stats: skipping malformed event payload' debug log. "
        f"Got debug calls: {debug_calls}"
    )


def test_count_calendar_event_types_returns_zeros_when_all_corrupt(db, user_model_store):
    """_count_calendar_event_types returns (0, 0) when all payloads are malformed."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    corrupt_rows = [{"payload": f"{{corrupt-{i}"} for i in range(3)]

    with patch.object(engine.db, "get_connection") as mock_conn:
        mock_ctx = mock_conn.return_value.__enter__.return_value
        mock_ctx.execute.return_value.fetchall.return_value = corrupt_rows

        all_day, timed = engine._count_calendar_event_types()

    assert all_day == 0
    assert timed == 0
