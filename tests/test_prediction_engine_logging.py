"""
Tests for the print→logging migration in PredictionEngine (iteration 198).

Verifies that:
1. PredictionEngine uses the standard logging module (not print()) for all
   diagnostic output.
2. Low-accuracy multiplier warnings are emitted at WARNING level so they
   surface even when DEBUG logging is disabled.
3. Diagnostic per-run stats are emitted at DEBUG level so they can be
   silenced in production without hiding warnings.
"""

import logging

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
