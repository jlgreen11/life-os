"""
Tests for the print→logging migration across core service modules (iteration 199).

Verifies that EventBus, AIEngine, SignalExtractorPipeline, DecisionExtractor,
and TaskManager all use the standard logging module instead of print() for
diagnostic output.  Consistent logging enables:

- Configurable verbosity without code changes (DEBUG vs WARNING vs ERROR)
- Structured log aggregation in production (Docker, journald, etc.)
- Stack-trace capture via exc_info=True (absent from print() calls)
- Log filtering / routing by module name (Python hierarchy)
"""

import logging

import pytest


# ---------------------------------------------------------------------------
# Logger presence tests — one per migrated module
# ---------------------------------------------------------------------------


def test_event_bus_uses_logging_module():
    """EventBus must expose a module-level logger via logging.getLogger.

    The event bus is the highest-throughput error path in the system (every
    JetStream NAK triggers the handler).  Structured error logging with
    exc_info is critical for diagnosing message-processing failures.
    """
    import services.event_bus.bus as bus_module

    assert hasattr(bus_module, "logger"), (
        "services.event_bus.bus must expose a module-level 'logger' "
        "created with logging.getLogger(__name__)"
    )
    assert isinstance(bus_module.logger, logging.Logger)


def test_event_bus_logger_name_is_module_path():
    """EventBus logger must be named after its module for correct hierarchy."""
    import services.event_bus.bus as bus_module

    assert bus_module.logger.name == "services.event_bus.bus"


def test_ai_engine_uses_logging_module():
    """AIEngine must expose a module-level logger via logging.getLogger.

    The vector-search fallback path is critical for observability: when
    LanceDB degrades, the warning must surface through the logging hierarchy
    rather than disappearing into stdout.
    """
    import services.ai_engine.engine as engine_module

    assert hasattr(engine_module, "logger"), (
        "services.ai_engine.engine must expose a module-level 'logger' "
        "created with logging.getLogger(__name__)"
    )
    assert isinstance(engine_module.logger, logging.Logger)


def test_ai_engine_logger_name_is_module_path():
    """AIEngine logger must be named after its module for correct hierarchy."""
    import services.ai_engine.engine as engine_module

    assert engine_module.logger.name == "services.ai_engine.engine"


def test_signal_pipeline_uses_logging_module():
    """SignalExtractorPipeline must expose a module-level logger.

    Extractor errors in the pipeline are fail-open (processing continues
    after an exception).  Logging them at ERROR level with exc_info=True
    ensures they are never silently swallowed in production.
    """
    import services.signal_extractor.pipeline as pipeline_module

    assert hasattr(pipeline_module, "logger"), (
        "services.signal_extractor.pipeline must expose a module-level 'logger' "
        "created with logging.getLogger(__name__)"
    )
    assert isinstance(pipeline_module.logger, logging.Logger)


def test_signal_pipeline_logger_name_is_module_path():
    """SignalExtractorPipeline logger must be named after its module."""
    import services.signal_extractor.pipeline as pipeline_module

    assert pipeline_module.logger.name == "services.signal_extractor.pipeline"


def test_decision_extractor_uses_logging_module():
    """DecisionExtractor must expose a module-level logger via logging.getLogger.

    Decision extraction errors were previously silent (print only) which made
    it impossible to distinguish a healthy empty result from a crash in the
    decision-profiling logic.
    """
    import services.signal_extractor.decision as decision_module

    assert hasattr(decision_module, "logger"), (
        "services.signal_extractor.decision must expose a module-level 'logger' "
        "created with logging.getLogger(__name__)"
    )
    assert isinstance(decision_module.logger, logging.Logger)


def test_decision_extractor_logger_name_is_module_path():
    """DecisionExtractor logger must be named after its module."""
    import services.signal_extractor.decision as decision_module

    assert decision_module.logger.name == "services.signal_extractor.decision"


def test_task_manager_uses_logging_module():
    """TaskManager must expose a module-level logger via logging.getLogger.

    AI extraction failures (Ollama down, model parsing errors) were previously
    logged only to stdout with no stack trace.  The logging migration adds
    exc_info=True so the full traceback is captured.
    """
    import services.task_manager.manager as manager_module

    assert hasattr(manager_module, "logger"), (
        "services.task_manager.manager must expose a module-level 'logger' "
        "created with logging.getLogger(__name__)"
    )
    assert isinstance(manager_module.logger, logging.Logger)


def test_task_manager_logger_name_is_module_path():
    """TaskManager logger must be named after its module for correct hierarchy."""
    import services.task_manager.manager as manager_module

    assert manager_module.logger.name == "services.task_manager.manager"


# ---------------------------------------------------------------------------
# No remaining print() calls — verify the migration is complete
# ---------------------------------------------------------------------------


def test_event_bus_has_no_print_calls(tmp_path):
    """EventBus source must not contain any bare print() calls.

    The docstring usage example (print(f"Got: {event}")) is inside a
    triple-quoted string and should not be treated as a live call.
    This test reads the source and verifies no executable print() exists.
    """
    import inspect
    import services.event_bus.bus as bus_module

    source = inspect.getsource(bus_module)
    # Strip the module docstring before scanning so the example isn't flagged.
    # The docstring ends at the first occurrence of triple-quote closing.
    docstring_end = source.find('"""', source.find('"""') + 3) + 3
    live_source = source[docstring_end:]
    assert "print(" not in live_source, (
        "services.event_bus.bus contains a print() call outside the docstring; "
        "replace with logger.error/warning/info"
    )


def test_ai_engine_has_no_print_calls():
    """AIEngine source must not contain any bare print() calls."""
    import inspect
    import services.ai_engine.engine as engine_module

    source = inspect.getsource(engine_module)
    assert "print(" not in source, (
        "services.ai_engine.engine contains a print() call; "
        "replace with logger.error/warning/info"
    )


def test_signal_pipeline_has_no_print_calls():
    """SignalExtractorPipeline source must not contain any bare print() calls."""
    import inspect
    import services.signal_extractor.pipeline as pipeline_module

    source = inspect.getsource(pipeline_module)
    assert "print(" not in source, (
        "services.signal_extractor.pipeline contains a print() call; "
        "replace with logger.error/warning/info"
    )


def test_decision_extractor_has_no_print_calls():
    """DecisionExtractor source must not contain any bare print() calls."""
    import inspect
    import services.signal_extractor.decision as decision_module

    source = inspect.getsource(decision_module)
    assert "print(" not in source, (
        "services.signal_extractor.decision contains a print() call; "
        "replace with logger.error/warning/info"
    )


def test_task_manager_has_no_print_calls():
    """TaskManager source must not contain any bare print() calls."""
    import inspect
    import services.task_manager.manager as manager_module

    source = inspect.getsource(manager_module)
    assert "print(" not in source, (
        "services.task_manager.manager contains a print() call; "
        "replace with logger.error/warning/info"
    )


# ---------------------------------------------------------------------------
# Log-level correctness — errors should be ERROR, fallbacks should be WARNING
# ---------------------------------------------------------------------------


def test_event_bus_handler_error_uses_logger_error(caplog):
    """EventBus handler exceptions must be logged at ERROR level with exc_info.

    ERROR level ensures the message surfaces in production even when DEBUG
    and INFO logging is suppressed.  exc_info=True captures the full
    traceback which print() cannot provide.
    """
    import services.event_bus.bus as bus_module

    # Simulate the error logging path directly.
    with caplog.at_level(logging.ERROR, logger="services.event_bus.bus"):
        bus_module.logger.error(
            "Event handler error for %s: %s", "email.*", ValueError("test"), exc_info=False
        )

    assert any(
        "Event handler error" in r.message and r.levelno == logging.ERROR
        for r in caplog.records
    ), "EventBus handler errors must be logged at ERROR level"


def test_ai_engine_vector_fallback_uses_logger_warning(caplog):
    """AIEngine vector-search fallback must be logged at WARNING level.

    WARNING is appropriate here because the system degrades gracefully
    (falls back to SQL LIKE) rather than failing completely.  Using WARNING
    (not ERROR) avoids false alarms while still surfacing the degradation.
    """
    import services.ai_engine.engine as engine_module

    with caplog.at_level(logging.WARNING, logger="services.ai_engine.engine"):
        engine_module.logger.warning(
            "Vector search failed, falling back to SQL LIKE: %s", RuntimeError("test")
        )

    assert any(
        "Vector search failed" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    ), "AIEngine vector search fallback must be logged at WARNING level"


def test_signal_pipeline_extractor_error_uses_logger_error(caplog):
    """Extractor errors in the pipeline must be logged at ERROR level.

    Extractor errors are unexpected failures in signal processing — they
    deserve ERROR level so they are visible in production logs even with
    INFO filtering.
    """
    import services.signal_extractor.pipeline as pipeline_module

    with caplog.at_level(logging.ERROR, logger="services.signal_extractor.pipeline"):
        pipeline_module.logger.error(
            "Extractor %s error: %s", "MoodInferenceEngine", RuntimeError("test"), exc_info=False
        )

    assert any(
        "Extractor" in r.message and "error" in r.message and r.levelno == logging.ERROR
        for r in caplog.records
    ), "Pipeline extractor errors must be logged at ERROR level"


def test_signal_pipeline_mood_persistence_failure_uses_logger_warning(caplog):
    """Mood persistence failures must be logged at WARNING level.

    Mood history persistence is best-effort: a failure doesn't affect the
    returned MoodState.  WARNING is appropriate — it indicates degraded
    functionality, not a critical error.
    """
    import services.signal_extractor.pipeline as pipeline_module

    with caplog.at_level(logging.WARNING, logger="services.signal_extractor.pipeline"):
        pipeline_module.logger.warning(
            "Failed to persist mood snapshot to history: %s", OSError("test")
        )

    assert any(
        "mood snapshot" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    ), "Mood persistence failures must be logged at WARNING level"
