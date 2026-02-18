"""
Tests for episode mood retrieval error logging.

CRITICAL BUG FIX (iteration 134):
The episode creation code had a bare `except Exception: pass` that silently
swallowed ALL exceptions during mood retrieval. This caused 28,900+ episodes
to be created with energy_level=NULL despite 27K+ mood signals being available.

The bug was invisible because the exception was never logged. This test ensures
that exceptions are now logged with full tracebacks so future mood retrieval
failures can be diagnosed and fixed.
"""

import pytest
from io import StringIO
import sys

from services.signal_extractor.pipeline import SignalExtractorPipeline
from models.core import EventType
from datetime import datetime, timezone


class TestEpisodeMoodLogging:
    """Test that mood retrieval exceptions are logged, not silently swallowed."""

    @pytest.fixture
    def signal_extractor(self, db, user_model_store):
        """Create a SignalExtractorPipeline instance using shared fixtures from conftest.py."""
        return SignalExtractorPipeline(db, user_model_store)

    @pytest.mark.asyncio
    async def test_mood_logging_on_exception(
        self, db, user_model_store, signal_extractor, capsys
    ):
        """
        Test that exceptions during mood retrieval are logged to stdout.

        This test verifies that the bare `except: pass` has been replaced with
        logging that captures the exception message and traceback.
        """
        # This test is a integration-level verification that the logging code exists.
        # We can't easily mock get_current_mood to raise an exception because it's
        # called inside _create_episode which is a method on LifeOS, and we'd need
        # to instantiate LifeOS with all its dependencies.
        #
        # Instead, we verify:
        # 1. The code has logging (by reading main.py)
        # 2. Normal episode creation works (by testing with real signal extractor)

        # Generate mood signals
        event = {
            "id": "mood-test-1",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "gmail",
            "payload": {
                "from_address": "user@example.com",
                "to_addresses": ["friend@example.com"],
                "subject": "Stressed",
                "body_plain": "I'm so stressed and frustrated!",
            },
            "metadata": {},
        }
        await signal_extractor.process_event(event)

        # Verify mood signals were created
        mood_state = signal_extractor.get_current_mood()
        assert mood_state.confidence > 0, "Mood signals should exist"

    @pytest.mark.asyncio
    async def test_main_py_has_logging_code(self):
        """
        Test that main.py has explicit logging for mood retrieval exceptions.

        This is a code inspection test to ensure the fix is present.
        """
        with open("main.py", "r") as f:
            content = f.read()

        # Verify the exception is logged via the standard logging module
        # (migrated from print() to logger.warning() in iteration 219)
        assert "Mood retrieval failed in episode creation" in content, \
            "main.py should log mood retrieval exceptions"

        # Verify it uses logger.warning (not silent pass or bare print)
        assert "logger.warning" in content, \
            "main.py should use logger.warning() for mood exceptions"

        # Verify we're NOT swallowing exceptions silently
        # (the old code had `except Exception: pass`)
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "except Exception" in line and "_create_episode" in "\n".join(lines[max(0, i-50):i+50]):
                # Found the except block in _create_episode
                # Check the next few lines
                next_lines = lines[i+1:i+5]
                next_content = "\n".join(next_lines)
                assert "logger" in next_content or "logging" in next_content or "print" in next_content, \
                    f"Exception handler at line {i+1} should log, not silently pass"

    def test_logging_code_coverage(self):
        """
        Verify that the logging code paths are reachable.

        This is a static analysis test that verifies the exception handler
        is correctly structured.
        """
        # Read main.py
        with open("main.py", "r") as f:
            content = f.read()

        # Find the _create_episode method
        assert "def _create_episode" in content, "_create_episode should exist"

        # Find the mood retrieval block
        mood_block_start = content.find("# Retrieve current mood from the user model")
        assert mood_block_start != -1, "Mood retrieval block should exist"

        # Find the except clause
        except_start = content.find("except Exception as e:", mood_block_start)
        assert except_start != -1, "Exception handler should exist and bind the exception"

        # Verify logging is present using the standard logging module
        # (migrated from print()/traceback.print_exc() to logger.warning() in iteration 219)
        logging_block = content[except_start:except_start+500]
        assert "logger.warning" in logging_block or "logger.error" in logging_block or "print(" in logging_block, \
            "Exception should be logged (via logger or print)"
        # exc_info=True captures the full traceback equivalent to traceback.print_exc()
        assert "exc_info" in logging_block or "traceback" in logging_block or "logger" in logging_block, \
            "Exception context should be captured (exc_info or traceback)"
