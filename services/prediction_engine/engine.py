"""
Life OS — Prediction Engine

Forward-looking intelligence. Continuously evaluates the current state
against learned patterns to predict what the user will need before they
know they need it.

This is what creates the "blown away" moments — the AI doing something
helpful that the user never asked for, at exactly the right time.

Prediction Types:
    NEED        — "You'll probably need X soon"
    CONFLICT    — "These two things overlap / contradict"
    OPPORTUNITY — "Good time to do X based on your patterns"
    RISK        — "Something might go wrong if you don't..."
    REMINDER    — "You haven't done X and it's been a while"

Confidence Gates:
    < 0.3  OBSERVE    — Watch silently, keep learning
    0.3-0.6 SUGGEST   — "Would you like me to..."
    0.6-0.8 DEFAULT   — Do it, but make it easy to undo
    > 0.8  AUTONOMOUS — Just handle it
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

from models.core import ConfidenceGate, Priority
from models.user_model import MoodState, Prediction, ReactionPrediction
from services.signal_extractor.marketing_filter import is_marketing_or_noreply
from storage.database import DatabaseManager, UserModelStore


def _parse_score_from_reasoning(reasoning: str) -> float | None:
    """Extract the numeric score from a predict_reaction() reasoning string.

    The reasoning has the form 'score=0.30, dismissals=0, ...'.
    Returns the parsed float or None if parsing fails.
    """
    try:
        # Find the score=X.XX segment
        for part in reasoning.split(","):
            part = part.strip()
            if part.startswith("score="):
                return float(part.split("=", 1)[1])
    except (ValueError, IndexError):
        pass
    return None


def _parse_penalty_frequency(
    reasoning: str,
    prediction_type: str,
    freq: dict[str, int],
) -> None:
    """Increment penalty frequency counters based on a reasoning string.

    Modifies *freq* in-place.  Each penalty is counted at most once per
    prediction (boolean presence, not magnitude).
    """
    try:
        for part in reasoning.split(","):
            part = part.strip()
            if part.startswith("stress_signals="):
                count = int(part.split("=", 1)[1])
                if count > 2:  # matches the >2 threshold in predict_reaction
                    freq["stress"] += 1
            elif part.startswith("dismissals="):
                count = int(part.split("=", 1)[1])
                if count > 5:  # matches the >5 threshold in predict_reaction
                    freq["dismissals"] += 1
            elif part.startswith("quiet_hours=True"):
                freq["quiet_hours"] += 1
            elif part.startswith("low_activity=True"):
                freq["low_activity"] += 1
        # Opportunity type penalty is applied in predict_reaction when type == "opportunity"
        if prediction_type == "opportunity":
            freq["opportunity_type"] += 1
    except (ValueError, IndexError):
        pass  # Non-fatal; penalty tracking is best-effort


class PredictionEngine:
    """
    Generates predictions about user needs by combining:
    - Current context (time, location, calendar, mood)
    - Signal profiles (behavioral patterns)
    - Semantic memory (known facts & preferences)
    - Episodic memory (past similar situations)
    """

    def __init__(self, db: DatabaseManager, ums: UserModelStore, timezone: str = "America/Los_Angeles"):
        self.db = db   # Database access for events, user_model, and preferences tables
        self.ums = ums  # User-model store for signal profiles and semantic memory
        self._tz_name = timezone
        self._last_event_cursor: int = 0  # rowid of last processed event
        self._last_time_based_run: Optional[datetime] = None  # Last time-based prediction run
        self._first_follow_up_run: bool = True  # First cycle uses wider lookback (72h vs 24h)

        # Diagnostic counters for monitoring prediction pipeline health.
        # Updated at the end of each generate_predictions() cycle and
        # queryable via get_diagnostics().
        self._last_run_diagnostics: dict[str, Any] = {}
        self._total_runs: int = 0
        self._total_predictions_generated: int = 0
        self._total_predictions_surfaced: int = 0
        self._consecutive_zero_runs: int = 0

        # Store failure tracking — makes silent prediction storage drops visible.
        # _store_failure_count is a lifetime counter; _last_store_errors is a
        # capped ring buffer of the most recent 10 errors for diagnostics.
        self._store_failure_count: int = 0
        self._last_store_errors: list[dict] = []

        # Post-store verification flag — set to True if predictions appear to
        # store successfully but are not found on a subsequent read.  Indicates
        # a silent persistence failure (e.g. WAL not checkpointed, DB recovery).
        self._persistence_failure_detected: bool = False

        # Per-check generation breakdown from the last generate_predictions() run.
        # Persisted to prediction_engine_state so it survives restarts and is
        # available in diagnostics for debugging 0-prediction anomalies.
        self._last_generation_stats: dict[str, Any] = {}
        self._last_generation_timestamp: Optional[str] = None

        # Tracks consecutive prediction cycles where the surfacing rate is 0%.
        # When this exceeds 3, the filtering log is escalated from DEBUG to
        # WARNING so operators notice that no predictions are reaching users.
        self._zero_surfacing_cycles: int = 0

        # Per-run surfacing diagnostics — tracks WHY predictions are filtered
        # so operators can diagnose 0% surfacing rates.  Populated at the end
        # of each generate_predictions() cycle and included in both
        # get_runtime_diagnostics() and get_diagnostics().
        self._surfacing_diagnostics: dict[str, Any] = self._empty_surfacing_diagnostics()

        # Lazy-loaded cache mapping lowercase email addresses → contact names
        # from the entities.db contacts table.  Refreshed every 30 minutes.
        self._contact_email_map: dict[str, str] = {}
        self._contact_email_map_loaded_at: Optional[datetime] = None

        # Ensure prediction_engine_state table exists and load any persisted state.
        # Both are wrapped in try/except so a corrupted user_model.db doesn't
        # prevent the engine from instantiating at all.
        try:
            self._ensure_state_table()
        except Exception as e:
            logger.warning("_ensure_state_table failed (will operate without persisted state): %s", e)

        try:
            self._load_persisted_state()
        except Exception as e:
            logger.warning("_load_persisted_state failed (using defaults: cursor=0, last_run=None): %s", e)

    def _ensure_state_table(self) -> None:
        """Create the prediction_engine_state table if it doesn't exist."""
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS prediction_engine_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )

    def _load_persisted_state(self) -> None:
        """Load persisted cursor and last-run timestamp from the state table.

        If no persisted state exists (fresh install or first boot), the defaults
        set in __init__ (cursor=0, last_run=None) are preserved.
        """
        with self.db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT key, value FROM prediction_engine_state"
            ).fetchall()

        state = {row["key"]: row["value"] for row in rows}

        if "last_event_cursor" in state:
            self._last_event_cursor = int(state["last_event_cursor"])
        if "last_time_based_run" in state:
            self._last_time_based_run = datetime.fromisoformat(state["last_time_based_run"])
        if "last_generation_stats" in state:
            try:
                self._last_generation_stats = json.loads(state["last_generation_stats"])
            except (json.JSONDecodeError, TypeError):
                pass  # Corrupt stats are non-fatal; keep empty dict default
        if "last_generation_timestamp" in state:
            self._last_generation_timestamp = state["last_generation_timestamp"]
        if "last_run_diagnostics" in state:
            try:
                self._last_run_diagnostics = json.loads(state["last_run_diagnostics"])
                # Restore aggregate counters from the persisted diagnostics snapshot
                self._total_runs = self._last_run_diagnostics.get("total_runs", 0)
                self._total_predictions_generated = self._last_run_diagnostics.get("total_generated", 0)
                self._total_predictions_surfaced = self._last_run_diagnostics.get("total_surfaced", 0)
                self._consecutive_zero_runs = self._last_run_diagnostics.get("consecutive_zero_runs", 0)
            except (json.JSONDecodeError, TypeError):
                pass  # Corrupt diagnostics are non-fatal; keep __init__ defaults

        if state:
            logger.info(
                "Prediction engine restored state: cursor=%d, last_time_run=%s",
                self._last_event_cursor,
                self._last_time_based_run,
            )
        else:
            logger.info("Prediction engine starting fresh (no persisted state)")

    def _persist_state(self, key: str, value: str) -> None:
        """Persist a single state key-value pair via INSERT OR REPLACE.

        Wrapped in try/except so a corrupted user_model.db doesn't crash the
        prediction pipeline. Failure to persist state is non-fatal — the engine
        will simply re-process some events on the next cycle.
        """
        try:
            with self.db.get_connection("user_model") as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO prediction_engine_state (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, value, datetime.now(timezone.utc).isoformat()),
                )
        except Exception as e:
            logger.warning("_persist_state(%s) failed (non-fatal, will retry next cycle): %s", key, e)

    def reset_state(self) -> None:
        """Reset in-memory state after DB recovery.

        Called by _db_health_loop after user_model.db rebuild wipes the
        prediction_engine_state table. Without this, in-memory cursors
        and counters retain stale values that prevent event-based
        predictions from regenerating.

        Thread safety: both the prediction loop and _db_health_loop are
        coroutines in the same asyncio event loop, so there is no true
        concurrency issue — no lock is needed.
        """
        logger.info(
            "PredictionEngine: resetting state (was: cursor=%d, last_run=%s, "
            "total_runs=%d, consecutive_zero=%d)",
            self._last_event_cursor,
            self._last_time_based_run,
            self._total_runs,
            self._consecutive_zero_runs,
        )
        self._last_event_cursor = 0
        self._last_time_based_run = None
        self._first_follow_up_run = True
        self._total_runs = 0
        self._total_predictions_generated = 0
        self._total_predictions_surfaced = 0
        self._consecutive_zero_runs = 0
        self._store_failure_count = 0
        self._persistence_failure_detected = False
        self._last_store_errors = []
        self._last_generation_stats = {}
        self._last_generation_timestamp = None
        self._last_run_diagnostics = {}
        self._zero_surfacing_cycles = 0
        self._surfacing_diagnostics = self._empty_surfacing_diagnostics()

    def _load_contact_email_map(self) -> None:
        """Lazily build a mapping from lowercase email addresses to contact names.

        Queries the entities.db contacts table and parses each contact's
        ``emails`` JSON array.  The result is cached in ``self._contact_email_map``
        and refreshed at most every 30 minutes so that newly-added contacts
        are picked up without requiring a restart.
        """
        now = datetime.now(timezone.utc)
        if (
            self._contact_email_map_loaded_at is not None
            and (now - self._contact_email_map_loaded_at).total_seconds() < 1800
        ):
            return  # Cache is still fresh

        email_map: dict[str, str] = {}
        try:
            with self.db.get_connection("entities") as conn:
                rows = conn.execute("SELECT name, emails FROM contacts").fetchall()
            for row in rows:
                name = row["name"]
                if not name:
                    continue
                try:
                    emails = json.loads(row["emails"]) if row["emails"] else []
                except (json.JSONDecodeError, TypeError):
                    emails = []
                for email in emails:
                    if isinstance(email, str) and email.strip():
                        email_map[email.strip().lower()] = name
        except Exception as e:
            logger.warning("_load_contact_email_map failed (will use email-prefix fallback): %s", e)

        self._contact_email_map = email_map
        self._contact_email_map_loaded_at = now

    def _resolve_contact_name(self, email: str) -> str:
        """Resolve an email address to a human-readable contact name.

        Looks up the address in the entities.db contacts cache.  Returns the
        stored contact name when available, otherwise falls back to the local
        part of the email address (everything before the ``@``).

        Args:
            email: The email address to resolve.

        Returns:
            The contact's display name, or the email prefix as a fallback.
        """
        try:
            self._load_contact_email_map()
            name = self._contact_email_map.get(email.lower().strip())
            if name:
                return name
        except Exception:
            pass  # Any error falls through to heuristic
        return email.split("@")[0] if "@" in email else email

    def get_runtime_diagnostics(self) -> dict[str, Any]:
        """Return prediction engine runtime diagnostic information for monitoring.

        Lightweight, synchronous method that reads only in-memory state.
        Designed for the admin dashboard and data-quality endpoint to query
        prediction pipeline health without hitting the database.

        For comprehensive per-prediction-type analysis, use the async
        get_diagnostics() method instead.

        Returns:
            dict with 'engine_state', 'run_statistics', and 'health' keys.
            'health' is 'degraded' after 4+ consecutive zero-prediction runs.
        """
        return {
            "engine_state": {
                "last_event_cursor": self._last_event_cursor,
                "last_time_based_run": (
                    self._last_time_based_run.isoformat() if self._last_time_based_run else None
                ),
            },
            "run_statistics": self._last_run_diagnostics,
            "health": "degraded" if self._consecutive_zero_runs >= 4 else "ok",
            "store_failures": {
                "total": self._store_failure_count,
                "recent_errors": self._last_store_errors,
            },
            "persistence_failure_detected": self._persistence_failure_detected,
            "zero_surfacing_cycles": self._zero_surfacing_cycles,
            "surfacing": self._surfacing_diagnostics,
            "last_generation_breakdown": self._last_generation_stats or None,
            "last_generation_timestamp": self._last_generation_timestamp,
        }

    @staticmethod
    def _empty_surfacing_diagnostics() -> dict[str, Any]:
        """Return a fresh surfacing diagnostics dict with zeroed counters.

        Called once at __init__ and again at the start of each
        generate_predictions() cycle to reset per-run tracking.
        """
        return {
            "total_generated": 0,
            "filtered_by_reaction": {"total": 0, "helpful": 0, "neutral": 0, "annoying": 0},
            "filtered_by_confidence": 0,
            "score_distribution": {"below_neg0.1": 0, "neg0.1_to_0.2": 0, "0.2_to_0.5": 0, "above_0.5": 0},
            "penalty_frequency": {
                "stress": 0,
                "dismissals": 0,
                "quiet_hours": 0,
                "low_activity": 0,
                "opportunity_type": 0,
            },
            "sample_filtered_reasons": [],
        }

    def _bucket_reaction_score(self, score: float) -> str:
        """Classify a reaction score into a histogram bucket.

        Buckets:
            < -0.1       → 'below_neg0.1'
            -0.1 to 0.2  → 'neg0.1_to_0.2'
            0.2 to 0.5   → '0.2_to_0.5'
            > 0.5        → 'above_0.5'
        """
        if score < -0.1:
            return "below_neg0.1"
        elif score < 0.2:
            return "neg0.1_to_0.2"
        elif score <= 0.5:
            return "0.2_to_0.5"
        else:
            return "above_0.5"

    def _has_new_events(self) -> bool:
        """Check if any new events have arrived since last prediction run.

        Updates the in-memory cursor but does NOT persist it to the database.
        The cursor is only persisted after generate_predictions() completes
        successfully, so that events are not permanently skipped if the
        prediction pipeline fails mid-way.
        """
        with self.db.get_connection("events") as conn:
            row = conn.execute(
                "SELECT MAX(rowid) as max_id FROM events"
            ).fetchone()
            current_max = row["max_id"] if row and row["max_id"] else 0

        if current_max <= self._last_event_cursor:
            return False

        self._last_event_cursor = current_max
        return True

    def _should_run_time_based_predictions(self) -> bool:
        """
        Check if time-based predictions should run.

        Time-based predictions check temporal conditions (time passing, approaching events,
        missed routines) rather than reacting to new events. They should run periodically
        even when no new events have arrived.

        Examples of time-based predictions:
        - Relationship maintenance (days since last contact increasing)
        - Routine deviations (expected routine didn't occur)
        - Preparation needs (event approaching in time)
        - Calendar conflicts (future events coming into 48h window)

        These run every 15 minutes to detect changes in temporal state.
        """
        now = datetime.now(timezone.utc)

        # First run always executes
        if self._last_time_based_run is None:
            self._last_time_based_run = now
            self._persist_state("last_time_based_run", now.isoformat())
            return True

        # Run if 15+ minutes have passed since last time-based check
        time_since_last = (now - self._last_time_based_run).total_seconds() / 60
        if time_since_last >= 15:
            self._last_time_based_run = now
            self._persist_state("last_time_based_run", now.isoformat())
            return True

        return False

    async def generate_predictions(self, current_context: dict) -> list[Prediction]:
        """
        Main prediction loop. Called periodically (every 15 min)
        and on significant context changes (location, calendar event start, etc.)

        Runs prediction generation when EITHER:
        1. New events have arrived (event-based predictions like follow-up needs)
        2. 15+ minutes have passed (time-based predictions like relationship maintenance)

        This dual-trigger approach ensures both reactive predictions (responding to
        new events) and proactive predictions (detecting temporal conditions) work
        correctly.
        """
        # Determine which trigger conditions are met.
        # Wrapped in try/except so DB errors (e.g. corrupted user_model.db)
        # default to True rather than aborting the entire pipeline.
        generation_stats = {}

        # --- Persistence failure recovery ---
        # If the previous cycle detected that predictions were lost after storage
        # (e.g., due to DB corruption recovery dropping the table), perform
        # corrective actions to ensure this cycle's predictions actually persist.
        if self._persistence_failure_detected:
            logger.warning(
                'Prediction persistence failure detected in previous cycle — '
                'running recovery: verifying DB, clearing pre-filter cache'
            )
            try:
                test_id = '__persistence_test__'
                with self.db.get_connection('user_model') as conn:
                    conn.execute(
                        'INSERT OR REPLACE INTO predictions '
                        '(id, prediction_type, description, confidence, confidence_gate, '
                        'resolved_at, user_response) '
                        'VALUES (?, ?, ?, ?, ?, datetime("now"), ?)',
                        (test_id, 'test', 'persistence_check', 0.0, 'OBSERVE', 'filtered'),
                    )
                with self.db.get_connection('user_model') as conn:
                    row = conn.execute(
                        'SELECT id FROM predictions WHERE id = ?', (test_id,)
                    ).fetchone()
                    if row:
                        conn.execute('DELETE FROM predictions WHERE id = ?', (test_id,))
                        logger.info('Persistence recovery: DB write test PASSED — clearing failure flag')
                        self._persistence_failure_detected = False
                    else:
                        logger.critical(
                            'Persistence recovery: DB write test FAILED — predictions '
                            'table is not persisting writes. Skipping this cycle.'
                        )
                        return []
            except Exception as e:
                logger.critical(
                    'Persistence recovery: DB write test raised %s: %s — '
                    'skipping this cycle', type(e).__name__, e
                )
                return []

            # Mark this cycle as running in recovery mode for diagnostics
            generation_stats['recovery_mode'] = True

        try:
            has_new_events = self._has_new_events()
        except Exception as e:
            logger.warning("_has_new_events failed (defaulting to True): %s", e)
            has_new_events = True
            generation_stats["trigger_errors"] = f"has_new_events: {e}"

        try:
            time_based_due = self._should_run_time_based_predictions()
        except Exception as e:
            logger.warning("_should_run_time_based_predictions failed (defaulting to True): %s", e)
            time_based_due = True
            prev = generation_stats.get("trigger_errors", "")
            generation_stats["trigger_errors"] = f"{prev}; time_based: {e}".lstrip("; ")

        # Skip entirely if neither trigger is active
        if not has_new_events and not time_based_due:
            logger.info(
                "Prediction engine skipped: no new events (cursor=%d) and "
                "time-based not due (last_run=%s, interval=15m)",
                self._last_event_cursor,
                self._last_time_based_run,
            )
            return []

        # Pre-load existing unresolved predictions to skip regenerating them.
        # This reduces the 16x dedup waste where identical predictions are
        # generated, processed through filtering, then discarded at storage time.
        existing_predictions: set[tuple[str, str]] = set()
        try:
            with self.db.get_connection('user_model') as conn:
                rows = conn.execute(
                    """SELECT prediction_type, description FROM predictions
                       WHERE resolved_at IS NULL
                          OR datetime(resolved_at) > datetime('now', '-24 hours')"""
                ).fetchall()
                existing_predictions = {(r[0], r[1]) for r in rows}
            if existing_predictions:
                logger.debug(
                    'Pre-filter: %d existing predictions will be skipped',
                    len(existing_predictions),
                )
        except Exception as e:
            logger.warning('Pre-filter query failed (proceeding without filter): %s', e)

        # Proactive persistence failure detection: if the table is empty
        # but we've had store failures, flag for recovery on next cycle.
        if not existing_predictions and self._store_failure_count > 0:
            if not self._persistence_failure_detected:
                logger.warning(
                    'Predictions table is empty but %d store failures recorded — '
                    'flagging for persistence recovery on next cycle',
                    self._store_failure_count,
                )
                self._persistence_failure_detected = True

        predictions = []

        # --- Prediction generation pipeline ---
        # Each _check_* method is a specialized detector that looks for one
        # category of predictable user need. They run independently and
        # return zero or more Prediction objects. The full set of prediction
        # types covers the most common "blown away" moments:

        # TIME-BASED predictions: Run when time passes (even without new events)
        # These check temporal conditions like approaching events, missed routines,
        # and relationship gaps growing wider.
        #
        # Each _check method is wrapped in its own try/except so that a failure
        # in one detector (e.g. sqlite3.DatabaseError from a corrupted DB) does
        # not abort the entire pipeline. This follows the fail-open pattern used
        # by master_event_handler in main.py.
        if time_based_due:
            try:
                calendar_preds = await self._check_calendar_conflicts(current_context)
                generation_stats['calendar_conflicts'] = len(calendar_preds)
                predictions.extend(calendar_preds)
            except Exception as e:
                generation_stats['calendar_conflicts'] = f'error: {e}'
                logger.error("calendar_conflicts check failed: %s", e)

            try:
                routine_preds = await self._check_routine_deviations(current_context)
                generation_stats['routine_deviations'] = len(routine_preds)
                predictions.extend(routine_preds)
            except Exception as e:
                generation_stats['routine_deviations'] = f'error: {e}'
                logger.error("routine_deviations check failed: %s", e)

            try:
                relationship_preds = await self._check_relationship_maintenance(current_context)
                generation_stats['relationship_maintenance'] = len(relationship_preds)
                predictions.extend(relationship_preds)
            except Exception as e:
                generation_stats['relationship_maintenance'] = f'error: {e}'
                logger.error("relationship_maintenance check failed: %s", e)

            try:
                prep_preds = await self._check_preparation_needs(current_context)
                generation_stats['preparation_needs'] = len(prep_preds)
                predictions.extend(prep_preds)
            except Exception as e:
                generation_stats['preparation_needs'] = f'error: {e}'
                logger.error("preparation_needs check failed: %s", e)

            try:
                reminder_preds = await self._check_calendar_event_reminders(current_context)
                generation_stats['calendar_reminders'] = len(reminder_preds)
                predictions.extend(reminder_preds)
            except Exception as e:
                generation_stats['calendar_reminders'] = f'error: {e}'
                logger.error("calendar_reminders check failed: %s", e)

            try:
                connector_preds = await self._check_connector_health(current_context)
                generation_stats['connector_health'] = len(connector_preds)
                predictions.extend(connector_preds)
            except Exception as e:
                generation_stats['connector_health'] = f'error: {e}'
                logger.error("connector_health check failed: %s", e)
        else:
            # Skip time-based predictions, mark as not run
            generation_stats['calendar_conflicts'] = '(skipped: no time trigger)'
            generation_stats['routine_deviations'] = '(skipped: no time trigger)'
            generation_stats['relationship_maintenance'] = '(skipped: no time trigger)'
            generation_stats['preparation_needs'] = '(skipped: no time trigger)'
            generation_stats['calendar_reminders'] = '(skipped: no time trigger)'
            generation_stats['connector_health'] = '(skipped: no time trigger)'

        # HYBRID predictions: Run on EITHER trigger (new events OR time-based).
        # Follow-up checks need to run on time-based triggers too, because emails
        # age past the 3-hour grace period even when no new events arrive (e.g.,
        # after a connector outage). The deduplication logic in _check_follow_up_needs
        # (already_predicted_messages set) prevents duplicate predictions.
        try:
            followup_preds = await self._check_follow_up_needs(current_context)
            generation_stats['follow_up_needs'] = len(followup_preds)
            predictions.extend(followup_preds)
        except Exception as e:
            generation_stats['follow_up_needs'] = f'error: {e}'
            logger.error("follow_up_needs check failed: %s", e)

        # EVENT-BASED predictions: Run when new events arrive
        # These react to specific events like new spending activity.
        if has_new_events:
            try:
                spending_preds = await self._check_spending_patterns(current_context)
                generation_stats['spending_patterns'] = len(spending_preds)
                predictions.extend(spending_preds)
            except Exception as e:
                generation_stats['spending_patterns'] = f'error: {e}'
                logger.error("spending_patterns check failed: %s", e)
        else:
            # Skip event-based predictions, mark as not run
            generation_stats['spending_patterns'] = '(skipped: no new events)'

        # Filter out predictions that match existing unresolved ones.
        # This mirrors the dedup check in store_prediction() but avoids
        # the overhead of per-prediction DB queries and telemetry events.
        if existing_predictions:
            before_count = len(predictions)
            predictions = [
                p for p in predictions
                if (p.prediction_type, p.description) not in existing_predictions
            ]
            skipped = before_count - len(predictions)
            if skipped:
                generation_stats['pre_filtered'] = skipped
                logger.debug('Pre-filter: skipped %d duplicate predictions', skipped)

        logger.info(
            "Generated predictions by type: %s (total=%d) [triggers: events=%s, time=%s]",
            generation_stats, len(predictions), has_new_events, time_based_due,
        )

        # --- Not-relevant suppression ---
        # Filter out predictions the user has explicitly marked as "not_relevant"
        # ("Not About Me"). Without this, dismissed predictions regenerate
        # indefinitely — the feedback is recorded but never acted upon.
        suppressed_keys = self._get_suppressed_prediction_keys()
        if suppressed_keys:
            before_suppression = len(predictions)
            predictions = [p for p in predictions if not self._is_suppressed(p, suppressed_keys)]
            suppressed_count = before_suppression - len(predictions)
            if suppressed_count:
                logger.debug(
                    "Suppressed %d predictions based on not_relevant feedback", suppressed_count,
                )

        # --- Accuracy-based confidence decay/boost ---
        # Adjust confidence based on historical accuracy for each prediction type.
        # This closes the feedback loop: predictions that keep getting dismissed
        # have their confidence reduced, eventually suppressing them entirely.
        #
        # For opportunity predictions we apply a *second* per-contact multiplier
        # on top of the global type multiplier.  Different contacts have very
        # different response patterns: the user may reliably reach out to their
        # mother but rarely act on suggestions to contact a distant acquaintance.
        # Knowing this per-contact history lets the engine suppress low-value
        # suggestions and surface high-value ones with boosted confidence.
        for pred in predictions:
            multiplier = self._get_accuracy_multiplier(pred.prediction_type)
            # For opportunity predictions, additionally scale by per-contact accuracy
            if pred.prediction_type == "opportunity" and pred.supporting_signals:
                contact_email = pred.supporting_signals.get("contact_email")
                if contact_email:
                    contact_multiplier = self._get_contact_accuracy_multiplier(contact_email)
                    multiplier *= contact_multiplier
            # Clamp to [0.0, 1.0] after multiplier to prevent NaN/inf propagation
            pred.confidence = max(0.0, min(1.0, pred.confidence * multiplier))
            pred.confidence_gate = self._gate_from_confidence(pred.confidence)

        # --- Reaction prediction gatekeeper ---
        # Before surfacing any prediction, ask: "Will the user find this
        # helpful or annoying right now?" This prevents piling on during
        # stressful moments or when the user has been dismissing alerts.
        #
        # Track reaction predictions for each prediction so we can log filter reasons
        reaction_map = {}
        filtered = []

        # Reset per-run surfacing diagnostics
        surf_diag = self._empty_surfacing_diagnostics()
        surf_diag["total_generated"] = len(predictions)

        for pred in predictions:
            reaction = await self.predict_reaction(pred, current_context)
            reaction_map[pred.id] = reaction

            # Parse the structured reasoning to extract penalty flags.
            # The reasoning string has the form:
            #   "score=X.XX, dismissals=N, stress_signals=N, quiet_hours=True/False, low_activity=True/False"
            reasoning = reaction.reasoning
            _parse_penalty_frequency(reasoning, pred.prediction_type, surf_diag["penalty_frequency"])

            # Extract the numeric score from reasoning for histogram bucketing
            reaction_score = _parse_score_from_reasoning(reasoning)
            if reaction_score is not None:
                bucket = self._bucket_reaction_score(reaction_score)
                surf_diag["score_distribution"][bucket] += 1

            # Track reaction classification counts
            surf_diag["filtered_by_reaction"][reaction.predicted_reaction] = (
                surf_diag["filtered_by_reaction"].get(reaction.predicted_reaction, 0) + 1
            )

            if reaction.predicted_reaction in ("helpful", "neutral"):
                filtered.append(pred)
            else:
                # Mark why this prediction was filtered (annoying reaction)
                pred.filter_reason = f"reaction:{reaction.predicted_reaction} ({reaction.reasoning})"
                surf_diag["filtered_by_reaction"]["total"] += 1
                # Capture sample filtered reasons (capped at 5)
                if len(surf_diag["sample_filtered_reasons"]) < 5:
                    surf_diag["sample_filtered_reasons"].append({
                        "prediction_type": pred.prediction_type,
                        "score": reaction_score,
                        "reaction": reaction.predicted_reaction,
                        "reason": f"reaction:{reaction.predicted_reaction}",
                        "reasoning": reasoning,
                    })

        # Confidence floor — don't surface anything below SUGGEST threshold (0.3).
        # This enables relationship maintenance, preparation, and other valuable
        # predictions that should be shown as suggestions ("Would you like...?")
        # even if confidence isn't high enough for autonomous action.
        filtered_after_confidence = []
        for p in filtered:
            if p.confidence >= 0.3:
                filtered_after_confidence.append(p)
            else:
                # Mark why this prediction was filtered (low confidence)
                p.filter_reason = f"confidence:{p.confidence:.3f} (threshold:0.3)"
                surf_diag["filtered_by_confidence"] += 1
                # Capture sample filtered reasons (capped at 5)
                if len(surf_diag["sample_filtered_reasons"]) < 5:
                    surf_diag["sample_filtered_reasons"].append({
                        "prediction_type": p.prediction_type,
                        "score": None,
                        "reaction": "passed_reaction",
                        "reason": f"confidence:{p.confidence:.3f}",
                        "reasoning": f"below 0.3 threshold (confidence={p.confidence:.3f})",
                    })
        filtered = filtered_after_confidence

        # Persist surfacing diagnostics for this run
        self._surfacing_diagnostics = surf_diag

        # Sort by confidence to prioritize delivery order (notification manager uses this).
        # We no longer cap the number of surfaced predictions - if a prediction passes
        # both reaction gating and confidence threshold, it should surface. The
        # notification manager handles batching/digest grouping to prevent spam.
        #
        # Previously, a hard-coded top-5 limit caused 6.6% of legitimate DEFAULT-gate
        # predictions (confidence 0.6-0.8) to be silently discarded purely due to
        # ranking position. This defeated the purpose of confidence gates.
        #
        # Ranking now serves its proper purpose: determining priority order for
        # delivery, not existence.
        filtered.sort(key=lambda p: p.confidence, reverse=True)

        # Log filtering results for observability
        filtered_by_reaction = len([p for p in predictions if p.filter_reason and p.filter_reason.startswith("reaction:")])
        filtered_by_confidence = len([p for p in predictions if p.filter_reason and p.filter_reason.startswith("confidence:")])

        # Track consecutive zero-surfacing cycles and escalate log level
        # when the surfacing rate stays at 0% for too long.
        if len(predictions) > 0 and len(filtered) == 0:
            self._zero_surfacing_cycles += 1
        else:
            self._zero_surfacing_cycles = 0

        if self._zero_surfacing_cycles >= 3:
            logger.warning(
                "Prediction surfacing rate has been 0%% for %d consecutive cycles — "
                "check reaction prediction and confidence gates. "
                "This run: %d raw → %d surfaced (filtered: %d by reaction, %d by confidence)",
                self._zero_surfacing_cycles,
                len(predictions), len(filtered), filtered_by_reaction, filtered_by_confidence,
            )
        else:
            logger.debug(
                "Filtering: %d raw → %d surfaced (filtered: %d by reaction, %d by confidence)",
                len(predictions), len(filtered), filtered_by_reaction, filtered_by_confidence,
            )

        # Store ALL predictions (including filtered-out ones) for accuracy
        # tracking. Mark which ones were actually surfaced so the feedback
        # loop can distinguish them via was_surfaced=1 in queries.
        #
        # CRITICAL: Predictions that don't pass reaction prediction or
        # confidence gates are immediately resolved as 'filtered'. This
        # prevents database bloat from hundreds of thousands of unsurfaced
        # predictions that will never be shown to the user.
        surfaced_ids = {p.id for p in filtered}
        now = datetime.now(timezone.utc).isoformat()
        run_store_failures = 0  # Per-run counter; resets each cycle
        stored_count = 0  # Tracks successful store_prediction() calls this run

        for pred in predictions:
            pred.was_surfaced = pred.id in surfaced_ids

            # If this prediction was filtered out, mark it as resolved
            # immediately with user_response='filtered'. This closes the
            # lifecycle for predictions that never surface, preventing them
            # from accumulating in the database indefinitely.
            if not pred.was_surfaced:
                pred.resolved_at = now
                pred.user_response = 'filtered'
                # Ensure filter_reason is set (if not already set by filtering logic above)
                if not pred.filter_reason:
                    pred.filter_reason = "unknown (should not happen)"

            try:
                self.ums.store_prediction(pred.model_dump())
                stored_count += 1
            except Exception as e:
                logger.error("Failed to store prediction %s: %s", pred.id, e)
                run_store_failures += 1
                self._store_failure_count += 1
                # Keep a capped ring buffer of the 10 most recent store errors
                # so diagnostics can show actionable details without unbounded growth.
                self._last_store_errors.append({
                    "timestamp": now,
                    "prediction_id": pred.id,
                    "prediction_type": pred.prediction_type if hasattr(pred, "prediction_type") else "unknown",
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                })
                if len(self._last_store_errors) > 10:
                    self._last_store_errors = self._last_store_errors[-10:]

        if run_store_failures > 0:
            logger.warning(
                "Prediction storage: %d/%d predictions failed to store",
                run_store_failures,
                len(predictions),
            )

        # --- Post-store verification ---
        # Detect the case where store_prediction() appeared to succeed but
        # data was lost (DB recovery, WAL issue, etc.).  This makes the
        # critical 'predictions generated but not persisted' anomaly visible
        # at runtime instead of only in periodic data-quality reports.
        #
        # We query by created_at (>= run start minus 60s buffer) rather than
        # resolved_at IS NULL, because filtered predictions have resolved_at
        # set before storage — a 0% surfacing rate is normal and should not
        # trigger the alarm.
        if stored_count > 0:
            try:
                run_start = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
                with self.db.get_connection("user_model") as conn:
                    actual_count = conn.execute(
                        "SELECT COUNT(*) FROM predictions WHERE created_at >= ?",
                        (run_start,),
                    ).fetchone()[0]
                if actual_count == 0:
                    logger.critical(
                        "PREDICTION PERSISTENCE FAILURE: stored %d predictions this run "
                        "but found 0 rows with created_at in the last 60s — data is being lost. "
                        "Check for DB recovery events or WAL checkpoint issues.",
                        stored_count,
                    )
                    self._persistence_failure_detected = True
            except Exception as e:
                logger.warning("Post-store verification query failed: %s", e)

        # Summary log for zero-prediction debugging: reports total output,
        # per-method breakdown, and signal profile availability so operators
        # can quickly identify why the intelligence layer isn't producing.
        profile_types = ["relationships", "cadence", "mood_signals", "linguistic", "topics"]
        try:
            available_profiles = [pt for pt in profile_types if self.ums.get_signal_profile(pt)]
        except Exception:
            available_profiles = []
        logger.info(
            "Prediction engine completed: %d predictions generated (from %d raw, %d filtered). "
            "Methods: %s. Signal profiles available: %d/%d (%s)",
            len(filtered),
            len(predictions),
            len(predictions) - len(filtered),
            generation_stats,
            len(available_profiles),
            len(profile_types),
            ", ".join(available_profiles) if available_profiles else "none",
        )

        # --- Persist diagnostics for programmatic monitoring ---
        self._total_runs += 1
        self._total_predictions_generated += len(predictions)
        self._total_predictions_surfaced += len(filtered)
        if len(filtered) == 0:
            self._consecutive_zero_runs += 1
        else:
            self._consecutive_zero_runs = 0

        # Fire a one-time warning when the engine enters degraded state
        # (4 consecutive cycles with zero surfaced predictions). Uses == 4
        # rather than >= 4 so the alert fires exactly once.
        if self._consecutive_zero_runs == 4:
            logger.warning(
                "Prediction engine DEGRADED: 4 consecutive cycles produced zero "
                "surfaced predictions. Last run stats: %s. Signal profiles: %s",
                generation_stats,
                available_profiles,
            )

        self._last_run_diagnostics = {
            "last_run_at": datetime.now(timezone.utc).isoformat(),
            "total_runs": self._total_runs,
            "total_generated": self._total_predictions_generated,
            "total_surfaced": self._total_predictions_surfaced,
            "consecutive_zero_runs": self._consecutive_zero_runs,
            "last_run_stats": generation_stats,
            "last_run_raw_count": len(predictions),
            "last_run_surfaced_count": len(filtered),
            "last_run_filtered_by_reaction": filtered_by_reaction,
            "last_run_filtered_by_confidence": filtered_by_confidence,
            "signal_profiles_available": available_profiles,
            "signal_profiles_total": len(profile_types),
            "triggers": {"has_new_events": has_new_events, "time_based_due": time_based_due},
            "stored_count": stored_count,
            "zero_surfacing_cycles": self._zero_surfacing_cycles,
            "store_failures_this_run": run_store_failures,
            "store_failures_total": self._store_failure_count,
            "surfacing": self._surfacing_diagnostics,
        }
        self._persist_state("last_run_diagnostics", json.dumps(self._last_run_diagnostics))

        # Persist per-check generation breakdown for zero-prediction debugging.
        # This lets diagnostics show exactly which _check_* methods returned 0
        # vs errored vs were skipped, even after a restart.
        self._last_generation_stats = generation_stats
        self._last_generation_timestamp = datetime.now(timezone.utc).isoformat()
        self._persist_state("last_generation_stats", json.dumps(generation_stats))
        self._persist_state("last_generation_timestamp", self._last_generation_timestamp)

        # Persist the event cursor AFTER the full pipeline completes
        # successfully. This ensures that if generate_predictions() crashes
        # mid-way, the cursor stays at the old persisted value and those
        # events will be reprocessed on the next cycle.
        self._persist_state("last_event_cursor", str(self._last_event_cursor))

        return filtered

    # -------------------------------------------------------------------
    # Prediction Generators
    # -------------------------------------------------------------------

    async def _check_calendar_conflicts(self, ctx: dict) -> list[Prediction]:
        """
        Detect scheduling conflicts and tight transitions.

        Scans a 48-hour lookahead window and compares consecutive events
        pairwise. Flags two scenarios:
            - Overlap (gap < 0 min)  -> CONFLICT at 0.95 confidence (0.8 for all-day)
            - Tight transition (<15 min gap) -> RISK at 0.70 confidence (timed only)

        All-day event handling:
            - All-day vs all-day: No conflict (multiple all-day markers are fine)
            - All-day vs timed: Conflict detected (different locations/contexts)
            - Timed vs timed: Full conflict detection with gap analysis

        CRITICAL FIX (iteration 132):
            The code was filtering out ALL all-day events (line 261-262), which
            meant 99.9% of calendar events (2,571 of 2,573 in production) were
            ignored. This completely broke calendar conflict detection, causing
            0 predictions despite having thousands of events in the database.

            Now we:
            - Include all-day events in the conflict detection pipeline
            - Skip all-day vs all-day comparisons (multiple markers are fine)
            - Detect all-day vs timed conflicts (e.g., meeting during travel day)
            - Enable downstream features that depend on all-day events

        CRITICAL FIX (iteration 117):
            The original implementation queried by event.timestamp (when the
            event was synced to the database) instead of the actual event
            start_time in the payload. This caused ALL calendar conflict
            predictions to be missed because synced events are timestamped
            in the past, even if the actual event is in the future.

            Now we:
            - Fetch all recent calendar events (last 30 days of syncs)
            - Parse start_time from each event's payload
            - Filter to events starting in the next 48 hours
            - Sort by actual start_time for accurate conflict detection
        """
        predictions = []

        with self.db.get_connection("events") as conn:
            # Fetch calendar events synced in the last 30 days.
            # This captures all events the CalDAV connector has loaded,
            # including future events that were synced recently.
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

            events = conn.execute(
                """SELECT * FROM events
                   WHERE type = 'calendar.event.created'
                   AND timestamp > ?""",
                (cutoff,),
            ).fetchall()

        # Diagnostic logging for observability
        if len(events) < 2:
            logger.info(
                "calendar_conflicts: %d events fetched (need ≥2 for conflict detection) — skipping",
                len(events),
            )
            return predictions  # Need at least two events to find conflicts

        # Parse event payloads and extract actual start/end times.
        # Filter to events that START in the next 48 hours.
        now = datetime.now(timezone.utc)
        lookahead = now + timedelta(hours=48)

        parsed_events = []
        for event in events:
            try:
                payload = json.loads(event["payload"])
                # Handle double-encoded JSON (rare but possible)
                if isinstance(payload, str):
                    payload = json.loads(payload)

                start_str = payload.get("start_time", "")
                end_str = payload.get("end_time", "")

                if not start_str or not end_str:
                    continue  # Skip events without time bounds

                # Parse ISO timestamps. Handle both 'Z' suffix and '+00:00' format.
                # CRITICAL FIX (iteration 128): fromisoformat() parses date-only
                # strings like "2026-02-14" successfully but creates timezone-NAIVE
                # datetimes. This caused all calendar predictions to fail when
                # comparing naive vs aware datetimes. Now we explicitly check and
                # add UTC timezone to any naive datetime.
                try:
                    start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    # If timezone-naive (all-day events), make it UTC-aware
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    # Completely unparseable — skip this event
                    continue

                try:
                    end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    # If timezone-naive (all-day events), make it UTC-aware
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    # Completely unparseable — skip this event
                    continue

                # Check if event falls within the 48-hour lookahead window.
                # CRITICAL FIX (iteration 143): All-day events have date-only timestamps
                # like "2026-02-16" which parse as midnight UTC. If it's currently 18:52 UTC,
                # today's all-day events appear to have started 18+ hours ago and fail the
                # `start_dt >= now` check. This caused 99.9% of calendar events to be excluded
                # from conflict/preparation predictions (2,571 all-day vs 2 timed events).
                #
                # Solution: For all-day events, check if their DATE falls within the window
                # (today through 2 days from now) rather than checking their midnight timestamp.
                # For timed events, include events that are still ongoing OR will start within 48h.
                is_all_day = payload.get("is_all_day", False)

                in_window = False
                if is_all_day:
                    # For all-day events: check if date falls within [today, today+2 days]
                    # This captures today's events (even if midnight has passed) and upcoming ones.
                    event_date = start_dt.date()
                    today = now.date()
                    lookahead_date = (now + timedelta(days=2)).date()
                    in_window = today <= event_date <= lookahead_date
                else:
                    # For timed events: include if EITHER:
                    # 1. Event hasn't ended yet (ongoing), OR
                    # 2. Event will start within the 48h window (upcoming)
                    # This ensures we catch conflicts with events happening right now.
                    event_ended = end_dt < now
                    event_starts_soon = start_dt <= lookahead
                    in_window = not event_ended and event_starts_soon

                if in_window:
                    parsed_events.append({
                        "start_dt": start_dt,
                        "end_dt": end_dt,
                        "payload": payload,
                        "event_id": event["id"],
                        "is_all_day": is_all_day,
                    })

            except Exception as e:
                # Fail-open: skip individual parse errors without breaking
                # conflict detection for other events.
                continue

        # Count event types for diagnostics
        all_day_count = sum(1 for e in parsed_events if e.get("is_all_day"))
        timed_count = len(parsed_events) - all_day_count

        if len(parsed_events) < 2:
            logger.info(
                "calendar_conflicts: %d events fetched, %d in 48h window "
                "(all_day=%d, timed=%d) — need ≥2 for conflict detection, skipping",
                len(events), len(parsed_events), all_day_count, timed_count,
            )
            return predictions

        # Sort by actual event start time (not sync timestamp)
        parsed_events.sort(key=lambda e: e["start_dt"])

        # Compare each consecutive pair of events for overlap or tight gaps.
        # Skip all-day event pairs (multiple all-day markers are fine), but DO
        # compare timed events with all-day events (e.g., a timed meeting during
        # an all-day conference in a different location IS a conflict).
        comparisons_made = 0
        skipped_all_day_pairs = 0
        for i in range(len(parsed_events) - 1):
            curr = parsed_events[i]
            next_evt = parsed_events[i + 1]

            # Skip if both events are all-day (no conflict between all-day markers)
            if curr.get("is_all_day") and next_evt.get("is_all_day"):
                skipped_all_day_pairs += 1
                continue

            comparisons_made += 1

            gap_minutes = (next_evt["start_dt"] - curr["end_dt"]).total_seconds() / 60

            if gap_minutes < 0:
                # Negative gap = events overlap in time.
                # For all-day events, only flag if one is timed (location conflict).
                # For timed events, always flag.
                is_all_day_conflict = curr.get("is_all_day") or next_evt.get("is_all_day")

                predictions.append(Prediction(
                    prediction_type="conflict",
                    description=(
                        f"Calendar overlap: '{curr['payload'].get('title', 'Event')}' "
                        f"and '{next_evt['payload'].get('title', 'Event')}' overlap"
                        + ("" if is_all_day_conflict else f" by {abs(int(gap_minutes))} minutes")
                    ),
                    confidence=0.8 if is_all_day_conflict else 0.95,
                    confidence_gate=ConfidenceGate.DEFAULT,
                    time_horizon="24_hours",
                    suggested_action="Reschedule one of the conflicting events",
                    # supporting_signals enables BehavioralAccuracyTracker._infer_conflict_accuracy()
                    # to detect when the user resolved the conflict by checking whether either
                    # conflicting_event_ids appears in calendar.event.updated/deleted events.
                    # Without this, _infer_conflict_accuracy returns None immediately (line 327)
                    # and conflict predictions remain unresolved indefinitely.
                    supporting_signals={
                        "conflicting_event_ids": [curr["event_id"], next_evt["event_id"]],
                        "event_titles": [
                            curr["payload"].get("title", "Event"),
                            next_evt["payload"].get("title", "Event"),
                        ],
                        "event_start_times": [
                            curr["start_dt"].isoformat(),
                            next_evt["start_dt"].isoformat(),
                        ],
                        "overlap_minutes": abs(int(gap_minutes)),
                        "is_all_day_conflict": is_all_day_conflict,
                    },
                ))
            elif gap_minutes < 15 and not (curr.get("is_all_day") or next_evt.get("is_all_day")):
                # Very tight transition (only for timed events — all-day events
                # don't have tight transitions by definition).
                predictions.append(Prediction(
                    prediction_type="risk",
                    description=(
                        f"Only {int(gap_minutes)} minutes between "
                        f"'{curr['payload'].get('title', 'Event')}' and "
                        f"'{next_evt['payload'].get('title', 'Event')}'"
                    ),
                    confidence=0.7,
                    confidence_gate=ConfidenceGate.SUGGEST,
                    time_horizon="24_hours",
                    suggested_action="Consider adding buffer time",
                    # supporting_signals enables BehavioralAccuracyTracker._infer_risk_accuracy()
                    # to look up whether the user modified either event to add buffer time.
                    # Without event IDs, the risk accuracy tracker cannot correlate this
                    # prediction with calendar updates.
                    supporting_signals={
                        "conflicting_event_ids": [curr["event_id"], next_evt["event_id"]],
                        "event_titles": [
                            curr["payload"].get("title", "Event"),
                            next_evt["payload"].get("title", "Event"),
                        ],
                        "event_start_times": [
                            curr["start_dt"].isoformat(),
                            next_evt["start_dt"].isoformat(),
                        ],
                        "gap_minutes": int(gap_minutes),
                    },
                ))

        # Diagnostic summary
        logger.info(
            "calendar_conflicts: Analyzed %d synced events → %d in 48h window "
            "(all_day=%d, timed=%d) → %d comparisons (skipped %d all-day pairs) → %d predictions",
            len(events), len(parsed_events), all_day_count, timed_count,
            comparisons_made, skipped_all_day_pairs, len(predictions),
        )

        return predictions

    async def _check_calendar_event_reminders(self, ctx: dict) -> list[Prediction]:
        """Generate REMINDER predictions for upcoming calendar events.

        Unlike other prediction generators, this works entirely from events.db
        and requires no signal profiles. This ensures the prediction engine
        produces basic useful output even when user_model.db is degraded.

        Generates reminders for:
        - Events starting in the next 2-24 hours (not imminent <2h, not far >24h)
        - Only timed events (skip all-day events — they don't need time reminders)
        - One reminder per event (deduplicated against existing predictions)
        """
        predictions = []
        now = datetime.now(timezone.utc)
        window_start = now + timedelta(hours=2)
        window_end = now + timedelta(hours=24)

        # Query calendar events from events.db (same approach as _check_calendar_conflicts)
        with self.db.get_connection("events") as conn:
            cutoff = (now - timedelta(days=30)).isoformat()
            events = conn.execute(
                """SELECT id, payload FROM events
                   WHERE type = 'calendar.event.created'
                   AND timestamp > ?""",
                (cutoff,),
            ).fetchall()

        if not events:
            return predictions

        # Parse and filter to events starting in the 2-24h window
        upcoming = []
        for event in events:
            try:
                payload = json.loads(event["payload"])
                # Handle double-encoded JSON
                if isinstance(payload, str):
                    payload = json.loads(payload)

                start_str = payload.get("start_time", "")
                if not start_str:
                    continue

                # Skip all-day events (date-only format like '2026-03-04')
                if len(start_str) <= 10:
                    continue

                start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)

                if window_start <= start_dt <= window_end:
                    upcoming.append({
                        "event_id": event["id"],
                        "title": payload.get("title", payload.get("summary", "Upcoming event")),
                        "start_dt": start_dt,
                        "location": payload.get("location", ""),
                        "payload": payload,
                    })
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        if not upcoming:
            logger.debug("calendar_reminders: no events in 2-24h window")
            return predictions

        # Deduplicate: check which events already have reminder predictions
        existing_event_ids: set[str] = set()
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT supporting_signals FROM predictions
                       WHERE prediction_type = 'reminder'
                       AND created_at > ?""",
                    ((now - timedelta(hours=48)).isoformat(),),
                ).fetchall()
                for row in rows:
                    try:
                        signals = json.loads(row["supporting_signals"] or "{}")
                        eid = signals.get("calendar_event_id")
                        if eid:
                            existing_event_ids.add(eid)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug("calendar_reminders: skipping malformed dedup entry: %s", e)
        except Exception as e:
            logger.debug("calendar_reminders: could not check existing predictions: %s", e)

        for evt in upcoming:
            if evt["event_id"] in existing_event_ids:
                continue

            hours_until = (evt["start_dt"] - now).total_seconds() / 3600
            location_str = f" at {evt['location']}" if evt["location"] else ""

            predictions.append(Prediction(
                id=str(uuid.uuid4()),
                prediction_type="reminder",
                description=f"Upcoming: {evt['title']}{location_str} in {hours_until:.0f} hours",
                confidence=0.85,  # High confidence — it's a real scheduled event
                confidence_gate=ConfidenceGate.DEFAULT,
                time_horizon="24_hours",
                suggested_action=f"Prepare for {evt['title']}",
                supporting_signals={
                    "calendar_event_id": evt["event_id"],
                    "event_title": evt["title"],
                    "event_start": evt["start_dt"].isoformat(),
                    "hours_until": round(hours_until, 1),
                    "location": evt["location"],
                },
            ))

        logger.info(
            "calendar_reminders: %d upcoming events in 2-24h window, "
            "%d new reminders (skipped %d already predicted)",
            len(upcoming), len(predictions),
            len(existing_event_ids & {e["event_id"] for e in upcoming}),
        )
        return predictions

    async def _check_follow_up_needs(self, ctx: dict) -> list[Prediction]:
        """
        Detect messages that need a reply -- things the user read but
        hasn't responded to, especially from priority contacts.

        Strategy:
            1. Fetch all inbound messages from the last 24 hours that are:
               a) Unreplied (not in replied_to_threads set)
               b) Not marketing/automated (filtered by _is_marketing_or_noreply)
               c) Older than 3 hours (grace period for user to respond)
            2. Check if we've already created a prediction for this message
            3. Create new predictions only for messages we haven't alerted about yet

        Priority Contact Detection:
            A "priority contact" is someone the user has sent outbound messages to
            (bidirectional relationship). These contacts receive boosted confidence
            (0.7 vs 0.4 baseline) because the user has an established communication
            relationship with them, not just a passive inbound-only contact.

            Priority detection uses the "relationships" signal profile, which tracks
            per-contact interaction history including outbound_count. A contact with
            outbound_count > 0 is someone the user has actively reached out to.

            CRITICAL FIX (iteration 183):
                The previous implementation checked metadata.get("related_contacts", [])
                from the event envelope. This field contains only the sending address
                itself (the email's own from_address), so the check:
                    any(from_addr in contacts for contacts in [metadata.get(...)])
                ... was structurally impossible to trigger: it was checking if the
                sender was in a list that only contained the sender themselves, not
                a curated list of priority contacts. This meant is_priority was always
                False and the 0.3 confidence boost for priority contacts NEVER fired —
                even when receiving an email from a close collaborator who the user
                consistently replies to.

                The fix loads the relationships signal profile once per prediction
                cycle and checks outbound_count > 0 for each sender.

        CRITICAL FIX (iteration 62):
            Previously, this method processed ALL emails from the last 48 hours on
            EVERY prediction cycle (every 15 min). This caused:
            - 9,086 duplicate predictions created in a single batch
            - Overwhelming the database with redundant reminders
            - Breaking the accuracy feedback loop with noise

            Now we:
            - Track which message IDs we've already created predictions for
            - Only create ONE prediction per unreplied message, ever
            - Prevent reprocessing the same emails repeatedly

        PERFORMANCE FIX (iteration 81):
            With 70K+ emails in the database, scanning 48 hours of emails every
            15 minutes caused massive overhead (37K predictions/hour, 73K email
            scans every cycle). This caused the prediction engine to consume
            100% CPU continuously.

            Optimizations:
            - Reduced lookback window from 48h → 24h (cuts scan volume in half)
            - Early exit after scanning existing predictions (avoid redundant work)
            - Only fetch message IDs first, then details for new predictions
        """
        predictions = []

        # Load the relationships signal profile ONCE per cycle (not per email).
        # We use this to identify "priority contacts" — people the user has sent
        # outbound messages to (outbound_count > 0), meaning there is an established
        # bidirectional communication relationship. Emails from these contacts get
        # a higher confidence boost because the user is more likely to need to reply.
        # Wrapped in try/except so a corrupted user_model.db doesn't prevent
        # follow-up predictions entirely — we can still generate them without
        # priority contact boosting.
        try:
            rel_profile = self.ums.get_signal_profile("relationships")
        except Exception as e:
            logger.warning("follow_up_needs: get_signal_profile('relationships') failed (disabling priority boost): %s", e)
            rel_profile = None
        rel_contacts = rel_profile["data"].get("contacts", {}) if rel_profile else {}

        if not rel_profile or not rel_contacts:
            logger.info(
                "follow_up_needs: relationships signal profile is %s — "
                "priority contact boost disabled",
                "empty" if rel_profile else "unavailable",
            )

        # Build a fast-lookup set of priority contact addresses.
        # A contact is "priority" if the user has sent at least one outbound message
        # to them, establishing a real two-way communication pattern.
        # We normalize to lowercase for case-insensitive matching.
        priority_contacts: set[str] = {
            addr.lower()
            for addr, data in rel_contacts.items()
            if data.get("outbound_count", 0) > 0
            and not self._is_marketing_or_noreply(addr, {})
        }

        # First, quickly check what messages we've already created predictions for
        # in the last 48 hours (wider than scan window to catch stragglers).
        # Wrapped in try/except so a corrupted user_model.db doesn't prevent
        # follow-up predictions entirely — better to risk duplicates than silence.
        already_predicted_messages: set[str] = set()
        try:
            prediction_cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
            with self.db.get_connection("user_model") as conn:
                existing_predictions = conn.execute(
                    """SELECT supporting_signals FROM predictions
                       WHERE prediction_type = 'reminder'
                       AND created_at > ?""",
                    (prediction_cutoff,),
                ).fetchall()

            for pred in existing_predictions:
                try:
                    signals = json.loads(pred["supporting_signals"]) if pred["supporting_signals"] else {}
                    msg_id = signals.get("message_id")
                    if msg_id:
                        already_predicted_messages.add(msg_id)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug("follow_up: skipping malformed dedup entry: %s", e)
        except Exception as e:
            logger.warning(
                "follow_up dedup query failed (skipping dedup, may produce duplicates): %s", e
            )

        # Determine lookback window: 72h on first cycle (to catch unreplied emails
        # from startup or after a connector outage), then 24h on subsequent cycles.
        # The 72h window keeps scan volume manageable (~3x the normal 24h, not the
        # full 70K+ that caused the original performance issue in iteration 81).
        if self._first_follow_up_run:
            lookback_hours = 72
            self._first_follow_up_run = False
            logger.debug("follow_up_needs: first cycle — using wider %dh lookback", lookback_hours)
        else:
            lookback_hours = 24

        with self.db.get_connection("events") as conn:
            # Inbound messages from the lookback window
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()

            inbound = conn.execute(
                """SELECT id, payload, timestamp FROM events
                   WHERE type IN ('email.received', 'message.received')
                   AND timestamp > ?
                   ORDER BY timestamp DESC""",
                (cutoff,),
            ).fetchall()

            # --- Stale-data fallback ---
            # When the standard lookback window yields 0 inbound messages AND
            # the most recent email.received event is older than 72 hours, it
            # likely means the connector has been down (e.g. Google auth failure).
            # Extend the lookback to the last active period so accumulated
            # unreplied emails still surface.  Capped at 14 days to keep scan
            # volume bounded.
            if len(inbound) == 0:
                latest_row = conn.execute(
                    """SELECT timestamp FROM events
                       WHERE type = 'email.received'
                       ORDER BY timestamp DESC LIMIT 1"""
                ).fetchone()

                if latest_row:
                    try:
                        latest_ts = datetime.fromisoformat(
                            latest_row["timestamp"].replace("Z", "+00:00")
                        )
                        staleness = datetime.now(timezone.utc) - latest_ts
                        max_lookback = timedelta(days=14)

                        if staleness > timedelta(hours=72):
                            # Re-query: scan the 24-hour window around the last
                            # active email period, capped at 14 days ago.
                            fallback_start = max(
                                latest_ts - timedelta(hours=24),
                                datetime.now(timezone.utc) - max_lookback,
                            )
                            fallback_cutoff = fallback_start.isoformat()
                            days_ago = staleness.days

                            logger.warning(
                                "follow_up_needs: no recent emails detected, "
                                "extending lookback to last active period "
                                "(%d days ago)",
                                days_ago,
                            )

                            inbound = conn.execute(
                                """SELECT id, payload, timestamp FROM events
                                   WHERE type IN ('email.received', 'message.received')
                                   AND timestamp > ?
                                   ORDER BY timestamp DESC""",
                                (fallback_cutoff,),
                            ).fetchall()
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "follow_up_needs: stale-data fallback failed "
                            "to parse latest timestamp: %s", e
                        )

            # Outbound messages in the same window (use widest cutoff to
            # cover both standard and fallback periods)
            outbound_cutoff = cutoff
            if len(inbound) > 0:
                # Use the oldest inbound timestamp as the outbound cutoff
                # to ensure we catch replies to fallback-window emails too.
                try:
                    oldest_inbound_ts = min(
                        msg["timestamp"] for msg in inbound
                    )
                    if oldest_inbound_ts < outbound_cutoff:
                        outbound_cutoff = oldest_inbound_ts
                except (ValueError, TypeError):
                    pass  # Keep original cutoff on parse errors

            outbound = conn.execute(
                """SELECT payload FROM events
                   WHERE type IN ('email.sent', 'message.sent')
                   AND timestamp > ?""",
                (outbound_cutoff,),
            ).fetchall()

        # Build a set of thread/message IDs we've already replied to,
        # so we can exclude them from the "needs follow-up" list.
        replied_to_threads = set()
        for msg in outbound:
            payload = json.loads(msg["payload"])
            if payload.get("in_reply_to"):
                replied_to_threads.add(payload["in_reply_to"])

        # Find unreplied inbound messages
        for msg in inbound:
            payload = json.loads(msg["payload"])
            message_id = payload.get("message_id", "")

            # Skip if already replied
            if message_id in replied_to_threads:
                continue

            # Skip if we've already created a prediction for this message
            # This is the critical fix to prevent duplicate predictions
            if message_id in already_predicted_messages:
                continue

            # Check if from a priority contact
            from_addr = payload.get("from_address", "")

            # Skip messages with missing or empty from_address — these are
            # malformed events that shouldn't generate predictions. Without
            # this check, empty addresses bypass the marketing filter entirely
            # and create broken predictions with blank sender fields.
            if not from_addr or not from_addr.strip():
                continue

            # Skip marketing/automated emails — no-reply senders, bulk
            # sender patterns, and messages containing "unsubscribe" in
            # snippet, body_plain, or body.
            if self._is_marketing_or_noreply(from_addr, payload):
                continue

            # Priority detection: check if this sender is in our bidirectional
            # contacts set (someone the user has actively sent messages to).
            # Normalize to lowercase for case-insensitive matching.
            is_priority = from_addr.lower() in priority_contacts

            # Calculate how long it's been
            try:
                msg_time = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
                hours_ago = (datetime.now(timezone.utc) - msg_time).total_seconds() / 3600
            except (ValueError, TypeError):
                hours_ago = 24

            # Don't nag about very recent messages — give the user time.
            # Priority contacts get a shorter grace period (1h vs 3h) because
            # important emails from established contacts deserve faster alerting.
            grace_hours = 1 if is_priority else 3
            if hours_ago < grace_hours:
                continue

            # --- Confidence scoring for follow-up predictions ---
            # Base confidence is low (0.4) to avoid false positives.
            # Boosted by: priority contact (+0.3), age > 24h (+0.2),
            # explicit "requires_response" flag (+0.2). Capped at 0.9.
            confidence = 0.4
            if is_priority:
                confidence = 0.7   # Priority contacts start higher
            if hours_ago > 24:
                confidence = min(confidence + 0.2, 0.9)  # Aging messages get more urgent
            if payload.get("requires_response"):
                confidence = min(confidence + 0.2, 0.9)  # Explicit response request

            subject = payload.get("subject", "No subject")
            resolved_name = self._resolve_contact_name(from_addr)
            predictions.append(Prediction(
                prediction_type="reminder",
                description=(
                    f"Unreplied message from {resolved_name}: \"{subject}\" "
                    f"({int(hours_ago)} hours ago)"
                ),
                confidence=confidence,
                confidence_gate=self._gate_from_confidence(confidence),
                time_horizon="2_hours",
                suggested_action=f"Reply to {resolved_name} ({from_addr})",
                relevant_contacts=[from_addr],
                supporting_signals={
                    "contact_email": from_addr,
                    "contact_name": resolved_name,
                    "message_id": message_id,
                    "hours_since_received": hours_ago,
                    "is_priority_contact": is_priority,
                    "requires_response": payload.get("requires_response", False),
                },
            ))

        return predictions

    async def _check_routine_deviations(self, ctx: dict) -> list[Prediction]:
        """
        Detect when the user deviates from their usual patterns.
        E.g., usually exercises Mon/Wed/Fri but hasn't today.

        Only considers routines with consistency_score > 0.6 — these are
        habits the user follows at least 60% of the time, so deviations
        are meaningful rather than noisy.

        CRITICAL FIX (iteration 114):
            Previously, this method had a stubbed implementation with two fatal bugs:
            1. Day-name matching logic was inverted: checked if day name IN trigger
               (e.g., "monday" in "morning") instead of checking the trigger's day pattern
            2. Never checked if routine was actually completed — created duplicate
               predictions every 15 minutes regardless of user behavior

            Now we:
            - First check if we've already created a prediction for this routine today
            - Parse routine steps to identify expected event types
            - Query events table to see if those event types occurred today
            - Only create prediction if routine hasn't been completed
            - Track routine_name in supporting_signals for deduplication
        """
        predictions = []

        # Routine data lives exclusively in user_model.db.  When that DB is
        # known to be corrupted (is_user_model_healthy() == False), skip the
        # query entirely to avoid noisy error logs from a DB we already know
        # is broken.  There is no events.db fallback for routines because
        # routine definitions are only stored in user_model.db.
        if not self.db.is_user_model_healthy():
            logger.debug(
                "routine_deviations: user_model.db is degraded — skipping "
                "(routines require user_model.db)"
            )
            return predictions

        # Load established routines from procedural memory
        with self.db.get_connection("user_model") as conn:
            routines = conn.execute(
                "SELECT * FROM routines WHERE consistency_score > 0.6"
            ).fetchall()

        if not routines:
            logger.info("routine_deviations: 0 routines with consistency_score > 0.6 — skipping")
            return predictions

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Check what routine deviation predictions we've already created today
        # to avoid duplicate reminders every 15 minutes.
        # Wrapped in try/except so a corrupted user_model.db doesn't prevent
        # routine deviation predictions entirely.
        already_predicted_routines: set[str] = set()
        try:
            with self.db.get_connection("user_model") as conn:
                existing_predictions = conn.execute(
                    """SELECT supporting_signals FROM predictions
                       WHERE prediction_type = 'routine_deviation'
                       AND created_at > ?""",
                    (today_start.isoformat(),),
                ).fetchall()

            for pred in existing_predictions:
                try:
                    signals = json.loads(pred["supporting_signals"]) if pred["supporting_signals"] else {}
                    routine_name = signals.get("routine_name")
                    if routine_name:
                        already_predicted_routines.add(routine_name)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug("routine_deviation: skipping malformed dedup entry: %s", e)
        except Exception as e:
            logger.warning(
                "routine_deviation dedup query failed (skipping dedup, may produce duplicates): %s", e
            )

        # For each routine, check if it should have been completed by now
        for routine in routines:
            routine_name = routine["name"]

            # Skip if we've already created a prediction for this routine today
            if routine_name in already_predicted_routines:
                continue

            # Parse the routine steps to identify what event types to look for
            try:
                steps = json.loads(routine["steps"])
                if not steps:
                    continue

                # Extract expected event types from the workflow steps
                # Steps are dicts with "action" keys (e.g., "email_received", "task_created")
                expected_actions = []
                for step in steps[:3]:  # Only check first 3 steps for performance
                    if isinstance(step, dict):
                        action = step.get("action", "")
                        if action:
                            expected_actions.append(action)

                if not expected_actions:
                    continue

                # Map routine action types to event types
                # Routine actions use underscore format (email_received) while
                # event types use dot format (email.received)
                # Complete mapping of all interaction types produced by
                # _classify_interaction_type() in main.py to their source event types.
                # Missing entries cause false deviation predictions because the
                # fallback (action.replace("_", ".")) generates invalid event types
                # that never match anything in events.db.
                event_type_mapping = {
                    "email_received": "email.received",
                    "email_sent": "email.sent",
                    "message_received": "message.received",
                    "message_sent": "message.sent",
                    "task_created": "task.created",
                    "task_completed": "task.completed",
                    "calendar_event_created": "calendar.event.created",
                    "call_answered": "call.received",
                    "call_missed": "call.missed",
                    "meeting_scheduled": "calendar.event.created",
                    "calendar_blocked": "calendar.event.created",
                    "calendar_reviewed": "calendar.event.updated",
                    "spending": "finance.transaction.new",
                    "income": "finance.transaction.new",
                    "location_arrived": "location.arrived",
                    "location_departed": "location.departed",
                    "location_changed": "location.changed",
                    "context_location": "context.location",
                    "context_activity": "context.activity",
                    "user_command": "system.user.command",
                }

                expected_event_types = []
                for action in expected_actions:
                    event_type = event_type_mapping.get(action, action.replace("_", "."))
                    expected_event_types.append(event_type)

                # Check if any of the expected event types occurred today
                with self.db.get_connection("events") as conn:
                    result = conn.execute(
                        f"""SELECT COUNT(*) as count FROM events
                           WHERE type IN ({','.join('?' * len(expected_event_types))})
                           AND timestamp > ?""",
                        (*expected_event_types, today_start.isoformat()),
                    ).fetchone()

                if result and result["count"] > 0:
                    # Routine was completed today — no deviation
                    continue

                # Routine hasn't been completed today — create a prediction
                # Confidence is based on consistency score but capped lower since
                # routines can have legitimate skip days
                confidence = min(routine["consistency_score"] * 0.5, 0.65)

                # Only surface if confidence meets SUGGEST threshold (0.3+)
                if confidence < 0.3:
                    continue

                predictions.append(Prediction(
                    prediction_type="routine_deviation",
                    description=f"You usually do your '{routine_name}' routine by now",
                    confidence=confidence,
                    confidence_gate=self._gate_from_confidence(confidence),
                    time_horizon=now.replace(hour=23, minute=59, second=59, microsecond=0).isoformat(),
                    suggested_action=f"Start {routine_name}",
                    supporting_signals={
                        "routine_name": routine_name,
                        "consistency_score": routine["consistency_score"],
                        "expected_actions": expected_actions,
                    },
                ))

            except (json.JSONDecodeError, TypeError, KeyError) as e:
                # Fail-open: skip routines with malformed data
                continue

        # Diagnostic summary
        logger.info(
            "routine_deviations: Analyzed %d routines (already_predicted_today=%d) → %d predictions",
            len(routines), len(already_predicted_routines), len(predictions),
        )

        return predictions

    async def _check_relationship_maintenance(self, ctx: dict) -> list[Prediction]:
        """
        Detect contacts the user hasn't been in touch with for longer
        than their typical frequency.

        Uses the "relationships" signal profile, which tracks per-contact
        interaction history. For each contact with enough data (5+
        interactions), we compute the average gap between interactions
        and flag when the current gap exceeds 1.5x the average.

        When the signal profile is unavailable (e.g. user_model.db is
        corrupt), falls back to computing basic contact data directly
        from events.db, which stores the raw email event log.
        """
        predictions = []

        # Load the relationships signal profile from the user model.
        # If unavailable (returns None) or inaccessible (user_model.db is
        # corrupted and the query throws), fall back to building contact
        # data directly from events.db.
        try:
            rel_profile = self.ums.get_signal_profile("relationships")
        except Exception as e:
            logger.warning(
                "relationship_maintenance: get_signal_profile('relationships') "
                "failed (falling back to events.db): %s", e,
            )
            rel_profile = None
        if not rel_profile:
            logger.info("relationship_maintenance: relationships signal profile unavailable — falling back to events.db")
            contacts = self._build_contacts_from_events()
            if not contacts:
                logger.info("relationship_maintenance: no contact data available from events.db either — skipping")
                return predictions
        else:
            contacts = rel_profile["data"].get("contacts", {})
        now = datetime.now(timezone.utc)

        for addr, data in contacts.items():
            last = data.get("last_interaction")
            count = data.get("interaction_count", 0)

            # Skip contacts with too little history — we need at least 5
            # interactions to establish a reliable frequency baseline.
            if not last or count < 5:
                continue

            # Skip marketing/automated senders — relationship maintenance is
            # for human connections, not bulk email subscriptions. Without
            # this filter, the system would generate "reach out" suggestions
            # for addresses like callofduty@comms.activision.com which are
            # marketing automations, not relationships to maintain.
            if self._is_marketing_or_noreply(addr, {}):
                continue

            # Skip purely inbound-only contacts — a "relationship" requires
            # bidirectionality.  If the user has *never* sent anything to this
            # address (outbound_count == 0), we have no evidence of an
            # intentional relationship; generating an "opportunity" prediction
            # would be a false positive.
            #
            # Common sources of inbound-only contacts that slip through the
            # marketing filter:
            #   - Low-volume mailing lists (1–3 emails/year) that look human
            #   - HR/payroll system notifications addressed personally
            #   - Automated alerts from SaaS tools using personal-looking names
            #
            # Once the user replies to any of these, outbound_count becomes ≥ 1
            # and they are eligible for future opportunity predictions.
            outbound_count = data.get("outbound_count", 0)
            if outbound_count == 0:
                continue

            try:
                last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                days_since = (now - last_dt).days
            except (ValueError, TypeError):
                continue

            # Estimate typical contact frequency from the last 10 timestamps.
            # We compute the average gap in days between consecutive interactions.
            #
            # CRITICAL FIX (iteration 166): Use fractional days instead of integer days.
            # The original implementation used .days which returns integers, causing all
            # gaps < 24 hours to become 0. For contacts with frequent interactions (daily
            # or multiple times per day), this produced avg_gap=0, threshold=0, making it
            # impossible for ANY contact to trigger a prediction (days_since > 0 is never
            # greater than threshold 0 * 1.5 = 0).
            #
            # This caused 316 eligible contacts (5+ interactions) to generate ZERO
            # relationship maintenance predictions. Switching to fractional days via
            # .total_seconds() / 86400 fixes this for all contact frequencies.
            timestamps = data.get("interaction_timestamps", [])
            if len(timestamps) >= 3:
                try:
                    dts = sorted([
                        datetime.fromisoformat(t.replace("Z", "+00:00"))
                        for t in timestamps[-10:]
                    ])
                    # Use fractional days for accurate gap calculation
                    gaps = [(dts[i + 1] - dts[i]).total_seconds() / 86400 for i in range(len(dts) - 1)]
                    avg_gap = sum(gaps) / len(gaps) if gaps else 30
                except (ValueError, TypeError):
                    avg_gap = 30

                # Flag if the current gap exceeds 1.5x the average AND
                # it's been at least 7 days (avoid nagging about daily contacts).
                # Confidence scales linearly with how far past the threshold.
                if days_since > avg_gap * 1.5 and days_since > 7:
                    confidence = min(0.6, 0.3 + (days_since / avg_gap - 1.5) * 0.2)
                    resolved_name = self._resolve_contact_name(addr)
                    predictions.append(Prediction(
                        prediction_type="opportunity",
                        description=(
                            f"It's been {days_since} days since you last "
                            f"contacted {resolved_name} (you usually connect every ~{int(avg_gap)} days)"
                        ),
                        confidence=confidence,
                        confidence_gate=self._gate_from_confidence(confidence),
                        time_horizon="this_week",
                        suggested_action=f"Reach out to {resolved_name} ({addr})",
                        relevant_contacts=[addr],
                        # supporting_signals enables BehavioralAccuracyTracker._infer_opportunity_accuracy()
                        # to reliably match this prediction to outbound emails/messages.
                        # Without these fields, the tracker must regex-parse the description,
                        # which can fail for email addresses with unusual formatting.
                        # Also enables the automated-sender fast-path (PR #189) to fire correctly.
                        supporting_signals={
                            "contact_email": addr,
                            "contact_name": resolved_name,
                            "days_since_last_contact": days_since,
                            "avg_contact_gap_days": round(avg_gap, 1),
                        },
                    ))

        # Diagnostic summary — includes breakdown of each filter stage so we can
        # observe the data quality of the relationships profile over time.
        total_contacts = len(contacts)
        eligible = sum(1 for data in contacts.values() if data.get("interaction_count", 0) >= 5)
        marketing_filtered = sum(1 for addr in contacts.keys() if self._is_marketing_or_noreply(addr, {}))
        # Count contacts that have 5+ interactions, passed marketing filter, but
        # were skipped because the user has never sent them anything (pure inbound).
        inbound_only_filtered = sum(
            1 for addr, data in contacts.items()
            if data.get("interaction_count", 0) >= 5
            and not self._is_marketing_or_noreply(addr, {})
            and data.get("outbound_count", 0) == 0
        )
        logger.info(
            "relationship_maintenance: Analyzed %d contacts "
            "(eligible=%d, marketing_filtered=%d, inbound_only_filtered=%d) → %d predictions",
            total_contacts, eligible, marketing_filtered, inbound_only_filtered, len(predictions),
        )

        return predictions

    def _build_contacts_from_events(self) -> dict:
        """Build basic contact interaction data from events.db when signal profiles are unavailable.

        Queries the raw email event log (last 90 days) to reconstruct per-contact
        interaction data with the same shape as the relationships signal profile:
        interaction_count, last_interaction, outbound_count, interaction_timestamps.

        For email.received events, the contact is in ``from_address``.
        For email.sent events, each address in ``to_addresses`` is a contact.

        Returns:
            dict mapping contact email addresses to interaction dicts, or empty
            dict on failure.
        """
        try:
            with self.db.get_connection("events") as conn:
                # Step 1: Aggregate inbound interactions per contact (from_address on received emails)
                inbound_rows = conn.execute("""
                    SELECT
                        json_extract(payload, '$.from_address') AS addr,
                        COUNT(*) AS inbound_count,
                        MAX(timestamp) AS last_inbound
                    FROM events
                    WHERE type = 'email.received'
                      AND timestamp > datetime('now', '-90 days')
                      AND json_extract(payload, '$.from_address') IS NOT NULL
                    GROUP BY addr
                """).fetchall()

                # Build initial contacts dict from inbound emails
                contacts: dict[str, dict] = {}
                for row in inbound_rows:
                    addr = row["addr"]
                    contacts[addr] = {
                        "interaction_count": row["inbound_count"],
                        "last_interaction": row["last_inbound"],
                        "outbound_count": 0,
                        "interaction_timestamps": [],
                    }

                # Step 2: Count outbound interactions per contact.
                # For sent emails, the contact addresses are in to_addresses (a JSON array).
                # We extract each element and aggregate.
                outbound_rows = conn.execute("""
                    SELECT
                        je.value AS addr,
                        COUNT(*) AS outbound_count,
                        MAX(e.timestamp) AS last_outbound
                    FROM events e,
                         json_each(json_extract(e.payload, '$.to_addresses')) je
                    WHERE e.type = 'email.sent'
                      AND e.timestamp > datetime('now', '-90 days')
                    GROUP BY addr
                """).fetchall()

                for row in outbound_rows:
                    addr = row["addr"]
                    if addr in contacts:
                        contacts[addr]["outbound_count"] = row["outbound_count"]
                        # Update last_interaction if the outbound is more recent
                        if row["last_outbound"] and row["last_outbound"] > contacts[addr]["last_interaction"]:
                            contacts[addr]["last_interaction"] = row["last_outbound"]
                        contacts[addr]["interaction_count"] += row["outbound_count"]
                    else:
                        # Contact only appears in outbound (user sent but never received)
                        contacts[addr] = {
                            "interaction_count": row["outbound_count"],
                            "last_interaction": row["last_outbound"],
                            "outbound_count": row["outbound_count"],
                            "interaction_timestamps": [],
                        }

                # Step 3: Filter to contacts with 5+ interactions, then fetch timestamps
                # for gap calculation. Done as a second pass to avoid N+1 queries for
                # low-interaction contacts.
                filtered = {addr: data for addr, data in contacts.items() if data["interaction_count"] >= 5}

                for addr, data in filtered.items():
                    # Fetch the 10 most recent interaction timestamps for this contact.
                    # Combines both inbound (from_address match) and outbound (to_addresses match).
                    ts_rows = conn.execute("""
                        SELECT timestamp FROM (
                            SELECT timestamp FROM events
                            WHERE type = 'email.received'
                              AND json_extract(payload, '$.from_address') = ?
                              AND timestamp > datetime('now', '-90 days')
                            UNION ALL
                            SELECT e.timestamp FROM events e,
                                   json_each(json_extract(e.payload, '$.to_addresses')) je
                            WHERE e.type = 'email.sent'
                              AND je.value = ?
                              AND e.timestamp > datetime('now', '-90 days')
                        )
                        ORDER BY timestamp DESC
                        LIMIT 10
                    """, (addr, addr)).fetchall()
                    data["interaction_timestamps"] = [row["timestamp"] for row in ts_rows]

                logger.info(
                    "_build_contacts_from_events: found %d contacts (%d with 5+ interactions) from events.db",
                    len(contacts), len(filtered),
                )
                return filtered

        except Exception as e:
            logger.warning("_build_contacts_from_events failed: %s", e)
            return {}

    async def _check_preparation_needs(self, ctx: dict) -> list[Prediction]:
        """
        Detect upcoming events that require preparation based on
        learned patterns. E.g., "You have a flight tomorrow -- pack tonight."

        Looks at events 12-48 hours out (the "preparation window") and
        checks for keywords that signal preparation needs:
            - Travel keywords -> packing & reservation checks
            - Large meetings (>3 attendees) -> agenda review

        CRITICAL FIX (iteration 122):
            The original implementation queried by event.timestamp (when the
            event was synced to the database) instead of the actual event
            start_time in the payload. This caused ALL preparation need
            predictions to be missed because synced events are timestamped
            in the past, even if the actual event is in the future.

            Now we:
            - Fetch all recent calendar events (last 30 days of syncs)
            - Parse start_time from each event's payload
            - Filter to events starting in the 12-48 hour preparation window
            - Generate predictions based on actual event timing

            This is the same bug that was fixed for calendar conflicts in
            iteration 117 (PR #131).
        """
        predictions = []

        with self.db.get_connection("events") as conn:
            # Fetch calendar events synced in the last 30 days.
            # This captures all events the CalDAV connector has loaded,
            # including future events that were synced recently.
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

            events = conn.execute(
                """SELECT * FROM events
                   WHERE type = 'calendar.event.created'
                   AND timestamp > ?""",
                (cutoff,),
            ).fetchall()

        if not events:
            logger.info("preparation_needs: 0 calendar events found — skipping")
            return predictions

        # Parse event payloads and extract actual start times.
        # Filter to events that START in the 12-48 hour preparation window.
        now = datetime.now(timezone.utc)
        window_start = now + timedelta(hours=12)
        window_end = now + timedelta(hours=48)

        parsed_events = []
        for event in events:
            try:
                payload = json.loads(event["payload"])
                # Handle double-encoded JSON (rare but possible)
                if isinstance(payload, str):
                    payload = json.loads(payload)

                start_time_str = payload.get("start_time")
                if not start_time_str:
                    continue

                # Parse the start time and check if it's in our preparation window
                # CRITICAL FIX (iteration 128): Same timezone-naive bug as calendar
                # conflicts — date-only strings parse but create naive datetimes.
                # CRITICAL FIX (iteration 143): All-day events with date-only timestamps
                # like "2026-02-16" parse as midnight UTC and fail time window checks if
                # it's already past midnight. Apply the same date-based window logic used
                # in calendar conflict detection.
                start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)

                is_all_day = payload.get("is_all_day", False)
                in_window = False

                if is_all_day:
                    # For all-day events: check if date falls within preparation window
                    # (tomorrow through 2 days from now for 12-48h window)
                    event_date = start_time.date()
                    window_start_date = window_start.date()
                    window_end_date = window_end.date()
                    in_window = window_start_date <= event_date <= window_end_date
                else:
                    # For timed events: check if start time falls in 12-48h window
                    in_window = window_start <= start_time <= window_end

                if in_window:
                    parsed_events.append({
                        "start_time": start_time,
                        "payload": payload,
                        # Include the event's database ID so supporting_signals can reference
                        # it. _infer_need_accuracy() first tries exact event_id match before
                        # falling back to fuzzy title matching, so providing this avoids
                        # false positives when multiple events have similar titles.
                        "event_id": payload.get("event_id") or event["id"],
                    })
            except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                # Skip events with malformed payloads or missing start_time
                continue

        # Generate predictions for events in the preparation window
        for event in parsed_events:
            payload = event["payload"]
            title = payload.get("title", "").lower()
            location = payload.get("location", "")
            start_time = event["start_time"]

            # Calculate hours until event for more precise messaging
            hours_until = (start_time - now).total_seconds() / 3600

            # --- Travel detection ---
            # Keyword-based check for travel-related events.
            travel_keywords = ["flight", "airport", "hotel", "travel", "trip"]
            if any(kw in title for kw in travel_keywords):
                predictions.append(Prediction(
                    prediction_type="need",
                    description=f"Upcoming travel in {int(hours_until)}h: '{payload.get('title')}'. Time to prepare.",
                    confidence=0.75,
                    confidence_gate=ConfidenceGate.DEFAULT,
                    time_horizon="24_hours",
                    suggested_action="Check packing list and confirm reservations",
                    # supporting_signals enables BehavioralAccuracyTracker._infer_need_accuracy()
                    # to check if the event occurred or was cancelled/rescheduled.
                    # Without event_start_time, _infer_need_accuracy() returns None immediately
                    # (line 413) and the prediction can never be resolved.
                    supporting_signals={
                        "event_id": event["event_id"],
                        "event_title": payload.get("title"),
                        "event_start_time": start_time.isoformat(),
                        "preparation_type": "travel",
                    },
                ))

            # --- Large meeting detection ---
            # Meetings with many attendees often need an agenda and talking points.
            attendees = payload.get("attendees", [])
            if len(attendees) > 3:
                predictions.append(Prediction(
                    prediction_type="need",
                    description=f"Large meeting in {int(hours_until)}h: '{payload.get('title')}' with {len(attendees)} attendees",
                    confidence=0.5,
                    confidence_gate=ConfidenceGate.SUGGEST,
                    time_horizon="24_hours",
                    suggested_action="Review agenda and prepare talking points",
                    # supporting_signals enables BehavioralAccuracyTracker._infer_need_accuracy()
                    # to check if the event occurred or was cancelled/rescheduled.
                    # Without event_start_time, _infer_need_accuracy() returns None immediately
                    # (line 413) and the prediction can never be resolved.
                    supporting_signals={
                        "event_id": event["event_id"],
                        "event_title": payload.get("title"),
                        "event_start_time": start_time.isoformat(),
                        "preparation_type": "large_meeting",
                        "attendee_count": len(attendees),
                    },
                ))

        # Diagnostic summary
        travel_events = sum(1 for e in parsed_events if any(kw in e["payload"].get("title", "").lower() for kw in ["flight", "airport", "hotel", "travel", "trip"]))
        large_meetings = sum(1 for e in parsed_events if len(e["payload"].get("attendees", [])) > 3)
        logger.info(
            "preparation_needs: Analyzed %d synced events → %d in 12-48h window "
            "(travel=%d, large_meetings=%d) → %d predictions",
            len(events), len(parsed_events), travel_events, large_meetings, len(predictions),
        )

        return predictions

    async def _check_spending_patterns(self, ctx: dict) -> list[Prediction]:
        """
        Detect spending anomalies and subscription waste.

        Aggregates the last 30 days of transactions by category and
        flags any single category that consumes >25% of total spend
        AND exceeds $200 absolute. The dual threshold avoids false
        positives for low overall spending or evenly-split budgets.
        """
        predictions = []

        with self.db.get_connection("events") as conn:
            # Transactions in the last 30 days
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

            transactions = conn.execute(
                """SELECT payload FROM events
                   WHERE type = 'finance.transaction.new'
                   AND timestamp > ?""",
                (cutoff,),
            ).fetchall()

        if len(transactions) < 5:
            logger.info(
                "spending_patterns: %d transactions found (need ≥5) — skipping",
                len(transactions),
            )
            return predictions  # Not enough data for meaningful patterns

        # Aggregate spending by category from transaction payloads
        by_category: dict[str, float] = {}
        for txn in transactions:
            payload = json.loads(txn["payload"])
            cat = payload.get("category", "uncategorized")
            amount = abs(payload.get("amount", 0))
            by_category[cat] = by_category.get(cat, 0) + amount

        total = sum(by_category.values())
        if total == 0:
            return predictions

        # Flag categories that dominate spending (>25% share AND >$200 absolute)
        for cat, amount in by_category.items():
            pct = amount / total
            if pct > 0.25 and amount > 200:
                predictions.append(Prediction(
                    prediction_type="risk",
                    description=(
                        f"Spending alert: ${amount:.0f} on '{cat}' this month "
                        f"({pct * 100:.0f}% of total)"
                    ),
                    confidence=0.5,
                    confidence_gate=ConfidenceGate.SUGGEST,
                    time_horizon="this_week",
                    suggested_action=f"Review {cat} spending",
                    # supporting_signals enables BehavioralAccuracyTracker._infer_risk_accuracy()
                    # to identify the flagged category and measure whether spending in that
                    # category decreased after the alert (lines 703-705 of tracker.py).
                    # Without these fields the tracker must fall back to regex-parsing the
                    # description, which can fail for category names containing special
                    # characters.  Providing structured signals is the authoritative path.
                    supporting_signals={
                        "category": cat,
                        "amount": round(amount, 2),
                        "percentage": round(pct * 100, 1),
                        "total_spending": round(total, 2),
                        "transaction_count": len(transactions),
                    },
                ))

        # Diagnostic summary
        high_spend_categories = sum(1 for cat, amt in by_category.items() if amt / total > 0.25 and amt > 200)
        logger.info(
            "spending_patterns: Analyzed %d transactions "
            "(total=$%.0f, categories=%d, high_spend=%d) → %d predictions",
            len(transactions), total, len(by_category), high_spend_categories, len(predictions),
        )

        return predictions

    async def _check_connector_health(self, ctx: dict) -> list[Prediction]:
        """Detect enabled connectors stuck in an error state and generate risk predictions.

        Queries the connector_state table in state.db for enabled connectors
        with status='error' and error_count >= 3 (to skip transient failures).
        Deduplicates against existing connector health predictions from the last
        7 days to avoid re-alerting for the same broken connector.

        Returns:
            List of Prediction objects for connectors that need user attention.
        """
        predictions: list[Prediction] = []
        now = datetime.now(timezone.utc)

        # Query broken connectors from state.db
        try:
            with self.db.get_connection("state") as conn:
                broken = conn.execute(
                    """SELECT connector_id, status, enabled, last_sync,
                              error_count, last_error, updated_at
                       FROM connector_state
                       WHERE enabled = 1
                         AND status = 'error'
                         AND error_count >= 3""",
                ).fetchall()
        except Exception as e:
            logger.warning("connector_health: failed to query connector_state (fail-open): %s", e)
            return predictions

        if not broken:
            logger.debug("connector_health: all enabled connectors healthy")
            return predictions

        # Deduplication: check for existing connector health predictions in last 7 days
        already_predicted_connectors: set[str] = set()
        try:
            dedup_cutoff = (now - timedelta(days=7)).isoformat()
            with self.db.get_connection("user_model") as conn:
                existing = conn.execute(
                    """SELECT supporting_signals FROM predictions
                       WHERE prediction_type = 'risk'
                       AND created_at > ?""",
                    (dedup_cutoff,),
                ).fetchall()

            for row in existing:
                try:
                    signals = json.loads(row["supporting_signals"])
                    cid = signals.get("connector_id")
                    # Only match predictions that are specifically connector health alerts
                    if cid and signals.get("prediction_source") == "connector_health":
                        already_predicted_connectors.add(cid)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug("connector_health: skipping malformed dedup entry: %s", e)
        except Exception as e:
            logger.warning(
                "connector_health dedup query failed (skipping dedup, may produce duplicates): %s", e
            )

        for row in broken:
            connector_id = row["connector_id"]

            # Skip if already predicted in the last 7 days
            if connector_id in already_predicted_connectors:
                continue

            # Calculate staleness from last_sync (or updated_at as fallback)
            reference_time = row["last_sync"] or row["updated_at"]
            try:
                ref_dt = datetime.fromisoformat(reference_time)
                if ref_dt.tzinfo is None:
                    ref_dt = ref_dt.replace(tzinfo=timezone.utc)
                staleness = now - ref_dt
                days_stale = max(1, int(staleness.total_seconds() / 86400))
            except (ValueError, TypeError):
                days_stale = 0  # Can't determine staleness, still report the error

            last_error = row["last_error"] or "Unknown error"
            error_count = row["error_count"] or 0

            if days_stale > 0:
                desc = f"{connector_id} connector has been failing for {days_stale} day{'s' if days_stale != 1 else ''} — {last_error}"
            else:
                desc = f"{connector_id} connector is in error state — {last_error}"

            predictions.append(Prediction(
                prediction_type="risk",
                description=desc,
                confidence=0.95,
                confidence_gate=ConfidenceGate.DEFAULT,
                time_horizon="this_week",
                suggested_action=f"Check {connector_id} connector configuration and reauthenticate if needed",
                supporting_signals={
                    "prediction_source": "connector_health",
                    "connector_id": connector_id,
                    "error_count": error_count,
                    "last_error": last_error,
                    "last_sync": row["last_sync"],
                    "days_stale": days_stale,
                },
            ))

        logger.info(
            "connector_health: %d broken connectors found, %d already predicted, %d new predictions",
            len(broken), len(already_predicted_connectors), len(predictions),
        )

        return predictions

    # -------------------------------------------------------------------
    # Reaction Prediction — Should we surface this?
    # -------------------------------------------------------------------

    async def predict_reaction(self, prediction: Prediction,
                                context: dict) -> ReactionPrediction:
        """
        Before surfacing a prediction, estimate whether the user will
        find it helpful, annoying, or intrusive right now.

        This is the reaction prediction gatekeeper. It scores each
        prediction on a -1.0 to +1.0 scale using multiple signals,
        then classifies the result:
            score > 0.3  -> "helpful"  (surface it)
            score > -0.1 -> "neutral"  (surface it, but lower priority)
            score <= -0.1 -> "annoying" (suppress it)

        CALIBRATION NOTE: The original thresholds (0.4 and 0.1) were too
        conservative, suppressing 99.95% of predictions and completely
        breaking the feedback loop. Recalibrated to allow more predictions
        through while still filtering truly annoying interruptions.
        """
        # --- Gather context signals ---
        # Current mood from the mood_signals profile.
        # Wrapped in try/except so a corrupted user_model.db doesn't crash
        # reaction prediction. Default to empty data (neutral mood).
        try:
            mood_profile = self.ums.get_signal_profile("mood_signals")
            mood_data = mood_profile["data"] if mood_profile else {}
        except Exception as e:
            logger.warning("predict_reaction: mood profile query failed (using defaults): %s", e)
            mood_data = {}

        # Count how many notifications the user dismissed in the last 2 hours.
        # A high count means they're in "leave me alone" mode.
        # Wrapped in try/except so preferences DB errors don't crash prediction.
        recent_dismissals = 0
        try:
            with self.db.get_connection("preferences") as conn:
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
                row = conn.execute(
                    """SELECT COUNT(*) as cnt FROM feedback_log
                       WHERE feedback_type = 'dismissed' AND timestamp > ?""",
                    (cutoff,),
                ).fetchone()
                recent_dismissals = row["cnt"] if row else 0
        except Exception as e:
            logger.warning("predict_reaction: dismissal count query failed (using 0): %s", e)

        # --- Scoring logic ---
        # Start at 0.3 (slightly positive) to bias toward surfacing by default.
        # The original 0.5 start was too high given the penalty magnitudes.
        score = 0.3

        # Stress detection: check the last 5 mood signals for negative
        # language or calendar overload. If stressed, reduce score to
        # avoid piling more onto an overwhelmed user.
        # REDUCED from −0.2 to −0.1 to avoid over-penalizing.
        stress_signals = mood_data.get("recent_signals", [])
        stress_count = sum(1 for s in stress_signals[-5:]
                          if s.get("signal_type") in ["negative_language", "calendar_density"])
        if stress_count > 2:
            score -= 0.1

        # Dismissal fatigue: >3 dismissals in 2 hours = strong "go away" signal.
        # REDUCED from −0.3 to −0.2 and increased threshold from >3 to >5
        # to avoid being too reactive to a few dismissals.
        if recent_dismissals > 5:
            score -= 0.2

        # High-confidence predictions are more likely to be genuinely helpful
        if prediction.confidence > 0.7:
            score += 0.2

        # Urgency weighting: conflicts and risks warrant interruption more
        # than opportunities (which are nice-to-have, not need-to-know).
        if prediction.prediction_type in ("conflict", "risk"):
            score += 0.2
        elif prediction.prediction_type == "opportunity":
            score -= 0.05  # REDUCED from −0.1 to allow opportunities through

        # Quiet hours check: suppress non-urgent predictions when the user is
        # in their configured quiet hours (e.g., 22:00–07:00 on weeknights).
        # Uses the same quiet_hours preference that NotificationManager uses,
        # so the prediction gate and the delivery gate agree on "do not disturb".
        #
        # Falls back to a broad low-activity heuristic from the cadence profile
        # when no explicit quiet hours are configured: if the current hour shows
        # very low historical activity (< 5% of peak), treat it as a natural
        # quiet period and apply the same penalty.
        now = datetime.now(timezone.utc)
        local_now = now.astimezone(ZoneInfo(self._tz_name))
        in_quiet_hours = self._is_quiet_hours(local_now)

        # Secondary: check cadence profile for naturally observed low-activity hours.
        # Only used when no explicit quiet hours are configured.
        in_low_activity_hour = False
        if not in_quiet_hours:
            try:
                cadence_profile = self.ums.get_signal_profile("cadence")
            except Exception as e:
                logger.warning("predict_reaction: cadence profile query failed (skipping): %s", e)
                cadence_profile = None
            if cadence_profile:
                hourly_activity: dict = cadence_profile["data"].get("hourly_activity", {})
                if hourly_activity:
                    peak = max(hourly_activity.values(), default=0)
                    current_hour_activity = hourly_activity.get(str(local_now.hour), 0)
                    # Flag as low-activity if current hour is below 5% of peak.
                    # This catches observed quiet periods (e.g., someone who never
                    # messages before 8am or after 10pm) without requiring the
                    # user to manually configure quiet hours.
                    if peak > 0 and current_hour_activity < peak * 0.05:
                        in_low_activity_hour = True

        # Apply penalty for quiet or low-activity hours — but only for non-urgent
        # prediction types. Conflicts and risks always get through.
        if (in_quiet_hours or in_low_activity_hour) and prediction.prediction_type not in ("conflict", "risk"):
            score -= 0.2

        # --- Classify the final score into a reaction label ---
        # RECALIBRATED: helpful >= 0.2, neutral > −0.1, else annoying.
        # The >= 0.2 threshold allows most "default" gate predictions (0.6-0.8
        # confidence) to surface unless they accumulate multiple penalties.
        # Round to 2 decimal places to avoid floating point precision issues
        # (e.g., 0.3 - 0.1 = 0.19999999... in Python).
        score = round(score, 2)
        predicted = "helpful" if score >= 0.2 else ("neutral" if score > -0.1 else "annoying")

        return ReactionPrediction(
            proposed_action=prediction.description,
            predicted_reaction=predicted,
            confidence=min(1.0, abs(score)),
            reasoning=(
                f"score={score:.2f}, dismissals={recent_dismissals}, "
                f"stress_signals={stress_count}, quiet_hours={in_quiet_hours}, "
                f"low_activity={in_low_activity_hour}"
            ),
        )

    def _is_quiet_hours(self, now: datetime) -> bool:
        """Check whether *now* falls within the user's configured quiet hours.

        ``now`` should already be in the user's local timezone so that
        ``.time()`` and ``.strftime("%A")`` return local values.

        Quiet hours are stored in the ``preferences`` database under the key
        ``quiet_hours`` as a JSON list of time-range objects::

            [{"start": "22:00", "end": "07:00", "days": ["monday", ...]}, ...]

        Multiple ranges are supported (e.g., different times on weekdays vs.
        weekends).  Overnight ranges (start > end, e.g., 22:00–07:00) are
        handled correctly.

        Returns ``False`` (fail-open) when no quiet hours are configured or
        when the stored data is malformed, so that a missing preference never
        prevents a prediction from surfacing.

        This mirrors the logic in ``NotificationManager._is_quiet_hours()`` so
        that the prediction gate and the notification delivery gate always agree
        on the user's "do not disturb" window.
        """
        try:
            with self.db.get_connection("preferences") as conn:
                row = conn.execute(
                    "SELECT value FROM user_preferences WHERE key = 'quiet_hours'"
                ).fetchone()

            if not row:
                return False

            quiet_hours_list = json.loads(row["value"])
            current_time = now.time()
            current_day = now.strftime("%A").lower()

            for qh in quiet_hours_list:
                # Skip ranges that don't apply to today.
                if current_day not in qh.get("days", []):
                    continue

                start = time.fromisoformat(qh["start"])
                end = time.fromisoformat(qh["end"])

                if start <= end:
                    # Same-day range (e.g., 09:00–17:00)
                    if start <= current_time <= end:
                        return True
                else:
                    # Overnight range (e.g., 22:00–07:00 crosses midnight)
                    if current_time >= start or current_time <= end:
                        return True

        except Exception as e:
            # Malformed data or DB error — fail open.
            logger.debug("quiet_hours: could not parse quiet hours config, failing open: %s", e)

        return False

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _get_accuracy_multiplier(self, prediction_type: str) -> float:
        """Compute confidence multiplier based on historical accuracy for this prediction type.

        The multiplier scales prediction confidence up or down based on how accurate
        that prediction type has been historically. This closes the feedback loop:
        types that are frequently wrong lose confidence; types that are usually right
        gain confidence.

        **Excludes automated-sender fast-path resolutions from the count.**

        Background: The BehavioralAccuracyTracker resolves predictions immediately as
        inaccurate when the contact is a marketing/automated sender (PRs #183–#189).
        This "fast-path" was introduced to clean up predictions generated before the
        marketing filter was robust — predictions that were structurally unfulfillable
        (the user can never reach out to noreply@company.com). These are bugs in
        prediction generation, not real user-behavior signals.

        Without this exclusion, the 174 fast-path-resolved opportunity predictions
        inflated the "inaccurate" count, producing a measured accuracy of 19% and a
        0.3 multiplier floor. Combined with the opportunity confidence cap of 0.6,
        this meant: 0.6 × 0.3 = 0.18 — below the 0.3 surfacing threshold, so
        ALL opportunity predictions were silently suppressed. The learning loop was
        broken: no predictions surfaced → no new accuracy data → can't recover.

        By excluding ``resolution_reason = 'automated_sender_fast_path'`` rows, the
        multiplier now reflects real user behavior (did the user reach out or not?)
        rather than historical prediction-generation bugs.

        Returns:
            0.3  — heavy penalty floor for very low accuracy (<20% after 10+ resolved).
                   Using 0.3 instead of 0.0 ensures the learning loop can recover
                   naturally as new, higher-quality predictions accumulate.  A hard
                   0.0 creates a death spiral: predictions are blocked → no new data
                   → accuracy never improves → permanently blocked.
            0.5-1.1 — scaled by accuracy rate (50% accuracy = 1.0x baseline)
            1.0  — insufficient data (<5 resolved predictions)

        Examples:
            _get_accuracy_multiplier("opportunity")
            # 41 accurate / 74 real-behavior samples (fast-path excluded) = 55%
            # → 0.5 + 0.55 * 0.6 = 0.83  (healthy multiplier, predictions surface)

            _get_accuracy_multiplier("reminder")  # 71% accuracy
            # → 0.5 + 0.71 * 0.6 = 0.926

            _get_accuracy_multiplier("routine_deviation")  # 100% accuracy
            # → 0.5 + 1.0 * 0.6 = 1.1

        On DB error, returns 1.0 (no adjustment) so predictions are not
        penalized or boosted when accuracy data is unavailable.
        """
        try:
            with self.db.get_connection("user_model") as conn:
                row = conn.execute(
                    """SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate
                       FROM predictions
                       WHERE prediction_type = ?
                         AND was_surfaced = 1
                         AND resolved_at IS NOT NULL
                         AND (resolution_reason IS NULL
                              OR resolution_reason != 'automated_sender_fast_path')""",
                    (prediction_type,),
                ).fetchone()
        except Exception as e:
            logger.warning("_get_accuracy_multiplier(%s) query failed (returning 1.0): %s", prediction_type, e)
            return 1.0

        total = row["total"] if row else 0
        accurate = row["accurate"] if row else 0

        if total < 5:
            return 1.0  # Not enough data to adjust

        accuracy_rate = accurate / total

        # Apply a heavy penalty for types with <20% accuracy after sufficient samples,
        # but use a floor of 0.3 rather than 0.0.  A 0.0 multiplier is a death switch:
        # it permanently blocks all predictions of this type, so no new data can ever
        # be collected and the accuracy can never recover.  With a 0.3 floor the type
        # continues to generate predictions at reduced confidence, allowing the feedback
        # loop to gradually rehabilitate genuinely useful types whose history was
        # polluted by now-fixed bugs (e.g. marketing-sender predictions, broken
        # deduplication, timezone errors in earlier iterations).
        if accuracy_rate < 0.2 and total >= 10:
            logger.warning(
                "accuracy_multiplier: %s has %.1f%% accuracy over %d samples "
                "(excluding automated-sender fast-path) — applying heavy-penalty floor (0.3)",
                prediction_type, accuracy_rate * 100, total,
            )
            return 0.3

        # Scale: 50% accuracy = 1.0x, 0% = 0.5x, 100% = 1.1x
        return 0.5 + (accuracy_rate * 0.6)

    def _get_contact_accuracy_multiplier(self, contact_email: str) -> float:
        """Compute a confidence multiplier for opportunity predictions targeting a specific contact.

        The global ``_get_accuracy_multiplier`` treats all opportunity predictions
        identically, but contact response rates vary enormously.  This method looks
        at the resolved accuracy history for a specific contact and returns a
        supplementary multiplier so the engine can:

        - **Boost** confidence for contacts the user consistently reaches out to
          (multiplier > 1.0): the pattern is reliable for this person.
        - **Suppress** confidence for contacts the user never reaches out to despite
          suggestions (multiplier < 1.0): stop nagging about this relationship.
        - **Return 1.0** (no adjustment) when there isn't enough history yet (< 3
          resolved predictions for this contact), so we don't over-fit on a single
          data point.

        The minimum floor is 0.5 (never fully suppress a contact — their behaviour
        may change) and the ceiling is 1.2 (modest boost; we can't exceed the
        ``_check_relationship_maintenance`` initial cap of 0.6 dramatically).

        **Only fast-path-excluded resolutions are considered**, matching the same
        exclusion logic used in ``_get_accuracy_multiplier``.  Automated-sender
        resolutions are structural bugs in prediction generation, not real signal
        about whether the user values this contact.

        Args:
            contact_email: Lowercase email address of the contact to look up.

        Returns:
            0.5  — contact has poor response rate (< 20% accuracy, 3+ samples).
                   Still surfaces predictions at reduced confidence; behaviour can
                   change and we don't want to miss it permanently.
            0.5–1.2 — scaled by per-contact accuracy rate.
                   Formula: 0.5 + (accuracy_rate * 0.7), capped at 1.2.
                   50% accuracy → 0.85x, 100% accuracy → 1.2x.
            1.0  — insufficient data (< 3 resolved predictions for this contact).

        Examples:
            _get_contact_accuracy_multiplier("alice@example.com")
            # 8 accurate out of 10 resolved for alice = 80% accuracy
            # → 0.5 + 0.8 * 0.7 = 1.06x  (modest boost)

            _get_contact_accuracy_multiplier("distant@example.com")
            # 0 accurate out of 5 resolved for distant = 0% accuracy
            # → 0.5 floor  (suppressed but not blocked)

            _get_contact_accuracy_multiplier("new@example.com")
            # Only 2 resolved predictions → not enough data
            # → 1.0  (no adjustment)

        On DB error, returns 1.0 (no adjustment) so predictions are not
        penalized or boosted when accuracy data is unavailable.
        """
        # Query resolved opportunity predictions where supporting_signals contains
        # this contact's email.  We use JSON_EXTRACT when available (SQLite >= 3.38)
        # with a string-contains fallback for older builds.
        try:
            with self.db.get_connection("user_model") as conn:
                row = conn.execute(
                    """SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate
                       FROM predictions
                       WHERE prediction_type = 'opportunity'
                         AND was_surfaced = 1
                         AND resolved_at IS NOT NULL
                         AND (resolution_reason IS NULL
                              OR resolution_reason != 'automated_sender_fast_path')
                         AND supporting_signals LIKE ?""",
                    (f'%"contact_email": "{contact_email.lower()}"%',),
                ).fetchone()
        except Exception as e:
            logger.warning("_get_contact_accuracy_multiplier(%s) query failed (returning 1.0): %s", contact_email, e)
            return 1.0

        total = row["total"] if row else 0
        accurate = row["accurate"] if row else 0

        if total < 3:
            # Not enough contact-specific history — defer to the type-level multiplier
            return 1.0

        accuracy_rate = accurate / total

        if accuracy_rate < 0.2:
            # Contact never responds to reach-out suggestions; reduce but don't
            # silence completely so we can detect when the relationship resumes.
            logger.warning(
                "contact_accuracy_multiplier: %s has %.1f%% accuracy over %d resolved predictions "
                "— applying per-contact suppression floor (0.5)",
                contact_email, accuracy_rate * 100, total,
            )
            return 0.5

        # Scale: 50% accuracy → 0.85x, 100% accuracy → 1.2x
        # Ceiling at 1.2 prevents runaway confidence amplification.
        return min(1.2, 0.5 + (accuracy_rate * 0.7))

    def _get_suppressed_prediction_keys(self) -> set[tuple[str, str | None]]:
        """Return a set of (prediction_type, contact_email_or_None) keys that the user
        has explicitly marked as ``not_relevant`` within the last 90 days.

        When a user clicks "Not About Me" on a prediction card, the frontend stores
        ``user_response = 'not_relevant'`` via ``resolve_prediction()``.  This method
        queries those resolved predictions and builds a suppression set so that
        ``generate_predictions()`` can filter out predictions the user has already
        told us are irrelevant.

        The 90-day window ensures suppressions are not permanent — user preferences
        change over time and we should re-check periodically.

        Returns:
            A set of ``(prediction_type, contact_email_or_None)`` tuples.  For
            predictions that targeted a specific contact (e.g. relationship
            maintenance), the contact_email is extracted from ``supporting_signals``.
            For non-contact predictions, the second element is ``None``.

        On DB error, returns an empty set (fail-open: no suppression rather
        than crashing the pipeline).
        """
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT prediction_type, supporting_signals
                       FROM predictions
                       WHERE user_response = 'not_relevant'
                         AND was_surfaced = 1
                         AND resolved_at IS NOT NULL
                         AND datetime(resolved_at) > datetime('now', '-90 days')""",
                ).fetchall()
        except Exception as e:
            logger.warning("_get_suppressed_prediction_keys query failed (skipping suppression): %s", e)
            return set()

        keys: set[tuple[str, str | None]] = set()
        for row in rows:
            pred_type = row["prediction_type"]
            contact_email = None
            if row["supporting_signals"]:
                try:
                    signals = json.loads(row["supporting_signals"])
                    contact_email = signals.get("contact_email")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug("suppression: skipping malformed supporting_signals: %s", e)
            keys.add((pred_type, contact_email))
        return keys

    @staticmethod
    def _is_suppressed(prediction: Prediction, suppressed_keys: set[tuple[str, str | None]]) -> bool:
        """Check whether a prediction matches any not_relevant suppression key.

        A prediction is suppressed if the user has previously marked a prediction
        of the same type (and, when applicable, the same contact) as ``not_relevant``.

        Matching rules:
          - (prediction_type, contact_email) — exact match for contact-specific predictions
          - (prediction_type, None) — type-only suppression for predictions without a contact

        Args:
            prediction: The candidate prediction to check.
            suppressed_keys: Set of suppressed keys from ``_get_suppressed_prediction_keys()``.

        Returns:
            True if the prediction should be suppressed (not surfaced).
        """
        pred_type = prediction.prediction_type
        contact_email = None
        if prediction.supporting_signals:
            contact_email = prediction.supporting_signals.get("contact_email")

        # Check exact match (type + contact)
        if (pred_type, contact_email) in suppressed_keys:
            return True

        # For contact predictions, also check if the type was suppressed generically
        # (user marked a non-contact prediction of this type as not_relevant)
        if contact_email and (pred_type, None) in suppressed_keys:
            return True

        return False

    @staticmethod
    def _is_marketing_or_noreply(from_addr: str, payload: dict) -> bool:
        """Delegate to the shared canonical marketing filter.

        The authoritative implementation lives in
        services.signal_extractor.marketing_filter.is_marketing_or_noreply().
        This wrapper exists so existing call sites (self._is_marketing_or_noreply)
        continue to work without modification.

        See marketing_filter.py for full documentation of the filter logic.
        """
        return is_marketing_or_noreply(from_addr, payload)

    def _count_calendar_event_types(self) -> tuple[int, int]:
        """Count all-day vs timed calendar events in the events database.

        Scans up to 1000 ``calendar.event.created`` event payloads and
        classifies each as all-day or timed based on the ``is_all_day``
        field.  Malformed or unparseable payloads are skipped with a debug log.

        Returns:
            Tuple of (all_day_count, timed_count).
        """
        all_day_count = 0
        timed_count = 0
        with self.db.get_connection("events") as conn:
            rows = conn.execute(
                "SELECT payload FROM events WHERE type = 'calendar.event.created' LIMIT 1000"
            ).fetchall()
        for row in rows:
            try:
                payload = json.loads(row["payload"])
                if payload.get("is_all_day"):
                    all_day_count += 1
                else:
                    timed_count += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.debug("calendar_stats: skipping malformed event payload: %s", e)
        return all_day_count, timed_count

    async def get_diagnostics(self) -> dict:
        """
        Comprehensive prediction engine diagnostics.

        Returns a detailed analysis of why each prediction type is or isn't
        generating predictions, including data availability, configuration gaps,
        and actionable recommendations.

        This is the single source of truth for understanding prediction engine
        behavior and debugging issues.

        Returns:
            Dictionary with structure:
            {
                "prediction_types": {
                    "reminder": {
                        "status": "active" | "limited" | "blocked",
                        "generated_last_7d": int,
                        "data_available": {
                            "unreplied_emails": int,
                            "recent_messages": int,
                            ...
                        },
                        "blockers": ["list", "of", "issues"],
                        "recommendations": ["actionable", "steps"]
                    },
                    ...
                },
                "overall": {
                    "total_predictions_7d": int,
                    "active_types": int,
                    "blocked_types": int,
                    "health": "healthy" | "degraded" | "broken"
                }
            }
        """
        diagnostics = {"prediction_types": {}, "overall": {}}
        now = datetime.now(timezone.utc)
        week_ago = (now - timedelta(days=7)).isoformat()

        # Get prediction counts for last 7 days
        with self.db.get_connection("user_model") as conn:
            pred_counts = conn.execute(
                """SELECT prediction_type, COUNT(*) as count
                   FROM predictions
                   WHERE created_at > ?
                   GROUP BY prediction_type""",
                (week_ago,),
            ).fetchall()

        prediction_type_counts = {row["prediction_type"]: row["count"] for row in pred_counts}

        # --- Data source health probes ---
        # Probe each signal profile the engine depends on so the admin can
        # see which are accessible vs corrupted or empty.
        data_source_health = {}
        for profile_name in ["relationships", "cadence", "mood_signals", "linguistic", "topics"]:
            try:
                profile = self.ums.get_signal_profile(profile_name)
                data_source_health[profile_name] = {
                    "status": "available" if profile else "empty",
                    "samples": profile.get("samples_count", 0) if profile else 0,
                }
            except Exception as e:
                data_source_health[profile_name] = {
                    "status": "error",
                    "error": str(e),
                }
        diagnostics["data_sources"] = data_source_health

        # --- Follow-up needs (reminder) ---
        with self.db.get_connection("events") as conn:
            unreplied_count = conn.execute(
                """SELECT COUNT(*) as count FROM events
                   WHERE type = 'email.received'
                   AND timestamp > ?""",
                ((now - timedelta(hours=24)).isoformat(),),
            ).fetchone()["count"]

            replied_count = conn.execute(
                """SELECT COUNT(*) as count FROM events
                   WHERE type = 'email.sent'
                   AND json_extract(payload, '$.in_reply_to') IS NOT NULL
                   AND timestamp > ?""",
                ((now - timedelta(hours=24)).isoformat(),),
            ).fetchone()["count"]

        actual_unreplied = unreplied_count - replied_count
        reminder_status = "active" if prediction_type_counts.get("reminder", 0) > 0 else "limited"
        reminder_blockers = []
        reminder_recommendations = []

        if actual_unreplied == 0:
            reminder_status = "blocked"
            reminder_blockers.append("No unreplied emails in last 24h")
            reminder_recommendations.append("Check if email connector is syncing correctly")
        elif prediction_type_counts.get("reminder", 0) == 0:
            reminder_blockers.append("Unreplied emails exist but 0 predictions generated")
            reminder_recommendations.append("Check marketing filter - may be filtering all emails")

        diagnostics["prediction_types"]["reminder"] = {
            "status": reminder_status,
            "generated_last_7d": prediction_type_counts.get("reminder", 0),
            "data_available": {
                "unreplied_emails_24h": actual_unreplied,
                "total_received_24h": unreplied_count,
                "replies_sent_24h": replied_count,
            },
            "blockers": reminder_blockers,
            "recommendations": reminder_recommendations,
        }

        # --- Calendar conflicts ---
        all_day_count, timed_count = self._count_calendar_event_types()
        with self.db.get_connection("events") as conn:
            calendar_events = conn.execute(
                "SELECT COUNT(*) as count FROM events WHERE type = 'calendar.event.created'"
            ).fetchone()["count"]

        conflict_status = "active" if prediction_type_counts.get("conflict", 0) > 0 else "blocked"
        conflict_blockers = []
        conflict_recommendations = []

        if calendar_events == 0:
            conflict_blockers.append("No calendar events in database")
            conflict_recommendations.append("Enable and configure CalDAV or Google Calendar connector")
        elif timed_count == 0:
            conflict_blockers.append("All calendar events are all-day (0 timed events for conflict detection)")
            conflict_recommendations.append("Verify calendar connector is correctly parsing timed events")
            conflict_recommendations.append("Check if calendar contains any actual appointments (not just birthdays/holidays)")

        diagnostics["prediction_types"]["conflict"] = {
            "status": conflict_status,
            "generated_last_7d": prediction_type_counts.get("conflict", 0),
            "data_available": {
                "total_calendar_events": calendar_events,
                "all_day_events": all_day_count,
                "timed_events": timed_count,
            },
            "blockers": conflict_blockers,
            "recommendations": conflict_recommendations,
        }

        # --- Relationship maintenance (opportunity) ---
        try:
            rel_profile = self.ums.get_signal_profile("relationships")
        except Exception as e:
            logger.warning("get_diagnostics: failed to read relationships profile: %s", e)
            rel_profile = None
        if rel_profile:
            contacts = rel_profile["data"].get("contacts", {})
            eligible_contacts = sum(1 for data in contacts.values()
                                   if data.get("interaction_count", 0) >= 5)
            total_contacts = len(contacts)
        else:
            contacts = {}
            eligible_contacts = 0
            total_contacts = 0

        opportunity_status = "active" if prediction_type_counts.get("opportunity", 0) > 0 else "blocked"
        opportunity_blockers = []
        opportunity_recommendations = []

        if total_contacts == 0:
            opportunity_blockers.append("No contacts tracked in relationships profile")
            opportunity_recommendations.append("Ensure email connector is running and processing messages")
        elif eligible_contacts == 0:
            opportunity_blockers.append("No contacts with 5+ interactions (need history to detect maintenance needs)")
            opportunity_recommendations.append("Wait for more email history to accumulate (need 5+ interactions per contact)")
        else:
            # Check if all contacts are marketing
            marketing_count = sum(1 for addr in contacts.keys()
                                 if self._is_marketing_or_noreply(addr, {}))
            if marketing_count / total_contacts > 0.9:
                opportunity_blockers.append(f"{marketing_count}/{total_contacts} contacts are marketing/automated (no human relationships)")
                opportunity_recommendations.append("This inbox appears to contain primarily marketing emails")
                opportunity_recommendations.append("Consider filtering marketing emails before they reach Life OS")

        diagnostics["prediction_types"]["opportunity"] = {
            "status": opportunity_status,
            "generated_last_7d": prediction_type_counts.get("opportunity", 0),
            "data_available": {
                "total_contacts": total_contacts,
                "eligible_contacts": eligible_contacts,
                "marketing_filtered": sum(1 for addr in contacts.keys()
                                         if self._is_marketing_or_noreply(addr, {}))
                                     if contacts else 0,
            },
            "blockers": opportunity_blockers,
            "recommendations": opportunity_recommendations,
        }

        # --- Preparation needs (need) ---
        # Reuse the pre-computed all_day_count / timed_count from the helper
        # so this section is independent of the conflict section above.
        total_calendar_events = all_day_count + timed_count

        need_status = "active" if prediction_type_counts.get("need", 0) > 0 else "blocked"
        need_blockers = []
        need_recommendations = []

        if total_calendar_events == 0:
            need_blockers.append("No calendar events available")
            need_recommendations.append("Enable calendar connector to track upcoming events")
        elif timed_count == 0:
            need_blockers.append("All events are all-day (preparation needs require timed events)")
            need_recommendations.append("Verify calendar contains actual appointments, not just reminders")

        diagnostics["prediction_types"]["need"] = {
            "status": need_status,
            "generated_last_7d": prediction_type_counts.get("need", 0),
            "data_available": {
                "total_events": total_calendar_events,
                "timed_events": timed_count,
            },
            "blockers": need_blockers,
            "recommendations": need_recommendations,
        }

        # --- Spending patterns (risk) ---
        with self.db.get_connection("events") as conn:
            transaction_count = conn.execute(
                """SELECT COUNT(*) as count FROM events
                   WHERE type = 'finance.transaction.new'
                   AND timestamp > ?""",
                ((now - timedelta(days=30)).isoformat(),),
            ).fetchone()["count"]

        risk_status = "active" if prediction_type_counts.get("risk", 0) > 0 else "blocked"
        risk_blockers = []
        risk_recommendations = []

        if transaction_count == 0:
            risk_blockers.append("No finance transactions in database")
            risk_recommendations.append("Enable a finance connector (Plaid, Mint, or similar)")
        elif transaction_count < 5:
            risk_blockers.append(f"Only {transaction_count} transactions (need ≥5 for pattern detection)")
            risk_recommendations.append("Wait for more transaction history to accumulate")

        diagnostics["prediction_types"]["risk"] = {
            "status": risk_status,
            "generated_last_7d": prediction_type_counts.get("risk", 0),
            "data_available": {
                "transactions_30d": transaction_count,
            },
            "blockers": risk_blockers,
            "recommendations": risk_recommendations,
        }

        # --- Routine deviations ---
        with self.db.get_connection("user_model") as conn:
            routine_count = conn.execute(
                "SELECT COUNT(*) as count FROM routines WHERE consistency_score > 0.6"
            ).fetchone()["count"]

        routine_status = "active" if prediction_type_counts.get("routine_deviation", 0) > 0 else "blocked"
        routine_blockers = []
        routine_recommendations = []

        if routine_count == 0:
            routine_blockers.append("No routines with consistency_score > 0.6")
            routine_recommendations.append("Routine detection requires consistent behavioral patterns over time")
            routine_recommendations.append("Wait for routine detection loop to identify patterns (runs hourly)")

        diagnostics["prediction_types"]["routine_deviation"] = {
            "status": routine_status,
            "generated_last_7d": prediction_type_counts.get("routine_deviation", 0),
            "data_available": {
                "established_routines": routine_count,
            },
            "blockers": routine_blockers,
            "recommendations": routine_recommendations,
        }

        # --- Overall health ---
        total_predictions_7d = sum(prediction_type_counts.values())
        active_types = sum(1 for v in diagnostics["prediction_types"].values()
                          if v["status"] == "active")
        blocked_types = sum(1 for v in diagnostics["prediction_types"].values()
                           if v["status"] == "blocked")

        if active_types >= 3:
            health = "healthy"
        elif active_types >= 1:
            health = "degraded"
        else:
            health = "broken"

        overall_blockers = []
        # Flag data source errors as an overall blocker
        errored_sources = [
            name for name, info in data_source_health.items()
            if info["status"] == "error"
        ]
        if errored_sources:
            overall_blockers.append(
                f"user_model.db may be corrupted (failed profiles: {', '.join(errored_sources)})"
            )
            # Data source errors cap health at degraded or worse
            if health == "healthy":
                health = "degraded"

        diagnostics["overall"] = {
            "total_predictions_7d": total_predictions_7d,
            "active_types": active_types,
            "blocked_types": blocked_types,
            "total_types": len(diagnostics["prediction_types"]),
            "health": health,
            "blockers": overall_blockers,
            "persistence_failure_detected": self._persistence_failure_detected,
        }

        # Include last-run surfacing diagnostics so the async diagnostics
        # endpoint also shows per-prediction filter breakdown.
        diagnostics["surfacing"] = self._surfacing_diagnostics

        return diagnostics

    @staticmethod
    def _gate_from_confidence(confidence: float) -> ConfidenceGate:
        """
        Map a numeric confidence score to a ConfidenceGate enum.

        Confidence gate thresholds:
            < 0.3  -> OBSERVE    (watch silently, keep learning)
            0.3-0.6 -> SUGGEST   (ask "would you like me to...")
            0.6-0.8 -> DEFAULT   (do it, but make it easy to undo)
            > 0.8  -> AUTONOMOUS (just handle it without asking)

        NaN, inf, and negative values are treated as invalid and
        conservatively mapped to OBSERVE to prevent dangerous
        autonomous actions on corrupted confidence scores.
        """
        # Guard against NaN/inf/negative — these must never escalate to AUTONOMOUS
        if math.isnan(confidence) or math.isinf(confidence) or confidence < 0:
            return ConfidenceGate.OBSERVE
        if confidence < 0.3:
            return ConfidenceGate.OBSERVE
        elif confidence < 0.6:
            return ConfidenceGate.SUGGEST
        elif confidence < 0.8:
            return ConfidenceGate.DEFAULT
        else:
            return ConfidenceGate.AUTONOMOUS
