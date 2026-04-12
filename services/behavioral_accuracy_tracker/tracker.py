"""
Life OS — Behavioral Prediction Accuracy Tracker

Automatically infers prediction accuracy from user behavior, closing the
feedback loop without requiring explicit user interaction with notifications.

The Problem:
    The prediction engine needs accuracy data to learn and improve, but this
    data only comes from explicit user actions (clicking "Act On" or "Dismiss"
    buttons) or from auto-resolution after 24 hours of ignoring a notification.
    This creates a cold-start problem: new systems have no accuracy data, so
    they can't learn or calibrate confidence gates.

The Solution:
    Track user behavior and automatically mark predictions as accurate when the
    user's actions align with what was predicted, even if they never interacted
    with the notification directly.

Examples:
    - Prediction: "Reply to Alice about dinner plans"
      Behavior: User sends a message to Alice within 6 hours
      → Mark prediction as ACCURATE

    - Prediction: "Calendar conflict: Team sync overlaps with dentist"
      Behavior: User reschedules one of the events within 24 hours
      → Mark prediction as ACCURATE

    - Prediction: "Prepare slides for Q4 planning meeting"
      Behavior: User opens/edits a file containing "slides" or "Q4" keywords
      → Mark prediction as ACCURATE

    - Prediction: "Follow up with Bob about the project"
      Behavior: 48 hours pass, no message sent to Bob
      → Mark prediction as INACCURATE

This allows the system to bootstrap its learning from observed behavior instead
of waiting for explicit feedback, dramatically accelerating the calibration loop.

Architecture:
    - Runs as a background task every 15 minutes (same cadence as prediction engine)
    - Queries unresolved surfaced predictions
    - Scans recent events for behavioral signals that confirm or refute each prediction
    - Updates predictions.was_accurate when confidence threshold is met
    - Preserves user_response = 'inferred' to distinguish from explicit feedback
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from services.signal_extractor.marketing_filter import is_marketing_or_noreply
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class BehavioralAccuracyTracker:
    """Infers prediction accuracy from user behavior patterns."""

    def __init__(self, db: DatabaseManager):
        """Initialize the behavioral accuracy tracker.

        Proactively applies the schema migration that adds the
        ``resolution_reason`` column (migration 3 → 4) if it is missing.
        This ensures the tracker works correctly even when the server has
        not been restarted since the migration was introduced — for example,
        if PR #197 was merged while the server was still running with the
        old schema.

        Without this guard the tracker's UPDATE statement fails with
        ``OperationalError: no such column: resolution_reason``, causing
        ``run_inference_cycle`` to raise on every cycle and the entire
        behavioral-accuracy learning loop to be silently broken.

        Also runs a recurring backfill to tag any inaccurate opportunity/
        reminder predictions for automated senders that were missed by the
        one-time migration guard (e.g., predictions created before
        supporting_signals was added in PR #190 have no contact_email in
        signals, so the migration guard skips them; this method falls back
        to parsing the description to recover the email address).

        Args:
            db: Database manager for accessing events and predictions.
        """
        self.db = db
        self._last_cycle_stats: dict[str, int] | None = None
        self._total_cycles: int = 0
        self._cycles_with_no_predictions: int = 0
        self._last_cycle_predictions_found: int = 0
        self._last_cycle_timestamp: str | None = None
        try:
            self._ensure_resolution_reason_column()
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            logger.warning("BehavioralAccuracyTracker: user_model.db unavailable, skipping schema migration")
        try:
            self._backfill_automated_sender_tags()
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            logger.warning("BehavioralAccuracyTracker: user_model.db unavailable, skipping sender tag backfill")

    # ------------------------------------------------------------------
    # Schema self-repair
    # ------------------------------------------------------------------

    def _ensure_resolution_reason_column(self) -> None:
        """Add the ``resolution_reason`` column to predictions if it is missing.

        This is a forward-compatibility guard for the schema migration
        introduced in ``storage/manager.py`` (migration 3 → 4, CURRENT_VERSION=4).
        That migration runs automatically when the DatabaseManager is first
        initialized *after* the code update — i.e., on the next server restart.
        However, if the server is still running with an in-memory
        DatabaseManager instance that was created before the migration code was
        merged, the column will be absent from the live database and every
        ``run_inference_cycle`` call will raise::

            sqlite3.OperationalError: no such column: resolution_reason

        By applying the ALTER TABLE here we ensure the tracker is operational
        immediately after the code update, without requiring a restart.

        The operation is idempotent: if the column already exists (normal path
        after a clean restart) the PRAGMA check short-circuits and no DDL is
        executed.

        Post-migration: sets ``resolution_reason = 'automated_sender_fast_path'``
        on all existing inaccurate opportunity/reminder predictions whose
        supporting_signals point to an automated sender.  This retroactively
        tags the historical pollution so that ``_get_accuracy_multiplier``
        excludes them from the accuracy denominator immediately, rather than
        waiting for the next BehavioralAccuracyTracker cycle to set the flag on
        newly-resolved predictions only.

        Usage:
            Called once from ``__init__``.  Safe to call multiple times.
        """
        try:
            with self.db.get_connection("user_model") as conn:
                # 1. Check whether the predictions table exists yet.
                #    If it doesn't (e.g., DatabaseManager initialized but
                #    initialize_all() not yet called, as happens in some tests),
                #    there is nothing to migrate — silently return.
                table_exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
                ).fetchone()
                if not table_exists:
                    return  # Schema not initialized yet; migration will run at startup

                # 2. Check whether the column already exists (schema v4+)
                columns = [row[1] for row in conn.execute(
                    "PRAGMA table_info(predictions)"
                ).fetchall()]

                if "resolution_reason" in columns:
                    return  # Already migrated — nothing to do

                # 3. Column is missing: apply the migration now
                logger.info(
                    "BehavioralAccuracyTracker: resolution_reason column missing "
                    "from predictions table — applying migration 3→4 inline "
                    "(server restart not required)"
                )
                conn.execute("ALTER TABLE predictions ADD COLUMN resolution_reason TEXT")

                # 4. Update schema_version table so the DatabaseManager won't
                #    re-run the same migration on next restart and log a confusing
                #    "column already exists" warning.
                try:
                    max_ver = conn.execute(
                        "SELECT MAX(version) FROM schema_version"
                    ).fetchone()[0] or 0
                    if max_ver < 4:
                        conn.execute(
                            "INSERT INTO schema_version (version) VALUES (4)"
                        )
                        logger.info(
                            "BehavioralAccuracyTracker: schema_version updated to 4"
                        )
                except Exception:
                    pass  # schema_version table absence is non-fatal

                # 5. Retroactively tag existing inaccurate predictions for automated
                #    senders as 'automated_sender_fast_path' so _get_accuracy_multiplier
                #    excludes them immediately rather than waiting for the next cycle.
                #
                #    We only tag rows that are:
                #    - already resolved (resolved_at IS NOT NULL)
                #    - marked inaccurate (was_accurate = 0)
                #    - were surfaced (was_accurate accuracy multiplier only counts surfaced)
                #    - have a contact_email in supporting_signals that is an automated sender
                #
                #    We do NOT use Python to loop over rows here because the table can have
                #    hundreds of thousands of rows.  Instead we use SQLite json_extract to
                #    pull the contact_email inline, then filter in Python for automated-sender
                #    patterns (there is no SQL REGEXP without loading an extension).
                backfill_rows = conn.execute(
                    """SELECT id, supporting_signals
                       FROM predictions
                       WHERE was_surfaced = 1
                         AND was_accurate = 0
                         AND resolved_at IS NOT NULL
                         AND resolution_reason IS NULL
                         AND prediction_type IN ('opportunity', 'reminder')"""
                ).fetchall()

                tagged = 0
                for row in backfill_rows:
                    try:
                        signals = json.loads(row["supporting_signals"] or "{}") or {}
                        if isinstance(signals, list):
                            signals = {}
                        contact_email = signals.get("contact_email", "")
                        if contact_email and self._is_automated_sender(contact_email):
                            conn.execute(
                                "UPDATE predictions SET resolution_reason = ? WHERE id = ?",
                                ("automated_sender_fast_path", row["id"]),
                            )
                            tagged += 1
                    except Exception:
                        continue  # Skip malformed rows; non-fatal

                if tagged:
                    logger.info(
                        f"BehavioralAccuracyTracker: retroactively tagged {tagged} "
                        f"automated-sender predictions as 'automated_sender_fast_path' "
                        f"to unblock accuracy multiplier recovery"
                    )
        except sqlite3.DatabaseError as e:
            logger.warning("_ensure_resolution_reason_column: user_model.db unavailable: %s", e)

    def _backfill_automated_sender_tags(self) -> None:
        """Tag all untagged inaccurate automated-sender predictions on every startup.

        The one-time migration guard in ``_ensure_resolution_reason_column`` tags
        predictions created before ``resolution_reason`` was added to the schema.
        But it has a blind spot: it only inspects ``supporting_signals`` for the
        contact email.  Predictions generated before PR #190 added
        ``supporting_signals`` have NULL signals, so the migration guard skips them
        entirely.  Those predictions continue to count as "inaccurate" in the
        accuracy calculation, artificially depressing opportunity prediction
        confidence toward the 0.3 multiplier floor.

        This method runs on **every** server startup (not just during migration) to
        catch predictions the migration guard missed.  For each untagged, resolved,
        inaccurate opportunity/reminder prediction it tries two strategies in order:

        1. **signals path** — read ``contact_email`` from ``supporting_signals``
           (works for predictions created after PR #190).
        2. **description fallback** — regex-extract an email address from the
           prediction description string (works for older predictions without
           supporting_signals, e.g. "It's been 45 days since you last contacted
           noreply@company.com (you usually connect every ~14 days)").

        If the extracted email is an automated sender, the prediction is tagged
        ``resolution_reason = 'automated_sender_fast_path'`` so it is excluded from
        the ``_get_accuracy_multiplier`` denominator.

        This is idempotent: predictions already tagged are excluded by the WHERE
        clause (``resolution_reason IS NULL``), so repeated calls are safe and
        cheap.

        Example scenario fixed:
            Before PR #190, opportunity predictions had descriptions like:
                "It's been 45 days since you last contacted noreply@company.com"
            The migration guard skipped these (no supporting_signals).
            This method finds them, extracts "noreply@company.com" from the
            description, detects it as automated, and tags it immediately.
            Result: 174 stale automated-sender predictions excluded from accuracy
            calculation → opportunity accuracy rises from 19% to ~55%.
        """
        import re

        try:
            with self.db.get_connection("user_model") as conn:
                # Guard: predictions table might not exist yet in tests that
                # initialize DatabaseManager but haven't called initialize_all().
                table_exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
                ).fetchone()
                if not table_exists:
                    return

                # Also guard: resolution_reason column must exist before we can
                # query or write it (the migration guard adds it, but may not have
                # run yet if the table was just created).
                columns = [row[1] for row in conn.execute(
                    "PRAGMA table_info(predictions)"
                ).fetchall()]
                if "resolution_reason" not in columns:
                    return  # Migration guard will handle this on the same startup

                # Find ALL untagged resolved-inaccurate opportunity/reminder predictions.
                # Include both was_surfaced=1 (counts toward accuracy) and was_surfaced=0
                # (filters; don't count toward accuracy but clean up for consistency).
                backfill_rows = conn.execute(
                    """SELECT id, supporting_signals, description
                       FROM predictions
                       WHERE was_accurate = 0
                         AND resolved_at IS NOT NULL
                         AND resolution_reason IS NULL
                         AND prediction_type IN ('opportunity', 'reminder')"""
                ).fetchall()
        except sqlite3.DatabaseError as e:
            logger.warning("_backfill_automated_sender_tags: user_model.db unavailable (query phase): %s", e)
            return

        if not backfill_rows:
            return  # Nothing to tag

        # Email address regex — same pattern used in _infer_opportunity_accuracy
        email_re = re.compile(
            r'([\w\.\-\+]+@[\w\.\-]+\.[\w\.]+)',
            re.IGNORECASE,
        )

        tagged = 0
        try:
            with self.db.get_connection("user_model") as conn:
                for row in backfill_rows:
                    try:
                        # Strategy 1: extract contact_email from supporting_signals
                        contact_email = ""
                        try:
                            signals = json.loads(row["supporting_signals"] or "{}") or {}
                            if isinstance(signals, list):
                                signals = {}
                            contact_email = signals.get("contact_email", "")
                        except (json.JSONDecodeError, TypeError):
                            signals = {}

                        # Strategy 2: fall back to parsing the description field.
                        # Handles predictions generated before PR #190 added supporting_signals.
                        # Example description: "It's been 45 days since you last contacted
                        # noreply@company.com (you usually connect every ~14 days)"
                        if not contact_email:
                            description = row["description"] or ""
                            email_match = email_re.search(description)
                            if email_match:
                                contact_email = email_match.group(1)

                        if contact_email and self._is_automated_sender(contact_email):
                            conn.execute(
                                "UPDATE predictions SET resolution_reason = ? WHERE id = ?",
                                ("automated_sender_fast_path", row["id"]),
                            )
                            tagged += 1
                    except Exception:
                        continue  # Skip malformed rows; non-fatal
        except sqlite3.DatabaseError as e:
            logger.warning("_backfill_automated_sender_tags: user_model.db unavailable (update phase): %s", e)
            return

        if tagged:
            logger.info(
                f"BehavioralAccuracyTracker: _backfill_automated_sender_tags tagged "
                f"{tagged} automated-sender predictions as 'automated_sender_fast_path' "
                f"(description-fallback included); opportunity accuracy multiplier can now recover"
            )

    async def run_inference_cycle(self) -> dict[str, int]:
        """Run one inference cycle over unresolved predictions.

        Processes both surfaced and filtered predictions to enable full learning loop:
        - Surfaced predictions: Check if user took predicted action (true positive)
        - Filtered predictions: Check if user STILL took action despite filter (false negative)

        This closes a critical gap: filtered predictions with was_accurate=NULL never
        contributed to the learning loop, preventing the system from discovering that
        its filters are rejecting valuable predictions.

        Returns:
            Dict with counts: {
                'marked_accurate': N,
                'marked_inaccurate': M,
                'surfaced': surfaced_count,
                'filtered': filtered_count
            }
        """
        stats = {
            'marked_accurate': 0,
            'marked_inaccurate': 0,
            'surfaced': 0,
            'filtered': 0,
        }

        # Process surfaced predictions that haven't been resolved yet
        try:
            with self.db.get_connection("user_model") as conn:
                surfaced_predictions = conn.execute(
                    """SELECT id, prediction_type, description, suggested_action,
                              supporting_signals, created_at, was_surfaced
                       FROM predictions
                       WHERE was_surfaced = 1
                         AND resolved_at IS NULL
                         AND created_at > ?""",
                    # Only look at predictions from the last 7 days (older ones are
                    # handled by auto-resolve stale predictions logic)
                    ((datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),),
                ).fetchall()
        except sqlite3.DatabaseError as e:
            logger.warning("run_inference_cycle: failed to query surfaced predictions: %s", e)
            return stats

        for pred in surfaced_predictions:
            # Try to infer accuracy from behavioral signals
            result = await self._infer_accuracy(dict(pred))

            if result is not None:
                # We have enough confidence to make an inference
                was_accurate = result
                now = datetime.now(timezone.utc).isoformat()

                # Determine WHY this prediction was resolved.  This distinguishes
                # historical-pollution resolutions (automated-sender fast-path) from
                # real user-behavior signals.  The accuracy multiplier in the prediction
                # engine excludes fast-path resolutions so that bug-era predictions
                # don't permanently depress confidence for an entire prediction type.
                resolution_reason = self._get_resolution_reason(dict(pred), was_accurate)

                try:
                    with self.db.get_connection("user_model") as conn:
                        conn.execute(
                            """UPDATE predictions SET
                               was_accurate = ?,
                               resolved_at = ?,
                               user_response = 'inferred',
                               resolution_reason = ?
                               WHERE id = ?""",
                            (1 if was_accurate else 0, now, resolution_reason, pred["id"]),
                        )
                except sqlite3.DatabaseError as e:
                    logger.warning("run_inference_cycle: failed to update surfaced prediction %s: %s", pred["id"], e)
                    continue

                if was_accurate:
                    stats['marked_accurate'] += 1
                else:
                    stats['marked_inaccurate'] += 1
                stats['surfaced'] += 1

        # Process filtered predictions to detect false negatives (filter mistakes)
        # These predictions were auto-filtered but might have been valuable!
        # If the user took the action anyway, the filter was WRONG (false negative).
        # If the user didn't take the action, the filter was RIGHT (true negative).
        try:
            with self.db.get_connection("user_model") as conn:
                filtered_predictions = conn.execute(
                    """SELECT id, prediction_type, description, suggested_action,
                              supporting_signals, created_at, was_surfaced
                       FROM predictions
                       WHERE was_surfaced = 0
                         AND user_response = 'filtered'
                         AND was_accurate IS NULL
                         AND created_at > ?
                         AND created_at < ?""",
                    # Look at filtered predictions from 48 hours to 7 days ago.
                    # - Must be 48+ hours old so we have time to observe behavior
                    # - Must be <7 days old to stay relevant
                    (
                        (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
                        (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat(),
                    ),
                ).fetchall()
        except sqlite3.DatabaseError as e:
            logger.warning("run_inference_cycle: failed to query filtered predictions: %s", e)
            return stats

        for pred in filtered_predictions:
            # Try to infer accuracy from behavioral signals
            result = await self._infer_accuracy(dict(pred))

            if result is not None:
                # We can now determine if the filter was correct!
                # - result=True: User DID take action → filter was WRONG (false negative)
                # - result=False: User didn't take action → filter was RIGHT (true negative)
                was_accurate = result
                now = datetime.now(timezone.utc).isoformat()

                # Tag resolution_reason the same way as surfaced predictions.
                # Without this, filtered automated-sender predictions resolved as
                # inaccurate lack the 'automated_sender_fast_path' tag and count
                # against accuracy scores in _get_accuracy_multiplier(), artificially
                # depressing confidence for the entire prediction type.
                resolution_reason = self._get_resolution_reason(dict(pred), was_accurate)

                try:
                    with self.db.get_connection("user_model") as conn:
                        conn.execute(
                            """UPDATE predictions SET
                               was_accurate = ?,
                               resolved_at = ?,
                               resolution_reason = ?
                               WHERE id = ?""",
                            # Keep user_response='filtered' to preserve provenance
                            (1 if was_accurate else 0, now, resolution_reason, pred["id"]),
                        )
                except sqlite3.DatabaseError as e:
                    logger.warning("run_inference_cycle: failed to update filtered prediction %s: %s", pred["id"], e)
                    continue

                if was_accurate:
                    stats['marked_accurate'] += 1
                else:
                    stats['marked_inaccurate'] += 1
                stats['filtered'] += 1

        # Calculate total predictions queried this cycle for diagnostics.
        # surfaced_predictions and filtered_predictions are counted separately.
        # We record a combined count so get_pipeline_health() can surface cold-start.
        predictions_queried = len(surfaced_predictions)
        self._last_cycle_predictions_found = predictions_queried

        if predictions_queried == 0:
            self._cycles_with_no_predictions += 1
            # Log at INFO every 10th empty cycle to avoid log spam while still
            # making cold-start / persistence problems observable.
            if self._total_cycles % 10 == 9:  # Pre-increment: total_cycles hasn't been bumped yet
                logger.info(
                    "BehavioralAccuracyTracker: cycle %d found 0 surfaced predictions "
                    "— accuracy learning loop idle (cold-start or prediction persistence issue)",
                    self._total_cycles + 1,
                )

        # Record timestamp of this cycle's completion.
        self._last_cycle_timestamp = datetime.now(timezone.utc).isoformat()

        # Cache cycle stats for diagnostics observability (include predictions_queried count).
        stats['predictions_queried'] = predictions_queried
        self._last_cycle_stats = dict(stats)
        self._total_cycles += 1

        return stats

    def get_diagnostics(self) -> dict:
        """Comprehensive prediction resolution diagnostics.

        Returns a detailed snapshot of the behavioral accuracy tracker's health
        including per-type resolution stats, resolution method breakdown,
        unresolved prediction details, inference cycle stats, and a health
        assessment. Follows the same diagnostic pattern used by
        PredictionEngine.get_diagnostics() and NotificationManager.get_diagnostics().

        Returns:
            Dictionary with structure:
            {
                "per_type_stats": {<prediction_type>: {total, resolved, accurate, inaccurate, unresolved_surfaced}},
                "resolution_methods": {<user_response>: count},
                "unresolved_details": [{prediction_type, description, created_at, age_hours, signal_keys, reason}],
                "inference_cycles": {"total_cycles": int, "last_cycle_stats": dict | None},
                "health": "healthy" | "degraded" | "stalled",
                "recommendations": [str]
            }
        """
        diagnostics: dict = {
            "per_type_stats": {},
            "resolution_methods": {},
            "unresolved_details": [],
            "inference_cycles": {
                "total_cycles": self._total_cycles,
                "last_cycle_stats": self._last_cycle_stats,
            },
            "health": "healthy",
            "recommendations": [],
        }

        # --- Per-type resolution stats (last 7 days) ---
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT prediction_type,
                              COUNT(*) as total,
                              SUM(CASE WHEN resolved_at IS NOT NULL THEN 1 ELSE 0 END) as resolved,
                              SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate,
                              SUM(CASE WHEN was_accurate = 0 THEN 1 ELSE 0 END) as inaccurate,
                              SUM(CASE WHEN resolved_at IS NULL AND was_surfaced = 1 THEN 1 ELSE 0 END) as unresolved_surfaced
                       FROM predictions
                       WHERE created_at > datetime('now', '-7 days')
                       GROUP BY prediction_type"""
                ).fetchall()
            diagnostics["per_type_stats"] = {
                row["prediction_type"]: {
                    "total": row["total"],
                    "resolved": row["resolved"],
                    "accurate": row["accurate"],
                    "inaccurate": row["inaccurate"],
                    "unresolved_surfaced": row["unresolved_surfaced"],
                }
                for row in rows
            }
        except Exception:
            logger.warning("Diagnostics: failed to query per-type resolution stats", exc_info=True)

        # --- Resolution method breakdown (last 7 days) ---
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT user_response, COUNT(*) as count
                       FROM predictions
                       WHERE resolved_at IS NOT NULL AND created_at > datetime('now', '-7 days')
                       GROUP BY user_response"""
                ).fetchall()
            diagnostics["resolution_methods"] = {
                (row["user_response"] or "unknown"): row["count"] for row in rows
            }
        except Exception:
            logger.warning("Diagnostics: failed to query resolution methods", exc_info=True)

        # --- Unresolved prediction details (limit 10) ---
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT prediction_type, description, created_at, supporting_signals
                       FROM predictions
                       WHERE resolved_at IS NULL AND was_surfaced = 1
                         AND created_at > datetime('now', '-7 days')
                       ORDER BY created_at DESC
                       LIMIT 10"""
                ).fetchall()

            now = datetime.now(timezone.utc)
            unresolved = []
            for row in rows:
                # Parse supporting_signals to extract keys
                signal_keys: list[str] = []
                try:
                    import json as _json
                    signals = _json.loads(row["supporting_signals"] or "[]")
                    if isinstance(signals, dict):
                        signal_keys = list(signals.keys())
                    elif isinstance(signals, list):
                        for s in signals:
                            if isinstance(s, dict):
                                signal_keys.extend(s.keys())
                except (json.JSONDecodeError, TypeError):
                    pass

                # Calculate age
                try:
                    created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                    age_hours = round((now - created).total_seconds() / 3600, 1)
                except (ValueError, TypeError):
                    age_hours = None

                # Determine why it hasn't been resolved
                reason = "unknown"
                if age_hours is not None:
                    if age_hours < 6:
                        reason = "within_observation_window"
                    elif not signal_keys:
                        reason = "missing_supporting_signals"
                    else:
                        reason = "no_matching_behavior_detected"

                unresolved.append({
                    "prediction_type": row["prediction_type"],
                    "description": (row["description"] or "")[:100],
                    "created_at": row["created_at"],
                    "age_hours": age_hours,
                    "signal_keys": signal_keys,
                    "reason": reason,
                })
            diagnostics["unresolved_details"] = unresolved
        except Exception:
            logger.warning("Diagnostics: failed to query unresolved prediction details", exc_info=True)

        # --- Health assessment ---
        total_all = 0
        resolved_all = 0
        for type_stats in diagnostics["per_type_stats"].values():
            total_all += type_stats.get("total", 0)
            resolved_all += type_stats.get("resolved", 0)

        resolution_rate = resolved_all / total_all if total_all > 0 else 0.0
        recommendations = []

        if total_all == 0:
            # No predictions at all — healthy by default (nothing to resolve)
            diagnostics["health"] = "healthy"
        elif resolution_rate > 0.5:
            diagnostics["health"] = "healthy"
        elif resolution_rate >= 0.1:
            diagnostics["health"] = "degraded"
            recommendations.append(
                f"Resolution rate is {resolution_rate:.0%} ({resolved_all}/{total_all} predictions resolved in 7d). "
                "Check if supporting_signals contain expected_actions for unresolved prediction types."
            )
        else:
            diagnostics["health"] = "stalled"
            recommendations.append(
                f"Resolution rate is {resolution_rate:.0%} ({resolved_all}/{total_all} predictions resolved in 7d). "
                "Behavioral inference is not resolving predictions — check strategy matching and event ingestion."
            )

        # Check for missing inference cycles
        if self._total_cycles == 0:
            recommendations.append(
                "No inference cycles have run yet. Ensure the background loop is calling run_inference_cycle()."
            )

        diagnostics["recommendations"] = recommendations

        return diagnostics

    def get_pipeline_health(self) -> dict:
        """Return a concise snapshot of the tracker's pipeline health.

        Designed for quick operational checks — distinguishes normal cold-start
        (no predictions yet) from a broken state where predictions exist but
        are not being tracked.

        Returns:
            Dictionary with structure::

                {
                    'total_cycles': int,              # Inference cycles run so far
                    'cycles_with_no_predictions': int,# Cycles that found 0 surfaced predictions
                    'last_cycle_stats': dict | None,  # Stats from most recent cycle
                    'last_cycle_timestamp': str | None, # ISO-8601 UTC timestamp of last cycle
                    'cold_start_detected': bool,      # True when ALL cycles found 0 predictions
                    'predictions_table_count': int,   # Total rows in predictions table
                }

        ``cold_start_detected=True`` means the tracker has run at least once but
        has never seen a prediction — indicating either a brand-new system
        (expected) or that predictions are not being persisted (needs fixing).
        """
        # Query the raw count of all rows in predictions table so the caller
        # can tell whether the table is empty or populated.
        predictions_table_count = 0
        try:
            with self.db.get_connection("user_model") as conn:
                row = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
                if row:
                    predictions_table_count = row[0]
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            pass  # Unavailable DB is non-fatal; caller sees count=0

        return {
            "total_cycles": self._total_cycles,
            "cycles_with_no_predictions": self._cycles_with_no_predictions,
            "last_cycle_stats": self._last_cycle_stats,
            "last_cycle_timestamp": self._last_cycle_timestamp,
            "cold_start_detected": (
                self._cycles_with_no_predictions == self._total_cycles
                and self._total_cycles > 0
            ),
            "predictions_table_count": predictions_table_count,
        }

    def _get_resolution_reason(self, prediction: dict, was_accurate: bool) -> Optional[str]:
        """Determine the machine-readable reason a prediction was resolved.

        Returns a well-known reason string for resolutions that are NOT driven by
        real user behavior.  The accuracy multiplier in the prediction engine
        excludes these from accuracy calculations so historical-bug pollution
        doesn't permanently suppress prediction types.

        Args:
            prediction: The prediction dict (must contain supporting_signals,
                        description, and prediction_type fields).
            was_accurate: Whether the prediction was inferred accurate or not.

        Returns:
            'automated_sender_fast_path'  — inaccurate because contact is a
                marketing/automated sender (structurally unfulfillable prediction
                generated before the marketing filter improvements in PRs #183–#189).
            None  — real user-behavior signal (user acted or didn't act within window).

        Usage example:
            reason = self._get_resolution_reason(prediction, was_accurate=False)
            # reason = 'automated_sender_fast_path' for noreply@company.com contacts
            # reason = None for real human contacts the user chose not to message

        Note:
            Falls back to regex-parsing the description when supporting_signals
            contains no contact_email.  This handles predictions created before
            PR #190 added supporting_signals (those predictions carry the contact
            address only in the human-readable description string).
        """
        import re

        # Only mark as fast-path if the prediction was inaccurate AND involves a contact
        if was_accurate:
            return None  # Accurate predictions are always real behavioral signals

        # Strategy 1: parse supporting_signals to check for contact email
        try:
            signals = json.loads(prediction.get("supporting_signals") or "{}") or {}
            if isinstance(signals, list):
                signals = {}
        except (json.JSONDecodeError, TypeError):
            signals = {}

        contact_email = signals.get("contact_email")

        # Strategy 2: fall back to regex-parsing the description.
        # Handles predictions created before PR #190 added supporting_signals —
        # e.g. "It's been 45 days since you last contacted noreply@company.com".
        if not contact_email:
            description = prediction.get("description") or ""
            email_match = re.search(
                r'([\w\.\-\+]+@[\w\.\-]+\.[\w\.]+)',
                description,
                re.IGNORECASE,
            )
            if email_match:
                contact_email = email_match.group(1)

        # For opportunity and reminder predictions: check if the contact is an
        # automated sender.  If so, this was an automated-sender fast-path resolution.
        if contact_email and self._is_automated_sender(contact_email):
            return "automated_sender_fast_path"

        return None  # Real user-behavior signal (timeout or confirmed no action)

    async def _infer_accuracy(self, prediction: dict) -> Optional[bool]:
        """Infer whether a prediction was accurate based on user behavior.

        Args:
            prediction: Dict with fields: id, prediction_type, description,
                       suggested_action, supporting_signals, created_at

        Returns:
            True if behavior confirms the prediction was accurate
            False if behavior confirms the prediction was inaccurate
            None if insufficient evidence to make a determination
        """
        pred_type = prediction["prediction_type"]
        created_at = datetime.fromisoformat(prediction["created_at"].replace('Z', '+00:00'))

        # Parse supporting_signals JSON to extract relevant context
        # Handle both old list format and new dict format for backward compatibility
        try:
            signals = json.loads(prediction["supporting_signals"]) if prediction["supporting_signals"] else {}
            # If it's a list (old format), convert to empty dict
            if isinstance(signals, list):
                signals = {}
        except (json.JSONDecodeError, TypeError):
            signals = {}

        # Dispatch to type-specific inference logic
        if pred_type == "reminder":
            return await self._infer_reminder_accuracy(prediction, signals, created_at)
        elif pred_type == "conflict":
            return await self._infer_conflict_accuracy(prediction, signals, created_at)
        elif pred_type == "need":
            return await self._infer_need_accuracy(prediction, signals, created_at)
        elif pred_type == "opportunity":
            return await self._infer_opportunity_accuracy(prediction, signals, created_at)
        elif pred_type == "risk":
            return await self._infer_risk_accuracy(prediction, signals, created_at)
        elif pred_type == "routine_deviation":
            return await self._infer_routine_deviation_accuracy(prediction, signals, created_at)
        else:
            return None  # Unknown prediction type

    async def _infer_reminder_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'reminder' predictions.

        Reminder predictions typically suggest: "Reply to X" or "Follow up with Y".
        We look for outbound messages to the mentioned contact within a reasonable
        timeframe (6-48 hours).

        Accuracy inference logic:
        - If the contact is an automated/marketing sender: immediately INACCURATE
          (the prediction was generated before the marketing filter was robust;
          the user will never "reply" to a no-reply mailer by definition)
        - If the user sends a message to the contact within 48 hours: ACCURATE
        - If 48+ hours pass with no reply: INACCURATE
        - If no contact info can be found AND 48+ hours have passed: INACCURATE
          (the prediction can never be confirmed so it's safe to resolve)
        - If still within the window: None (wait)

        The automated-sender fast-path resolves stale predictions from before
        the marketing filter improvements within minutes instead of waiting
        48 hours, keeping accuracy stats clean and the learning loop tight.
        """
        # Extract contact email/name from signals (new dict format)
        contact_email = signals.get("contact_email")
        contact_name = signals.get("contact_name")

        # Fallback: try old keys for backward compatibility
        if not contact_email:
            contact_email = signals.get("contact_id")

        # If no contact info in signals, try to extract from description
        if not contact_email and not contact_name:
            # Extract contact info from common description patterns.
            # Handles both old descriptions with email addresses and future
            # descriptions that may use names.
            import re

            # Pattern 1: "Unreplied message from EMAIL" (most common)
            # Example: "Unreplied message from alice@example.com: \"Subject\" (3 hours ago)"
            # Handles complex emails: john.doe+work@company-name.co.uk
            email_match = re.search(r'from\s+([\w\.\-\+]+@[\w\.\-]+\.[\w\.]+)', prediction["description"], re.IGNORECASE)
            if email_match:
                contact_email = email_match.group(1)

            # Pattern 2: "Reply to NAME" or "Follow up with NAME" (for future compatibility)
            # Example: "Follow up with Alice about the project"
            # Two-stage match: trigger phrase is case-insensitive, but name must be
            # properly capitalized to avoid false matches (e.g., "Grace" not "about")
            if not contact_email:
                trigger_match = re.search(r'(reply to|follow up with|message)\s+', prediction["description"], re.IGNORECASE)
                if trigger_match:
                    # Extract properly capitalized name after the trigger
                    rest = prediction["description"][trigger_match.end():]
                    name_match = re.match(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', rest)
                    if name_match:
                        contact_name = name_match.group(1)

        now = datetime.now(timezone.utc)

        if not contact_email and not contact_name:
            # No contact information found in signals or description.
            # A reminder without a contact can never be confirmed (we have no
            # way to look for outbound messages). Once the 48-hour window has
            # elapsed, resolve as inaccurate so the prediction doesn't remain
            # permanently pending and pollute the unresolved count.
            if now - created_at > timedelta(hours=48):
                return False  # Unresolvable after timeout → mark inaccurate
            return None  # Still within window, wait before giving up

        # Fast-path: if the contact is an automated/marketing sender, the
        # prediction was wrong by definition. These were generated before the
        # marketing filter was robust enough to catch all automated addresses.
        # The user will never "reply" to a no-reply mailer, so we mark them
        # inaccurate immediately rather than waiting the full 48-hour window.
        # This prevents accuracy stats from being polluted by structurally
        # unfulfillable predictions and keeps the learning loop accurate.
        if contact_email and self._is_automated_sender(contact_email):
            return False  # Automated sender → prediction was inaccurate

        # Look for outbound messages to this contact within 6-48 hours of prediction
        window_start = created_at
        window_end = created_at + timedelta(hours=48)

        with self.db.get_connection("events") as conn:
            # Check for email.sent or message.sent events to this contact
            events = conn.execute(
                """SELECT type, payload, timestamp
                   FROM events
                   WHERE (type = 'email.sent' OR type = 'message.sent')
                     AND timestamp >= ?
                     AND timestamp <= ?
                   ORDER BY timestamp ASC""",
                (window_start.isoformat(), window_end.isoformat()),
            ).fetchall()

        for event in events:
            try:
                payload = json.loads(event["payload"])

                # Build a unified recipient string covering To, CC, and BCC fields.
                # Previously only to_addresses was checked, which caused false
                # INACCURATE outcomes when the user replied-all to a group thread
                # and the predicted contact was in CC rather than To.  The same
                # fix was applied to _infer_opportunity_accuracy() in PR #201.
                # Both ProtonMail and Gmail connectors include cc_addresses /
                # bcc_addresses in their email payloads (see
                # connectors/proton_mail/connector.py and
                # connectors/google/connector.py), so historical events already
                # contain this data — no connector changes required.
                all_recipients: list[str] = []
                for field in ("to_addresses", "cc_addresses", "bcc_addresses"):
                    value = payload.get(field) or []
                    if isinstance(value, list):
                        all_recipients.extend(value)
                    elif isinstance(value, str) and value:
                        all_recipients.append(value)

                # Legacy fallback: some connectors emit a plain "to" string
                # instead of a list (e.g. older iMessage events).
                if not all_recipients:
                    legacy_to = payload.get("to", "")
                    if legacy_to:
                        all_recipients.append(legacy_to)

                # Single lowercase string for substring matching
                recipients_str = ", ".join(all_recipients).lower()

                if contact_email and contact_email.lower() in recipients_str:
                    return True  # User DID follow up — prediction was accurate
                if contact_name and contact_name.lower() in recipients_str:
                    return True
            except (json.JSONDecodeError, TypeError):
                continue

        # Check if enough time has passed to infer inaccuracy.
        # If 48+ hours have passed with no action, prediction was likely wrong —
        # the user chose not to reply (or already replied outside our tracking window).
        # `now` was computed above before the event scan to avoid repeated syscalls.
        if now - created_at > timedelta(hours=48):
            return False  # No action taken → prediction was inaccurate

        return None  # Still within the window, can't determine yet

    async def _infer_conflict_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'conflict' predictions.

        Conflict predictions alert about calendar overlaps. We check if the user
        took corrective action (rescheduled, cancelled, or shortened one of the
        conflicting events).
        """
        # Extract event IDs from signals
        event_ids = signals.get("conflicting_event_ids", [])
        if not event_ids:
            return None

        # Look for calendar.event.updated or calendar.event.deleted events
        # for either of the conflicting events within 24 hours
        window_end = created_at + timedelta(hours=24)

        with self.db.get_connection("events") as conn:
            updates = conn.execute(
                """SELECT type, payload
                   FROM events
                   WHERE (type = 'calendar.event.updated' OR type = 'calendar.event.deleted')
                     AND timestamp >= ?
                     AND timestamp <= ?""",
                (created_at.isoformat(), window_end.isoformat()),
            ).fetchall()

        for update in updates:
            try:
                payload = json.loads(update["payload"])
                event_id = payload.get("event_id")
                if event_id in event_ids:
                    return True  # User resolved the conflict — prediction was accurate
            except (json.JSONDecodeError, TypeError):
                continue

        # If 24+ hours passed and conflict still exists, prediction was correct
        # but user chose to ignore it (still counts as accurate prediction)
        now = datetime.now(timezone.utc)
        if now - created_at > timedelta(hours=24):
            return True  # Conflict was real, even if user didn't fix it

        return None  # Still within resolution window

    async def _infer_need_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'need' predictions.

        Need predictions suggest: "You'll probably need X soon". The most common
        'need' predictions are preparation needs for upcoming events (travel,
        large meetings). We check if the event actually occurred and wasn't
        cancelled/rescheduled away.

        Accuracy inference logic:
        - If the calendar event occurred (not cancelled/rescheduled): ACCURATE
        - If the event was cancelled/rescheduled before it happened: INACCURATE
        - If not enough time has passed to know: None (wait)

        This works for preparation_needs predictions generated by
        PredictionEngine._check_preparation_needs().
        """
        # Extract event information from signals
        event_id = signals.get("event_id")
        event_title = signals.get("event_title")
        event_start_time_str = signals.get("event_start_time")

        # If no event info in signals, try to extract from description
        # Handles: "Upcoming travel in 24h: 'Flight to Boston'. Time to prepare."
        # Handles: "Large meeting in 36h: 'Q4 Planning' with 5 attendees"
        if not event_title:
            import re
            # Pattern: "...: 'EVENT_TITLE'"
            title_match = re.search(r":\s*'([^']+)'", prediction["description"])
            if title_match:
                event_title = title_match.group(1)

        if not event_title and not event_id:
            # Can't track without event information
            return None

        # Parse event start time to know when to check if it happened
        if event_start_time_str:
            try:
                event_start_time = datetime.fromisoformat(
                    event_start_time_str.replace("Z", "+00:00")
                )
                if event_start_time.tzinfo is None:
                    event_start_time = event_start_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                event_start_time = None
        else:
            event_start_time = None

        # If we don't have the event start time, we can't determine if enough
        # time has passed. Be conservative and wait.
        if not event_start_time:
            return None

        now = datetime.now(timezone.utc)

        # If the event hasn't happened yet, we can't determine accuracy
        if now < event_start_time:
            return None

        # Event time has passed. Check if it was cancelled or rescheduled.
        # Look for calendar.event.deleted or calendar.event.updated events
        # that modify/remove this event BEFORE its scheduled start time.
        with self.db.get_connection("events") as conn:
            modifications = conn.execute(
                """SELECT type, payload, timestamp
                   FROM events
                   WHERE (type = 'calendar.event.updated' OR type = 'calendar.event.deleted')
                     AND timestamp >= ?
                     AND timestamp <= ?
                   ORDER BY timestamp ASC""",
                (created_at.isoformat(), event_start_time.isoformat()),
            ).fetchall()

        for mod in modifications:
            try:
                payload = json.loads(mod["payload"])
                mod_event_id = payload.get("event_id")
                mod_event_title = payload.get("title")

                # Check if this modification applies to our event
                if event_id and mod_event_id == event_id:
                    # Event was modified or deleted before it happened
                    if mod["type"] == "calendar.event.deleted":
                        # Event was cancelled → prediction was inaccurate
                        return False
                    elif mod["type"] == "calendar.event.updated":
                        # Check if start time was rescheduled away
                        new_start_time_str = payload.get("start_time")
                        if new_start_time_str:
                            new_start_time = datetime.fromisoformat(
                                new_start_time_str.replace("Z", "+00:00")
                            )
                            if new_start_time.tzinfo is None:
                                new_start_time = new_start_time.replace(tzinfo=timezone.utc)
                            # If rescheduled to a different day, prediction was for the
                            # wrong timing → inaccurate
                            if abs((new_start_time - event_start_time).total_seconds()) > 3600:
                                return False
                elif event_title and mod_event_title and event_title.lower() in mod_event_title.lower():
                    # Fuzzy match by title (for when we don't have event_id)
                    if mod["type"] == "calendar.event.deleted":
                        return False
                    elif mod["type"] == "calendar.event.updated":
                        new_start_time_str = payload.get("start_time")
                        if new_start_time_str:
                            new_start_time = datetime.fromisoformat(
                                new_start_time_str.replace("Z", "+00:00")
                            )
                            if new_start_time.tzinfo is None:
                                new_start_time = new_start_time.replace(tzinfo=timezone.utc)
                            if abs((new_start_time - event_start_time).total_seconds()) > 3600:
                                return False
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        # Event time has passed and no cancellation/major reschedule was detected
        # → Event likely occurred as planned → Prediction was accurate
        return True

    @staticmethod
    def _is_automated_sender(email: str) -> bool:
        """Check if an email address belongs to an automated/marketing sender.

        These addresses were generated by the prediction engine before the marketing
        filter was enhanced. Because automated senders are never human relationship
        contacts, any opportunity prediction about them was incorrect by definition —
        the user will never "reach out" to a no-reply mailer or transactional system.

        Resolving these predictions as inaccurate immediately (rather than waiting
        the full 7-day window) keeps the learning loop accurate and prevents the
        accuracy stats from being polluted by predictions that are structurally
        impossible to fulfill.

        Delegates to the shared canonical implementation in
        services.signal_extractor.marketing_filter.is_marketing_or_noreply(),
        which is the single source of truth for marketing/automated-sender detection
        across the entire system.

        Args:
            email: Email address to check (any case).

        Returns:
            True if the email is clearly automated/marketing, False if it could be human.
        """
        return is_marketing_or_noreply(email)

    async def _infer_opportunity_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'opportunity' predictions.

        Opportunity predictions suggest: "Good time to do X based on your patterns".
        The most common 'opportunity' predictions are relationship maintenance
        suggestions ("Reach out to X — it's been Y days"). We check if the user
        actually contacted the person within a reasonable timeframe.

        Accuracy inference logic:
        - If the contact is an automated/marketing sender: immediately INACCURATE
          (the prediction was generated before the marketing filter was enhanced;
          the user will never reach out to an automated mailer by definition)
        - If user contacts the person within 7 days: ACCURATE
        - If 7+ days pass with no contact: INACCURATE
        - If still within the window: None (wait)

        The automated-sender fast-path resolves stale predictions from before the
        marketing filter improvements (PRs #183, #186) within minutes instead of
        waiting 7 days, keeping accuracy stats clean and the learning loop tight.

        This works for relationship_maintenance predictions generated by
        PredictionEngine._check_relationship_maintenance().
        """
        import re

        # Extract contact information from signals
        contact_email = signals.get("contact_email")
        contact_name = signals.get("contact_name")
        days_since_contact = signals.get("days_since_last_contact")

        # If no contact info in signals, try to extract from description
        # Handles: "Reach out to alice@example.com — it's been 45 days"
        # Handles: "Consider reaching out to Bob — last contact was 60 days ago"
        if not contact_email and not contact_name:
            # Pattern 1: Email address in description
            email_match = re.search(
                r'([\w\.\-\+]+@[\w\.\-]+\.[\w\.]+)',
                prediction["description"],
                re.IGNORECASE
            )
            if email_match:
                contact_email = email_match.group(1)

            # Pattern 2: "Reach out to NAME" or "reaching out to NAME"
            if not contact_email:
                trigger_match = re.search(
                    r'(reach out to|reaching out to)\s+',
                    prediction["description"],
                    re.IGNORECASE
                )
                if trigger_match:
                    rest = prediction["description"][trigger_match.end():]
                    # Extract name (must be capitalized)
                    name_match = re.match(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', rest)
                    if name_match:
                        contact_name = name_match.group(1)

        if not contact_email and not contact_name:
            # Can't track without contact information
            return None

        # Fast-path: if the contact is an automated/marketing sender, the prediction
        # was wrong by definition. These were generated before the marketing filter
        # was enhanced in PRs #183 and #186. We mark them inaccurate immediately
        # rather than waiting 7 days, preventing pollution of the accuracy stats.
        if contact_email and self._is_automated_sender(contact_email):
            return False  # Automated sender — prediction was inaccurate

        # Look for outbound messages to this contact within 7 days of prediction
        window_start = created_at
        window_end = created_at + timedelta(days=7)
        now = datetime.now(timezone.utc)

        with self.db.get_connection("events") as conn:
            # Check for email.sent or message.sent events to this contact
            events = conn.execute(
                """SELECT type, payload, timestamp
                   FROM events
                   WHERE (type = 'email.sent' OR type = 'message.sent')
                     AND timestamp >= ?
                     AND timestamp <= ?
                   ORDER BY timestamp ASC""",
                (window_start.isoformat(), min(window_end, now).isoformat()),
            ).fetchall()

        for event in events:
            try:
                payload = json.loads(event["payload"])

                # Build a unified recipient string covering To, CC, and BCC fields.
                # Previously only to_addresses was checked, which caused false
                # INACCURATE outcomes when the user replied-all to a group thread
                # and the predicted contact was in CC rather than To. Both ProtonMail
                # and Gmail connectors include cc_addresses in all email payloads
                # (connectors/proton_mail/connector.py:253 and
                #  connectors/google/connector.py:284), so historical events already
                # contain this data — no connector changes required.
                all_recipients: list[str] = []
                for field in ("to_addresses", "cc_addresses", "bcc_addresses"):
                    value = payload.get(field) or []
                    if isinstance(value, list):
                        all_recipients.extend(value)
                    elif isinstance(value, str) and value:
                        all_recipients.append(value)

                # Legacy fallback: some connectors emit a plain "to" string
                # instead of a list (e.g. older iMessage events).
                if not all_recipients:
                    legacy_to = payload.get("to", "")
                    if legacy_to:
                        all_recipients.append(legacy_to)

                # Single lowercase string for substring matching
                recipients_str = ", ".join(all_recipients).lower()

                if contact_email and contact_email.lower() in recipients_str:
                    return True  # User DID reach out — prediction was accurate
                if contact_name and contact_name.lower() in recipients_str:
                    return True
            except (json.JSONDecodeError, TypeError):
                continue

        # Check if enough time has passed to infer inaccuracy
        # If 7+ days have passed with no contact, prediction was likely wrong
        # (user didn't feel the need to reach out)
        if now > window_end:
            return False  # No contact made → prediction was inaccurate

        return None  # Still within the 7-day window, can't determine yet

    async def _infer_risk_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'risk' predictions.

        Risk predictions warn: "Something might go wrong if you don't..."
        The most common 'risk' predictions are spending alerts ("$X on category Y
        this month (Z% of total)"). We check if spending on that category continues
        to be anomalously high or if the user corrected the pattern.

        Accuracy inference logic for spending risks:
        - If spending in the flagged category decreased in the following 2 weeks:
          ACCURATE (user acknowledged the risk and adjusted)
        - If spending in the flagged category stayed high or increased:
          ACCURATE (risk was real, whether or not user acted on it)
        - If not enough time has passed: None (wait)

        Note: For spending risks, we consider the prediction ACCURATE if the
        spending pattern was genuinely anomalous, regardless of whether the user
        corrected it. The prediction's job is to identify the risk, not to force
        behavior change.

        This works for spending pattern predictions generated by
        PredictionEngine._check_spending_patterns().
        """
        # Extract spending category from signals
        category = signals.get("category")
        flagged_amount = signals.get("amount")
        flagged_percentage = signals.get("percentage")

        # If no category in signals, try to extract from description
        # Handles: "Spending alert: $450 on 'groceries' this month (35% of total)"
        if not category:
            import re
            # Pattern: "on 'CATEGORY'" or "on \"CATEGORY\""
            category_match = re.search(r"on\s+['\"]([^'\"]+)['\"]", prediction["description"])
            if category_match:
                category = category_match.group(1)

            # Also extract amount if not in signals
            if not flagged_amount:
                amount_match = re.search(r'\$(\d+)', prediction["description"])
                if amount_match:
                    flagged_amount = float(amount_match.group(1))

        if not category:
            # Can't track without category information
            return None

        # Wait at least 14 days after the prediction to see if spending behavior changed
        now = datetime.now(timezone.utc)
        wait_period = created_at + timedelta(days=14)

        if now < wait_period:
            # Not enough time has passed to evaluate behavioral response
            return None

        # Analyze spending in the flagged category during the 14 days AFTER prediction
        # to see if the user corrected their behavior
        window_start = created_at
        window_end = created_at + timedelta(days=14)

        with self.db.get_connection("events") as conn:
            # Get all transactions in the flagged category during the 2-week window
            transactions = conn.execute(
                """SELECT payload FROM events
                   WHERE type = 'finance.transaction.new'
                     AND timestamp >= ?
                     AND timestamp <= ?""",
                (window_start.isoformat(), window_end.isoformat()),
            ).fetchall()

        # Calculate spending in the flagged category during follow-up period
        category_spend = 0.0
        for txn in transactions:
            try:
                payload = json.loads(txn["payload"])
                txn_category = payload.get("category", "uncategorized")
                if txn_category.lower() == category.lower():
                    category_spend += abs(payload.get("amount", 0))
            except (json.JSONDecodeError, TypeError):
                continue

        # The prediction is ACCURATE if:
        # 1. The original flagged amount was genuinely high (>$200 in a month)
        # 2. This indicates the prediction correctly identified a spending anomaly
        #
        # We don't penalize the prediction if the user didn't change behavior—
        # the prediction's job is to surface the risk, not to guarantee action.
        #
        # If the original flagged amount was low (<$200), it was likely a false
        # alarm → INACCURATE
        if flagged_amount and flagged_amount >= 200:
            # High spending was correctly identified → prediction was accurate
            return True
        else:
            # Spending alert was for a small amount → likely false positive
            return False

    async def _infer_routine_deviation_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'routine_deviation' predictions.

        Routine deviation predictions suggest: "You usually do your '<routine>'
        routine by now" — meaning the user has deviated from a detected pattern.
        The prediction engine generates these when expected routine events haven't
        occurred by the time the routine typically starts.

        Accuracy inference logic:
          - Look for events matching the routine's expected_actions within 2 hours
            of the prediction being created.
          - If the user performed the expected routine actions within 2 hours:
            ACCURATE (the nudge was timely — routine was late, user did it)
          - If 4 hours pass with no matching events:
            INACCURATE (the routine truly didn't happen — probably a legitimate
            skip day rather than a delay the user needed to be reminded about)
          - If still within the 4-hour observation window: None (wait)

        The 2-hour ACCURATE window captures "late starts" where the prediction
        nudge coincided with the user eventually doing the routine. The 4-hour
        INACCURATE threshold avoids penalising legitimate off-day skips while
        still resolving predictions that will never be confirmed.

        Supporting signals structure (from PredictionEngine):
            {
                "routine_name": "morning_email_review",
                "consistency_score": 0.85,
                "expected_actions": ["email_received", "task_created"],
            }
        """
        import re as _re

        # Extract routine metadata from signals
        routine_name = signals.get("routine_name")
        expected_actions = signals.get("expected_actions") or []

        # Build the list of event types to look for.  Routine actions use
        # underscore format ("email_received") while stored events use dot
        # format ("email.received").  Map between the two so our query
        # finds actual events in the events table.
        action_to_event = {
            "email_received": "email.received",
            "email_sent": "email.sent",
            "message_received": "message.received",
            "message_sent": "message.sent",
            "task_created": "task.created",
            "task_completed": "task.completed",
            "calendar_event_created": "calendar.event.created",
        }
        expected_event_types = [
            action_to_event.get(a, a.replace("_", "."))
            for a in expected_actions
        ]

        # If we have neither a routine name nor expected actions, fall back to
        # trying to parse them from the prediction description.
        # Handles: "You usually do your 'morning_email_review' routine by now"
        if not routine_name and not expected_event_types:
            name_match = _re.search(r"'([^']+)'", prediction.get("description", ""))
            if name_match:
                routine_name = name_match.group(1)

        if not expected_event_types:
            # Without event types we can only resolve by timeout, not by activity.
            # Use a 24-hour window to avoid permanent stagnation.
            now = datetime.now(timezone.utc)
            if now > created_at + timedelta(hours=24):
                # No event type information; conservatively mark as unresolvable.
                # Return False so the prediction doesn't stay pending indefinitely.
                return False
            return None

        now = datetime.now(timezone.utc)
        accurate_window_end = created_at + timedelta(hours=2)
        inaccurate_window_end = created_at + timedelta(hours=4)

        # Query events table for any of the expected routine event types within
        # the 4-hour observation window.
        with self.db.get_connection("events") as conn:
            placeholders = ",".join("?" * len(expected_event_types))
            events = conn.execute(
                f"""SELECT type, timestamp FROM events
                    WHERE type IN ({placeholders})
                      AND timestamp >= ?
                      AND timestamp <= ?
                    ORDER BY timestamp ASC
                    LIMIT 1""",
                (
                    *expected_event_types,
                    created_at.isoformat(),
                    min(inaccurate_window_end, now).isoformat(),
                ),
            ).fetchall()

        if events:
            # At least one expected routine event occurred within the observation
            # window — the routine was performed (possibly after a delay).
            first_event_time = datetime.fromisoformat(
                events[0]["timestamp"].replace("Z", "+00:00")
            )
            if first_event_time <= accurate_window_end:
                # User completed the routine within 2 hours of the prediction →
                # the prediction correctly identified a late-start deviation.
                return True
            else:
                # User completed the routine between 2-4 hours later — still
                # did it, so the prediction was valid (deviation detected correctly).
                return True

        # No matching events found so far.
        if now > inaccurate_window_end:
            # 4-hour window elapsed with no routine activity — this was likely
            # a legitimate skip day, not a delay the user needed prompting for.
            return False

        # Still within the observation window — too early to determine.
        return None
