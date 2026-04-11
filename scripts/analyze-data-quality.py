"""
Life OS — Data Quality Analysis

Run by the Claude Code improvement agent to understand current data quality
and identify areas for improvement. Outputs a JSON report covering:
  - Event volume and distribution
  - Prediction accuracy and resolution
  - Signal profile health
  - Insight feedback
  - Notification noise
  - Task pipeline health
  - Connector sync status
  - User model depth (episodes, facts, routines)
  - Error event tracking
  - Source weight drift

Usage: python scripts/analyze-data-quality.py [--data-dir ./data]
"""

import argparse
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Module-level list that accumulates query errors during analyze().
# Consumers can inspect report["query_errors"] to distinguish "no data" from "query failed".
_errors: list[dict] = []

# Maps each signal profile name to the event types its extractor actually processes.
# Mirrors the same constant in services/signal_extractor/pipeline.py.
# Used to distinguish "missing because no qualifying events" from "missing despite having data".
PROFILE_EVENT_TYPES: dict[str, list[str]] = {
    "linguistic": ["email.sent", "message.sent", "system.user.command"],
    "linguistic_inbound": ["email.received", "message.received"],
    "cadence": ["email.sent", "message.sent", "email.received", "message.received"],
    "mood_signals": [
        "email.received", "email.sent", "message.received", "message.sent",
        "health.metric.updated", "health.sleep.recorded",
        "calendar.event.created", "finance.transaction.new",
        "location.changed", "system.user.command",
    ],
    "relationships": ["email.received", "email.sent", "message.received", "message.sent"],
    "temporal": [
        "email.sent", "message.sent",
        "email.received", "message.received",
        "calendar.event.created", "calendar.event.updated",
        "task.created", "task.completed", "task.updated",
        "system.user.command",
    ],
    "topics": ["email.received", "email.sent", "message.received", "message.sent", "system.user.command"],
    "spatial": [
        "calendar.event.created", "calendar.event.updated",
        "ios.context.update", "system.user.location_update", "email.received",
    ],
    "decision": [
        "task.completed", "task.created", "email.sent", "message.sent",
        "email.received", "message.received",
        "calendar.event.created", "calendar.event.updated", "finance.transaction.new",
    ],
}


def _query(conn, sql, default=None):
    """Execute a query and return results, or default on error."""
    try:
        return conn.execute(sql).fetchall()
    except Exception as e:
        logger.warning("Query failed: %s — SQL: %s", e, sql[:200])
        _errors.append({"sql": sql[:200], "error": str(e)})
        return default


def _query_one(conn, sql, default=None):
    """Execute a query and return first row, or default on error."""
    try:
        return conn.execute(sql).fetchone()
    except Exception as e:
        logger.warning("Query failed: %s — SQL: %s", e, sql[:200])
        _errors.append({"sql": sql[:200], "error": str(e)})
        return default


def _query_params(conn, sql, params=(), default=None):
    """Execute a parameterized query and return results, or default on error.

    Unlike _query(), this accepts a params tuple for safe SQL parameterization.
    Use this whenever query values come from runtime data rather than literals.
    """
    try:
        return conn.execute(sql, params).fetchall()
    except Exception as e:
        logger.warning("Query failed: %s — SQL: %s", e, sql[:200])
        _errors.append({"sql": sql[:200], "error": str(e)})
        return default


def _query_one_params(conn, sql, params=(), default=None):
    """Execute a parameterized query and return first row, or default on error.

    Unlike _query_one(), this accepts a params tuple for safe SQL parameterization.
    """
    try:
        return conn.execute(sql, params).fetchone()
    except Exception as e:
        logger.warning("Query failed: %s — SQL: %s", e, sql[:200])
        _errors.append({"sql": sql[:200], "error": str(e)})
        return default


def _connect(db_path):
    """Connect to a SQLite database with row factory, or return None."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.warning("Could not connect to %s: %s", db_path, e)
        return None


def analyze(data_dir: str = "./data") -> dict:
    """Analyze data quality across all Life OS databases.

    Returns a report dict with sections for each database area. When a database
    connection or query fails, the affected sections contain an 'error' key
    (instead of being silently absent), and all individual query failures are
    collected in report["query_errors"].
    """
    _errors.clear()
    data_path = Path(data_dir)
    report = {"generated_at": datetime.now(UTC).isoformat(), "sections": {}}

    # -----------------------------------------------------------------------
    # Database health — run PRAGMA checks on each database independently
    # -----------------------------------------------------------------------
    db_names = ["events", "user_model", "state", "preferences", "entities"]
    health = {}
    for db_name in db_names:
        db_conn = _connect(data_path / f"{db_name}.db")
        if db_conn:
            try:
                # Use thorough integrity_check for user_model.db (corruption
                # there is the #1 blocker for semantic facts and predictions).
                # Use fast quick_check(1) for all other databases.
                if db_name == "user_model":
                    result = db_conn.execute("PRAGMA integrity_check").fetchone()
                else:
                    result = db_conn.execute("PRAGMA quick_check(1)").fetchone()

                status_text = result[0] if result else "unknown"
                health[db_name] = {
                    "status": "ok" if status_text == "ok" else "corrupt",
                    "detail": status_text,
                }
            except Exception as e:
                health[db_name] = {"status": "corrupt", "detail": str(e)}
            finally:
                db_conn.close()
        else:
            health[db_name] = {"status": "corrupt", "detail": "could not connect"}
    report["sections"]["database_health"] = health

    # -----------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------
    events_conn = _connect(data_path / "events.db")
    if events_conn:
        try:
            total = _query_one(events_conn, "SELECT COUNT(*) as c FROM events")
            by_type = _query(
                events_conn, "SELECT type, COUNT(*) as c FROM events GROUP BY type ORDER BY c DESC LIMIT 20", []
            )
            last_24h = _query_one(
                events_conn, "SELECT COUNT(*) as c FROM events WHERE timestamp > datetime('now', '-1 day')"
            )
            last_7d = _query_one(
                events_conn, "SELECT COUNT(*) as c FROM events WHERE timestamp > datetime('now', '-7 days')"
            )

            # Error events — high counts signal broken connectors or services
            error_events = _query(
                events_conn,
                """SELECT type, COUNT(*) as c FROM events
                   WHERE type LIKE '%.error%' OR type LIKE '%.failed%'
                   GROUP BY type ORDER BY c DESC LIMIT 10""",
                [],
            )

            # Event sources — which connectors are actually producing data
            event_sources = _query(
                events_conn,
                """SELECT source, COUNT(*) as c,
                          MAX(timestamp) as last_event
                   FROM events GROUP BY source ORDER BY c DESC""",
                [],
            )

            report["sections"]["events"] = {
                "total": total["c"] if total else 0,
                "last_24h": last_24h["c"] if last_24h else 0,
                "last_7d": last_7d["c"] if last_7d else 0,
                "top_types": {r["type"]: r["c"] for r in by_type},
                "error_events": {r["type"]: r["c"] for r in error_events},
                "sources": {r["source"]: {"count": r["c"], "last_event": r["last_event"]} for r in event_sources},
            }
        except Exception as e:
            report["sections"]["events"] = {"error": str(e)}
        finally:
            events_conn.close()
    else:
        report["sections"]["events"] = {"error": "could not connect to events.db"}

    # -----------------------------------------------------------------------
    # Prediction accuracy and resolution
    # -----------------------------------------------------------------------
    um_conn = _connect(data_path / "user_model.db")
    if um_conn:
        # Each section is wrapped independently so a failure in one
        # does not silently suppress all subsequent sections.
        try:
            pred_stats = _query(
                um_conn,
                """SELECT prediction_type,
                    COUNT(*) as total,
                    SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate,
                    SUM(CASE WHEN was_accurate = 0
                              AND (resolution_reason IS NULL
                                   OR resolution_reason != 'automated_sender_fast_path')
                         THEN 1 ELSE 0 END) as inaccurate,
                    SUM(CASE WHEN was_accurate IS NULL THEN 1 ELSE 0 END) as unresolved,
                    SUM(CASE WHEN resolution_reason = 'automated_sender_fast_path'
                         THEN 1 ELSE 0 END) as auto_excluded
                   FROM predictions
                   WHERE was_surfaced = 1
                   GROUP BY prediction_type""",
                [],
            )

            report["sections"]["prediction_accuracy"] = {
                r["prediction_type"]: {
                    "total": r["total"],
                    "accurate": r["accurate"],
                    "inaccurate": r["inaccurate"],
                    "unresolved": r["unresolved"],
                    "auto_excluded": r["auto_excluded"],
                    "accuracy_rate": r["accurate"] / max(r["accurate"] + r["inaccurate"], 1),
                }
                for r in pred_stats
            }

            # Unresolved predictions aging — predictions that were surfaced but
            # never got resolved. High counts mean the feedback loop is broken.
            stale_predictions = _query_one(
                um_conn,
                """SELECT COUNT(*) as c FROM predictions
                   WHERE was_surfaced = 1
                     AND was_accurate IS NULL
                     AND created_at < datetime('now', '-7 days')""",
            )
            total_unresolved = _query_one(
                um_conn,
                """SELECT COUNT(*) as c FROM predictions
                   WHERE was_surfaced = 1 AND was_accurate IS NULL""",
            )
            report["sections"]["prediction_resolution"] = {
                "total_unresolved": total_unresolved["c"] if total_unresolved else 0,
                "stale_over_7d": stale_predictions["c"] if stale_predictions else 0,
            }
        except Exception as e:
            report["sections"]["prediction_accuracy"] = {"error": str(e)}

        # ---------------------------------------------------------------
        # Prediction pipeline diagnostics — full visibility into why
        # predictions are or aren't surfacing, independent of accuracy.
        # ---------------------------------------------------------------
        try:
            pipeline_stats = _query_one(
                um_conn,
                """SELECT
                    COUNT(*) as total_generated,
                    SUM(CASE WHEN was_surfaced = 1 THEN 1 ELSE 0 END) as surfaced,
                    SUM(CASE WHEN was_surfaced = 0 THEN 1 ELSE 0 END) as filtered,
                    SUM(CASE WHEN resolved_at IS NOT NULL THEN 1 ELSE 0 END) as resolved,
                    SUM(CASE WHEN user_response = 'acted_on' THEN 1 ELSE 0 END) as acted_on,
                    SUM(CASE WHEN user_response = 'dismissed' THEN 1 ELSE 0 END) as dismissed,
                    SUM(CASE WHEN user_response = 'filtered' THEN 1 ELSE 0 END) as auto_filtered
                   FROM predictions""",
            )

            filter_reasons = _query(
                um_conn,
                """SELECT
                    CASE
                        WHEN filter_reason LIKE 'confidence:%' THEN 'low_confidence'
                        WHEN filter_reason LIKE 'reaction:%' THEN 'reaction_gate'
                        WHEN filter_reason LIKE 'duplicate:%' THEN 'duplicate'
                        WHEN filter_reason IS NOT NULL THEN filter_reason
                        ELSE 'none'
                    END as reason_category,
                    COUNT(*) as count
                   FROM predictions
                   WHERE was_surfaced = 0
                   GROUP BY reason_category
                   ORDER BY count DESC""",
                [],
            )

            total = (pipeline_stats["total_generated"] or 0) if pipeline_stats else 0
            surfaced = (pipeline_stats["surfaced"] or 0) if pipeline_stats else 0
            filtered = (pipeline_stats["filtered"] or 0) if pipeline_stats else 0
            resolved = (pipeline_stats["resolved"] or 0) if pipeline_stats else 0
            acted_on = (pipeline_stats["acted_on"] or 0) if pipeline_stats else 0
            dismissed = (pipeline_stats["dismissed"] or 0) if pipeline_stats else 0
            auto_filtered = (pipeline_stats["auto_filtered"] or 0) if pipeline_stats else 0

            report["sections"]["prediction_pipeline"] = {
                "total_generated": total,
                "surfaced": surfaced,
                "filtered": filtered,
                "surfacing_rate": round(surfaced / max(total, 1), 3),
                "resolved": resolved,
                "user_acted_on": acted_on,
                "user_dismissed": dismissed,
                "auto_filtered": auto_filtered,
                "filter_reasons": {r["reason_category"]: r["count"] for r in filter_reasons}
                if filter_reasons
                else {},
            }
        except Exception as e:
            report["sections"]["prediction_pipeline"] = {"error": str(e)}

        # ---------------------------------------------------------------
        # Prediction detail diagnostics — confidence distribution, per-type
        # breakdown, and recent filter reasons for root-cause analysis.
        # ---------------------------------------------------------------
        try:
            pp_section = report["sections"].setdefault("prediction_pipeline", {})

            # 1. Confidence histogram: 10 buckets (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
            confidence_buckets = _query(
                um_conn,
                """SELECT CAST(confidence * 10 AS INTEGER) as bucket,
                          COUNT(*) as count
                   FROM predictions
                   GROUP BY bucket
                   ORDER BY bucket""",
                [],
            )
            histogram = {}
            for r in (confidence_buckets or []):
                # Bucket 10 means confidence=1.0 exactly; merge into 0.9-1.0
                b = min(r["bucket"] or 0, 9)
                label = f"{b / 10:.1f}-{(b + 1) / 10:.1f}"
                histogram[label] = histogram.get(label, 0) + r["count"]

            # 2. Per-type breakdown: generated vs surfaced per prediction_type
            type_breakdown = _query(
                um_conn,
                """SELECT prediction_type,
                          COUNT(*) as total,
                          SUM(CASE WHEN was_surfaced = 1 THEN 1 ELSE 0 END) as surfaced
                   FROM predictions
                   GROUP BY prediction_type
                   ORDER BY total DESC""",
                [],
            )

            # 3. Recent filter reasons: last 10 filtered predictions with detail
            recent_filtered = _query(
                um_conn,
                """SELECT prediction_type, confidence, filter_reason, created_at
                   FROM predictions
                   WHERE user_response = 'filtered'
                   ORDER BY created_at DESC
                   LIMIT 10""",
                [],
            )

            # 4. Dedup ratio context: compare generation events to stored predictions
            stored_count = _query_one(
                um_conn, "SELECT COUNT(*) as c FROM predictions"
            )

            pp_section["prediction_detail"] = {
                "confidence_histogram": histogram,
                "type_breakdown": {
                    r["prediction_type"]: {
                        "total": r["total"],
                        "surfaced": r["surfaced"],
                    }
                    for r in (type_breakdown or [])
                },
                "recent_filtered": [
                    {
                        "prediction_type": r["prediction_type"],
                        "confidence": r["confidence"],
                        "filter_reason": r["filter_reason"],
                        "created_at": r["created_at"],
                    }
                    for r in (recent_filtered or [])
                ],
                "stored_prediction_count": stored_count["c"] if stored_count else 0,
            }
        except Exception as e:
            pp_section = report["sections"].setdefault("prediction_pipeline", {})
            pp_section["prediction_detail"] = {"error": str(e)}

        # ---------------------------------------------------------------
        # Event-sourced prediction activity — cross-reference the events
        # table to detect pipeline activity even when the predictions
        # table is empty (e.g. after cleanup runs).
        # ---------------------------------------------------------------
        try:
            pp_section = report["sections"].setdefault("prediction_pipeline", {})
            ev_conn = _connect(data_path / "events.db")
            if ev_conn:
                try:
                    pred_events = _query(
                        ev_conn,
                        """SELECT type, COUNT(*) as count
                           FROM events
                           WHERE type LIKE 'usermodel.prediction.%'
                           GROUP BY type
                           ORDER BY count DESC""",
                        [],
                    )
                    pp_section["event_activity"] = (
                        {r["type"]: r["count"] for r in pred_events} if pred_events else {}
                    )

                    last_pred = _query_one(
                        ev_conn,
                        """SELECT timestamp FROM events
                           WHERE type = 'usermodel.prediction.generated'
                           ORDER BY timestamp DESC LIMIT 1""",
                    )
                    pp_section["last_generation_event"] = last_pred["timestamp"] if last_pred else None
                finally:
                    ev_conn.close()
            else:
                pp_section["event_activity"] = {"error": "could not connect to events.db"}
                pp_section["last_generation_event"] = None
        except Exception as e:
            pp_section = report["sections"].setdefault("prediction_pipeline", {})
            pp_section["event_activity"] = {"error": str(e)}
            pp_section["last_generation_event"] = None

        try:
            # Signal profiles freshness
            profiles = _query(um_conn, "SELECT profile_type, samples_count, updated_at FROM signal_profiles", [])
            existing_types = {r["profile_type"] for r in profiles}
            expected_types = [
                "linguistic",
                "linguistic_inbound",
                "cadence",
                "mood_signals",
                "relationships",
                "temporal",
                "topics",
                "spatial",
                "decision",
            ]
            missing = [t for t in expected_types if t not in existing_types]

            # For each missing profile, cross-reference with events.db to determine
            # whether the absence is actionable ("qualifying events exist but profile
            # is still empty") or informational ("no qualifying events can ever populate it").
            # This prevents false-positive warnings for profiles like 'linguistic' that
            # only get data from outbound events (email.sent, message.sent) when a
            # connector is broken or no outbound activity has occurred.
            missing_profile_detail: dict[str, str] = {}
            if missing:
                ev_conn_profiles = _connect(data_path / "events.db")
                if ev_conn_profiles:
                    try:
                        for profile_type in missing:
                            qualifying_types = PROFILE_EVENT_TYPES.get(profile_type, [])
                            if not qualifying_types:
                                # No known qualifying types — treat as informational
                                missing_profile_detail[profile_type] = "no_qualifying_event_types_defined"
                                continue
                            placeholders = ",".join("?" * len(qualifying_types))
                            row = _query_one_params(
                                ev_conn_profiles,
                                f"SELECT COUNT(*) as c FROM events WHERE type IN ({placeholders})",
                                tuple(qualifying_types),
                            )
                            count = row["c"] if row else 0
                            if count == 0:
                                # Profile can't be populated — no qualifying events exist
                                missing_profile_detail[profile_type] = "no_qualifying_events"
                            else:
                                # Qualifying events exist but the profile was never written
                                missing_profile_detail[profile_type] = f"{count}_qualifying_events_exist"
                    finally:
                        ev_conn_profiles.close()

            report["sections"]["signal_profiles"] = {
                "profiles": {
                    r["profile_type"]: {"samples": r["samples_count"], "last_updated": r["updated_at"]}
                    for r in profiles
                },
                "missing_profiles": missing,
                "missing_profile_detail": missing_profile_detail,
            }
        except Exception as e:
            report["sections"]["signal_profiles"] = {"error": str(e)}

        try:
            # Insight feedback
            insight_stats = _query(
                um_conn,
                """SELECT type, feedback, COUNT(*) as c
                   FROM insights
                   GROUP BY type, feedback""",
                [],
            )
            report["sections"]["insight_feedback"] = [
                {"type": r["type"], "feedback": r["feedback"], "count": r["c"]} for r in insight_stats
            ]
        except Exception as e:
            report["sections"]["insight_feedback"] = {"error": str(e)}

        try:
            # User model depth — episodes, facts, routines
            # Track query failures separately so corruption/errors are
            # distinguishable from genuinely empty tables.
            query_errors = []

            episode_count = _query_one(um_conn, "SELECT COUNT(*) as c FROM episodes")
            if episode_count is None:
                query_errors.append("episodes")

            fact_count = _query_one(um_conn, "SELECT COUNT(*) as c FROM semantic_facts")
            if fact_count is None:
                query_errors.append("semantic_facts")

            routine_count = _query_one(um_conn, "SELECT COUNT(*) as c FROM routines")
            if routine_count is None:
                query_errors.append("routines")

            # Facts by category
            fact_categories = _query(
                um_conn,
                """SELECT category, COUNT(*) as c
                   FROM semantic_facts GROUP BY category ORDER BY c DESC""",
                [],
            )

            # Workflow and communication template counts — tables may not
            # exist in older schemas, so wrap each independently.
            workflow_count = 0
            try:
                wf_row = _query_one(um_conn, "SELECT COUNT(*) as c FROM workflows")
                workflow_count = wf_row["c"] if wf_row else 0
            except Exception:
                query_errors.append("workflows")

            template_count = 0
            try:
                tpl_row = _query_one(um_conn, "SELECT COUNT(*) as c FROM communication_templates")
                template_count = tpl_row["c"] if tpl_row else 0
            except Exception:
                query_errors.append("communication_templates")

            report["sections"]["user_model"] = {
                "episodes": episode_count["c"] if episode_count else 0,
                "semantic_facts": fact_count["c"] if fact_count else 0,
                "routines": routine_count["c"] if routine_count else 0,
                "workflows": workflow_count,
                "communication_templates": template_count,
                "fact_categories": {r["category"]: r["c"] for r in fact_categories},
                "query_errors": query_errors,
            }
        except Exception as e:
            report["sections"]["user_model"] = {"error": str(e)}

        um_conn.close()
    else:
        um_error = {"error": "could not connect to user_model.db"}
        report["sections"]["prediction_accuracy"] = um_error
        report["sections"]["prediction_resolution"] = um_error
        report["sections"]["prediction_pipeline"] = um_error
        report["sections"]["signal_profiles"] = um_error
        report["sections"]["insight_feedback"] = um_error
        report["sections"]["user_model"] = um_error

    # -----------------------------------------------------------------------
    # Workflow detection diagnostics — data supporting workflow discovery
    # -----------------------------------------------------------------------
    ev_conn = _connect(data_path / "events.db")
    um_conn2 = _connect(data_path / "user_model.db")
    try:
        wf_diag: dict = {}

        # Workflow detector thresholds (from WorkflowDetector defaults)
        wf_diag["thresholds"] = {
            "min_occurrences": 3,
            "min_completions": 2,
            "min_steps": 2,
            "max_step_gap_hours": 12,
        }

        # Email workflow data
        if ev_conn:
            email_received_30d = _query_one(
                ev_conn,
                """SELECT COUNT(*) as c FROM events
                   WHERE type = 'email.received'
                     AND julianday(timestamp) > julianday(datetime('now', '-30 days'))""",
            )
            email_sent_30d = _query_one(
                ev_conn,
                """SELECT COUNT(*) as c FROM events
                   WHERE type = 'email.sent'
                     AND julianday(timestamp) > julianday(datetime('now', '-30 days'))""",
            )
            top_senders = _query(
                ev_conn,
                """SELECT json_extract(payload, '$.email_from') as sender, COUNT(*) as c
                   FROM events
                   WHERE type = 'email.received'
                     AND julianday(timestamp) > julianday(datetime('now', '-30 days'))
                     AND json_extract(payload, '$.email_from') IS NOT NULL
                   GROUP BY sender ORDER BY c DESC LIMIT 5""",
                [],
            )
            wf_diag["email"] = {
                "received_30d": email_received_30d["c"] if email_received_30d else 0,
                "sent_30d": email_sent_30d["c"] if email_sent_30d else 0,
                "top_senders": {r["sender"]: r["c"] for r in top_senders} if top_senders else {},
            }

            # Task workflow data
            tasks_created_30d = _query_one(
                ev_conn,
                """SELECT COUNT(*) as c FROM events
                   WHERE type = 'task.created'
                     AND julianday(timestamp) > julianday(datetime('now', '-30 days'))""",
            )
            wf_diag["tasks"] = {
                "created_30d": tasks_created_30d["c"] if tasks_created_30d else 0,
            }

            # Calendar workflow data
            calendar_events_30d = _query_one(
                ev_conn,
                """SELECT COUNT(*) as c FROM events
                   WHERE type = 'calendar.event.created'
                     AND julianday(timestamp) > julianday(datetime('now', '-30 days'))""",
            )
            wf_diag["calendar"] = {
                "events_created_30d": calendar_events_30d["c"] if calendar_events_30d else 0,
            }
        else:
            wf_diag["email"] = {"error": "could not connect to events.db"}
            wf_diag["tasks"] = {"error": "could not connect to events.db"}
            wf_diag["calendar"] = {"error": "could not connect to events.db"}

        # Episode interaction type distribution
        if um_conn2:
            interaction_types = _query(
                um_conn2,
                """SELECT interaction_type, COUNT(*) as c
                   FROM episodes
                   GROUP BY interaction_type
                   ORDER BY c DESC""",
                [],
            )
            total_episodes = sum(r["c"] for r in interaction_types) if interaction_types else 0
            type_dist = {}
            null_unknown_comm = 0
            for r in (interaction_types or []):
                itype = r["interaction_type"]
                type_dist[str(itype)] = r["c"]
                if itype is None or itype in ("unknown", "communication"):
                    null_unknown_comm += r["c"]

            wf_diag["episode_interaction_types"] = {
                "total_episodes": total_episodes,
                "distribution": type_dist,
                "null_unknown_communication_count": null_unknown_comm,
                "null_unknown_communication_pct": round(null_unknown_comm / max(total_episodes, 1), 3),
            }
        else:
            wf_diag["episode_interaction_types"] = {"error": "could not connect to user_model.db"}

        report["sections"]["workflow_diagnostics"] = wf_diag
    except Exception as e:
        report["sections"]["workflow_diagnostics"] = {"error": str(e)}
    finally:
        if ev_conn:
            ev_conn.close()
        if um_conn2:
            um_conn2.close()

    # -----------------------------------------------------------------------
    # Notification dismissal rate
    # -----------------------------------------------------------------------
    state_conn = _connect(data_path / "state.db")
    if state_conn:
        try:
            notif_stats = _query(state_conn, "SELECT status, COUNT(*) as c FROM notifications GROUP BY status", [])
            report["sections"]["notifications"] = {r["status"]: r["c"] for r in notif_stats}

            # Task pipeline health
            task_stats = _query(state_conn, "SELECT status, COUNT(*) as c FROM tasks GROUP BY status", [])
            stale_tasks = _query_one(
                state_conn,
                """SELECT COUNT(*) as c FROM tasks
                   WHERE status = 'pending'
                     AND created_at < datetime('now', '-7 days')""",
            )
            report["sections"]["tasks"] = {
                "by_status": {r["status"]: r["c"] for r in task_stats},
                "stale_pending_over_7d": stale_tasks["c"] if stale_tasks else 0,
            }

            # Connector sync status — last sync times and error states
            connector_states = _query(
                state_conn,
                """SELECT connector_id, status, last_sync, last_error
                   FROM connector_state ORDER BY connector_id""",
                [],
            )
            report["sections"]["connectors"] = (
                {
                    r["connector_id"]: {
                        "status": r["status"],
                        "last_sync": r["last_sync"],
                        "error": r["last_error"],
                    }
                    for r in connector_states
                }
                if connector_states
                else {}
            )
        except Exception as e:
            report["sections"]["notifications"] = {"error": str(e)}
        finally:
            state_conn.close()
    else:
        state_error = {"error": "could not connect to state.db"}
        report["sections"]["notifications"] = state_error
        report["sections"]["tasks"] = state_error
        report["sections"]["connectors"] = state_error

    # -----------------------------------------------------------------------
    # Feedback log and source weights
    # -----------------------------------------------------------------------
    pref_conn = _connect(data_path / "preferences.db")
    if pref_conn:
        try:
            feedback = _query(
                pref_conn,
                """SELECT action_type, feedback_type, COUNT(*) as c
                   FROM feedback_log
                   GROUP BY action_type, feedback_type
                   ORDER BY c DESC""",
                [],
            )
            report["sections"]["feedback"] = [
                {"action_type": r["action_type"], "feedback_type": r["feedback_type"], "count": r["c"]}
                for r in feedback
            ]

            # Count only explicit user-initiated notification dismissals (not auto-resolved
            # / timed-out ones).  Auto-expired predictions are stored with
            # context = '{"auto_resolved": true, ...}' and should not count as real
            # negative feedback for source weight wiring checks.
            #
            # The context column was added after the initial schema — check column
            # existence via PRAGMA before running the granular query so older and
            # test databases degrade gracefully without polluting query_errors.
            fl_cols = {
                row[1]
                for row in (pref_conn.execute("PRAGMA table_info(feedback_log)").fetchall() or [])
            }
            if "context" in fl_cols:
                explicit_dismissal_rows = _query(
                    pref_conn,
                    """SELECT COUNT(*) as c FROM feedback_log
                       WHERE action_type = 'notification'
                         AND feedback_type = 'dismissed'
                         AND (context IS NULL
                              OR json_extract(context, '$.auto_resolved') IS NOT 1)""",
                    [],
                )
                report["sections"]["explicit_user_dismissals"] = (
                    explicit_dismissal_rows[0]["c"] if explicit_dismissal_rows else 0
                )
            else:
                # Schema predates context column — fallback signals caller to use total count
                report["sections"]["explicit_user_dismissals"] = None

            # Source weights — high drift indicates the system is losing
            # confidence in some data sources
            source_weights = _query(
                pref_conn,
                """SELECT source_key, user_weight, ai_drift, ai_updated_at,
                          interactions, engagements, dismissals
                   FROM source_weights ORDER BY ai_drift DESC""",
                [],
            )
            report["sections"]["source_weights"] = (
                {
                    r["source_key"]: {
                        "weight": r["user_weight"],
                        "drift": r["ai_drift"],
                        "updated_at": r["ai_updated_at"],
                        "interactions": r["interactions"],
                        "engagements": r["engagements"],
                        "dismissals": r["dismissals"],
                    }
                    for r in source_weights
                }
                if source_weights
                else {}
            )
        except Exception as e:
            report["sections"]["feedback"] = {"error": str(e)}
        finally:
            pref_conn.close()
    else:
        pref_error = {"error": "could not connect to preferences.db"}
        report["sections"]["feedback"] = pref_error
        report["sections"]["source_weights"] = pref_error

    report["query_errors"] = list(_errors)

    # Anomaly detection and health scoring
    anomalies = detect_anomalies(report["sections"])
    report["anomalies"] = anomalies
    report["health_score"] = compute_health_score(anomalies)

    return report


def detect_anomalies(sections: dict) -> list[dict]:
    """Analyze collected report sections and detect common failure patterns.

    Each anomaly is a dict with:
      - severity: 'critical' | 'warning' | 'info'
      - category: short category string
      - message: human-readable description
      - recommendation: actionable suggestion
    """
    anomalies: list[dict] = []

    # --- (a) Prediction table empty despite generation events ---
    pipeline = sections.get("prediction_pipeline", {})
    total_generated = pipeline.get("total_generated", 0)
    event_activity = pipeline.get("event_activity", {})
    gen_events = event_activity.get("usermodel.prediction.generated", 0) if isinstance(event_activity, dict) else 0

    if total_generated == 0 and gen_events > 0:
        anomalies.append({
            "severity": "critical",
            "category": "prediction_persistence",
            "message": (
                f"Predictions table has 0 rows but {gen_events} generation events exist "
                "— predictions are being generated but not persisted"
            ),
            "recommendation": (
                "Check store_prediction() errors in logs; possible schema mismatch after migration"
            ),
        })

    # --- (a2) All predictions clustered below surfacing threshold ---
    pred_detail = pipeline.get("prediction_detail", {})
    if isinstance(pred_detail, dict) and "error" not in pred_detail:
        histogram = pred_detail.get("confidence_histogram", {})
        if histogram:
            total_preds = sum(histogram.values())
            # Count predictions in buckets below 0.3 (surfacing threshold)
            below_threshold = sum(
                count for label, count in histogram.items()
                if label < "0.3"  # "0.0-0.1", "0.1-0.2", "0.2-0.3" all sort before "0.3"
            )
            if total_preds > 5 and below_threshold == total_preds:
                anomalies.append({
                    "severity": "warning",
                    "category": "prediction_low_confidence",
                    "message": (
                        f"All {total_preds} predictions have confidence below 0.3 "
                        "(the surfacing threshold) — none can be surfaced to the user"
                    ),
                    "recommendation": (
                        "Check prediction confidence calibration; accuracy multipliers "
                        "may be too aggressive, or supporting signal data may be insufficient"
                    ),
                })

        # --- (a3) All predictions are the same type ---
        type_breakdown = pred_detail.get("type_breakdown", {})
        if len(type_breakdown) == 1 and total_generated > 5:
            single_type = next(iter(type_breakdown))
            anomalies.append({
                "severity": "warning",
                "category": "prediction_type_monoculture",
                "message": (
                    f"All {total_generated} predictions are type '{single_type}' "
                    "— prediction engine may be stuck on one signal source"
                ),
                "recommendation": (
                    "Review prediction generation triggers; ensure multiple prediction "
                    "types (NEED, RISK, OPPORTUNITY, REMINDER) are being considered"
                ),
            })

    # --- (b) High dedup ratio ---
    dedup_events = event_activity.get("usermodel.prediction.deduplicated", 0) if isinstance(event_activity, dict) else 0
    gen_events_for_ratio = gen_events if gen_events > 0 else 1
    if gen_events > 0 and dedup_events > 10 * gen_events:
        ratio = round(dedup_events / gen_events_for_ratio, 1)
        anomalies.append({
            "severity": "warning",
            "category": "prediction_deduplication",
            "message": f"Prediction deduplication rate is {ratio}x — most predictions are duplicates",
            "recommendation": (
                "Review prediction generation logic for duplicate triggers; "
                "consider increasing dedup window or reducing generation frequency"
            ),
        })

    # --- (c) Zero routines with sufficient episodes ---
    user_model = sections.get("user_model", {})
    routines = user_model.get("routines", 0)
    episodes = user_model.get("episodes", 0)

    if routines == 0 and episodes > 100:
        anomalies.append({
            "severity": "warning",
            "category": "routine_detection",
            "message": f"No routines detected despite {episodes} episodes",
            "recommendation": "Check routine_detector diagnostics for interaction_type distribution",
        })

    # --- (d) Zero workflows ---
    workflows = user_model.get("workflows", 0)
    if workflows == 0 and episodes > 100:
        anomalies.append({
            "severity": "warning",
            "category": "workflow_detection",
            "message": f"No workflows detected despite {episodes} episodes",
            "recommendation": "Check workflow detection logic; ensure episodes have sufficient variety",
        })

    # --- (d2) Workflow diagnostics anomalies ---
    wf_diag = sections.get("workflow_diagnostics", {})
    if isinstance(wf_diag, dict) and "error" not in wf_diag:
        email_data = wf_diag.get("email", {})
        if isinstance(email_data, dict) and "error" not in email_data:
            received = email_data.get("received_30d", 0)
            sent = email_data.get("sent_30d", 0)
            if received > 100 and sent < 5:
                anomalies.append({
                    "severity": "warning",
                    "category": "workflow_email_imbalance",
                    "message": (
                        f"Low outbound email volume ({sent} sent vs {received} received in 30d) "
                        "limits email workflow detection — workflows require send actions to complete"
                    ),
                    "recommendation": (
                        "Verify email.sent events are being captured by the email connector; "
                        "check connector sync for outbound mail"
                    ),
                })

        ep_types = wf_diag.get("episode_interaction_types", {})
        if isinstance(ep_types, dict) and "error" not in ep_types:
            pct = ep_types.get("null_unknown_communication_pct", 0)
            if pct > 0.5:
                count = ep_types.get("null_unknown_communication_count", 0)
                total = ep_types.get("total_episodes", 0)
                anomalies.append({
                    "severity": "warning",
                    "category": "workflow_stale_interaction_types",
                    "message": (
                        f"{count}/{total} episodes ({pct:.0%}) have NULL/unknown/communication "
                        "interaction_type — stale types block interaction-based workflow detection"
                    ),
                    "recommendation": (
                        "Run episode interaction_type backfill to reclassify episodes "
                        "with specific types (email, task, calendar, etc.)"
                    ),
                })

    # --- (e) Connector errors ---
    connectors = sections.get("connectors", {})
    if isinstance(connectors, dict) and "error" not in connectors:
        for connector_id, info in connectors.items():
            if isinstance(info, dict) and info.get("status") == "error":
                error_msg = info.get("error", "unknown error")
                last_sync = info.get("last_sync")
                anomalies.append({
                    "severity": "critical",
                    "category": "connector_error",
                    "message": (
                        f"Connector '{connector_id}' is in error state: {error_msg}"
                        f" (last_sync: {last_sync or 'never'})"
                    ),
                    "recommendation": f"Check connector '{connector_id}' configuration and credentials",
                })

    # --- (f) Stale data sources ---
    # Only flag external user-facing data connectors as stale.  Internal system
    # event sources (rules engine, user model store, etc.) emit events only while
    # the process is running — flagging them as stale just adds noise when the
    # system is stopped.
    _INTERNAL_EVENT_SOURCES = frozenset({
        "user_model_store",
        "rules_engine",
        "system",
        "notification_manager",
        "routine_detector",
        "connector_health_monitor",
        "db_health_loop",
        "feedback_collector",
        "prediction_engine",
        "signal_extractor",
        "insight_engine",
        "test-service",
    })
    events = sections.get("events", {})
    sources = events.get("sources", {})
    if isinstance(sources, dict):
        now = datetime.now(UTC)
        stale_threshold = now - timedelta(days=7)
        for source_name, source_info in sources.items():
            if source_name in _INTERNAL_EVENT_SOURCES:
                continue  # Internal system sources are not user-facing connectors
            if isinstance(source_info, dict):
                last_event = source_info.get("last_event")
                if last_event:
                    try:
                        # Parse ISO timestamp — handle both with and without timezone
                        last_dt = datetime.fromisoformat(last_event.replace("Z", "+00:00"))
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=UTC)
                        if last_dt < stale_threshold:
                            days_ago = (now - last_dt).days
                            anomalies.append({
                                "severity": "warning",
                                "category": "stale_source",
                                "message": (
                                    f"Source '{source_name}' last produced data {days_ago} days ago"
                                ),
                                "recommendation": (
                                    f"Check if connector for '{source_name}' is still running and authenticated"
                                ),
                            })
                    except (ValueError, TypeError):
                        pass  # Skip unparseable timestamps

    # --- (g) No prediction accuracy data ---
    pred_accuracy = sections.get("prediction_accuracy", {})
    if isinstance(pred_accuracy, dict) and not pred_accuracy:
        anomalies.append({
            "severity": "info",
            "category": "prediction_accuracy",
            "message": "No prediction accuracy data available — predictions may not have been resolved yet",
            "recommendation": "Wait for predictions to be resolved through user interaction or time-based expiry",
        })

    # --- (h) Pending notification backlog ---
    notifications = sections.get("notifications", {})
    if isinstance(notifications, dict) and "error" not in notifications:
        pending = notifications.get("pending", 0)
        if pending > 50:
            anomalies.append({
                "severity": "warning",
                "category": "notification_backlog",
                "message": f"Notification backlog has {pending} pending notifications",
                "recommendation": (
                    "Review notification generation rate; consider auto-expiring old notifications "
                    "or adjusting notification thresholds"
                ),
            })

    # --- (i) Source weight learning activity ---
    sw_section = sections.get("source_weights", {})
    if isinstance(sw_section, dict) and "error" not in sw_section:
        total_interactions = sum(v.get("interactions", 0) for v in sw_section.values())
        total_dismissals = sum(v.get("dismissals", 0) for v in sw_section.values())
        total_engagements = sum(v.get("engagements", 0) for v in sw_section.values())
        event_total = events.get("total", 0) if isinstance(events, dict) else 0

        if total_interactions == 0 and event_total > 100:
            anomalies.append({
                "severity": "warning",
                "category": "source_weight_learning",
                "message": (
                    f"Source weights have 0 interactions despite {event_total} events "
                    "— event classification may not be reaching source_weights"
                ),
                "recommendation": (
                    "Check that SourceWeightManager.record_interaction() is being called "
                    "in master_event_handler and that classify_event() returns keys "
                    "matching source_weights table rows"
                ),
            })
        elif total_interactions > 0 and total_dismissals == 0:
            # Count only explicit user dismissals — auto-resolved (timed-out) notifications
            # are recorded in feedback_log with auto_resolved=true in the context JSON but
            # should NOT trigger a source weight warning because they were never shown to the
            # user.  Only user-initiated dismissals (via the UI dismiss button) represent
            # real negative feedback that should propagate to source weights.
            feedback_list = sections.get("feedback", [])
            if isinstance(feedback_list, list):
                # The feedback section aggregates by (action_type, feedback_type) without
                # the context JSON, so we use the explicit_user_dismissals sub-key if
                # the report was built with the granular query; otherwise fall back to
                # the total which may be inflated by auto-resolved entries.
                explicit_dismissals = sections.get("explicit_user_dismissals", None)
                if explicit_dismissals is not None:
                    feedback_dismissals = explicit_dismissals
                else:
                    # Heuristic: if the total "dismissed" count is high but all known
                    # notification dismissals have auto_resolved context, treat as 0.
                    # This prevents false positives when the system auto-expires stale
                    # prediction notifications that the user never saw.
                    feedback_dismissals = sum(
                        f.get("count", 0) for f in feedback_list
                        if f.get("feedback_type") == "dismissed"
                    )
                if feedback_dismissals > 5:
                    anomalies.append({
                        "severity": "warning",
                        "category": "source_weight_feedback",
                        "message": (
                            f"Source weights recorded {total_interactions} interactions "
                            f"but 0 dismissals despite {feedback_dismissals} explicit user "
                            "notification dismissals — feedback-to-weight wiring may be broken"
                        ),
                        "recommendation": (
                            "Check _classify_notification_source() in web/routes.py "
                            "— dismissed notifications may lack source_event_id or "
                            "map to unknown source_keys"
                        ),
                    })

    # --- (j) Missing signal profiles — severity depends on qualifying event availability ---
    # A profile missing with qualifying events is a pipeline bug (warning).
    # A profile missing with NO qualifying events is expected/informational (info).
    signal_profiles = sections.get("signal_profiles", {})
    if isinstance(signal_profiles, dict) and "error" not in signal_profiles:
        missing_profiles = signal_profiles.get("missing_profiles", [])
        missing_detail = signal_profiles.get("missing_profile_detail", {})

        for profile_type in missing_profiles:
            detail = missing_detail.get(profile_type, "unknown")
            if detail in ("no_qualifying_events", "no_qualifying_event_types_defined"):
                # No qualifying events can populate this profile — informational only
                anomalies.append({
                    "severity": "info",
                    "category": "missing_profile",
                    "message": (
                        f"Signal profile '{profile_type}' is missing — "
                        "no qualifying events exist to populate it"
                    ),
                    "recommendation": (
                        f"No action needed unless you expect {profile_type}-related "
                        "data to be ingested (e.g., check connector status)"
                    ),
                })
            elif "qualifying_events_exist" in detail:
                # Qualifying events exist but the profile was never written — pipeline issue
                qualifying_count = detail.split("_qualifying_events_exist")[0]
                anomalies.append({
                    "severity": "warning",
                    "category": "missing_profile",
                    "message": (
                        f"Signal profile '{profile_type}' is missing despite "
                        f"{qualifying_count} qualifying events existing in events.db"
                    ),
                    "recommendation": (
                        f"Check signal extractor pipeline for '{profile_type}'; "
                        "run profile rebuild from /admin or check for extractor errors in logs"
                    ),
                })
            else:
                # Unknown detail — report as warning to be safe
                anomalies.append({
                    "severity": "warning",
                    "category": "missing_profile",
                    "message": f"Signal profile '{profile_type}' is missing",
                    "recommendation": (
                        "Check signal extractor pipeline and run profile rebuild if needed"
                    ),
                })

    # --- (k) Root-cause annotation: link stale-data anomalies to connector errors ---
    # When a connector is in error state, many downstream anomalies are symptoms of that
    # root cause rather than independent problems.  Annotate those anomalies so the analyst
    # can focus on the real issue (fix the connector) rather than chasing symptoms.
    connector_errors: dict[str, str] = {}  # connector_id → days_in_error
    connectors = sections.get("connectors", {})
    if isinstance(connectors, dict) and "error" not in connectors:
        now = datetime.now(UTC)
        for connector_id, info in connectors.items():
            if isinstance(info, dict) and info.get("status") == "error":
                last_sync = info.get("last_sync")
                if last_sync:
                    try:
                        last_dt = datetime.fromisoformat(last_sync.replace("Z", "+00:00"))
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=UTC)
                        days_down = (now - last_dt).days
                        connector_errors[connector_id] = f"{days_down} days"
                    except (ValueError, TypeError):
                        connector_errors[connector_id] = "unknown duration"
                else:
                    connector_errors[connector_id] = "never synced"

    if connector_errors:
        # Build a human-readable note listing all errored connectors and their downtime
        error_parts = [f"'{cid}' (down {dur})" for cid, dur in sorted(connector_errors.items())]
        root_cause_hint = (
            f"This may be caused by connector(s) in error state: {', '.join(error_parts)}. "
            "When a connector is down, no new data flows in, so lookback-window checks "
            "and profile/routine detections will fail."
        )
        # Categories whose anomalies are likely downstream effects of connector outage
        stale_related_categories = {
            "routine_detection",
            "workflow_detection",
            "stale_source",
            "prediction_persistence",
            "workflow_email_imbalance",
            "missing_profile",
        }
        for anomaly in anomalies:
            # Only annotate stale-related anomalies that haven't already been annotated (idempotent).
            if anomaly.get("category") in stale_related_categories and "root_cause_hint" not in anomaly:
                anomaly["root_cause_hint"] = root_cause_hint

    return anomalies


def compute_health_score(anomalies: list[dict]) -> int:
    """Compute a health score (0-100) based on anomaly count and severity.

    Starts at 100 and subtracts:
      - 20 per critical anomaly
      - 10 per warning anomaly
      - 2 per info anomaly

    Score is clamped to [0, 100].
    """
    score = 100
    for anomaly in anomalies:
        severity = anomaly.get("severity", "info")
        if severity == "critical":
            score -= 20
        elif severity == "warning":
            score -= 10
        else:
            score -= 2
    return max(0, min(100, score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Life OS data quality analysis")
    parser.add_argument("--data-dir", default="./data", help="Path to data directory")
    args = parser.parse_args()
    result = analyze(args.data_dir)
    print(json.dumps(result, indent=2))
