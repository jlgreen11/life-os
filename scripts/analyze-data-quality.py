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
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Module-level list that accumulates query errors during analyze().
# Consumers can inspect report["query_errors"] to distinguish "no data" from "query failed".
_errors: list[dict] = []


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

            report["sections"]["signal_profiles"] = {
                "profiles": {
                    r["profile_type"]: {"samples": r["samples_count"], "last_updated": r["updated_at"]}
                    for r in profiles
                },
                "missing_profiles": missing,
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

            # Source weights — high drift indicates the system is losing
            # confidence in some data sources
            source_weights = _query(
                pref_conn,
                """SELECT source_key, user_weight, ai_drift, ai_updated_at
                   FROM source_weights ORDER BY ai_drift DESC""",
                [],
            )
            report["sections"]["source_weights"] = (
                {
                    r["source_key"]: {
                        "weight": r["user_weight"],
                        "drift": r["ai_drift"],
                        "updated_at": r["ai_updated_at"],
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
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Life OS data quality analysis")
    parser.add_argument("--data-dir", default="./data", help="Path to data directory")
    args = parser.parse_args()
    result = analyze(args.data_dir)
    print(json.dumps(result, indent=2))
