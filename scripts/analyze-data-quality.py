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

logger = logging.getLogger(__name__)


def _query(conn, sql, default=None):
    """Execute a query and return results, or default on error."""
    try:
        return conn.execute(sql).fetchall()
    except Exception as e:
        logger.warning("Query failed: %s — SQL: %s", e, sql[:200])
        return default


def _query_one(conn, sql, default=None):
    """Execute a query and return first row, or default on error."""
    try:
        return conn.execute(sql).fetchone()
    except Exception as e:
        logger.warning("Query failed: %s — SQL: %s", e, sql[:200])
        return default


def _connect(db_path):
    """Connect to a SQLite database with row factory, or return None."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def analyze(data_dir: str = "./data") -> dict:
    data_path = Path(data_dir)
    report = {"generated_at": datetime.now(UTC).isoformat(), "sections": {}}

    # -----------------------------------------------------------------------
    # Database health — run PRAGMA integrity_check on each database
    # -----------------------------------------------------------------------
    db_names = ["events", "user_model", "state", "preferences", "entities"]
    health = {}
    for db_name in db_names:
        db_conn = _connect(data_path / f"{db_name}.db")
        if db_conn:
            try:
                result = db_conn.execute("PRAGMA integrity_check").fetchone()
                health[db_name] = result[0] if result else "unknown"
            except Exception as e:
                health[db_name] = f"error: {e}"
            finally:
                db_conn.close()
        else:
            health[db_name] = "could not connect"
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

        try:
            # Signal profiles freshness
            profiles = _query(um_conn, "SELECT profile_type, samples_count, updated_at FROM signal_profiles", [])
            report["sections"]["signal_profiles"] = {
                r["profile_type"]: {"samples": r["samples_count"], "last_updated": r["updated_at"]} for r in profiles
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

            report["sections"]["user_model"] = {
                "episodes": episode_count["c"] if episode_count else 0,
                "semantic_facts": fact_count["c"] if fact_count else 0,
                "routines": routine_count["c"] if routine_count else 0,
                "fact_categories": {r["category"]: r["c"] for r in fact_categories},
                "query_errors": query_errors,
            }
        except Exception as e:
            report["sections"]["user_model"] = {"error": str(e)}

        um_conn.close()
    else:
        report["sections"]["prediction_accuracy"] = {"error": "could not connect to user_model.db"}

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
        report["sections"]["notifications"] = {"error": "could not connect to state.db"}

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
        report["sections"]["feedback"] = {"error": "could not connect to preferences.db"}

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Life OS data quality analysis")
    parser.add_argument("--data-dir", default="./data", help="Path to data directory")
    args = parser.parse_args()
    result = analyze(args.data_dir)
    print(json.dumps(result, indent=2))
