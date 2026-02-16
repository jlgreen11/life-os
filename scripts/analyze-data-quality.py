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
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def _query(conn, sql, default=None):
    """Execute a query and return results, or default on error."""
    try:
        return conn.execute(sql).fetchall()
    except Exception:
        return default


def _query_one(conn, sql, default=None):
    """Execute a query and return first row, or default on error."""
    try:
        return conn.execute(sql).fetchone()
    except Exception:
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
        try:
            # Accuracy by prediction type
            pred_stats = _query(
                um_conn,
                """SELECT prediction_type,
                    COUNT(*) as total,
                    SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate,
                    SUM(CASE WHEN was_accurate = 0 THEN 1 ELSE 0 END) as inaccurate,
                    SUM(CASE WHEN was_accurate IS NULL THEN 1 ELSE 0 END) as unresolved
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
                    "accuracy_rate": r["accurate"] / max(r["total"] - r["unresolved"], 1),
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

            # Signal profiles freshness
            profiles = _query(um_conn, "SELECT profile_type, samples_count, updated_at FROM signal_profiles", [])
            report["sections"]["signal_profiles"] = {
                r["profile_type"]: {"samples": r["samples_count"], "last_updated": r["updated_at"]} for r in profiles
            }

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

            # User model depth — episodes, facts, routines
            episode_count = _query_one(um_conn, "SELECT COUNT(*) as c FROM episodes")
            fact_count = _query_one(um_conn, "SELECT COUNT(*) as c FROM semantic_facts")
            routine_count = _query_one(um_conn, "SELECT COUNT(*) as c FROM routines")

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
            }
        except Exception as e:
            report["sections"]["prediction_accuracy"] = {"error": str(e)}
        finally:
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
                """SELECT connector_id, status, last_sync, error_message
                   FROM connector_state ORDER BY connector_id""",
                [],
            )
            report["sections"]["connectors"] = (
                {
                    r["connector_id"]: {
                        "status": r["status"],
                        "last_sync": r["last_sync"],
                        "error": r["error_message"],
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
                """SELECT source_key, weight, drift, updated_at
                   FROM source_weights ORDER BY drift DESC""",
                [],
            )
            report["sections"]["source_weights"] = (
                {
                    r["source_key"]: {
                        "weight": r["weight"],
                        "drift": r["drift"],
                        "updated_at": r["updated_at"],
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
