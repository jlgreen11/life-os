"""
Life OS — Data Quality Analysis

Run by the Claude Code improvement agent to understand current data quality
and identify areas for improvement. Outputs a JSON report.

Usage: python scripts/analyze-data-quality.py [--data-dir ./data]
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def analyze(data_dir: str = "./data") -> dict:
    data_path = Path(data_dir)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "sections": {}}

    # --- Event volume ---
    events_db = str(data_path / "events.db")
    try:
        events_conn = sqlite3.connect(events_db)
        events_conn.row_factory = sqlite3.Row
        try:
            total = events_conn.execute("SELECT COUNT(*) as c FROM events").fetchone()["c"]
            by_type = events_conn.execute(
                "SELECT type, COUNT(*) as c FROM events GROUP BY type ORDER BY c DESC LIMIT 20"
            ).fetchall()
            last_24h = events_conn.execute(
                "SELECT COUNT(*) as c FROM events WHERE timestamp > datetime('now', '-1 day')"
            ).fetchone()["c"]

            report["sections"]["events"] = {
                "total": total,
                "last_24h": last_24h,
                "top_types": {r["type"]: r["c"] for r in by_type},
            }
        finally:
            events_conn.close()
    except Exception as e:
        report["sections"]["events"] = {"error": str(e)}

    # --- Prediction accuracy ---
    um_db = str(data_path / "user_model.db")
    try:
        um_conn = sqlite3.connect(um_db)
        um_conn.row_factory = sqlite3.Row
    except Exception as e:
        um_conn = None
        err = f"connection failed: {e}"
        report["sections"]["prediction_accuracy"] = {"error": err}
        report["sections"]["signal_profiles"] = {"error": err}
        report["sections"]["insight_feedback"] = {"error": err}

    if um_conn:
        try:
            try:
                # Compute accuracy excluding "automated_sender_fast_path" resolutions.
                #
                # Background: PR #197 introduced resolution_reason to tag predictions that
                # were resolved as INACCURATE not because the prediction was wrong, but
                # because the contact was an automated/marketing sender (e.g. noreply@).
                # The user was never going to "reach out" to a no-reply address, so these
                # predictions were structurally unfulfillable — counting them as inaccurate
                # misleads the accuracy metric and depresses confidence in a prediction type
                # that is actually performing well.
                #
                # The prediction engine's _get_accuracy_multiplier() already excludes these
                # rows when adjusting confidence.  This script now uses the same exclusion so
                # the reported accuracy matches what the engine actually acts on.
                #
                # Fields:
                #   auto_excluded — predictions with resolution_reason='automated_sender_fast_path'
                #                   (unfulfillable by design; excluded from numerator AND denominator)
                #   real_inaccurate — inaccurate predictions with no fast-path exclusion
                #                     (genuine misses that count against accuracy)
                #   accuracy_rate  — accurate / (accurate + real_inaccurate)
                #                    (same formula used by _get_accuracy_multiplier)
                pred_stats = um_conn.execute(
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
                       GROUP BY prediction_type"""
                ).fetchall()

                report["sections"]["prediction_accuracy"] = {
                    r["prediction_type"]: {
                        "total": r["total"],
                        "accurate": r["accurate"],
                        # Real misses (excludes automated-sender fast-path exclusions).
                        "inaccurate": r["inaccurate"],
                        "unresolved": r["unresolved"],
                        # Structurally-unfulfillable predictions excluded from the rate
                        # (automated senders that will never receive a reply by definition).
                        "auto_excluded": r["auto_excluded"],
                        # accuracy_rate = accurate / (accurate + real_inaccurate)
                        # Matches _get_accuracy_multiplier() denominator logic.
                        "accuracy_rate": r["accurate"] / max(r["accurate"] + r["inaccurate"], 1),
                    }
                    for r in pred_stats
                }
            except Exception as e:
                report["sections"]["prediction_accuracy"] = {"error": str(e)}

            # Signal profiles freshness
            try:
                profiles = um_conn.execute(
                    "SELECT profile_type, samples_count, updated_at FROM signal_profiles"
                ).fetchall()
                report["sections"]["signal_profiles"] = {
                    r["profile_type"]: {"samples": r["samples_count"], "last_updated": r["updated_at"]}
                    for r in profiles
                }
            except Exception as e:
                report["sections"]["signal_profiles"] = {"error": str(e)}

            # Insight feedback
            try:
                insight_stats = um_conn.execute(
                    """SELECT type, feedback, COUNT(*) as c
                       FROM insights
                       GROUP BY type, feedback"""
                ).fetchall()
                report["sections"]["insight_feedback"] = [
                    {"type": r["type"], "feedback": r["feedback"], "count": r["c"]}
                    for r in insight_stats
                ]
            except Exception as e:
                report["sections"]["insight_feedback"] = {"error": str(e)}
        finally:
            um_conn.close()

    # --- Notification dismissal rate ---
    state_db = str(data_path / "state.db")
    try:
        state_conn = sqlite3.connect(state_db)
        state_conn.row_factory = sqlite3.Row
        try:
            notif_stats = state_conn.execute(
                """SELECT status, COUNT(*) as c FROM notifications GROUP BY status"""
            ).fetchall()
            report["sections"]["notifications"] = {r["status"]: r["c"] for r in notif_stats}
        finally:
            state_conn.close()
    except Exception as e:
        report["sections"]["notifications"] = {"error": str(e)}

    # --- Feedback log ---
    pref_db = str(data_path / "preferences.db")
    try:
        pref_conn = sqlite3.connect(pref_db)
        pref_conn.row_factory = sqlite3.Row
        try:
            feedback = pref_conn.execute(
                """SELECT action_type, feedback_type, COUNT(*) as c
                   FROM feedback_log
                   GROUP BY action_type, feedback_type
                   ORDER BY c DESC"""
            ).fetchall()
            report["sections"]["feedback"] = [
                {"action_type": r["action_type"], "feedback_type": r["feedback_type"], "count": r["c"]}
                for r in feedback
            ]
        finally:
            pref_conn.close()
    except Exception as e:
        report["sections"]["feedback"] = {"error": str(e)}

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Life OS data quality analysis")
    parser.add_argument("--data-dir", default="./data", help="Path to data directory")
    args = parser.parse_args()
    result = analyze(args.data_dir)
    print(json.dumps(result, indent=2))
