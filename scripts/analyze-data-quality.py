"""
Life OS — Data Quality Analysis

Run by the Claude Code improvement agent to understand current data quality
and identify areas for improvement. Outputs a JSON report.

Usage: python scripts/analyze-data-quality.py [--data-dir ./data]
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


def analyze(data_dir: str = "./data") -> dict:
    data_path = Path(data_dir)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "sections": {}}

    # --- Event volume ---
    events_db = str(data_path / "events.db")
    try:
        conn = sqlite3.connect(events_db)
        conn.row_factory = sqlite3.Row

        total = conn.execute("SELECT COUNT(*) as c FROM events").fetchone()["c"]
        by_type = conn.execute(
            "SELECT type, COUNT(*) as c FROM events GROUP BY type ORDER BY c DESC LIMIT 20"
        ).fetchall()
        last_24h = conn.execute(
            "SELECT COUNT(*) as c FROM events WHERE timestamp > datetime('now', '-1 day')"
        ).fetchone()["c"]

        report["sections"]["events"] = {
            "total": total,
            "last_24h": last_24h,
            "top_types": {r["type"]: r["c"] for r in by_type},
        }
        conn.close()
    except Exception as e:
        report["sections"]["events"] = {"error": str(e)}

    # --- Prediction accuracy ---
    um_db = str(data_path / "user_model.db")
    try:
        conn = sqlite3.connect(um_db)
        conn.row_factory = sqlite3.Row

        pred_stats = conn.execute(
            """SELECT prediction_type,
                COUNT(*) as total,
                SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate,
                SUM(CASE WHEN was_accurate = 0 THEN 1 ELSE 0 END) as inaccurate,
                SUM(CASE WHEN was_accurate IS NULL THEN 1 ELSE 0 END) as unresolved
               FROM predictions
               WHERE was_surfaced = 1
               GROUP BY prediction_type"""
        ).fetchall()

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

        # Signal profiles freshness
        profiles = conn.execute(
            "SELECT profile_type, samples_count, updated_at FROM signal_profiles"
        ).fetchall()
        report["sections"]["signal_profiles"] = {
            r["profile_type"]: {"samples": r["samples_count"], "last_updated": r["updated_at"]}
            for r in profiles
        }

        # Insight feedback
        insight_stats = conn.execute(
            """SELECT type, feedback, COUNT(*) as c
               FROM insights
               GROUP BY type, feedback"""
        ).fetchall()
        report["sections"]["insight_feedback"] = [
            {"type": r["type"], "feedback": r["feedback"], "count": r["c"]}
            for r in insight_stats
        ]

        conn.close()
    except Exception as e:
        report["sections"]["prediction_accuracy"] = {"error": str(e)}

    # --- Notification dismissal rate ---
    state_db = str(data_path / "state.db")
    try:
        conn = sqlite3.connect(state_db)
        conn.row_factory = sqlite3.Row

        notif_stats = conn.execute(
            """SELECT status, COUNT(*) as c FROM notifications GROUP BY status"""
        ).fetchall()
        report["sections"]["notifications"] = {r["status"]: r["c"] for r in notif_stats}

        conn.close()
    except Exception as e:
        report["sections"]["notifications"] = {"error": str(e)}

    # --- Feedback log ---
    pref_db = str(data_path / "preferences.db")
    try:
        conn = sqlite3.connect(pref_db)
        conn.row_factory = sqlite3.Row

        feedback = conn.execute(
            """SELECT action_type, feedback_type, COUNT(*) as c
               FROM feedback_log
               GROUP BY action_type, feedback_type
               ORDER BY c DESC"""
        ).fetchall()
        report["sections"]["feedback"] = [
            {"action_type": r["action_type"], "feedback_type": r["feedback_type"], "count": r["c"]}
            for r in feedback
        ]

        conn.close()
    except Exception as e:
        report["sections"]["feedback"] = {"error": str(e)}

    return report


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    result = analyze(data_dir)
    print(json.dumps(result, indent=2))
