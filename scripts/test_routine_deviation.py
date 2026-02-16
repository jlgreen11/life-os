#!/usr/bin/env python3
"""Test why routine_deviations returns 0 predictions."""

import asyncio
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore
from services.prediction_engine.engine import PredictionEngine


async def test_routine():
    """Diagnose routine deviation detection."""
    print("=== Routine Deviation Diagnostic ===\n")

    db = DatabaseManager("data")
    ums = UserModelStore(db)

    # Check routines in database
    with db.get_connection("user_model") as conn:
        routines = conn.execute("SELECT * FROM routines").fetchall()

    print(f"Found {len(routines)} routine(s)\n")

    for routine in routines:
        print(f"Routine: {routine['name']}")
        print(f"  Trigger: {routine['trigger_condition']}")
        print(f"  Consistency: {routine['consistency_score']}")
        print(f"  Times observed: {routine['times_observed']}")

        steps = json.loads(routine['steps'])
        print(f"  Steps ({len(steps)}):")
        for i, step in enumerate(steps[:5]):
            print(f"    {i}. {step.get('action', 'unknown')}")

        # Check if routine was completed today
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        # Parse routine steps to get expected event types
        expected_actions = []
        for step in steps[:3]:
            if isinstance(step, dict):
                action = step.get("action", "")
                if action:
                    expected_actions.append(action)

        print(f"  Expected actions (first 3): {expected_actions}")

        # Map to event types
        event_type_mapping = {
            "email_received": "email.received",
            "email_sent": "email.sent",
            "message_received": "message.received",
            "message_sent": "message.sent",
            "task_created": "task.created",
            "task_completed": "task.completed",
            "calendar_event_created": "calendar.event.created",
        }

        expected_event_types = []
        for action in expected_actions:
            event_type = event_type_mapping.get(action, action.replace("_", "."))
            expected_event_types.append(event_type)

        print(f"  Expected event types: {expected_event_types}")

        # Check if events occurred today
        with db.get_connection("events") as conn:
            for et in expected_event_types:
                count = conn.execute(
                    """SELECT COUNT(*) as cnt FROM events
                       WHERE type = ?
                       AND timestamp > ?""",
                    (et, today_start.isoformat()),
                ).fetchone()["cnt"]
                print(f"    {et}: {count} events today")

        # Check if routine trigger matches current time
        trigger = routine['trigger_condition']
        print(f"\n  Trigger condition: '{trigger}'")
        print(f"  Current hour (UTC): {datetime.now(timezone.utc).hour}")

        # Test if a prediction would be created
        engine = PredictionEngine(db, ums)
        predictions = await engine._check_routine_deviations({})
        print(f"\n  Predictions generated: {len(predictions)}")

        if predictions:
            for pred in predictions:
                print(f"    - {pred.description}")
                print(f"      Confidence: {pred.confidence:.2f}")


if __name__ == "__main__":
    asyncio.run(test_routine())
