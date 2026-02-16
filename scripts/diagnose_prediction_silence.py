#!/usr/bin/env python3
"""
Diagnostic script to understand why 5 of 6 prediction types are generating 0 predictions.

We know:
- reminder: 2,275 predictions (working)
- calendar_conflicts: 0 predictions (2,581 calendar events exist)
- routine_deviations: 0 predictions (4 routines exist)
- relationship_maintenance: 0 predictions (115K+ relationship samples)
- preparation_needs: 0 predictions (2,581 calendar events exist)
- follow_up_needs: 0 predictions (101K+ emails exist)
- spending_patterns: 0 predictions (0 finance events)
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.database import DatabaseManager
from storage.user_model_store import UserModelStore
from services.prediction_engine.engine import PredictionEngine


async def main():
    """Run diagnostics on all prediction types."""
    db = DatabaseManager("/Users/jeremygreenwood/life-os/data")
    ums = UserModelStore(db)
    engine = PredictionEngine(db, ums)

    print("=== Prediction Engine Diagnostics ===\n")

    # Create a minimal context
    context = {
        "current_location": None,
        "current_activity": None,
        "time_of_day": "afternoon"
    }

    # Test each prediction type individually
    print("1. Testing calendar conflicts...")
    try:
        conflicts = await engine._check_calendar_conflicts(context)
        print(f"   Result: {len(conflicts)} predictions")
        if conflicts:
            for c in conflicts[:3]:
                print(f"   - {c.description}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n2. Testing routine deviations...")
    try:
        routines = await engine._check_routine_deviations(context)
        print(f"   Result: {len(routines)} predictions")
        if routines:
            for r in routines[:3]:
                print(f"   - {r.description}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n3. Testing relationship maintenance...")
    try:
        relationships = await engine._check_relationship_maintenance(context)
        print(f"   Result: {len(relationships)} predictions")
        if relationships:
            for rel in relationships[:3]:
                print(f"   - {rel.description}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n4. Testing preparation needs...")
    try:
        prep = await engine._check_preparation_needs(context)
        print(f"   Result: {len(prep)} predictions")
        if prep:
            for p in prep[:3]:
                print(f"   - {p.description}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n5. Testing follow-up needs...")
    try:
        followup = await engine._check_follow_up_needs(context)
        print(f"   Result: {len(followup)} predictions")
        if followup:
            for f in followup[:3]:
                print(f"   - {f.description}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n6. Testing spending patterns...")
    try:
        spending = await engine._check_spending_patterns(context)
        print(f"   Result: {len(spending)} predictions")
        if spending:
            for s in spending[:3]:
                print(f"   - {s.description}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Check database state
    print("\n=== Database State ===")

    with db.get_connection("events") as conn:
        # Calendar events
        total_cal = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE type = 'calendar.event.created'"
        ).fetchone()["cnt"]

        future_cal = conn.execute(
            """SELECT COUNT(*) as cnt FROM events
               WHERE type = 'calendar.event.created'
               AND json_extract(payload, '$.start_time') > datetime('now')"""
        ).fetchone()["cnt"]

        print(f"Calendar events: {total_cal} total, {future_cal} future")

        # Email events
        emails = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE type = 'email.received'"
        ).fetchone()["cnt"]
        print(f"Email events: {emails}")

    with db.get_connection("user_model") as conn:
        # Routines
        routines = conn.execute(
            "SELECT COUNT(*) as cnt FROM routines"
        ).fetchone()["cnt"]
        print(f"Routines: {routines}")

        # Relationships profile
        profile = conn.execute(
            "SELECT data, samples_count FROM signal_profiles WHERE profile_type = 'relationships'"
        ).fetchone()

        if profile:
            data = json.loads(profile["data"])
            contacts_count = len(data.get("contacts", []))
            print(f"Relationship profile: {profile['samples_count']} samples, {contacts_count} contacts")


if __name__ == "__main__":
    asyncio.run(main())
