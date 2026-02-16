#!/usr/bin/env python3
"""
Diagnostic script to identify why only 'reminder' predictions are generated.

Tests each prediction type method independently against the production database
to determine which ones are failing to generate predictions and why.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore
from services.prediction_engine.engine import PredictionEngine


async def diagnose():
    """Run diagnostic tests on each prediction type."""
    print("=== Prediction Engine Diagnostic ===\n")

    # Connect to production database
    db = DatabaseManager("data")
    ums = UserModelStore(db)
    engine = PredictionEngine(db, ums)

    # Test context (empty for now)
    ctx = {}

    # Test each prediction type independently
    prediction_methods = [
        ("calendar_conflicts", engine._check_calendar_conflicts),
        ("follow_up_needs", engine._check_follow_up_needs),
        ("routine_deviations", engine._check_routine_deviations),
        ("relationship_maintenance", engine._check_relationship_maintenance),
        ("preparation_needs", engine._check_preparation_needs),
        ("spending_patterns", engine._check_spending_patterns),
    ]

    for name, method in prediction_methods:
        print(f"Testing {name}...")
        try:
            predictions = await method(ctx)
            print(f"  ✓ Generated {len(predictions)} predictions")
            if predictions:
                print(f"    Sample: {predictions[0].description[:80]}")
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
        print()

    # Check database stats
    print("\n=== Database Stats ===")
    with db.get_connection("events") as conn:
        # Email stats
        email_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE type IN ('email.received', 'message.received')"
        ).fetchone()["cnt"]
        print(f"Total inbound messages: {email_count:,}")

        # Calendar stats
        calendar_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE type = 'calendar.event.created'"
        ).fetchone()["cnt"]
        print(f"Total calendar events: {calendar_count:,}")

        # Transaction stats
        transaction_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE type = 'finance.transaction.new'"
        ).fetchone()["cnt"]
        print(f"Total transactions: {transaction_count:,}")

    with db.get_connection("user_model") as conn:
        # Routine stats
        routine_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM routines"
        ).fetchone()["cnt"]
        print(f"Total routines: {routine_count}")

        # Signal profile stats
        profile_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM signal_profiles WHERE profile_type = 'relationships'"
        ).fetchone()["cnt"]
        print(f"Relationship profiles: {profile_count}")

    print("\n=== Prediction Stats ===")
    with db.get_connection("user_model") as conn:
        pred_stats = conn.execute(
            """SELECT prediction_type, COUNT(*) as total,
                      SUM(CASE WHEN was_surfaced = 1 THEN 1 ELSE 0 END) as surfaced
               FROM predictions
               GROUP BY prediction_type"""
        ).fetchall()

        for row in pred_stats:
            print(f"{row['prediction_type']}: {row['total']:,} total, {row['surfaced']} surfaced")


if __name__ == "__main__":
    asyncio.run(diagnose())
