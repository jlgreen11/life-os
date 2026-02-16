"""
Life OS — Backfill Reminder Prediction Contact Information

This script fixes the 165 surfaced reminder predictions that were created with
empty supporting_signals ('[]') before the contact extraction logic was fixed
in iterations 66 and 68.

The Problem:
    The behavioral accuracy tracker (services/behavioral_accuracy_tracker/tracker.py)
    needs contact email/name in supporting_signals to automatically resolve reminder
    predictions. But predictions created before iteration 68 have supporting_signals
    set to the old list format '[]', making them unresolvable.

The Solution:
    Re-extract contact information from the prediction description field using the
    same regex patterns that the tracker uses (lines 174-196 of tracker.py), then
    update supporting_signals with a proper dict containing contact_email and/or
    contact_name.

Usage:
    python scripts/backfill_reminder_contacts.py [--data-dir ./data] [--dry-run]

Examples:
    # Preview changes without writing to database
    python scripts/backfill_reminder_contacts.py --dry-run

    # Apply the backfill
    python scripts/backfill_reminder_contacts.py

Expected Output:
    Backfilled 165 reminder predictions with contact information
    - 165 predictions now have contact_email
    - 0 predictions now have contact_name
    - 0 predictions remain unresolvable (no extractable contact info)
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Optional


def extract_contact_info(description: str) -> dict[str, str]:
    """Extract contact email and/or name from a reminder prediction description.

    Uses the same regex patterns as BehavioralAccuracyTracker._infer_reminder_accuracy
    to ensure consistency.

    Args:
        description: The prediction description field

    Returns:
        Dict with 'contact_email' and/or 'contact_name' if found, else empty dict
    """
    contact_info = {}

    # Pattern 1: "Unreplied message from EMAIL" (most common)
    # Example: "Unreplied message from alice@example.com: \"Subject\" (3 hours ago)"
    # Handles complex emails: john.doe+work@company-name.co.uk
    email_match = re.search(
        r'from\s+([\w\.\-\+]+@[\w\.\-]+\.[\w\.]+)', description, re.IGNORECASE
    )
    if email_match:
        contact_info["contact_email"] = email_match.group(1)

    # Pattern 2: "Reply to NAME" or "Follow up with NAME" (for future compatibility)
    # Example: "Follow up with Alice about the project"
    # Two-stage match: trigger phrase is case-insensitive, but name must be
    # properly capitalized to avoid false matches (e.g., "Grace" not "about")
    if not contact_info.get("contact_email"):
        trigger_match = re.search(
            r'(reply to|follow up with|message)\s+', description, re.IGNORECASE
        )
        if trigger_match:
            # Extract properly capitalized name after the trigger
            rest = description[trigger_match.end():]
            name_match = re.match(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', rest)
            if name_match:
                contact_info["contact_name"] = name_match.group(1)

    return contact_info


def backfill_reminder_contacts(data_dir: str, dry_run: bool = False) -> dict:
    """Backfill contact information for reminder predictions with empty supporting_signals.

    Args:
        data_dir: Path to the data directory containing user_model.db
        dry_run: If True, show what would be updated but don't write to database

    Returns:
        Dict with stats: {'updated': N, 'email_only': X, 'name_only': Y, 'unresolvable': Z}
    """
    db_path = Path(data_dir) / "user_model.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    stats = {
        "updated": 0,
        "email_only": 0,
        "name_only": 0,
        "both": 0,
        "unresolvable": 0,
    }

    try:
        # Find all reminder predictions with empty or list-formatted supporting_signals
        predictions = conn.execute(
            """SELECT id, description, supporting_signals
               FROM predictions
               WHERE prediction_type = 'reminder'
                 AND was_surfaced = 1
                 AND resolved_at IS NULL
                 AND (supporting_signals = '[]' OR supporting_signals IS NULL)"""
        ).fetchall()

        print(f"Found {len(predictions)} reminder predictions to backfill")

        for pred in predictions:
            # Extract contact info from description
            contact_info = extract_contact_info(pred["description"])

            if not contact_info:
                # No extractable contact info — mark as unresolvable
                stats["unresolvable"] += 1
                if dry_run:
                    print(f"  [SKIP] {pred['id'][:8]}... - No contact info in: {pred['description'][:60]}...")
                continue

            # Categorize the type of contact info found
            has_email = "contact_email" in contact_info
            has_name = "contact_name" in contact_info

            if has_email and has_name:
                stats["both"] += 1
            elif has_email:
                stats["email_only"] += 1
            elif has_name:
                stats["name_only"] += 1

            # Update supporting_signals with the extracted contact info
            new_signals = json.dumps(contact_info)

            if dry_run:
                print(f"  [DRY-RUN] {pred['id'][:8]}... - Would add: {contact_info}")
            else:
                conn.execute(
                    "UPDATE predictions SET supporting_signals = ? WHERE id = ?",
                    (new_signals, pred["id"]),
                )
                stats["updated"] += 1

        if not dry_run:
            conn.commit()
            print(f"\nBackfilled {stats['updated']} reminder predictions")
        else:
            print(f"\nDry run complete - would backfill {len(predictions) - stats['unresolvable']} predictions")

        print(f"  - {stats['email_only']} with contact_email only")
        print(f"  - {stats['name_only']} with contact_name only")
        print(f"  - {stats['both']} with both email and name")
        print(f"  - {stats['unresolvable']} unresolvable (no contact info in description)")

    finally:
        conn.close()

    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill contact information for reminder predictions"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Path to data directory (default: ./data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to database",
    )
    args = parser.parse_args()

    try:
        stats = backfill_reminder_contacts(args.data_dir, args.dry_run)
        return 0 if stats["updated"] > 0 or args.dry_run else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
