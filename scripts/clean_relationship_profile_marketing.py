#!/usr/bin/env python3
"""
Clean marketing contacts from relationships signal profile.

PROBLEM:
Before iteration 129 (PR #143), the RelationshipExtractor tracked ALL email
senders as "contacts", including marketing emails. This polluted the
relationships profile with 800+ marketing contacts when only ~20 were actual
humans.

PR #143 added a marketing filter to PREVENT new marketing contacts from being
tracked, but it didn't clean up the EXISTING pollution. This script removes
marketing contacts that were tracked before the filter was added.

IMPACT:
- Reduces relationships profile from 800+ contacts to ~20 real humans
- Enables relationship maintenance predictions to actually generate
- Improves prediction engine performance (no longer loops through 800+ marketing contacts)
- Cleans up storage bloat in signal_profiles table
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to import Life OS modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.manager import DatabaseManager


def is_marketing_or_noreply(from_addr: str) -> bool:
    """
    Check if an email address is marketing/automated.

    This is a simplified version of the filter from RelationshipExtractor
    and PredictionEngine._is_marketing_or_noreply, adapted to work without
    email body text (since we only have the address in the profile).

    Returns:
        True if the address is marketing/automated and should be removed
    """
    addr_lower = from_addr.lower()

    # No-reply and automated system senders
    noreply_patterns = (
        "no-reply@", "noreply@", "do-not-reply@", "donotreply@",
        "mailer-daemon@", "postmaster@", "daemon@", "auto-reply@",
        "autoreply@", "automated@",
    )
    if any(pattern in addr_lower for pattern in noreply_patterns):
        return True

    # Common bulk sender local-parts (the part before @)
    bulk_localpart_patterns = (
        "newsletter@", "notifications@", "updates@", "digest@",
        "mailer@", "bulk@", "promo@", "marketing@",
        "reply@", "email@", "news@", "offers@", "deals@",
        "hello@", "info@", "support@", "help@",
        "service@", "discover@", "alert@", "alerts@", "notification@",
        "orders@", "order@", "receipts@", "receipt@",
        "auto-confirm@", "autoconfirm@", "confirmation@",
        "shipment-tracking@", "shipping@", "delivery@",
        "accountservice@", "account-service@",
        "communications@", "development@", "fundraising@",
        "rewards@", "loyalty@",
    )
    if any(addr_lower.startswith(pattern) for pattern in bulk_localpart_patterns):
        return True

    # Embedded notification patterns (middle of local-part)
    embedded_notification_patterns = (
        "-notification", "-notifications", "-alert", "-alerts",
        "-update", "-updates", "-digest",
    )
    local_part = addr_lower.split("@")[0] if "@" in addr_lower else addr_lower
    if any(pattern in local_part for pattern in embedded_notification_patterns):
        return True

    # Marketing domain patterns (the part after @)
    marketing_domain_patterns = (
        "@news-", "@email.", "@reply.", "@mailing.",
        "@newsletters.", "@promo.", "@marketing.",
        "@em.", "@mg.", "@mail.",
        "@engage.", "@iluv.", "@e.", "@e2.",
        "@comms.",  # Critical addition: callofduty@comms.activision.com pattern
    )
    if any(pattern in addr_lower for pattern in marketing_domain_patterns):
        return True

    # Marketing service provider subdomains
    domain = addr_lower.split("@")[1] if "@" in addr_lower else ""
    marketing_service_patterns = (
        ".e2ma.net", ".sendgrid.net", ".mailchimp.com",
        ".constantcontact.com", ".hubspot.com", ".marketo.com",
        ".pardot.com", ".eloqua.com",
    )
    if any(domain.endswith(pattern) for pattern in marketing_service_patterns):
        return True

    return False


def clean_relationship_profile(db: DatabaseManager, dry_run: bool = False) -> dict:
    """
    Remove marketing contacts from the relationships signal profile.

    Args:
        db: Database manager instance
        dry_run: If True, report what would be removed but don't modify the database

    Returns:
        dict with statistics about the cleanup (total, removed, remaining)
    """
    print("=" * 80)
    print("RELATIONSHIP PROFILE CLEANUP")
    print("=" * 80)
    print()

    # Load the current relationships profile
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            """SELECT profile_type, data, samples_count, updated_at
               FROM signal_profiles
               WHERE profile_type = 'relationships'"""
        ).fetchone()

    if not row:
        print("✗ No relationships profile found")
        return {"total": 0, "removed": 0, "remaining": 0}

    # Parse the profile data
    try:
        data = json.loads(row["data"])
    except (json.JSONDecodeError, TypeError):
        print("✗ Failed to parse relationships profile data")
        return {"total": 0, "removed": 0, "remaining": 0}

    contacts = data.get("contacts", {})
    total_contacts = len(contacts)

    print(f"Current profile state:")
    print(f"  Total contacts: {total_contacts}")
    print(f"  Samples count: {row['samples_count']}")
    print(f"  Last updated: {row['updated_at']}")
    print()

    # Identify marketing contacts
    marketing_contacts = []
    human_contacts = []

    for addr in contacts.keys():
        if is_marketing_or_noreply(addr):
            marketing_contacts.append(addr)
        else:
            human_contacts.append(addr)

    print(f"Analysis:")
    print(f"  Marketing/automated contacts: {len(marketing_contacts)}")
    print(f"  Human contacts: {len(human_contacts)}")
    print()

    if len(marketing_contacts) > 0:
        print(f"Sample marketing contacts to remove (showing first 10):")
        for addr in marketing_contacts[:10]:
            count = contacts[addr].get("interaction_count", 0)
            print(f"  - {addr} ({count} interactions)")
        if len(marketing_contacts) > 10:
            print(f"  ... and {len(marketing_contacts) - 10} more")
        print()

    if dry_run:
        print("DRY RUN: No changes made to database")
        print(f"Would remove {len(marketing_contacts)} marketing contacts")
        return {
            "total": total_contacts,
            "removed": len(marketing_contacts),
            "remaining": len(human_contacts),
        }

    # Remove marketing contacts from the data
    for addr in marketing_contacts:
        del contacts[addr]

    # Update the profile with cleaned data
    cleaned_data = {"contacts": contacts}
    cleaned_json = json.dumps(cleaned_data)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """UPDATE signal_profiles
               SET data = ?,
                   updated_at = ?
               WHERE profile_type = 'relationships'""",
            (cleaned_json, datetime.now(timezone.utc).isoformat()),
        )

    print("✓ Relationships profile cleaned successfully")
    print(f"  Removed: {len(marketing_contacts)} marketing contacts")
    print(f"  Remaining: {len(human_contacts)} human contacts")
    print()

    return {
        "total": total_contacts,
        "removed": len(marketing_contacts),
        "remaining": len(human_contacts),
    }


def main():
    """Run the cleanup script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean marketing contacts from relationships signal profile"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed but don't modify the database",
    )
    args = parser.parse_args()

    db = DatabaseManager("data")
    stats = clean_relationship_profile(db, dry_run=args.dry_run)

    print("=" * 80)
    print("CLEANUP COMPLETE")
    print("=" * 80)
    print(f"Total contacts before: {stats['total']}")
    print(f"Marketing contacts removed: {stats['removed']}")
    print(f"Human contacts remaining: {stats['remaining']}")

    if stats["removed"] > 0 and not args.dry_run:
        print()
        print("IMPORTANT: Relationship maintenance predictions should now generate.")
        print("Run the prediction engine to verify:")
        print("  python scripts/diagnose_prediction_failures.py")


if __name__ == "__main__":
    main()
