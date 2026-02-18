"""
Tests for automated-sender fast-path in opportunity prediction accuracy inference.

Problem (pre-fix):
    The marketing filter in the prediction engine was enhanced in PRs #183 and #186
    to block automated/transactional senders from generating opportunity predictions.
    However, 207 predictions for automated senders had already been generated and
    surfaced before those fixes. These predictions were structurally impossible to
    fulfill (the user will never "reach out to" Fidelity's automated mailer), yet the
    tracker waited the full 7-day window before marking them as inaccurate.

    This left 207 unresolved predictions in the database, distorting the 'opportunity'
    accuracy_rate stat and slowing down the confidence calibration loop by 7 days.

Fix (this iteration):
    Added _is_automated_sender() static method to BehavioralAccuracyTracker and
    integrated it as a fast-path in _infer_opportunity_accuracy(). When the extracted
    contact email matches known automated/marketing sender patterns, the prediction is
    immediately resolved as INACCURATE instead of waiting 7 days.

    This closes the learning loop ~7 days earlier for the backlog of stale predictions
    and keeps accuracy stats clean going forward.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


# ============================================================================
# _is_automated_sender() unit tests
# ============================================================================


def test_is_automated_sender_noreply_variants():
    """All common no-reply patterns must be detected as automated."""
    variants = [
        "noreply@example.com",
        "no-reply@example.com",
        "no_reply@example.com",
        "donotreply@example.com",
        "do-not-reply@example.com",
        "autoreply@example.com",
        "auto-reply@example.com",
        "mailer-daemon@example.com",
        "postmaster@example.com",
        "automated@example.com",
    ]
    for addr in variants:
        assert BehavioralAccuracyTracker._is_automated_sender(addr), (
            f"Expected {addr!r} to be detected as automated"
        )


def test_is_automated_sender_bulk_prefixes():
    """Common bulk/transactional local-part prefixes must be detected."""
    automated = [
        "newsletter@company.com",
        "notifications@service.com",
        "notification@service.com",
        "updates@platform.com",
        "marketing@brand.com",
        "orders@shop.com",
        "confirmation@checkout.com",
        "shipping@logistics.com",
        "rewards@loyalty.com",
        "edelivery@brokerage.com",
        # Financial senders from iteration 171
        "fidelity@investments.com",
        "vanguard@fund.com",
        "schwab@broker.com",
        "paypal@payments.com",
    ]
    for addr in automated:
        assert BehavioralAccuracyTracker._is_automated_sender(addr), (
            f"Expected {addr!r} to be detected as automated"
        )


def test_is_automated_sender_embedded_patterns():
    """Embedded notification patterns in local-part must be detected."""
    automated = [
        "hoa-notifications@community.org",
        "system-alert@platform.io",
        "team-updates@corp.com",
        "morning-digest@news.com",
        "lafconews@lafc.com",
    ]
    for addr in automated:
        assert BehavioralAccuracyTracker._is_automated_sender(addr), (
            f"Expected {addr!r} to be detected as automated"
        )


def test_is_automated_sender_marketing_subdomains():
    """Marketing subdomain patterns (email.*, ifly.*, trx.*) must be detected."""
    automated = [
        "user@email.example.com",
        "passenger@ifly.southwest.com",
        "Fidelity.Investments@shareholderdocs.fidelity.com",
        "points@points-mail.ihg.com",
        "eDelivery@etradefrommorganstanley.com",
        "receipt@transaction@info.samsclub.com",
    ]
    for addr in automated:
        assert BehavioralAccuracyTracker._is_automated_sender(addr), (
            f"Expected {addr!r} to be detected as automated"
        )


def test_is_automated_sender_real_humans_not_flagged():
    """Real human email addresses must NOT be detected as automated."""
    humans = [
        "alice@gmail.com",
        "bob.smith@company.com",
        "jane.doe@startup.io",
        "john@outlook.com",
        "mary@protonmail.com",
        "contact@mycompany.com",   # 'contact@' is NOT in bulk_prefixes
        "team@startup.io",         # 'team@' is NOT in bulk_prefixes
        "admin@company.com",       # 'admin@' is NOT in bulk_prefixes
        "sales@partner.biz",       # 'sales@' — not a marketing keyword
    ]
    for addr in humans:
        assert not BehavioralAccuracyTracker._is_automated_sender(addr), (
            f"Expected {addr!r} to NOT be detected as automated (false positive)"
        )


# ============================================================================
# _infer_opportunity_accuracy() integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_opportunity_automated_sender_immediately_inaccurate(db):
    """Opportunity predictions for automated senders are marked INACCURATE immediately.

    This is the core regression test: predictions generated before the marketing
    filter improvements should resolve in the same tracker cycle, not 7 days later.
    """
    tracker = BehavioralAccuracyTracker(db)

    # Simulate a stale prediction from before the marketing filter fix:
    # no supporting_signals (old empty-list format) and a no-reply sender in description
    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)  # Only 2 hours old

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "opportunity",
                "It's been 33 days since you last contacted "
                "Fidelity.Investments.email@shareholderdocs.fidelity.com "
                "(you usually connect every ~8 days)",
                0.50,
                "suggest",
                "7_days",
                "Consider reaching out",
                "[]",  # Old empty-list format (no signals dict)
                1,
                created_at.isoformat(),
            ),
        )

    stats = await tracker.run_inference_cycle()

    # Should be resolved in this cycle (not waiting 7 days)
    assert stats["marked_inaccurate"] >= 1, (
        "Expected automated-sender prediction to be marked inaccurate immediately"
    )

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None
    assert row["was_accurate"] == 0, "Automated sender prediction should be INACCURATE"
    assert row["resolved_at"] is not None, "Prediction should be resolved"


@pytest.mark.asyncio
async def test_opportunity_real_human_waits_for_window(db):
    """Opportunity predictions for real humans still wait for the 7-day window.

    The automated-sender fast-path must not affect human contact predictions.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    # Created 2 hours ago — within the 7-day window, no contact made yet
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "opportunity",
                "It's been 30 days since you last contacted alice@gmail.com "
                "(you usually connect every ~14 days)",
                0.60,
                "suggest",
                "7_days",
                "Consider reaching out to Alice",
                json.dumps({"contact_email": "alice@gmail.com", "contact_name": "Alice"}),
                1,
                created_at.isoformat(),
            ),
        )

    stats = await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    # Should NOT be resolved yet — still within the 7-day window
    assert row["was_accurate"] is None, (
        "Human contact prediction should not be resolved within the 7-day window"
    )
    assert row["resolved_at"] is None, (
        "Human contact prediction should remain unresolved during the window"
    )


@pytest.mark.asyncio
async def test_opportunity_real_human_accurate_after_contact(db):
    """Opportunity predictions for real humans are ACCURATE when user sends outbound message."""
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    event_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=4)
    sent_at = datetime.now(timezone.utc) - timedelta(hours=1)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "opportunity",
                "It's been 25 days since you last contacted bob@gmail.com",
                0.65,
                "suggest",
                "7_days",
                "Reach out to Bob",
                json.dumps({"contact_email": "bob@gmail.com", "contact_name": "Bob"}),
                1,
                created_at.isoformat(),
            ),
        )

    # Simulate user sending an email to Bob after the prediction
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                "email.sent",
                "gmail",
                sent_at.isoformat(),
                "normal",
                json.dumps({"to_addresses": ["bob@gmail.com"], "subject": "Hey!"}),
                "{}",
            ),
        )

    stats = await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["was_accurate"] == 1, "Human contact prediction should be ACCURATE after outbound message"
    assert row["resolved_at"] is not None


@pytest.mark.asyncio
async def test_opportunity_automated_sender_via_signals_dict(db):
    """Automated sender check works when contact_email is in the signals dict."""
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=3)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "opportunity",
                "It's been 45 days since you last contacted an email list",
                0.50,
                "suggest",
                "7_days",
                "Consider reaching out",
                json.dumps({
                    "contact_email": "newsletter@brand.com",
                    "contact_name": "Brand Newsletter",
                }),
                1,
                created_at.isoformat(),
            ),
        )

    stats = await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["was_accurate"] == 0, (
        "Automated sender (via signals dict) should be immediately marked INACCURATE"
    )


@pytest.mark.asyncio
async def test_opportunity_no_contact_info_returns_none(db):
    """Opportunity predictions with no extractable contact info return None (unchanged behavior)."""
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=1)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "opportunity",
                "Good time to do some relationship maintenance",  # No email/name
                0.40,
                "suggest",
                "7_days",
                "Reach out to someone",
                "{}",
                1,
                created_at.isoformat(),
            ),
        )

    await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    # Should remain unresolved — no contact info to track
    assert row["was_accurate"] is None
    assert row["resolved_at"] is None


@pytest.mark.asyncio
async def test_opportunity_multiple_automated_senders_bulk_resolved(db):
    """Multiple automated-sender predictions are all resolved in a single inference cycle.

    This is the primary production scenario: 207 stale opportunity predictions for
    automated senders were created before the marketing filter improvements. This test
    verifies that all of them can be resolved in one cycle.
    """
    tracker = BehavioralAccuracyTracker(db)

    automated_emails = [
        ("noreply@company.com", "It's been 45 days since you last contacted noreply@company.com"),
        ("newsletter@brand.com", "It's been 30 days since you last contacted newsletter@brand.com"),
        ("vanguard@fund.com", "It's been 22 days since you last contacted vanguard@fund.com"),
        ("edelivery@etrade.com", "It's been 15 days since you last contacted edelivery@etrade.com"),
    ]

    pred_ids = []
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        for email, desc in automated_emails:
            pred_id = str(uuid.uuid4())
            pred_ids.append(pred_id)
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    time_horizon, suggested_action, supporting_signals,
                    was_surfaced, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pred_id, "opportunity", desc, 0.50, "suggest", "7_days",
                    "Consider reaching out", "[]", 1, created_at.isoformat(),
                ),
            )

    stats = await tracker.run_inference_cycle()

    assert stats["marked_inaccurate"] >= len(automated_emails), (
        f"Expected all {len(automated_emails)} automated-sender predictions to be resolved, "
        f"got {stats['marked_inaccurate']}"
    )

    with db.get_connection("user_model") as conn:
        for pred_id in pred_ids:
            row = conn.execute(
                "SELECT was_accurate FROM predictions WHERE id = ?",
                (pred_id,),
            ).fetchone()
            assert row["was_accurate"] == 0, (
                f"Prediction {pred_id} should be marked INACCURATE"
            )
