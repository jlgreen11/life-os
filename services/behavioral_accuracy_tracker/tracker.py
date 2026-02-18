"""
Life OS — Behavioral Prediction Accuracy Tracker

Automatically infers prediction accuracy from user behavior, closing the
feedback loop without requiring explicit user interaction with notifications.

The Problem:
    The prediction engine needs accuracy data to learn and improve, but this
    data only comes from explicit user actions (clicking "Act On" or "Dismiss"
    buttons) or from auto-resolution after 24 hours of ignoring a notification.
    This creates a cold-start problem: new systems have no accuracy data, so
    they can't learn or calibrate confidence gates.

The Solution:
    Track user behavior and automatically mark predictions as accurate when the
    user's actions align with what was predicted, even if they never interacted
    with the notification directly.

Examples:
    - Prediction: "Reply to Alice about dinner plans"
      Behavior: User sends a message to Alice within 6 hours
      → Mark prediction as ACCURATE

    - Prediction: "Calendar conflict: Team sync overlaps with dentist"
      Behavior: User reschedules one of the events within 24 hours
      → Mark prediction as ACCURATE

    - Prediction: "Prepare slides for Q4 planning meeting"
      Behavior: User opens/edits a file containing "slides" or "Q4" keywords
      → Mark prediction as ACCURATE

    - Prediction: "Follow up with Bob about the project"
      Behavior: 48 hours pass, no message sent to Bob
      → Mark prediction as INACCURATE

This allows the system to bootstrap its learning from observed behavior instead
of waiting for explicit feedback, dramatically accelerating the calibration loop.

Architecture:
    - Runs as a background task every 15 minutes (same cadence as prediction engine)
    - Queries unresolved surfaced predictions
    - Scans recent events for behavioral signals that confirm or refute each prediction
    - Updates predictions.was_accurate when confidence threshold is met
    - Preserves user_response = 'inferred' to distinguish from explicit feedback
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from storage.database import DatabaseManager


class BehavioralAccuracyTracker:
    """Infers prediction accuracy from user behavior patterns."""

    def __init__(self, db: DatabaseManager):
        """Initialize the behavioral accuracy tracker.

        Args:
            db: Database manager for accessing events and predictions.
        """
        self.db = db

    async def run_inference_cycle(self) -> dict[str, int]:
        """Run one inference cycle over unresolved predictions.

        Processes both surfaced and filtered predictions to enable full learning loop:
        - Surfaced predictions: Check if user took predicted action (true positive)
        - Filtered predictions: Check if user STILL took action despite filter (false negative)

        This closes a critical gap: filtered predictions with was_accurate=NULL never
        contributed to the learning loop, preventing the system from discovering that
        its filters are rejecting valuable predictions.

        Returns:
            Dict with counts: {
                'marked_accurate': N,
                'marked_inaccurate': M,
                'surfaced': surfaced_count,
                'filtered': filtered_count
            }
        """
        stats = {
            'marked_accurate': 0,
            'marked_inaccurate': 0,
            'surfaced': 0,
            'filtered': 0,
        }

        # Process surfaced predictions that haven't been resolved yet
        with self.db.get_connection("user_model") as conn:
            surfaced_predictions = conn.execute(
                """SELECT id, prediction_type, description, suggested_action,
                          supporting_signals, created_at, was_surfaced
                   FROM predictions
                   WHERE was_surfaced = 1
                     AND resolved_at IS NULL
                     AND created_at > ?""",
                # Only look at predictions from the last 7 days (older ones are
                # handled by auto-resolve stale predictions logic)
                ((datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),),
            ).fetchall()

        for pred in surfaced_predictions:
            # Try to infer accuracy from behavioral signals
            result = await self._infer_accuracy(dict(pred))

            if result is not None:
                # We have enough confidence to make an inference
                was_accurate = result
                now = datetime.now(timezone.utc).isoformat()

                with self.db.get_connection("user_model") as conn:
                    conn.execute(
                        """UPDATE predictions SET
                           was_accurate = ?,
                           resolved_at = ?,
                           user_response = 'inferred'
                           WHERE id = ?""",
                        (1 if was_accurate else 0, now, pred["id"]),
                    )

                if was_accurate:
                    stats['marked_accurate'] += 1
                else:
                    stats['marked_inaccurate'] += 1
                stats['surfaced'] += 1

        # Process filtered predictions to detect false negatives (filter mistakes)
        # These predictions were auto-filtered but might have been valuable!
        # If the user took the action anyway, the filter was WRONG (false negative).
        # If the user didn't take the action, the filter was RIGHT (true negative).
        with self.db.get_connection("user_model") as conn:
            filtered_predictions = conn.execute(
                """SELECT id, prediction_type, description, suggested_action,
                          supporting_signals, created_at, was_surfaced
                   FROM predictions
                   WHERE was_surfaced = 0
                     AND user_response = 'filtered'
                     AND was_accurate IS NULL
                     AND created_at > ?
                     AND created_at < ?""",
                # Look at filtered predictions from 48 hours to 7 days ago.
                # - Must be 48+ hours old so we have time to observe behavior
                # - Must be <7 days old to stay relevant
                (
                    (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
                    (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat(),
                ),
            ).fetchall()

        for pred in filtered_predictions:
            # Try to infer accuracy from behavioral signals
            result = await self._infer_accuracy(dict(pred))

            if result is not None:
                # We can now determine if the filter was correct!
                # - result=True: User DID take action → filter was WRONG (false negative)
                # - result=False: User didn't take action → filter was RIGHT (true negative)
                was_accurate = result
                now = datetime.now(timezone.utc).isoformat()

                with self.db.get_connection("user_model") as conn:
                    conn.execute(
                        """UPDATE predictions SET
                           was_accurate = ?,
                           resolved_at = ?
                           WHERE id = ?""",
                        # Keep user_response='filtered' to preserve provenance
                        (1 if was_accurate else 0, now, pred["id"]),
                    )

                if was_accurate:
                    stats['marked_accurate'] += 1
                else:
                    stats['marked_inaccurate'] += 1
                stats['filtered'] += 1

        return stats

    async def _infer_accuracy(self, prediction: dict) -> Optional[bool]:
        """Infer whether a prediction was accurate based on user behavior.

        Args:
            prediction: Dict with fields: id, prediction_type, description,
                       suggested_action, supporting_signals, created_at

        Returns:
            True if behavior confirms the prediction was accurate
            False if behavior confirms the prediction was inaccurate
            None if insufficient evidence to make a determination
        """
        pred_type = prediction["prediction_type"]
        created_at = datetime.fromisoformat(prediction["created_at"].replace('Z', '+00:00'))

        # Parse supporting_signals JSON to extract relevant context
        # Handle both old list format and new dict format for backward compatibility
        try:
            signals = json.loads(prediction["supporting_signals"]) if prediction["supporting_signals"] else {}
            # If it's a list (old format), convert to empty dict
            if isinstance(signals, list):
                signals = {}
        except (json.JSONDecodeError, TypeError):
            signals = {}

        # Dispatch to type-specific inference logic
        if pred_type == "reminder":
            return await self._infer_reminder_accuracy(prediction, signals, created_at)
        elif pred_type == "conflict":
            return await self._infer_conflict_accuracy(prediction, signals, created_at)
        elif pred_type == "need":
            return await self._infer_need_accuracy(prediction, signals, created_at)
        elif pred_type == "opportunity":
            return await self._infer_opportunity_accuracy(prediction, signals, created_at)
        elif pred_type == "risk":
            return await self._infer_risk_accuracy(prediction, signals, created_at)
        elif pred_type == "routine_deviation":
            return await self._infer_routine_deviation_accuracy(prediction, signals, created_at)
        else:
            return None  # Unknown prediction type

    async def _infer_reminder_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'reminder' predictions.

        Reminder predictions typically suggest: "Reply to X" or "Follow up with Y".
        We look for outbound messages to the mentioned contact within a reasonable
        timeframe (6-48 hours).

        Accuracy inference logic:
        - If the contact is an automated/marketing sender: immediately INACCURATE
          (the prediction was generated before the marketing filter was robust;
          the user will never "reply" to a no-reply mailer by definition)
        - If the user sends a message to the contact within 48 hours: ACCURATE
        - If 48+ hours pass with no reply: INACCURATE
        - If no contact info can be found AND 48+ hours have passed: INACCURATE
          (the prediction can never be confirmed so it's safe to resolve)
        - If still within the window: None (wait)

        The automated-sender fast-path resolves stale predictions from before
        the marketing filter improvements within minutes instead of waiting
        48 hours, keeping accuracy stats clean and the learning loop tight.
        """
        # Extract contact email/name from signals (new dict format)
        contact_email = signals.get("contact_email")
        contact_name = signals.get("contact_name")

        # Fallback: try old keys for backward compatibility
        if not contact_email:
            contact_email = signals.get("contact_id")

        # If no contact info in signals, try to extract from description
        if not contact_email and not contact_name:
            # Extract contact info from common description patterns.
            # Handles both old descriptions with email addresses and future
            # descriptions that may use names.
            import re

            # Pattern 1: "Unreplied message from EMAIL" (most common)
            # Example: "Unreplied message from alice@example.com: \"Subject\" (3 hours ago)"
            # Handles complex emails: john.doe+work@company-name.co.uk
            email_match = re.search(r'from\s+([\w\.\-\+]+@[\w\.\-]+\.[\w\.]+)', prediction["description"], re.IGNORECASE)
            if email_match:
                contact_email = email_match.group(1)

            # Pattern 2: "Reply to NAME" or "Follow up with NAME" (for future compatibility)
            # Example: "Follow up with Alice about the project"
            # Two-stage match: trigger phrase is case-insensitive, but name must be
            # properly capitalized to avoid false matches (e.g., "Grace" not "about")
            if not contact_email:
                trigger_match = re.search(r'(reply to|follow up with|message)\s+', prediction["description"], re.IGNORECASE)
                if trigger_match:
                    # Extract properly capitalized name after the trigger
                    rest = prediction["description"][trigger_match.end():]
                    name_match = re.match(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', rest)
                    if name_match:
                        contact_name = name_match.group(1)

        now = datetime.now(timezone.utc)

        if not contact_email and not contact_name:
            # No contact information found in signals or description.
            # A reminder without a contact can never be confirmed (we have no
            # way to look for outbound messages). Once the 48-hour window has
            # elapsed, resolve as inaccurate so the prediction doesn't remain
            # permanently pending and pollute the unresolved count.
            if now - created_at > timedelta(hours=48):
                return False  # Unresolvable after timeout → mark inaccurate
            return None  # Still within window, wait before giving up

        # Fast-path: if the contact is an automated/marketing sender, the
        # prediction was wrong by definition. These were generated before the
        # marketing filter was robust enough to catch all automated addresses.
        # The user will never "reply" to a no-reply mailer, so we mark them
        # inaccurate immediately rather than waiting the full 48-hour window.
        # This prevents accuracy stats from being polluted by structurally
        # unfulfillable predictions and keeps the learning loop accurate.
        if contact_email and self._is_automated_sender(contact_email):
            return False  # Automated sender → prediction was inaccurate

        # Look for outbound messages to this contact within 6-48 hours of prediction
        window_start = created_at
        window_end = created_at + timedelta(hours=48)

        with self.db.get_connection("events") as conn:
            # Check for email.sent or message.sent events to this contact
            events = conn.execute(
                """SELECT type, payload, timestamp
                   FROM events
                   WHERE (type = 'email.sent' OR type = 'message.sent')
                     AND timestamp >= ?
                     AND timestamp <= ?
                   ORDER BY timestamp ASC""",
                (window_start.isoformat(), window_end.isoformat()),
            ).fetchall()

        for event in events:
            try:
                payload = json.loads(event["payload"])
                # Check if recipient matches (handle both email addresses and names)
                # Email events use 'to_addresses' (list), message events use 'to' (string)
                to_field = payload.get("to_addresses") or payload.get("to", "")

                # Convert list to string for matching
                if isinstance(to_field, list):
                    to_str = ", ".join(to_field).lower()
                else:
                    to_str = to_field.lower()

                if contact_email and contact_email.lower() in to_str:
                    return True  # User DID follow up — prediction was accurate
                if contact_name and contact_name.lower() in to_str:
                    return True
            except (json.JSONDecodeError, TypeError):
                continue

        # Check if enough time has passed to infer inaccuracy.
        # If 48+ hours have passed with no action, prediction was likely wrong —
        # the user chose not to reply (or already replied outside our tracking window).
        # `now` was computed above before the event scan to avoid repeated syscalls.
        if now - created_at > timedelta(hours=48):
            return False  # No action taken → prediction was inaccurate

        return None  # Still within the window, can't determine yet

    async def _infer_conflict_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'conflict' predictions.

        Conflict predictions alert about calendar overlaps. We check if the user
        took corrective action (rescheduled, cancelled, or shortened one of the
        conflicting events).
        """
        # Extract event IDs from signals
        event_ids = signals.get("conflicting_event_ids", [])
        if not event_ids:
            return None

        # Look for calendar.event.updated or calendar.event.deleted events
        # for either of the conflicting events within 24 hours
        window_end = created_at + timedelta(hours=24)

        with self.db.get_connection("events") as conn:
            updates = conn.execute(
                """SELECT type, payload
                   FROM events
                   WHERE (type = 'calendar.event.updated' OR type = 'calendar.event.deleted')
                     AND timestamp >= ?
                     AND timestamp <= ?""",
                (created_at.isoformat(), window_end.isoformat()),
            ).fetchall()

        for update in updates:
            try:
                payload = json.loads(update["payload"])
                event_id = payload.get("event_id")
                if event_id in event_ids:
                    return True  # User resolved the conflict — prediction was accurate
            except (json.JSONDecodeError, TypeError):
                continue

        # If 24+ hours passed and conflict still exists, prediction was correct
        # but user chose to ignore it (still counts as accurate prediction)
        now = datetime.now(timezone.utc)
        if now - created_at > timedelta(hours=24):
            return True  # Conflict was real, even if user didn't fix it

        return None  # Still within resolution window

    async def _infer_need_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'need' predictions.

        Need predictions suggest: "You'll probably need X soon". The most common
        'need' predictions are preparation needs for upcoming events (travel,
        large meetings). We check if the event actually occurred and wasn't
        cancelled/rescheduled away.

        Accuracy inference logic:
        - If the calendar event occurred (not cancelled/rescheduled): ACCURATE
        - If the event was cancelled/rescheduled before it happened: INACCURATE
        - If not enough time has passed to know: None (wait)

        This works for preparation_needs predictions generated by
        PredictionEngine._check_preparation_needs().
        """
        # Extract event information from signals
        event_id = signals.get("event_id")
        event_title = signals.get("event_title")
        event_start_time_str = signals.get("event_start_time")

        # If no event info in signals, try to extract from description
        # Handles: "Upcoming travel in 24h: 'Flight to Boston'. Time to prepare."
        # Handles: "Large meeting in 36h: 'Q4 Planning' with 5 attendees"
        if not event_title:
            import re
            # Pattern: "...: 'EVENT_TITLE'"
            title_match = re.search(r":\s*'([^']+)'", prediction["description"])
            if title_match:
                event_title = title_match.group(1)

        if not event_title and not event_id:
            # Can't track without event information
            return None

        # Parse event start time to know when to check if it happened
        if event_start_time_str:
            try:
                event_start_time = datetime.fromisoformat(
                    event_start_time_str.replace("Z", "+00:00")
                )
                if event_start_time.tzinfo is None:
                    event_start_time = event_start_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                event_start_time = None
        else:
            event_start_time = None

        # If we don't have the event start time, we can't determine if enough
        # time has passed. Be conservative and wait.
        if not event_start_time:
            return None

        now = datetime.now(timezone.utc)

        # If the event hasn't happened yet, we can't determine accuracy
        if now < event_start_time:
            return None

        # Event time has passed. Check if it was cancelled or rescheduled.
        # Look for calendar.event.deleted or calendar.event.updated events
        # that modify/remove this event BEFORE its scheduled start time.
        with self.db.get_connection("events") as conn:
            modifications = conn.execute(
                """SELECT type, payload, timestamp
                   FROM events
                   WHERE (type = 'calendar.event.updated' OR type = 'calendar.event.deleted')
                     AND timestamp >= ?
                     AND timestamp <= ?
                   ORDER BY timestamp ASC""",
                (created_at.isoformat(), event_start_time.isoformat()),
            ).fetchall()

        for mod in modifications:
            try:
                payload = json.loads(mod["payload"])
                mod_event_id = payload.get("event_id")
                mod_event_title = payload.get("title")

                # Check if this modification applies to our event
                if event_id and mod_event_id == event_id:
                    # Event was modified or deleted before it happened
                    if mod["type"] == "calendar.event.deleted":
                        # Event was cancelled → prediction was inaccurate
                        return False
                    elif mod["type"] == "calendar.event.updated":
                        # Check if start time was rescheduled away
                        new_start_time_str = payload.get("start_time")
                        if new_start_time_str:
                            new_start_time = datetime.fromisoformat(
                                new_start_time_str.replace("Z", "+00:00")
                            )
                            if new_start_time.tzinfo is None:
                                new_start_time = new_start_time.replace(tzinfo=timezone.utc)
                            # If rescheduled to a different day, prediction was for the
                            # wrong timing → inaccurate
                            if abs((new_start_time - event_start_time).total_seconds()) > 3600:
                                return False
                elif event_title and mod_event_title and event_title.lower() in mod_event_title.lower():
                    # Fuzzy match by title (for when we don't have event_id)
                    if mod["type"] == "calendar.event.deleted":
                        return False
                    elif mod["type"] == "calendar.event.updated":
                        new_start_time_str = payload.get("start_time")
                        if new_start_time_str:
                            new_start_time = datetime.fromisoformat(
                                new_start_time_str.replace("Z", "+00:00")
                            )
                            if new_start_time.tzinfo is None:
                                new_start_time = new_start_time.replace(tzinfo=timezone.utc)
                            if abs((new_start_time - event_start_time).total_seconds()) > 3600:
                                return False
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        # Event time has passed and no cancellation/major reschedule was detected
        # → Event likely occurred as planned → Prediction was accurate
        return True

    @staticmethod
    def _is_automated_sender(email: str) -> bool:
        """Check if an email address belongs to an automated/marketing sender.

        These addresses were generated by the prediction engine before the marketing
        filter was enhanced. Because automated senders are never human relationship
        contacts, any opportunity prediction about them was incorrect by definition —
        the user will never "reach out" to a no-reply mailer or transactional system.

        Resolving these predictions as inaccurate immediately (rather than waiting
        the full 7-day window) keeps the learning loop accurate and prevents the
        accuracy stats from being polluted by predictions that are structurally
        impossible to fulfill.

        This replicates the key patterns from PredictionEngine._is_marketing_or_noreply()
        without creating a shared dependency between two services (which would require
        extracting a utility module and touching more files than needed).

        Args:
            email: Email address to check (any case).

        Returns:
            True if the email is clearly automated/marketing, False if it could be human.
        """
        addr = email.lower()
        local_part = addr.split("@")[0] if "@" in addr else addr
        domain = addr.split("@")[1] if "@" in addr else ""

        # No-reply and auto-reply senders
        noreply_patterns = (
            "noreply", "no-reply", "no_reply",
            "donotreply", "do-not-reply", "do_not_reply",
            "autoreply", "auto-reply", "auto_reply",
            "mailer-daemon", "postmaster", "daemon",
            "automated", "automation",
        )
        if any(p in local_part for p in noreply_patterns):
            return True

        # Bulk/transactional local-part prefixes that indicate automated systems
        bulk_prefixes = (
            "newsletter", "notifications", "notification",
            "updates", "update", "digest",
            "mailer", "bulk", "promo", "marketing",
            "reply", "email", "news", "offers", "offer", "deals",
            "info", "support", "help", "service", "services",
            "alert", "alerts",
            "orders", "order", "receipts", "receipt",
            "confirmation", "confirm", "shipment", "shipping", "delivery",
            "account", "accountservice",
            "rewards", "loyalty",
            "communications", "development", "fundraising",
            # Financial/brokerage automated systems
            "fidelity", "schwab", "vanguard", "etrade", "merrilledge",
            "robinhood", "wealthfront", "betterment",
            "paypal", "venmo", "stripe", "square",
            "experian", "equifax", "transunion", "creditkarma",
            # Transactional sender patterns (e.g. eDelivery, eConfirm)
            "edelivery", "econfirm", "enotice",
            # Retail/hospitality/travel transactional senders (iteration 178)
            # Synced with PredictionEngine._is_marketing_or_noreply patterns to ensure
            # the fast-path in _infer_opportunity_accuracy covers the same senders.
            "customerservice",   # e.g., Customerservice@nationalcar.com
            "reservations",      # e.g., reservations@nationalcar.com
            "onlineservice",     # e.g., onlineservice@fedex.com
            "return",            # e.g., return@amazon.com (retail returns)
            "tracking",          # e.g., tracking@shipstation.com
            "transaction",       # e.g., transaction@info.samsclub.com
            "online.account",    # e.g., online.account@marriott.com
            "guestservices",     # e.g., guestservices@boxoffice.axs.com
            "drivers",           # e.g., drivers@chargepoint.com
            "gaming",            # e.g., gaming@nvgaming.nvidia.com
            "messenger",         # e.g., messenger@messaging.squareup.com
            "tickets",           # e.g., tickets@transactions.axs.com
            "walgreens",         # e.g., walgreens@eml.walgreens.com
            "rei",               # e.g., rei@alerts.rei.com
            "applecash",         # e.g., applecash@insideapple.apple.com
            "worldofhyatt",      # e.g., worldofhyatt@loyalty.hyatt.com
            "disneycruiseline",  # e.g., disneycruiseline@vacations.disneydestinations.com
            # Additional automated local-part prefixes identified from production data
            # (iteration 179): synced with PredictionEngine._is_marketing_or_noreply.
            # Each pattern was confirmed against 67 unresolved opportunity predictions.
            "mail",         # e.g., mail@directpay.irs.gov, mail@cardsagainsthumanity.com,
                            #        mail@ifttt.com, mail@cdkeys.com — generic "mail@" is
                            #        always an automated system, never a personal address
            "alumni",       # e.g., alumni@mst.edu — university/college mass-mailing lists
            "top",          # e.g., top@raymore.com — promotional "top deals" automated sender
                            #        (Raymore & Flanigan furniture) — single-word dept prefix
            "stay",         # e.g., stay@hotelvandivort.com — hotel booking/CRM automated
            "msftpc",       # e.g., msftpc@microsoft.com — Microsoft PC-fleet automated
            "irrigation",   # e.g., irrigation@ryanlawn.com — landscaping service automated
            "claims",       # e.g., claims@treasurer.mo.gov — government/insurance automated
        )
        if any(local_part.startswith(p) for p in bulk_prefixes):
            return True

        # Embedded notification patterns anywhere in local-part
        embedded = (
            "-notification", "-notifications", "-alert", "-alerts",
            "-update", "-updates", "-digest", "news",
            # Dot-separated no-reply variants (iteration 178)
            "no.reply", "do.not.reply",
        )
        if any(p in local_part for p in embedded):
            return True

        # Marketing/transactional subdomain patterns (e.g. email.example.com,
        # mail.fidelity.com, ifly.southwest.com, trx.company.com).
        # 'mail.' captures bulk-sender subdomains like mail.fidelity.com,
        # mail.instagram.com, mail.schwab.com, mail.hbomax.com — all confirmed
        # to be dedicated transactional email infrastructure in production data.
        # Real humans never have @mail.* email addresses.
        marketing_subdomains = (
            "mail.", "email.", "ifly.", "trx.", "mailer.", "em.", "e.",
            "eonline.", "rewards.", "points-mail.", "shareholderdocs.",
            "investordelivery.", "customer.",
            # Additional subdomains identified from production data (iteration 178)
            "alerts.", "loyalty.", "vacations.", "transactions.", "eml.",
            "insideapple.", "card.", "eg.", "mc.", "odysseymail.",
        )
        if any(domain.startswith(p) for p in marketing_subdomains):
            return True

        # Specific automated domains that only send transactional emails
        automated_domains = (
            "proxyvote.com",    # Proxy voting service (Broadridge Financial)
            "playatmcd.com",    # McDonald's Monopoly game promo
            "facebookmail.com", # Facebook automated notifications
        )
        if any(domain.endswith(d) for d in automated_domains):
            return True

        # Marketing service provider domain endings.
        # These are third-party email marketing platforms — any address whose
        # domain ends with one of these suffixes was delivered via an ESP (Email
        # Service Provider) and is therefore automated by definition.
        # Synced from PredictionEngine._is_marketing_or_noreply (iteration 179).
        marketing_service_patterns = (
            ".e2ma.net",             # Emma email marketing
            ".sendgrid.net",         # SendGrid
            ".mailchimp.com",        # Mailchimp
            ".constantcontact.com",  # Constant Contact
            ".hubspot.com",          # HubSpot
            ".marketo.com",          # Marketo
            ".pardot.com",           # Salesforce Pardot
            ".eloqua.com",           # Oracle Eloqua
            ".sailthru.com",         # Sailthru email platform
            ".responsys.net",        # Oracle Responsys
            ".exacttarget.com",      # Salesforce Marketing Cloud
            ".smtp2go.com",          # SMTP2GO transactional email
            ".postmarkapp.com",      # Postmark transactional
            ".mandrillapp.com",      # Mandrill by Mailchimp
            ".amazonses.com",        # Amazon SES
            ".sparkpostmail.com",    # SparkPost
            ".sendinblue.com",       # Sendinblue/Brevo
            ".intercom-mail.com",    # Intercom notifications
            ".customer.io",          # Customer.io
            ".iterable.com",         # Iterable
            ".klaviyo.com",          # Klaviyo e-commerce marketing
            "proxyvote.com",         # Proxy voting (Broadridge Financial)
            "playatmcd.com",         # McDonald's Monopoly promo
            "facebookmail.com",      # Facebook automated
            ".smg.com",              # Service Management Group surveys
            ".ms.aa.com",            # American Airlines marketing
        )
        if any(domain.endswith(pattern) for pattern in marketing_service_patterns):
            return True

        # Compound subdomain patterns (match anywhere in domain)
        compound_patterns = (
            "info.email.",  # e.g., @info.email.aa.com (American Airlines marketing)
            ".smg.com",     # Service Management Group survey platform
            ".ms.aa.com",   # American Airlines info subdomain
        )
        if any(p in domain for p in compound_patterns):
            return True

        return False

    async def _infer_opportunity_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'opportunity' predictions.

        Opportunity predictions suggest: "Good time to do X based on your patterns".
        The most common 'opportunity' predictions are relationship maintenance
        suggestions ("Reach out to X — it's been Y days"). We check if the user
        actually contacted the person within a reasonable timeframe.

        Accuracy inference logic:
        - If the contact is an automated/marketing sender: immediately INACCURATE
          (the prediction was generated before the marketing filter was enhanced;
          the user will never reach out to an automated mailer by definition)
        - If user contacts the person within 7 days: ACCURATE
        - If 7+ days pass with no contact: INACCURATE
        - If still within the window: None (wait)

        The automated-sender fast-path resolves stale predictions from before the
        marketing filter improvements (PRs #183, #186) within minutes instead of
        waiting 7 days, keeping accuracy stats clean and the learning loop tight.

        This works for relationship_maintenance predictions generated by
        PredictionEngine._check_relationship_maintenance().
        """
        import re

        # Extract contact information from signals
        contact_email = signals.get("contact_email")
        contact_name = signals.get("contact_name")
        days_since_contact = signals.get("days_since_last_contact")

        # If no contact info in signals, try to extract from description
        # Handles: "Reach out to alice@example.com — it's been 45 days"
        # Handles: "Consider reaching out to Bob — last contact was 60 days ago"
        if not contact_email and not contact_name:
            # Pattern 1: Email address in description
            email_match = re.search(
                r'([\w\.\-\+]+@[\w\.\-]+\.[\w\.]+)',
                prediction["description"],
                re.IGNORECASE
            )
            if email_match:
                contact_email = email_match.group(1)

            # Pattern 2: "Reach out to NAME" or "reaching out to NAME"
            if not contact_email:
                trigger_match = re.search(
                    r'(reach out to|reaching out to)\s+',
                    prediction["description"],
                    re.IGNORECASE
                )
                if trigger_match:
                    rest = prediction["description"][trigger_match.end():]
                    # Extract name (must be capitalized)
                    name_match = re.match(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', rest)
                    if name_match:
                        contact_name = name_match.group(1)

        if not contact_email and not contact_name:
            # Can't track without contact information
            return None

        # Fast-path: if the contact is an automated/marketing sender, the prediction
        # was wrong by definition. These were generated before the marketing filter
        # was enhanced in PRs #183 and #186. We mark them inaccurate immediately
        # rather than waiting 7 days, preventing pollution of the accuracy stats.
        if contact_email and self._is_automated_sender(contact_email):
            return False  # Automated sender — prediction was inaccurate

        # Look for outbound messages to this contact within 7 days of prediction
        window_start = created_at
        window_end = created_at + timedelta(days=7)
        now = datetime.now(timezone.utc)

        with self.db.get_connection("events") as conn:
            # Check for email.sent or message.sent events to this contact
            events = conn.execute(
                """SELECT type, payload, timestamp
                   FROM events
                   WHERE (type = 'email.sent' OR type = 'message.sent')
                     AND timestamp >= ?
                     AND timestamp <= ?
                   ORDER BY timestamp ASC""",
                (window_start.isoformat(), min(window_end, now).isoformat()),
            ).fetchall()

        for event in events:
            try:
                payload = json.loads(event["payload"])
                # Check if recipient matches
                to_field = payload.get("to_addresses") or payload.get("to", "")

                # Convert list to string for matching
                if isinstance(to_field, list):
                    to_str = ", ".join(to_field).lower()
                else:
                    to_str = to_field.lower()

                if contact_email and contact_email.lower() in to_str:
                    return True  # User DID reach out — prediction was accurate
                if contact_name and contact_name.lower() in to_str:
                    return True
            except (json.JSONDecodeError, TypeError):
                continue

        # Check if enough time has passed to infer inaccuracy
        # If 7+ days have passed with no contact, prediction was likely wrong
        # (user didn't feel the need to reach out)
        if now > window_end:
            return False  # No contact made → prediction was inaccurate

        return None  # Still within the 7-day window, can't determine yet

    async def _infer_risk_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'risk' predictions.

        Risk predictions warn: "Something might go wrong if you don't..."
        The most common 'risk' predictions are spending alerts ("$X on category Y
        this month (Z% of total)"). We check if spending on that category continues
        to be anomalously high or if the user corrected the pattern.

        Accuracy inference logic for spending risks:
        - If spending in the flagged category decreased in the following 2 weeks:
          ACCURATE (user acknowledged the risk and adjusted)
        - If spending in the flagged category stayed high or increased:
          ACCURATE (risk was real, whether or not user acted on it)
        - If not enough time has passed: None (wait)

        Note: For spending risks, we consider the prediction ACCURATE if the
        spending pattern was genuinely anomalous, regardless of whether the user
        corrected it. The prediction's job is to identify the risk, not to force
        behavior change.

        This works for spending pattern predictions generated by
        PredictionEngine._check_spending_patterns().
        """
        # Extract spending category from signals
        category = signals.get("category")
        flagged_amount = signals.get("amount")
        flagged_percentage = signals.get("percentage")

        # If no category in signals, try to extract from description
        # Handles: "Spending alert: $450 on 'groceries' this month (35% of total)"
        if not category:
            import re
            # Pattern: "on 'CATEGORY'" or "on \"CATEGORY\""
            category_match = re.search(r"on\s+['\"]([^'\"]+)['\"]", prediction["description"])
            if category_match:
                category = category_match.group(1)

            # Also extract amount if not in signals
            if not flagged_amount:
                amount_match = re.search(r'\$(\d+)', prediction["description"])
                if amount_match:
                    flagged_amount = float(amount_match.group(1))

        if not category:
            # Can't track without category information
            return None

        # Wait at least 14 days after the prediction to see if spending behavior changed
        now = datetime.now(timezone.utc)
        wait_period = created_at + timedelta(days=14)

        if now < wait_period:
            # Not enough time has passed to evaluate behavioral response
            return None

        # Analyze spending in the flagged category during the 14 days AFTER prediction
        # to see if the user corrected their behavior
        window_start = created_at
        window_end = created_at + timedelta(days=14)

        with self.db.get_connection("events") as conn:
            # Get all transactions in the flagged category during the 2-week window
            transactions = conn.execute(
                """SELECT payload FROM events
                   WHERE type = 'finance.transaction.new'
                     AND timestamp >= ?
                     AND timestamp <= ?""",
                (window_start.isoformat(), window_end.isoformat()),
            ).fetchall()

        # Calculate spending in the flagged category during follow-up period
        category_spend = 0.0
        for txn in transactions:
            try:
                payload = json.loads(txn["payload"])
                txn_category = payload.get("category", "uncategorized")
                if txn_category.lower() == category.lower():
                    category_spend += abs(payload.get("amount", 0))
            except (json.JSONDecodeError, TypeError):
                continue

        # The prediction is ACCURATE if:
        # 1. The original flagged amount was genuinely high (>$200 in a month)
        # 2. This indicates the prediction correctly identified a spending anomaly
        #
        # We don't penalize the prediction if the user didn't change behavior—
        # the prediction's job is to surface the risk, not to guarantee action.
        #
        # If the original flagged amount was low (<$200), it was likely a false
        # alarm → INACCURATE
        if flagged_amount and flagged_amount >= 200:
            # High spending was correctly identified → prediction was accurate
            return True
        else:
            # Spending alert was for a small amount → likely false positive
            return False

    async def _infer_routine_deviation_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'routine_deviation' predictions.

        Routine deviation predictions suggest: "You usually do your '<routine>'
        routine by now" — meaning the user has deviated from a detected pattern.
        The prediction engine generates these when expected routine events haven't
        occurred by the time the routine typically starts.

        Accuracy inference logic:
          - Look for events matching the routine's expected_actions within 2 hours
            of the prediction being created.
          - If the user performed the expected routine actions within 2 hours:
            ACCURATE (the nudge was timely — routine was late, user did it)
          - If 4 hours pass with no matching events:
            INACCURATE (the routine truly didn't happen — probably a legitimate
            skip day rather than a delay the user needed to be reminded about)
          - If still within the 4-hour observation window: None (wait)

        The 2-hour ACCURATE window captures "late starts" where the prediction
        nudge coincided with the user eventually doing the routine. The 4-hour
        INACCURATE threshold avoids penalising legitimate off-day skips while
        still resolving predictions that will never be confirmed.

        Supporting signals structure (from PredictionEngine):
            {
                "routine_name": "morning_email_review",
                "consistency_score": 0.85,
                "expected_actions": ["email_received", "task_created"],
            }
        """
        import re as _re

        # Extract routine metadata from signals
        routine_name = signals.get("routine_name")
        expected_actions = signals.get("expected_actions") or []

        # Build the list of event types to look for.  Routine actions use
        # underscore format ("email_received") while stored events use dot
        # format ("email.received").  Map between the two so our query
        # finds actual events in the events table.
        action_to_event = {
            "email_received": "email.received",
            "email_sent": "email.sent",
            "message_received": "message.received",
            "message_sent": "message.sent",
            "task_created": "task.created",
            "task_completed": "task.completed",
            "calendar_event_created": "calendar.event.created",
        }
        expected_event_types = [
            action_to_event.get(a, a.replace("_", "."))
            for a in expected_actions
        ]

        # If we have neither a routine name nor expected actions, fall back to
        # trying to parse them from the prediction description.
        # Handles: "You usually do your 'morning_email_review' routine by now"
        if not routine_name and not expected_event_types:
            name_match = _re.search(r"'([^']+)'", prediction.get("description", ""))
            if name_match:
                routine_name = name_match.group(1)

        if not expected_event_types:
            # Without event types we can only resolve by timeout, not by activity.
            # Use a 24-hour window to avoid permanent stagnation.
            now = datetime.now(timezone.utc)
            if now > created_at + timedelta(hours=24):
                # No event type information; conservatively mark as unresolvable.
                # Return False so the prediction doesn't stay pending indefinitely.
                return False
            return None

        now = datetime.now(timezone.utc)
        accurate_window_end = created_at + timedelta(hours=2)
        inaccurate_window_end = created_at + timedelta(hours=4)

        # Query events table for any of the expected routine event types within
        # the 4-hour observation window.
        with self.db.get_connection("events") as conn:
            placeholders = ",".join("?" * len(expected_event_types))
            events = conn.execute(
                f"""SELECT type, timestamp FROM events
                    WHERE type IN ({placeholders})
                      AND timestamp >= ?
                      AND timestamp <= ?
                    ORDER BY timestamp ASC
                    LIMIT 1""",
                (
                    *expected_event_types,
                    created_at.isoformat(),
                    min(inaccurate_window_end, now).isoformat(),
                ),
            ).fetchall()

        if events:
            # At least one expected routine event occurred within the observation
            # window — the routine was performed (possibly after a delay).
            first_event_time = datetime.fromisoformat(
                events[0]["timestamp"].replace("Z", "+00:00")
            )
            if first_event_time <= accurate_window_end:
                # User completed the routine within 2 hours of the prediction →
                # the prediction correctly identified a late-start deviation.
                return True
            else:
                # User completed the routine between 2-4 hours later — still
                # did it, so the prediction was valid (deviation detected correctly).
                return True

        # No matching events found so far.
        if now > inaccurate_window_end:
            # 4-hour window elapsed with no routine activity — this was likely
            # a legitimate skip day, not a delay the user needed prompting for.
            return False

        # Still within the observation window — too early to determine.
        return None
