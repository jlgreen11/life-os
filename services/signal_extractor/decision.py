"""
Life OS — Decision Signal Extractor

Tracks how the user makes decisions — speed, research depth, risk tolerance,
delegation patterns, and decision fatigue indicators.

This extractor builds a DecisionProfile by analyzing:
- Decision speed: Time from question/choice to action
- Research depth: Number of search queries or messages before deciding
- Delegation patterns: When they ask for opinions vs. decide alone
- Decision fatigue: Patterns of "whatever you decide" or increased delegation

The profile enables decision-aware predictions like:
- "You typically research tech purchases for 2-3 days before buying"
- "Decision fatigue detected (15+ decisions today), delegating this choice"
- "You always defer restaurant choices to your partner"
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

from models.core import EventType
from models.user_model import DecisionProfile
from services.signal_extractor.base import BaseExtractor
from services.signal_extractor.marketing_filter import is_marketing_or_noreply


class DecisionExtractor(BaseExtractor):
    """
    Tracks how the user makes decisions — speed, research depth, risk tolerance,
    delegation patterns, and decision fatigue indicators.

    Decision-making patterns are inferred from:
    - Task creation → completion (decision speed for action items)
    - Question patterns in messages → action taken (delegation vs. self-decision)
    - Search/research activity before purchases or commitments
    - Time-of-day patterns in decisive language vs. deferring language
    """

    def __init__(self, db, user_model_store):
        """Initialize DecisionExtractor.

        Extends BaseExtractor to add a write-attempt counter used by the
        defensive health check in _update_profile.  After the first successful
        persistence call we expect every subsequent call to also write
        successfully; if the profile disappears mid-stream a CRITICAL is logged
        immediately so operators are alerted.

        Args:
            db: DatabaseManager for raw event history.
            user_model_store: UserModelStore for signal profile persistence.
        """
        super().__init__(db, user_model_store)
        # Counts the number of times _update_profile has attempted to persist
        # the decision profile.  Used to distinguish "first write" (profile
        # doesn't exist yet, which is normal) from "Nth write" (profile should
        # exist, so a None result from get_signal_profile is a persistence
        # failure).
        self._profile_write_count = 0

    def can_process(self, event: dict) -> bool:
        """
        Process events that indicate decision-making activity.

        We track:
        - Task completion (decision execution)
        - Outbound messages with decision-related patterns
        - Calendar event creation (commitment decisions)
        - Email responses (decision communication)
        """
        return event.get("type") in [
            # Task events show decision execution
            EventType.TASK_COMPLETED.value,
            EventType.TASK_CREATED.value,
            # Outbound communication shows decision patterns
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_SENT.value,
            # Inbound communication shows decision response patterns
            # (approvals, rejections, input requests from contacts)
            EventType.EMAIL_RECEIVED.value,
            EventType.MESSAGE_RECEIVED.value,
            # Calendar commitments are decisions
            EventType.CALENDAR_EVENT_CREATED.value,
            # Calendar updates indicate decision revisions
            EventType.CALENDAR_EVENT_UPDATED.value,
            # Financial transactions are purchase decisions
            EventType.TRANSACTION_NEW.value,
        ]

    def extract(self, event: dict) -> list[dict]:
        """
        Extract decision-making signals from the event.

        Returns signal dicts and updates the DecisionProfile as a side-effect.
        """
        event_type = event.get("type", "")
        payload = event.get("payload", {})
        timestamp = event.get("timestamp", "")

        signals = []

        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            # Track task completion speed (decision → execution time)
            if event_type == EventType.TASK_COMPLETED.value:
                decision_speed_signal = self._track_task_decision_speed(payload, dt)
                if decision_speed_signal:
                    signals.append(decision_speed_signal)

            # Detect delegation patterns in outbound messages.
            # When no delegation is detected we still emit a synthetic
            # "outbound_message" signal so _update_profile can keep the
            # _total_outbound_count denominator accurate for delegation_comfort.
            if event_type in [EventType.EMAIL_SENT.value, EventType.MESSAGE_SENT.value]:
                delegation_signal = self._detect_delegation_patterns(payload, dt)
                if delegation_signal:
                    signals.append(delegation_signal)
                else:
                    # Non-delegation outbound message: counted for the ratio only
                    signals.append({
                        "type": "outbound_nondelegation",
                        "timestamp": dt.isoformat(),
                    })

            # Detect decision response patterns in inbound messages.
            # Stakeholders may approve/reject the user's proposals or seek
            # the user's input — both are decision-relevant signals.
            if event_type in [EventType.EMAIL_RECEIVED.value, EventType.MESSAGE_RECEIVED.value]:
                response_signal = self._detect_decision_response_patterns(payload, dt)
                if response_signal:
                    signals.append(response_signal)

            # Detect broader inbound-email decision signals: urgency language,
            # action requests, and information-gathering patterns.  These cover
            # the majority of email.received events that previously fell through
            # without any signal because they didn't match the narrow
            # approval/rejection/input_request patterns above.
            if event_type == EventType.EMAIL_RECEIVED.value:
                urgency_signal = self._detect_urgency_patterns(payload, dt, event_type)
                if urgency_signal:
                    signals.append(urgency_signal)

                action_signal = self._detect_action_request_patterns(payload, dt, event_type)
                if action_signal:
                    signals.append(action_signal)

                info_signal = self._detect_information_gathering_patterns(payload, dt, event_type)
                if info_signal:
                    signals.append(info_signal)

            # Fallback: inbound email/message with no specific pattern match emits
            # 'decision_information_processing' so every event produces a visible
            # signal for extraction quality diagnostics.
            # Marketing/noreply senders are excluded (same filter as the specific
            # pattern detectors above) to avoid inflating counts with automated mail.
            if event_type in [EventType.EMAIL_RECEIVED.value, EventType.MESSAGE_RECEIVED.value]:
                from_address = payload.get("from_address", "") or payload.get("from", "")
                if not is_marketing_or_noreply(from_address, payload):
                    has_specific = any(
                        s.get("type") in {"decision_response", "decision_signal"}
                        for s in signals
                    )
                    if not has_specific:
                        body = payload.get("body", "") or payload.get("body_plain", "") or ""
                        signals.append({
                            "type": "decision_information_processing",
                            "source_type": event_type,
                            "has_attachments": bool(payload.get("attachments")),
                            "word_count": len(body.split()) if body else 0,
                            "timestamp": dt.isoformat(),
                        })

            # Track calendar commitment speed (how far in advance they schedule)
            if event_type == EventType.CALENDAR_EVENT_CREATED.value:
                commitment_signal = self._track_commitment_patterns(payload, dt)
                if commitment_signal:
                    signals.append(commitment_signal)

                # Multi-attendee calendar events represent social commitments — a
                # distinct decision type separate from the planning-horizon metric.
                commitment_decision = self._detect_calendar_commitment_signal(payload, dt)
                if commitment_decision:
                    signals.append(commitment_decision)

            # Track calendar event updates as decision revisions
            if event_type == EventType.CALENDAR_EVENT_UPDATED.value:
                revision_signal = self._track_decision_revision(payload, dt)
                if revision_signal:
                    signals.append(revision_signal)

            # Track financial transactions as purchase decisions
            if event_type == EventType.TRANSACTION_NEW.value:
                purchase_signal = self._track_purchase_decision(payload, dt)
                if purchase_signal:
                    signals.append(purchase_signal)

            # Update the profile with aggregated signals (including the internal
            # outbound_nondelegation accounting token when present)
            self._update_profile(signals, dt)

        except Exception as e:
            # Fail-open: decision extraction should never block the pipeline
            logger.error("DecisionExtractor error: %s", e, exc_info=True)

        # Strip internal accounting tokens before returning; callers only need
        # semantically meaningful signals (delegation_pattern, decision_speed, etc.)
        return [s for s in signals if s.get("type") != "outbound_nondelegation"]

    def _track_task_decision_speed(self, payload: dict, completed_at: datetime) -> Optional[dict]:
        """
        Track how long from task creation to completion (decision → execution).

        This reveals decision speed by domain:
        - Quick tasks (<1 hour): impulsive/urgent
        - Same-day tasks (1-8 hours): normal pace
        - Multi-day tasks (>24 hours): deliberative
        """
        task_id = payload.get("task_id")
        if not task_id:
            return None

        # Query the database for task creation event
        query = """
            SELECT timestamp, json_extract(payload, '$.title') as title,
                   json_extract(payload, '$.domain') as domain
            FROM events
            WHERE type = ? AND json_extract(payload, '$.task_id') = ?
            ORDER BY timestamp ASC
            LIMIT 1
        """

        with self.db.get_connection("events") as conn:
            row = conn.execute(query, (EventType.TASK_CREATED.value, task_id)).fetchone()

        if not row:
            return None

        created_at = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
        title = row[1] or ""
        domain = row[2] or "general"

        # Calculate decision speed (seconds from creation to completion)
        decision_time_seconds = (completed_at - created_at).total_seconds()

        # Classify decision speed
        if decision_time_seconds < 3600:
            speed_category = "immediate"
        elif decision_time_seconds < 28800:  # 8 hours
            speed_category = "same_day"
        elif decision_time_seconds < 86400:  # 24 hours
            speed_category = "next_day"
        else:
            speed_category = "multi_day"

        return {
            "type": "decision_speed",
            "domain": domain,
            "decision_time_seconds": decision_time_seconds,
            "speed_category": speed_category,
            "task_title": title,
            "timestamp": completed_at.isoformat(),
        }

    def _detect_delegation_patterns(self, payload: dict, sent_at: datetime) -> Optional[dict]:
        """
        Detect delegation patterns in outbound messages.

        Looks for:
        - Asking for opinions: "what do you think?", "should I...?"
        - Deferring decisions: "you decide", "whatever you prefer"
        - Seeking consensus: "does that work for you?"
        """
        content = payload.get("content", "") or payload.get("body", "")
        if not content:
            return None

        content_lower = content.lower()

        # Delegation indicators
        delegation_patterns = [
            "you decide", "you choose", "you pick", "up to you",
            "whatever you prefer", "whatever you want", "your call",
            "i don't care", "doesn't matter to me", "either way",
        ]

        # Opinion-seeking patterns (soft delegation)
        opinion_patterns = [
            "what do you think", "what's your opinion", "should i",
            "would you", "do you think i should", "any thoughts",
            "what would you do", "advice", "suggestions",
        ]

        is_delegating = any(pattern in content_lower for pattern in delegation_patterns)
        is_seeking_opinion = any(pattern in content_lower for pattern in opinion_patterns)

        if not (is_delegating or is_seeking_opinion):
            return None

        # Extract recipient for per-contact delegation tracking
        recipient = payload.get("to", "") or payload.get("recipient", "")

        delegation_type = "full" if is_delegating else "seeking_input"

        return {
            "type": "delegation_pattern",
            "delegation_type": delegation_type,
            "recipient": recipient,
            "hour": sent_at.hour,
            "timestamp": sent_at.isoformat(),
        }

    def _detect_decision_response_patterns(self, payload: dict, received_at: datetime) -> Optional[dict]:
        """
        Detect decision response patterns in inbound messages.

        Analyzes received email/message content for signals about how others
        respond to the user's decisions or how they seek the user's input:
        - Approval patterns: stakeholder accepts/endorses the user's proposal
        - Rejection patterns: stakeholder pushes back on the user's proposal
        - Input-seeking patterns: contact defers the decision to the user

        Filters out marketing/automated senders to avoid learning decision
        patterns from newsletters or transactional emails.

        Args:
            payload: Event payload dict containing message body and sender info.
            received_at: Timestamp of message receipt.

        Returns:
            A signal dict with type='decision_response', or None if no
            decision-relevant pattern is found.
        """
        # Skip marketing and automated senders — they never carry genuine
        # decision responses; filtering prevents profile pollution.
        from_address = payload.get("from_address", "") or payload.get("from", "")
        if is_marketing_or_noreply(from_address, payload):
            return None

        # Use the same body-extraction pattern as linguistic.py (line 228).
        content = payload.get("body", "") or payload.get("body_plain", "")
        if not content:
            return None

        content_lower = content.lower()

        # Patterns indicating a stakeholder approved the user's decision/proposal.
        # Avoid generic words like "proceed" that can appear in rejection context
        # (e.g. "reconsider before proceeding") — use unambiguous approval phrases.
        approval_patterns = [
            "looks good", "sounds good", "that works", "i agree",
            "approved", "approve", "go ahead",
            "great idea", "love it", "perfect", "exactly right",
            "fully support", "on board", "let's do it", "let's go with",
            "i'm in", "works for me", "fine with me", "happy with that",
            "thumbs up", "yes, let's", "yes, please proceed",
            "please proceed", "go ahead and proceed",
        ]

        # Patterns indicating a stakeholder rejected or questioned the user's decision
        rejection_patterns = [
            "i disagree", "disagree", "i don't think", "not sure about",
            "concerns about", "concerned about", "reconsider", "think again",
            "not advisable", "bad idea", "won't work", "doesn't work",
            "problem with", "issue with", "pushback", "push back",
            "hold off", "wait on", "let's not", "i'd rather not",
            "not a good fit", "not the right time",
        ]

        # Patterns indicating the contact is seeking the user's decision/input
        input_seeking_patterns = [
            "what do you think", "your thoughts", "your opinion",
            "need your input", "need your feedback", "need your approval",
            "your call", "up to you", "your decision", "you decide",
            "what would you like", "what's your preference",
            "let me know what you think", "thoughts?", "opinion?",
            "can you weigh in", "what do you prefer",
        ]

        is_approval = any(pattern in content_lower for pattern in approval_patterns)
        is_rejection = any(pattern in content_lower for pattern in rejection_patterns)
        is_input_request = any(pattern in content_lower for pattern in input_seeking_patterns)

        if not (is_approval or is_rejection or is_input_request):
            return None

        # Determine the primary response type (approval takes precedence over
        # rejection; input_request is a separate decision-involvement signal).
        if is_input_request:
            response_type = "input_request"
        elif is_approval:
            response_type = "approval"
        else:
            response_type = "rejection"

        return {
            "type": "decision_response",
            "response_type": response_type,
            "from_contact": from_address,
            "hour": received_at.hour,
            "timestamp": received_at.isoformat(),
        }

    def _detect_urgency_patterns(self, payload: dict, dt: datetime, event_type: str) -> Optional[dict]:
        """
        Detect urgency/priority signals in inbound emails.

        Looks for time-pressure language in the subject or body that indicates
        the user may need to make a decision quickly.  Subject-line matches get
        a higher confidence score (0.6) than body-only matches (0.4) because
        senders who front-load urgency in the subject are more likely to be
        communicating a genuine time constraint.

        Filters out marketing and automated senders to avoid false positives
        from "URGENT: Limited time offer!" promotions.

        Args:
            payload:    Event payload dict with subject, body, and sender fields.
            dt:         Timestamp of the received message.
            event_type: Source event type string (always EMAIL_RECEIVED here).

        Returns:
            A signal dict with type='decision_signal' and
            decision_type='urgency_response', or None if no urgency found.
        """
        from_address = payload.get("from_address", "") or payload.get("from", "")
        if is_marketing_or_noreply(from_address, payload):
            return None

        subject = payload.get("subject", "") or ""
        body = payload.get("body", "") or payload.get("body_plain", "") or ""

        if not (subject or body):
            return None

        subject_lower = subject.lower()
        body_lower = body.lower()

        urgency_keywords = [
            "urgent", "asap", "as soon as possible", "deadline",
            "immediately", "time-sensitive", "time sensitive",
            "critical", "emergency", "rush", "overdue",
            "action required", "response required",
        ]

        subject_match = any(kw in subject_lower for kw in urgency_keywords)
        body_match = any(kw in body_lower for kw in urgency_keywords)

        if not (subject_match or body_match):
            return None

        # Subject-line urgency is a stronger signal than body-only urgency
        confidence = 0.6 if subject_match else 0.4

        return {
            "type": "decision_signal",
            "decision_type": "urgency_response",
            "confidence": confidence,
            "event_type": event_type,
            "timestamp": dt.isoformat(),
        }

    def _detect_action_request_patterns(self, payload: dict, dt: datetime, event_type: str) -> Optional[dict]:
        """
        Detect action-request patterns in inbound emails.

        Many inbound emails represent pending decisions: someone is asking the
        user to review a document, approve a request, or provide input.  These
        "please X" and "can you X" patterns indicate the user has a decision to
        make — even if they are not yet aware of it.

        Focuses on the email body since action requests are rarely expressed
        only in the subject line.  Filters out marketing senders.

        Args:
            payload:    Event payload dict with body and sender fields.
            dt:         Timestamp of the received message.
            event_type: Source event type string.

        Returns:
            A signal dict with type='decision_signal' and
            decision_type='action_request', or None if no request detected.
        """
        from_address = payload.get("from_address", "") or payload.get("from", "")
        if is_marketing_or_noreply(from_address, payload):
            return None

        body = payload.get("body", "") or payload.get("body_plain", "") or ""
        if not body:
            return None

        body_lower = body.lower()

        action_patterns = [
            "please review", "can you", "could you", "need your",
            "waiting for", "waiting on", "would you",
            "please confirm", "please respond", "please advise",
            "please let me know", "kindly review",
            "i need you to", "we need you to", "could you please",
            "please provide", "your input needed", "need your help",
            "need your approval", "need your feedback",
            "action required", "action needed",
        ]

        if not any(pattern in body_lower for pattern in action_patterns):
            return None

        return {
            "type": "decision_signal",
            "decision_type": "action_request",
            "confidence": 0.4,
            "event_type": event_type,
            "timestamp": dt.isoformat(),
        }

    def _detect_information_gathering_patterns(self, payload: dict, dt: datetime, event_type: str) -> Optional[dict]:
        """
        Detect information-gathering signals in inbound emails.

        When the user receives replies to their own questions (RE: threads),
        forwarded documents, or explicitly-requested information, they are in
        an active decision process.  Identifying these patterns lets the
        prediction engine surface relevant context when a decision is imminent.

        Two complementary checks:
        1. Subject starts with "re:", "fwd:", or "fw:" → thread reply
        2. Body contains information-delivery language ("as requested",
           "please find attached", etc.)

        Info-delivery body patterns get higher confidence (0.5) because they
        are more specific than a bare RE: prefix (0.3).  Filters out marketing
        senders before any content inspection.

        Args:
            payload:    Event payload dict with subject, body, and sender fields.
            dt:         Timestamp of the received message.
            event_type: Source event type string.

        Returns:
            A signal dict with type='decision_signal' and
            decision_type='information_gathering', or None if not detected.
        """
        from_address = payload.get("from_address", "") or payload.get("from", "")
        if is_marketing_or_noreply(from_address, payload):
            return None

        subject = payload.get("subject", "") or ""
        body = payload.get("body", "") or payload.get("body_plain", "") or ""

        subject_lower = subject.lower()
        body_lower = body.lower()

        # Thread-reply prefix is the most common form of info gathering
        is_thread_reply = subject_lower.startswith(("re:", "fwd:", "fw:"))

        # Explicit information-delivery phrases in the body
        info_delivery_patterns = [
            "as requested", "as discussed", "attached is", "attached are",
            "here is the", "here are the", "please find attached",
            "please find enclosed", "i've attached", "i have attached",
            "forwarding the", "fyi", "for your information",
            "per your request", "following up",
        ]
        has_info_delivery = any(p in body_lower for p in info_delivery_patterns)

        if not (is_thread_reply or has_info_delivery):
            return None

        # Explicit info delivery is a stronger signal than a thread-reply prefix
        confidence = 0.5 if has_info_delivery else 0.3

        return {
            "type": "decision_signal",
            "decision_type": "information_gathering",
            "confidence": confidence,
            "event_type": event_type,
            "timestamp": dt.isoformat(),
        }

    def _detect_calendar_commitment_signal(self, payload: dict, dt: datetime) -> Optional[dict]:
        """
        Detect social commitment decisions in calendar events with multiple attendees.

        A calendar event with two or more attendees is a social commitment — the
        user has (or will) negotiate a shared time slot with others, which is a
        distinct decision type from a solo planning-horizon signal.  This emits
        a 'decision_signal' so the profile tracks how often the user makes
        multi-party commitment decisions, separate from the planning-horizon
        metric captured by commitment_pattern signals.

        Args:
            payload: Calendar event payload with optional 'attendees' list.
            dt:      Timestamp of the event creation.

        Returns:
            A signal dict with type='decision_signal' and
            decision_type='commitment', or None if fewer than 2 attendees.
        """
        attendees = payload.get("attendees", [])

        # Require at least 2 attendees to indicate a genuine social commitment
        if not attendees or len(attendees) < 2:
            return None

        return {
            "type": "decision_signal",
            "decision_type": "commitment",
            "confidence": 0.5,
            "event_type": EventType.CALENDAR_EVENT_CREATED.value,
            "attendee_count": len(attendees),
            "timestamp": dt.isoformat(),
        }

    def _track_commitment_patterns(self, payload: dict, created_at: datetime) -> Optional[dict]:
        """
        Track calendar commitment patterns (planning horizon and spontaneity).

        Reveals risk tolerance and planning depth:
        - Same-day scheduling: spontaneous/reactive
        - Week-ahead scheduling: normal planning
        - Month+ scheduling: deliberative/cautious
        """
        start_time_str = payload.get("start_time", "")
        if not start_time_str:
            return None

        try:
            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            # Date-only strings (e.g. '2026-02-15' for all-day events) produce
            # naive datetimes with no tzinfo.  created_at is always timezone-aware,
            # so subtracting them would raise TypeError.  Treat date-only times as
            # midnight UTC so we can still compute a valid planning horizon.
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError, TypeError):
            return None

        # Calculate planning horizon (how far in advance they scheduled)
        planning_horizon_seconds = (start_time - created_at).total_seconds()

        # Ignore past events (likely imported or synced)
        if planning_horizon_seconds < 0:
            return None

        # Classify planning horizon
        if planning_horizon_seconds < 3600:  # < 1 hour
            horizon_category = "immediate"
        elif planning_horizon_seconds < 86400:  # < 1 day
            horizon_category = "same_day"
        elif planning_horizon_seconds < 604800:  # < 1 week
            horizon_category = "week_ahead"
        else:
            horizon_category = "long_term"

        summary = payload.get("summary", "") or payload.get("title", "")
        domain = self._classify_event_domain(summary)

        return {
            "type": "commitment_pattern",
            "domain": domain,
            "planning_horizon_seconds": planning_horizon_seconds,
            "horizon_category": horizon_category,
            "event_summary": summary,
            "timestamp": created_at.isoformat(),
        }

    def _track_purchase_decision(self, payload: dict, transaction_at: datetime) -> Optional[dict]:
        """
        Track financial transactions as purchase decisions.

        Reveals spending decision patterns:
        - Amount relative to historical average indicates risk tolerance
        - Merchant category maps to decision domains (dining, shopping, etc.)
        """
        amount = payload.get("amount")
        if amount is None:
            return None

        try:
            amount = float(amount)
        except (ValueError, TypeError):
            return None

        merchant = payload.get("merchant_name") or payload.get("description") or "unknown"
        domain = self._classify_merchant_domain(merchant)

        return {
            "type": "purchase_decision",
            "domain": domain,
            "amount": amount,
            "merchant": merchant,
            "timestamp": transaction_at.isoformat(),
        }

    def _track_decision_revision(self, payload: dict, updated_at: datetime) -> Optional[dict]:
        """
        Track calendar event updates as decision revisions.

        Changing a calendar event (time, location, cancellation) indicates
        the user reversed or refined a previous decision. Frequent revisions
        suggest lower confidence in initial decisions.
        """
        # Determine what kind of change was made
        changes = payload.get("changes", {})
        summary = payload.get("summary", "") or payload.get("title", "")

        # Classify the revision type from available change data
        revision_type = "general"
        if isinstance(changes, dict):
            if "start_time" in changes or "end_time" in changes or "time" in changes:
                revision_type = "time_change"
            elif "location" in changes:
                revision_type = "location_change"
            elif "status" in changes and changes.get("status") == "cancelled":
                revision_type = "cancellation"

        return {
            "type": "decision_revision",
            "revision_type": revision_type,
            "event_summary": summary,
            "timestamp": updated_at.isoformat(),
        }

    def _classify_merchant_domain(self, merchant: str) -> str:
        """
        Classify a merchant/transaction description into a decision domain.

        Maps common merchant categories to domains for risk tolerance tracking.
        """
        merchant_lower = merchant.lower()

        if any(word in merchant_lower for word in ["restaurant", "cafe", "coffee", "pizza", "food", "dining", "eat"]):
            return "dining"
        elif any(word in merchant_lower for word in ["amazon", "walmart", "target", "shop", "store", "buy"]):
            return "shopping"
        elif any(word in merchant_lower for word in ["netflix", "spotify", "subscription", "membership"]):
            return "subscriptions"
        elif any(word in merchant_lower for word in ["uber", "lyft", "gas", "fuel", "transit", "airline", "travel"]):
            return "transport"
        elif any(word in merchant_lower for word in ["rent", "mortgage", "insurance", "utility", "bill"]):
            return "housing"
        else:
            return "general"

    def _classify_event_domain(self, summary: str) -> str:
        """
        Classify calendar event into decision domain.

        Domains help track risk tolerance by area:
        - work: professional commitments
        - social: personal relationships
        - health: medical, fitness
        - finance: money-related
        """
        summary_lower = summary.lower()

        # Simple keyword-based classification
        # Check specific domains first (finance, health) before generic work terms
        if any(word in summary_lower for word in ["bank", "payment", "tax", "budget"]):
            return "finance"
        elif any(word in summary_lower for word in ["workout", "gym", "doctor", "dentist", "therapy"]):
            return "health"
        elif any(word in summary_lower for word in ["dinner", "lunch", "coffee", "hangout", "party"]):
            return "social"
        elif any(word in summary_lower for word in ["meeting", "standup", "call", "sync", "review"]):
            return "work"
        else:
            return "general"

    def _update_profile(self, signals: list[dict], timestamp: datetime):
        """
        Update the DecisionProfile with aggregated signals.

        Incrementally builds the profile by:
        1. Loading existing profile from the database
        2. Merging new signals into domain-specific metrics
        3. Persisting updated profile back to signal_profiles table
        """
        if not signals:
            return

        # Load existing profile
        profile_data = self.ums.get_signal_profile("decision")
        if profile_data and profile_data.get("data"):
            profile_dict = profile_data["data"]
        else:
            # Initialize empty profile
            profile_dict = DecisionProfile().model_dump()

        # Aggregate decision speed signals
        for signal in signals:
            if signal["type"] == "decision_speed":
                domain = signal["domain"]
                decision_time = signal["decision_time_seconds"]

                # Update decision_speed_by_domain with exponential moving average
                current_speeds = profile_dict.get("decision_speed_by_domain", {})
                if domain in current_speeds:
                    # EMA: 0.3 weight on new observation
                    current_speeds[domain] = 0.7 * current_speeds[domain] + 0.3 * decision_time
                else:
                    current_speeds[domain] = decision_time
                profile_dict["decision_speed_by_domain"] = current_speeds

            elif signal["type"] == "delegation_pattern":
                recipient = signal["recipient"]
                delegation_type = signal["delegation_type"]
                hour = signal["hour"]

                # ----------------------------------------------------------------
                # Update delegation_comfort (0=micromanage, 1=fully delegated).
                #
                # We maintain a running tally of:
                #   delegation_event_count  – outbound messages that delegate/seek input
                #   total_outbound_count    – all outbound messages processed so far
                #
                # Both counters are stored in the profile dict so they survive
                # across process restarts.  The comfort score is their ratio,
                # smoothed with an EMA (α=0.3) so that one burst of delegation
                # doesn't permanently spike the score.
                # ----------------------------------------------------------------
                prev_delegation_count = profile_dict.get("_delegation_event_count", 0)
                prev_total_count = profile_dict.get("_total_outbound_count", 0)

                prev_delegation_count += 1   # this signal IS a delegation event
                prev_total_count += 1        # it is also an outbound message
                profile_dict["_delegation_event_count"] = prev_delegation_count
                profile_dict["_total_outbound_count"] = prev_total_count

                # Instantaneous ratio for this update
                instant_ratio = prev_delegation_count / prev_total_count

                # EMA blend into the stored comfort score (α=0.3)
                current_comfort = profile_dict.get("delegation_comfort", 0.5)
                profile_dict["delegation_comfort"] = round(
                    0.7 * current_comfort + 0.3 * instant_ratio, 4
                )

                # ----------------------------------------------------------------
                # Update delegation_by_domain — per-recipient delegation tendency.
                #
                # The "domain" here is the normalized recipient identifier (email
                # address or contact name).  We store a per-recipient EMA so the
                # prediction engine can say "you always defer restaurant choices
                # to your partner."
                # ----------------------------------------------------------------
                if recipient:
                    current_by_domain = profile_dict.get("delegation_by_domain", {})
                    if recipient in current_by_domain:
                        # EMA: new delegation toward this recipient
                        current_by_domain[recipient] = round(
                            0.7 * current_by_domain[recipient] + 0.3 * 1.0, 4
                        )
                    else:
                        current_by_domain[recipient] = 1.0  # first delegation to them
                    profile_dict["delegation_by_domain"] = current_by_domain

                    # ----------------------------------------------------------------
                    # Update defers_to — domain-keyed list of recipients the user
                    # trusts with a given decision category.  We infer the decision
                    # category from the hour and delegation type:
                    #   - full delegation at evening hours → social/personal category
                    #   - seeking_input during work hours → work category
                    # ----------------------------------------------------------------
                    defers_to = profile_dict.get("defers_to", {})
                    if delegation_type == "full":
                        category = "personal" if hour >= 17 else "general"
                    else:
                        category = "work" if 8 <= hour < 17 else "general"

                    recipients_for_category = defers_to.get(category, [])
                    if recipient not in recipients_for_category:
                        recipients_for_category.append(recipient)
                    defers_to[category] = recipients_for_category
                    profile_dict["defers_to"] = defers_to

                # ----------------------------------------------------------------
                # Track decision fatigue by hour (late hours = more delegation).
                # We record the earliest hour at which delegation consistently
                # starts, which is a proxy for when decision fatigue sets in.
                # ----------------------------------------------------------------
                if hour >= 20:  # After 8pm
                    # Detect fatigue patterns
                    current_fatigue_hour = profile_dict.get("fatigue_time_of_day")
                    if current_fatigue_hour is None or hour < current_fatigue_hour:
                        profile_dict["fatigue_time_of_day"] = hour

            elif signal["type"] == "outbound_nondelegation":
                # ----------------------------------------------------------------
                # Non-delegation outbound message: increment the denominator so
                # that delegation_comfort reflects a true ratio (delegation events
                # / total outbound messages) rather than drifting toward 1.0.
                # The numerator (_delegation_event_count) is unchanged here.
                # ----------------------------------------------------------------
                prev_total_count = profile_dict.get("_total_outbound_count", 0)
                prev_total_count += 1
                profile_dict["_total_outbound_count"] = prev_total_count

                # Recompute comfort with the updated denominator
                delegation_count = profile_dict.get("_delegation_event_count", 0)
                instant_ratio = delegation_count / prev_total_count if prev_total_count > 0 else 0.0
                current_comfort = profile_dict.get("delegation_comfort", 0.5)
                profile_dict["delegation_comfort"] = round(
                    0.7 * current_comfort + 0.3 * instant_ratio, 4
                )

            elif signal["type"] == "commitment_pattern":
                domain = signal["domain"]
                planning_horizon = signal["planning_horizon_seconds"]

                # Short planning horizons = higher risk tolerance (spontaneity)
                # Long planning horizons = lower risk tolerance (cautious)
                # Map planning horizon to risk tolerance: shorter = higher risk
                risk_score = 1.0 - min(planning_horizon / 2592000, 1.0)  # 30 days = 0.0 risk

                current_risk = profile_dict.get("risk_tolerance_by_domain", {})
                if domain in current_risk:
                    # EMA: 0.3 weight on new observation
                    current_risk[domain] = 0.7 * current_risk[domain] + 0.3 * risk_score
                else:
                    current_risk[domain] = risk_score
                profile_dict["risk_tolerance_by_domain"] = current_risk

            elif signal["type"] == "purchase_decision":
                # ----------------------------------------------------------------
                # Update risk_tolerance_by_domain based on transaction amounts.
                # Higher amounts relative to a $100 baseline indicate higher risk
                # tolerance in the merchant's domain.
                # ----------------------------------------------------------------
                domain = signal["domain"]
                amount = signal["amount"]

                # Map amount to risk score: $0=0.0, $200+=1.0
                risk_score = min(amount / 200.0, 1.0)

                current_risk = profile_dict.get("risk_tolerance_by_domain", {})
                if domain in current_risk:
                    current_risk[domain] = round(0.7 * current_risk[domain] + 0.3 * risk_score, 4)
                else:
                    current_risk[domain] = round(risk_score, 4)
                profile_dict["risk_tolerance_by_domain"] = current_risk

            elif signal["type"] == "decision_revision":
                # ----------------------------------------------------------------
                # Increment mind_change_frequency using EMA.
                # Each revision nudges the frequency toward 1.0 (frequent changes).
                # The absence of revisions is handled implicitly: as more non-revision
                # events arrive, commitment_pattern signals dilute the score.
                # ----------------------------------------------------------------
                current_freq = profile_dict.get("mind_change_frequency", 0.1)
                profile_dict["mind_change_frequency"] = round(
                    0.7 * current_freq + 0.3 * 1.0, 4
                )

            elif signal["type"] == "decision_signal":
                # ----------------------------------------------------------------
                # Aggregate inbound-email and calendar decision signals.
                #
                # These signals are emitted by the new broad-pattern detectors
                # (urgency_response, action_request, information_gathering,
                # commitment).  We track two things:
                #
                # 1. _decision_signal_counts — raw event count per decision_type,
                #    stored in the profile dict.  The prediction engine uses this
                #    to bootstrap decision-pattern intelligence from historical email
                #    data that previously produced no signals at all.
                #
                # 2. decision_signal_confidence — per-type EMA (α=0.3) of
                #    confidence scores.  This lets the prediction engine weight
                #    high-confidence signals (e.g. subject-line "URGENT:") more
                #    heavily than low-confidence ones (bare RE: thread replies).
                # ----------------------------------------------------------------
                decision_type = signal.get("decision_type", "unknown")
                confidence = signal.get("confidence", 0.3)

                # Increment raw count for this decision type
                signal_counts = profile_dict.get("_decision_signal_counts", {})
                signal_counts[decision_type] = signal_counts.get(decision_type, 0) + 1
                profile_dict["_decision_signal_counts"] = signal_counts

                # EMA blend of per-type confidence scores (α=0.3)
                signal_confidence = profile_dict.get("decision_signal_confidence", {})
                if decision_type in signal_confidence:
                    signal_confidence[decision_type] = round(
                        0.7 * signal_confidence[decision_type] + 0.3 * confidence, 4
                    )
                else:
                    signal_confidence[decision_type] = round(confidence, 4)
                profile_dict["decision_signal_confidence"] = signal_confidence

            elif signal["type"] == "decision_response":
                # ----------------------------------------------------------------
                # Track stakeholder approval rate using EMA (α=0.3).
                #
                # approval   → nudge stakeholder_approval_rate toward 1.0
                # rejection  → nudge toward 0.0
                # input_request → counts as involvement, but doesn't change
                #   the approval/rejection ratio; it IS counted in
                #   _response_total_count so the denominator is accurate.
                #
                # Per-contact response counts let the prediction engine surface
                # insights like "Alice usually approves your proposals".
                # ----------------------------------------------------------------
                response_type = signal["response_type"]
                from_contact = signal.get("from_contact", "")

                # Update per-contact response count regardless of type.
                if from_contact:
                    contact_counts = profile_dict.get("_contact_response_counts", {})
                    contact_counts[from_contact] = contact_counts.get(from_contact, 0) + 1
                    profile_dict["_contact_response_counts"] = contact_counts

                # Track the approval/rejection ratio via EMA; skip input_request
                # since it carries no polarity on the user's prior decisions.
                if response_type in ("approval", "rejection"):
                    approval_value = 1.0 if response_type == "approval" else 0.0
                    current_rate = profile_dict.get("stakeholder_approval_rate", 0.5)
                    profile_dict["stakeholder_approval_rate"] = round(
                        0.7 * current_rate + 0.3 * approval_value, 4
                    )

                # Increment total response count for informational use.
                profile_dict["_response_total_count"] = (
                    profile_dict.get("_response_total_count", 0) + 1
                )

            elif signal["type"] == "decision_information_processing":
                # ----------------------------------------------------------------
                # Track fallback inbound events: events that didn't match any
                # specific decision pattern but still need to be counted for
                # extraction quality diagnostics.
                #
                # _inbound_processing_counts — per-source-type event count
                # ----------------------------------------------------------------
                source_type = signal.get("source_type", "unknown")
                inbound_counts = profile_dict.get("_inbound_processing_counts", {})
                inbound_counts[source_type] = inbound_counts.get(source_type, 0) + 1
                profile_dict["_inbound_processing_counts"] = inbound_counts

        # Update timestamp
        profile_dict["last_updated"] = timestamp.isoformat()

        # Persist the updated profile with defensive write pattern.
        # Every 10 writes, verify the profile row actually exists; if it is
        # missing, update_signal_profile() is silently failing and a CRITICAL
        # is logged so operators are alerted immediately.
        # A passive WAL checkpoint is issued after each write to flush
        # accumulated WAL frames to the main DB file, reducing data-loss
        # exposure between the store-level throttled checkpoints (every 50).
        self._profile_write_count += 1
        try:
            self.ums.update_signal_profile("decision", profile_dict)
        except Exception as e:
            logger.error(
                "decision_extractor: update_signal_profile raised on write #%d: %s",
                self._profile_write_count,
                e,
                exc_info=True,
            )
            return

        # Post-write verification every 10 writes (avoids per-event read
        # overhead on high-volume rebuilds while still catching silent failures).
        if self._profile_write_count > 1 and self._profile_write_count % 10 == 0:
            try:
                profile = self.ums.get_signal_profile("decision")
                if profile is None:
                    logger.critical(
                        "decision_extractor: profile MISSING after %d writes — "
                        "update_signal_profile() is silently failing to persist data. "
                        "Check user_model.db health and disk space.",
                        self._profile_write_count,
                    )
            except Exception as e:
                logger.warning(
                    "decision_extractor: post-write profile verification failed: %s", e
                )

        # Forced WAL checkpoint after each profile write.
        try:
            with self.ums.db.get_connection("user_model") as conn:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except Exception as e:
            logger.warning(
                "decision_extractor: WAL checkpoint after profile write failed: %s", e
            )
