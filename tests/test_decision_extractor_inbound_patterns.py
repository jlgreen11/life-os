"""
Tests for DecisionExtractor — inbound email and calendar commitment signal patterns.

Verifies that the new broad-pattern detectors correctly produce signals for:
1. Urgency/priority language in received emails  (urgency_response)
2. Action-request language in received emails    (action_request)
3. Information-gathering patterns                 (information_gathering)
4. Calendar events with multiple attendees        (commitment)

Also verifies false-positive prevention:
- Plain emails with no qualifying language produce no new decision_signal
- Marketing / noreply senders are filtered before content inspection

These tests target the patterns added to bootstrap the decision profile from
the 13,500+ email.received events that previously fell through without signals.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from models.core import EventType
from services.signal_extractor.decision import DecisionExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_email_event(
    body: str = "",
    subject: str = "",
    from_address: str = "alice@example.com",
    event_type: str = EventType.EMAIL_RECEIVED.value,
    timestamp: datetime | None = None,
) -> dict:
    """Build a minimal EMAIL_RECEIVED event dict for testing."""
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()
    return {
        "type": event_type,
        "timestamp": ts,
        "payload": {
            "from_address": from_address,
            "subject": subject,
            "body": body,
        },
    }


def _make_calendar_event(
    attendees: list | None = None,
    summary: str = "Team meeting",
    start_offset_hours: int = 24,
    timestamp: datetime | None = None,
) -> dict:
    """Build a minimal CALENDAR_EVENT_CREATED event dict for testing."""
    now = timestamp or datetime.now(timezone.utc)
    from datetime import timedelta
    start_time = now + timedelta(hours=start_offset_hours)
    payload: dict = {
        "summary": summary,
        "start_time": start_time.isoformat(),
    }
    if attendees is not None:
        payload["attendees"] = attendees
    return {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": now.isoformat(),
        "payload": payload,
    }


def _decision_signals_of_type(signals: list[dict], decision_type: str) -> list[dict]:
    """Return signals with type='decision_signal' and the given decision_type."""
    return [
        s for s in signals
        if s.get("type") == "decision_signal" and s.get("decision_type") == decision_type
    ]


# ---------------------------------------------------------------------------
# Urgency / priority signal tests
# ---------------------------------------------------------------------------


def test_urgency_signal_from_urgent_subject(db, user_model_store):
    """Email with 'URGENT' in the subject should produce an urgency_response signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        subject="URGENT: deadline tomorrow for budget approval",
        body="Hi, we need your sign-off on the budget by end of day.",
    )

    signals = extractor.extract(event)

    urgency = _decision_signals_of_type(signals, "urgency_response")
    assert len(urgency) == 1
    assert urgency[0]["confidence"] == pytest.approx(0.6)  # subject-line match


def test_urgency_signal_from_asap_body(db, user_model_store):
    """Email with 'ASAP' in the body (but not subject) should produce lower-confidence signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        subject="Following up on the proposal",
        body="Can you please review and respond ASAP? We're waiting on your decision.",
    )

    signals = extractor.extract(event)

    urgency = _decision_signals_of_type(signals, "urgency_response")
    assert len(urgency) == 1
    assert urgency[0]["confidence"] == pytest.approx(0.4)  # body-only match


def test_urgency_signal_from_deadline_keyword(db, user_model_store):
    """'deadline' in the email subject should trigger urgency_response."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        subject="deadline for contract submission",
        body="Just a reminder that the contract needs to be signed this week.",
    )

    signals = extractor.extract(event)

    urgency = _decision_signals_of_type(signals, "urgency_response")
    assert len(urgency) == 1
    assert urgency[0]["event_type"] == EventType.EMAIL_RECEIVED.value


def test_urgency_signal_not_produced_for_marketing_sender(db, user_model_store):
    """Marketing senders are filtered before urgency detection."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        subject="URGENT: Limited time offer — 50% off!",
        body="This is a time-sensitive offer. Act now before the deadline!",
        from_address="noreply@newsletter.example.com",
    )

    signals = extractor.extract(event)

    urgency = _decision_signals_of_type(signals, "urgency_response")
    assert len(urgency) == 0


# ---------------------------------------------------------------------------
# Action-request signal tests
# ---------------------------------------------------------------------------


def test_action_request_signal_from_review_email(db, user_model_store):
    """Email asking user to 'please review' should produce an action_request signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        subject="Proposal for Q3 roadmap",
        body="Hi, please review the attached proposal and let us know your thoughts.",
    )

    signals = extractor.extract(event)

    action = _decision_signals_of_type(signals, "action_request")
    assert len(action) == 1
    assert action[0]["confidence"] == pytest.approx(0.4)


def test_action_request_signal_from_can_you_email(db, user_model_store):
    """'Can you' phrasing in an email body should produce an action_request signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        body="Can you take a look at this before Friday and give me your feedback?",
    )

    signals = extractor.extract(event)

    action = _decision_signals_of_type(signals, "action_request")
    assert len(action) == 1


def test_action_request_signal_from_need_your_approval(db, user_model_store):
    """'Need your approval' phrase should produce an action_request signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        body="We need your approval before we can proceed with the order.",
    )

    signals = extractor.extract(event)

    action = _decision_signals_of_type(signals, "action_request")
    assert len(action) == 1


def test_action_request_not_produced_for_noreply_sender(db, user_model_store):
    """noreply senders are filtered out before action-request detection."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        body="Please review and confirm your account settings.",
        from_address="noreply@service.example.com",
    )

    signals = extractor.extract(event)

    action = _decision_signals_of_type(signals, "action_request")
    assert len(action) == 0


# ---------------------------------------------------------------------------
# Information-gathering signal tests
# ---------------------------------------------------------------------------


def test_information_gathering_signal_from_re_thread(db, user_model_store):
    """Reply thread (subject starts with 'Re:') should produce an info-gathering signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        subject="Re: vendor selection for Q4",
        body="Thanks for reaching out. Here is the information you requested.",
    )

    signals = extractor.extract(event)

    info = _decision_signals_of_type(signals, "information_gathering")
    assert len(info) == 1


def test_information_gathering_signal_from_fwd_thread(db, user_model_store):
    """Forwarded email (subject starts with 'Fwd:') should produce an info-gathering signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        subject="Fwd: legal review comments on the contract",
        body="Forwarding this for your reference.",
    )

    signals = extractor.extract(event)

    info = _decision_signals_of_type(signals, "information_gathering")
    assert len(info) == 1


def test_information_gathering_signal_from_as_requested(db, user_model_store):
    """'As requested' in body should produce info-gathering with higher confidence."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        subject="Data analysis results",
        body="As requested, please find the data analysis results attached.",
    )

    signals = extractor.extract(event)

    info = _decision_signals_of_type(signals, "information_gathering")
    assert len(info) == 1
    # 'please find attached' + 'as requested' both match — confidence should be 0.5
    assert info[0]["confidence"] == pytest.approx(0.5)


def test_information_gathering_signal_from_please_find_attached(db, user_model_store):
    """'Please find attached' should produce an information_gathering signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        body="Please find attached the documents you asked for last week.",
    )

    signals = extractor.extract(event)

    info = _decision_signals_of_type(signals, "information_gathering")
    assert len(info) == 1


def test_re_thread_confidence_is_lower_than_info_delivery(db, user_model_store):
    """RE: prefix alone (no delivery phrases) gets lower confidence than info-delivery body."""
    extractor = DecisionExtractor(db, user_model_store)

    # Subject-only RE: with no delivery phrases in body
    event = _make_email_event(
        subject="Re: office hours next week",
        body="See you then!",
    )

    signals = extractor.extract(event)

    info = _decision_signals_of_type(signals, "information_gathering")
    assert len(info) == 1
    # Bare thread reply, no info-delivery phrases → lower confidence
    assert info[0]["confidence"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Calendar commitment signal tests
# ---------------------------------------------------------------------------


def test_commitment_signal_from_calendar_event_with_two_attendees(db, user_model_store):
    """Calendar event with 2+ attendees should produce a commitment decision signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_calendar_event(
        attendees=["alice@example.com", "bob@example.com"],
        summary="Project kickoff",
    )

    signals = extractor.extract(event)

    commitment = _decision_signals_of_type(signals, "commitment")
    assert len(commitment) == 1
    assert commitment[0]["confidence"] == pytest.approx(0.5)
    assert commitment[0]["attendee_count"] == 2
    assert commitment[0]["event_type"] == EventType.CALENDAR_EVENT_CREATED.value


def test_commitment_signal_from_calendar_event_with_many_attendees(db, user_model_store):
    """Calendar event with many attendees should still produce exactly one commitment signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_calendar_event(
        attendees=["a@ex.com", "b@ex.com", "c@ex.com", "d@ex.com"],
        summary="All-hands meeting",
    )

    signals = extractor.extract(event)

    commitment = _decision_signals_of_type(signals, "commitment")
    assert len(commitment) == 1
    assert commitment[0]["attendee_count"] == 4


def test_no_commitment_signal_for_calendar_event_without_attendees(db, user_model_store):
    """Calendar event with no attendees should NOT produce a commitment signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_calendar_event(attendees=[])

    signals = extractor.extract(event)

    commitment = _decision_signals_of_type(signals, "commitment")
    assert len(commitment) == 0


def test_no_commitment_signal_for_calendar_event_with_one_attendee(db, user_model_store):
    """Calendar event with only 1 attendee (solo block) should NOT produce a commitment signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_calendar_event(attendees=["self@example.com"])

    signals = extractor.extract(event)

    commitment = _decision_signals_of_type(signals, "commitment")
    assert len(commitment) == 0


# ---------------------------------------------------------------------------
# False-positive prevention: plain emails produce no decision_signal
# ---------------------------------------------------------------------------


def test_no_decision_signal_for_plain_newsletter_email(db, user_model_store):
    """Plain email with no action language should produce no decision_signal at all."""
    extractor = DecisionExtractor(db, user_model_store)

    # Regular human sender, completely generic content — no urgency, no action
    # requests, no RE:, no FW:, no info-delivery phrases.
    event = _make_email_event(
        subject="Hope you're doing well",
        body="Hey! Just checking in. Haven't heard from you in a while. How are things going?",
    )

    signals = extractor.extract(event)

    # No decision_signal of any type should be produced
    decision_signals = [s for s in signals if s.get("type") == "decision_signal"]
    assert len(decision_signals) == 0


def test_no_decision_signal_for_casual_update_email(db, user_model_store):
    """Casual informational email with no qualifying patterns produces no signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_email_event(
        subject="Team lunch next Thursday",
        body="Just letting everyone know lunch is at noon in the main conference room.",
    )

    signals = extractor.extract(event)

    decision_signals = [s for s in signals if s.get("type") == "decision_signal"]
    assert len(decision_signals) == 0


def test_marketing_sender_produces_no_decision_signal(db, user_model_store):
    """Marketing senders are filtered before ANY decision signal is produced."""
    extractor = DecisionExtractor(db, user_model_store)

    # Even though the body contains urgency + action + info-delivery phrases,
    # the noreply sender should block all three detectors.
    event = _make_email_event(
        subject="URGENT: Please review the attached proposal",
        body="As requested, please find attached the documents. ASAP action required.",
        from_address="noreply@promotions.example.com",
    )

    signals = extractor.extract(event)

    decision_signals = [s for s in signals if s.get("type") == "decision_signal"]
    assert len(decision_signals) == 0


# ---------------------------------------------------------------------------
# Profile update tests for decision_signal type
# ---------------------------------------------------------------------------


def test_profile_updated_with_decision_signal_counts(db, user_model_store):
    """_decision_signal_counts should be populated in the profile after signals."""
    extractor = DecisionExtractor(db, user_model_store)

    # Trigger an urgency signal
    event = _make_email_event(
        subject="urgent: contract needs signature",
        body="Please sign before EOD.",
    )
    extractor.extract(event)

    # Trigger an action_request signal
    event2 = _make_email_event(
        body="Could you please review the attached report and give feedback?",
    )
    extractor.extract(event2)

    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None
    counts = profile["data"].get("_decision_signal_counts", {})
    assert counts.get("urgency_response", 0) >= 1
    assert counts.get("action_request", 0) >= 1


def test_profile_updated_with_decision_signal_confidence(db, user_model_store):
    """decision_signal_confidence should reflect EMA of per-type confidence scores."""
    extractor = DecisionExtractor(db, user_model_store)

    # Two urgency signals: first is subject-line (0.6), second is body-only (0.4)
    extractor.extract(_make_email_event(
        subject="URGENT: please respond",
        body="We need your input immediately.",
    ))
    extractor.extract(_make_email_event(
        subject="Follow-up on the proposal",
        body="Please respond ASAP.",
    ))

    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None
    confidence = profile["data"].get("decision_signal_confidence", {})
    # After two observations: 0.6 then EMA(0.6, 0.4) = 0.7*0.6 + 0.3*0.4 = 0.54
    assert "urgency_response" in confidence
    assert 0.4 <= confidence["urgency_response"] <= 0.65
