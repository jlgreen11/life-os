"""
Life OS — Cadence Thread Metrics Test Suite

Tests for thread_completion_rate and avg_thread_length computation in
CadenceExtractor.  These metrics reveal whether the user follows through
on conversations (completion rate) and how deep their conversations go
(average thread length).
"""

from datetime import datetime, timedelta, timezone

import pytest

from services.signal_extractor.cadence import CadenceExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type, payload, source="test", ts=None):
    """Build a minimal event dict for CadenceExtractor."""
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    return {
        "id": f"evt-{id(payload)}",
        "type": event_type,
        "source": source,
        "timestamp": ts,
        "priority": "normal",
        "payload": payload,
        "metadata": {},
    }


def _old_ts(hours_ago=48):
    """Return an ISO timestamp N hours in the past (default 48h, well past the 24h cutoff)."""
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _recent_ts():
    """Return an ISO timestamp 1 hour ago (within the 24h exclusion window)."""
    return (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()


# ---------------------------------------------------------------------------
# Thread completion rate tests
# ---------------------------------------------------------------------------


def test_replied_threads_produce_nonzero_completion_rate(db, user_model_store):
    """Inbound then outbound on the same thread_id yields completion_rate > 0."""
    ext = CadenceExtractor(db, user_model_store)

    # Inbound email starts the thread
    ext.extract(_make_event("email.received", {
        "sender": "alice@example.com",
        "thread_id": "thread-A",
        "subject": "Hello",
        "body": "Hi there",
    }, ts=_old_ts(72)))

    # User replies on the same thread
    ext.extract(_make_event("email.sent", {
        "to_addresses": ["alice@example.com"],
        "thread_id": "thread-A",
        "subject": "Re: Hello",
        "body": "Hey!",
        "is_reply": True,
        "in_reply_to": "msg-1",
    }, ts=_old_ts(48)))

    profile = user_model_store.get_signal_profile("cadence")
    assert profile is not None
    data = profile["data"]
    assert data["thread_completion_rate"] == 1.0
    assert data["avg_thread_length"] == 2.0


def test_unreplied_threads_produce_zero_completion_rate(db, user_model_store):
    """Inbound-only threads (no user reply) yield completion_rate = 0."""
    ext = CadenceExtractor(db, user_model_store)

    for i in range(3):
        ext.extract(_make_event("email.received", {
            "sender": "bob@example.com",
            "thread_id": f"thread-{i}",
            "subject": f"Topic {i}",
            "body": f"Message {i}",
        }, ts=_old_ts(48 + i)))

    profile = user_model_store.get_signal_profile("cadence")
    data = profile["data"]
    assert data["thread_completion_rate"] == 0.0
    assert data["avg_thread_length"] == 1.0


def test_mixed_replied_and_unreplied_threads(db, user_model_store):
    """A mix of replied and unreplied threads gives the correct ratio."""
    ext = CadenceExtractor(db, user_model_store)

    # 2 threads with replies, 3 without → 2/5 = 0.4
    for i in range(5):
        ext.extract(_make_event("email.received", {
            "sender": f"contact{i}@example.com",
            "thread_id": f"mix-thread-{i}",
            "subject": f"Topic {i}",
            "body": f"Inbound {i}",
        }, ts=_old_ts(72 + i)))

    # Reply to threads 0 and 1 only
    for i in range(2):
        ext.extract(_make_event("email.sent", {
            "to_addresses": [f"contact{i}@example.com"],
            "thread_id": f"mix-thread-{i}",
            "subject": f"Re: Topic {i}",
            "body": f"Reply {i}",
            "is_reply": True,
            "in_reply_to": f"ref-{i}",
        }, ts=_old_ts(48 + i)))

    profile = user_model_store.get_signal_profile("cadence")
    data = profile["data"]
    assert abs(data["thread_completion_rate"] - 0.4) < 1e-9
    # Replied threads have 2 messages each, unreplied have 1 → (2+2+1+1+1)/5 = 1.4
    assert abs(data["avg_thread_length"] - 1.4) < 1e-9


# ---------------------------------------------------------------------------
# avg_thread_length tests
# ---------------------------------------------------------------------------


def test_avg_thread_length_multi_message_threads(db, user_model_store):
    """Threads with varying message counts produce the correct average length."""
    ext = CadenceExtractor(db, user_model_store)

    # Thread A: 4 messages (2 inbound, 2 outbound)
    for j in range(2):
        ext.extract(_make_event("email.received", {
            "sender": "alice@example.com",
            "thread_id": "length-A",
            "subject": "Discussion",
            "body": f"Inbound {j}",
        }, ts=_old_ts(72 - j)))
        ext.extract(_make_event("email.sent", {
            "to_addresses": ["alice@example.com"],
            "thread_id": "length-A",
            "subject": "Re: Discussion",
            "body": f"Outbound {j}",
            "is_reply": True,
            "in_reply_to": f"ref-{j}",
        }, ts=_old_ts(71 - j)))

    # Thread B: 1 message (inbound only)
    ext.extract(_make_event("email.received", {
        "sender": "bob@example.com",
        "thread_id": "length-B",
        "subject": "Quick note",
        "body": "Just one message",
    }, ts=_old_ts(48)))

    profile = user_model_store.get_signal_profile("cadence")
    data = profile["data"]
    # Thread A = 4 messages, Thread B = 1 message → avg = 2.5
    assert abs(data["avg_thread_length"] - 2.5) < 1e-9


# ---------------------------------------------------------------------------
# 24-hour exclusion window tests
# ---------------------------------------------------------------------------


def test_recent_threads_excluded_from_rate_calculation(db, user_model_store):
    """Threads younger than 24 hours should NOT affect thread_completion_rate."""
    ext = CadenceExtractor(db, user_model_store)

    # Old thread (eligible) — unreplied → rate should be 0.0
    ext.extract(_make_event("email.received", {
        "sender": "old@example.com",
        "thread_id": "old-thread",
        "subject": "Old topic",
        "body": "Old message",
    }, ts=_old_ts(48)))

    # Recent thread (excluded) — replied
    ext.extract(_make_event("email.received", {
        "sender": "new@example.com",
        "thread_id": "new-thread",
        "subject": "New topic",
        "body": "New message",
    }, ts=_recent_ts()))
    ext.extract(_make_event("email.sent", {
        "to_addresses": ["new@example.com"],
        "thread_id": "new-thread",
        "subject": "Re: New topic",
        "body": "Reply",
        "is_reply": True,
        "in_reply_to": "ref-new",
    }, ts=_recent_ts()))

    profile = user_model_store.get_signal_profile("cadence")
    data = profile["data"]
    # Only the old thread is eligible; it's unreplied → 0.0
    assert data["thread_completion_rate"] == 0.0


# ---------------------------------------------------------------------------
# Thread eviction tests
# ---------------------------------------------------------------------------


def test_thread_eviction_at_500_limit(db, user_model_store):
    """Exceeding 500 threads evicts the oldest by last_message_ts."""
    ext = CadenceExtractor(db, user_model_store)

    # Insert 502 threads with increasing timestamps
    for i in range(502):
        ext.extract(_make_event("email.received", {
            "sender": f"user{i}@example.com",
            "thread_id": f"evict-thread-{i:04d}",
            "subject": f"Topic {i}",
            "body": f"Message {i}",
        }, ts=_old_ts(600 - i)))

    profile = user_model_store.get_signal_profile("cadence")
    threads = profile["data"]["thread_tracking"]["threads"]
    assert len(threads) <= 500

    # The oldest threads (0 and 1, with the earliest timestamps) should be evicted.
    assert "evict-thread-0000" not in threads
    assert "evict-thread-0001" not in threads
    # The newest thread should still be present.
    assert "evict-thread-0501" in threads


# ---------------------------------------------------------------------------
# Thread key fallback tests
# ---------------------------------------------------------------------------


def test_thread_key_uses_subject_as_fallback(db, user_model_store):
    """When thread_id and in_reply_to are absent, subject is used as thread key."""
    ext = CadenceExtractor(db, user_model_store)

    ext.extract(_make_event("email.received", {
        "sender": "alice@example.com",
        "subject": "Budget Review Q1",
        "body": "Let's review the budget.",
    }, ts=_old_ts(48)))

    ext.extract(_make_event("email.sent", {
        "to_addresses": ["alice@example.com"],
        "subject": "Budget Review Q1",
        "body": "Looks good to me.",
    }, ts=_old_ts(47)))

    profile = user_model_store.get_signal_profile("cadence")
    threads = profile["data"]["thread_tracking"]["threads"]
    assert "Budget Review Q1" in threads
    assert threads["Budget Review Q1"]["message_count"] == 2
    assert threads["Budget Review Q1"]["has_user_reply"] is True


def test_thread_key_prefers_thread_id_over_subject(db, user_model_store):
    """thread_id takes priority over in_reply_to and subject."""
    ext = CadenceExtractor(db, user_model_store)

    ext.extract(_make_event("email.received", {
        "sender": "bob@example.com",
        "thread_id": "tid-123",
        "in_reply_to": "ref-456",
        "subject": "Some subject",
        "body": "Hello",
    }, ts=_old_ts(48)))

    profile = user_model_store.get_signal_profile("cadence")
    threads = profile["data"]["thread_tracking"]["threads"]
    assert "tid-123" in threads
    assert "ref-456" not in threads
    assert "Some subject" not in threads


# ---------------------------------------------------------------------------
# Message-type coverage
# ---------------------------------------------------------------------------


def test_message_events_also_tracked(db, user_model_store):
    """message.received and message.sent events produce thread activity signals."""
    ext = CadenceExtractor(db, user_model_store)

    ext.extract(_make_event("message.received", {
        "sender": "+1234567890",
        "thread_id": "sms-thread-1",
        "text": "Hey, are you free?",
    }, ts=_old_ts(48)))

    ext.extract(_make_event("message.sent", {
        "to_addresses": ["+1234567890"],
        "thread_id": "sms-thread-1",
        "text": "Yes!",
    }, ts=_old_ts(47)))

    profile = user_model_store.get_signal_profile("cadence")
    threads = profile["data"]["thread_tracking"]["threads"]
    assert "sms-thread-1" in threads
    assert threads["sms-thread-1"]["message_count"] == 2
    assert threads["sms-thread-1"]["has_user_reply"] is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_thread_key_produces_no_thread_signal(db, user_model_store):
    """Events without thread_id, in_reply_to, or subject produce no thread tracking."""
    ext = CadenceExtractor(db, user_model_store)

    ext.extract(_make_event("email.received", {
        "sender": "nope@example.com",
        "body": "No subject or thread info at all",
    }, ts=_old_ts(48)))

    profile = user_model_store.get_signal_profile("cadence")
    threads = profile["data"]["thread_tracking"]["threads"]
    assert len(threads) == 0


def test_empty_profile_has_no_thread_metrics(db, user_model_store):
    """When no threads exist, thread_completion_rate and avg_thread_length are not set."""
    ext = CadenceExtractor(db, user_model_store)

    # Process an event that has no thread key
    ext.extract(_make_event("email.received", {
        "sender": "x@example.com",
        "body": "Bare message",
    }, ts=_old_ts(48)))

    profile = user_model_store.get_signal_profile("cadence")
    data = profile["data"]
    # _compute_thread_metrics returns early with no threads, so these keys
    # should not be present (CadenceProfile defaults will apply).
    assert "thread_completion_rate" not in data
    assert "avg_thread_length" not in data
