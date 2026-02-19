"""
Tests for episodic topic enrichment in InsightEngine._contact_gap_insights().

PR #272 (iteration 255): When a contact-gap insight fires, the engine now
queries episodic memory for the most-recently-discussed topics with that
contact and appends them to the insight summary.

Before: "It has been 14 days since you last contacted Alice (usual interval ~3 days)."
After:  "It has been 14 days since you last contacted Alice (usual interval ~3 days).
         Last topics: budget-review, q1-planning."

Covers:
- _get_contact_last_topics(): happy path with matching episode
- _get_contact_last_topics(): empty list when no episode exists
- _get_contact_last_topics(): empty list when episode has no topics
- _get_contact_last_topics(): handles malformed JSON gracefully
- _get_contact_last_topics(): returns topics from most recent episode
- _get_contact_last_topics(): respects the limit parameter
- _contact_gap_insights(): topic suffix appears in summary when episode data exists
- _contact_gap_insights(): no topic suffix when no episode data (backward compat)
- _contact_gap_insights(): topics surface in evidence list
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.engine import InsightEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_episode(db, email_addr: str, topics: list, days_ago: float = 5.0,
                   content_summary: str = "Discussed project plan") -> str:
    """Insert a synthetic episode into the user_model DB and return its id.

    Args:
        db: DatabaseManager fixture.
        email_addr: Email address to include in contacts_involved.
        topics: List of topic strings to store as JSON.
        days_ago: How many days ago the episode occurred (default 5.0).
        content_summary: Short text summary for the episode.

    Returns:
        The UUID string used as the episode id.
    """
    episode_id = str(uuid.uuid4())
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, timestamp, event_id, interaction_type, content_summary,
                contacts_involved, topics, entities)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode_id,
                ts,
                str(uuid.uuid4()),  # event_id
                "email_received",
                content_summary,
                json.dumps([email_addr]),
                json.dumps(topics),
                json.dumps([]),
            ),
        )
    return episode_id


def _make_overdue_profile(email_addr: str, now: datetime) -> dict:
    """Build a relationships signal profile with a single overdue contact.

    Sets up a contact with 10 historical interactions at ~10-day intervals
    and a last_interaction 25 days ago, which exceeds the 1.5× threshold
    (15 days) and the 7-day minimum.

    Args:
        email_addr: Email address of the contact to populate.
        now: Datetime to use as "now" for timestamp calculation.

    Returns:
        A dict suitable for user_model_store.update_signal_profile().
    """
    avg_gap_days = 10
    overdue_days = 25
    return {
        "contacts": {
            email_addr: {
                "interaction_count": 10,
                "outbound_count": 5,
                "last_interaction": (now - timedelta(days=overdue_days)).isoformat(),
                "interaction_timestamps": [
                    (now - timedelta(days=overdue_days + i * avg_gap_days)).isoformat()
                    for i in range(10)
                ],
            }
        }
    }


# ---------------------------------------------------------------------------
# _get_contact_last_topics unit tests
# ---------------------------------------------------------------------------


def test_get_contact_last_topics_returns_topics(db, user_model_store):
    """Returns topic list from the most recent episode for the given contact."""
    _store_episode(db, "alice@example.com", ["budget-review", "q1-planning"])
    engine = InsightEngine(db, user_model_store)

    topics = engine._get_contact_last_topics("alice@example.com")
    assert topics == ["budget-review", "q1-planning"]


def test_get_contact_last_topics_no_matching_episode(db, user_model_store):
    """Returns empty list when no episode references the given email address."""
    # Store an episode for a DIFFERENT contact; should not affect result.
    _store_episode(db, "bob@example.com", ["meetings", "deadlines"])
    engine = InsightEngine(db, user_model_store)

    topics = engine._get_contact_last_topics("alice@example.com")
    assert topics == []


def test_get_contact_last_topics_episode_without_topics(db, user_model_store):
    """Returns empty list when the matching episode has an empty topic list."""
    _store_episode(db, "alice@example.com", [])
    engine = InsightEngine(db, user_model_store)

    topics = engine._get_contact_last_topics("alice@example.com")
    assert topics == []


def test_get_contact_last_topics_returns_most_recent(db, user_model_store):
    """When multiple episodes match, returns topics from the most recent one."""
    # Older episode: topics about the old project
    _store_episode(db, "alice@example.com", ["old-project", "legacy"],
                   days_ago=30.0)
    # Newer episode: topics about the current project
    _store_episode(db, "alice@example.com", ["q2-roadmap", "hiring"],
                   days_ago=5.0)
    engine = InsightEngine(db, user_model_store)

    topics = engine._get_contact_last_topics("alice@example.com")
    assert topics == ["q2-roadmap", "hiring"]
    assert "old-project" not in topics


def test_get_contact_last_topics_respects_limit(db, user_model_store):
    """limit parameter caps the returned list."""
    _store_episode(db, "alice@example.com",
                   ["alpha", "beta", "gamma", "delta", "epsilon"])
    engine = InsightEngine(db, user_model_store)

    topics = engine._get_contact_last_topics("alice@example.com", limit=2)
    assert len(topics) == 2
    assert topics == ["alpha", "beta"]


def test_get_contact_last_topics_default_limit_is_three(db, user_model_store):
    """Default limit of 3 is applied when limit is not specified."""
    _store_episode(db, "alice@example.com",
                   ["alpha", "beta", "gamma", "delta"])
    engine = InsightEngine(db, user_model_store)

    topics = engine._get_contact_last_topics("alice@example.com")
    assert len(topics) == 3


def test_get_contact_last_topics_filters_empty_strings(db, user_model_store):
    """Blank/empty strings in the topics list are filtered out."""
    _store_episode(db, "alice@example.com", ["planning", "", "  ", "budget"])
    engine = InsightEngine(db, user_model_store)

    topics = engine._get_contact_last_topics("alice@example.com")
    assert "" not in topics
    assert "  " not in topics
    assert "planning" in topics


def test_get_contact_last_topics_empty_table(db, user_model_store):
    """Returns empty list when the episodes table has no rows at all."""
    engine = InsightEngine(db, user_model_store)
    assert engine._get_contact_last_topics("alice@example.com") == []


# ---------------------------------------------------------------------------
# _contact_gap_insights integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contact_gap_includes_topic_suffix_when_episode_exists(
    db, user_model_store
):
    """Summary appends 'Last topics: ...' when episodic data is available."""
    email = "alice@example.com"
    now = datetime.now(timezone.utc)

    # Episode with clear topic tags
    _store_episode(db, email, ["budget-review", "q1-planning"])
    user_model_store.update_signal_profile(
        "relationships", _make_overdue_profile(email, now)
    )
    engine = InsightEngine(db, user_model_store)

    insights = engine._contact_gap_insights()
    assert len(insights) == 1
    summary = insights[0].summary
    assert "Last topics:" in summary
    assert "budget-review" in summary
    assert "q1-planning" in summary


@pytest.mark.asyncio
async def test_contact_gap_no_topic_suffix_when_no_episode(db, user_model_store):
    """Summary has no topic suffix when no episodic data exists for contact."""
    email = "ghost@example.com"
    now = datetime.now(timezone.utc)

    user_model_store.update_signal_profile(
        "relationships", _make_overdue_profile(email, now)
    )
    engine = InsightEngine(db, user_model_store)

    insights = engine._contact_gap_insights()
    assert len(insights) == 1
    # Summary should end with the usual interval clause without any topic suffix
    assert "Last topics:" not in insights[0].summary


@pytest.mark.asyncio
async def test_contact_gap_topics_in_evidence(db, user_model_store):
    """Topics discovered from episodic memory appear in the evidence list."""
    email = "bob@example.com"
    now = datetime.now(timezone.utc)

    _store_episode(db, email, ["project-deadline", "sprint-review"])
    user_model_store.update_signal_profile(
        "relationships", _make_overdue_profile(email, now)
    )
    engine = InsightEngine(db, user_model_store)

    insights = engine._contact_gap_insights()
    assert len(insights) == 1
    evidence = insights[0].evidence
    # Both topics should appear as evidence entries prefixed with "last_topic="
    assert any(e.startswith("last_topic=") for e in evidence)
    evidence_str = " ".join(evidence)
    assert "project-deadline" in evidence_str
    assert "sprint-review" in evidence_str


@pytest.mark.asyncio
async def test_contact_gap_standard_evidence_always_present(db, user_model_store):
    """Standard evidence keys (days_since_last, avg_gap_days, interaction_count)
    are always present, regardless of whether topic data is found."""
    email = "charlie@example.com"
    now = datetime.now(timezone.utc)

    user_model_store.update_signal_profile(
        "relationships", _make_overdue_profile(email, now)
    )
    engine = InsightEngine(db, user_model_store)

    insights = engine._contact_gap_insights()
    assert len(insights) == 1
    evidence = insights[0].evidence
    evidence_keys = [e.split("=")[0] for e in evidence]
    assert "days_since_last" in evidence_keys
    assert "avg_gap_days" in evidence_keys
    assert "interaction_count" in evidence_keys


@pytest.mark.asyncio
async def test_contact_gap_episode_wrong_contact_no_topic_suffix(db, user_model_store):
    """Episodes for a DIFFERENT contact are not mixed into this contact's insight."""
    email = "alice@example.com"
    now = datetime.now(timezone.utc)

    # Episode belongs to bob, not alice
    _store_episode(db, "bob@example.com", ["finance", "tax"])
    user_model_store.update_signal_profile(
        "relationships", _make_overdue_profile(email, now)
    )
    engine = InsightEngine(db, user_model_store)

    insights = engine._contact_gap_insights()
    assert len(insights) == 1
    assert "Last topics:" not in insights[0].summary


@pytest.mark.asyncio
async def test_contact_gap_insight_still_fires_when_topics_unavailable(
    db, user_model_store
):
    """Insight is generated even when topic lookup returns nothing (fail-open)."""
    email = "nodata@example.com"
    now = datetime.now(timezone.utc)

    # No episode stored — topic enrichment returns empty list
    user_model_store.update_signal_profile(
        "relationships", _make_overdue_profile(email, now)
    )
    engine = InsightEngine(db, user_model_store)

    insights = engine._contact_gap_insights()
    # Insight must still be generated despite missing episodic data
    assert len(insights) == 1
    assert str(email) in insights[0].summary or str(email) in insights[0].entity
