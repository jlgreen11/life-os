"""
Tests for the default email notification rules — 'Notify on urgent emails'
and 'Notify on direct reply requests'.

These rules ensure the notification system generates output for the most
common event type (email.received).  Marketing emails are protected by the
suppress-before-notify pipeline ordering in main.py: the existing 'Archive
marketing emails' rule sets event["_suppressed"] = True before any notify
action runs, so suppressed emails never produce notifications.

Test coverage:
    - Both new rules are present in DEFAULT_RULES
    - install_default_rules installs all 6 rules
    - Urgent rule matches case-insensitive subject keywords
    - Urgent rule does NOT match normal emails without keywords
    - Reply-request rule matches body phrases
    - Reply-request rule does NOT match normal emails without phrases
    - Notify actions have the expected priority
    - Marketing emails (containing 'unsubscribe') are suppressed before notify
"""

import pytest

from services.rules_engine.engine import DEFAULT_RULES, RulesEngine, install_default_rules


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_email_event(
    subject: str = "Hello",
    body_plain: str = "Normal email content",
    event_id: str = "evt-test",
) -> dict:
    """Build a minimal email.received event for testing rules."""
    return {
        "id": event_id,
        "type": "email.received",
        "payload": {
            "subject": subject,
            "body_plain": body_plain,
            "from_address": "sender@example.com",
        },
    }


# ---------------------------------------------------------------
# DEFAULT_RULES list tests
# ---------------------------------------------------------------

def test_urgent_rule_in_default_rules():
    """Verify 'Notify on urgent emails' is present in DEFAULT_RULES."""
    names = [r["name"] for r in DEFAULT_RULES]
    assert "Notify on urgent emails" in names


def test_reply_request_rule_in_default_rules():
    """Verify 'Notify on direct reply requests' is present in DEFAULT_RULES."""
    names = [r["name"] for r in DEFAULT_RULES]
    assert "Notify on direct reply requests" in names


def test_default_rules_count():
    """DEFAULT_RULES should contain exactly 6 rules."""
    assert len(DEFAULT_RULES) == 6


# ---------------------------------------------------------------
# install_default_rules
# ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_install_default_rules_installs_all_six(db):
    """install_default_rules should create all 6 rules in the database."""
    await install_default_rules(db)

    engine = RulesEngine(db)
    rules = engine.get_all_rules()
    assert len(rules) == 6

    rule_names = {r["name"] for r in rules}
    assert "Notify on urgent emails" in rule_names
    assert "Notify on direct reply requests" in rule_names


# ---------------------------------------------------------------
# Urgent email rule — matching
# ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_urgent_rule_matches_uppercase_subject(db):
    """The urgent rule should match 'URGENT' in subject (case-insensitive)."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = _make_email_event(subject="URGENT: Board meeting rescheduled")
    actions = await engine.evaluate(event)

    notify_actions = [a for a in actions if a["type"] == "notify" and a["rule_name"] == "Notify on urgent emails"]
    assert len(notify_actions) == 1
    assert notify_actions[0]["priority"] == "high"


@pytest.mark.asyncio
async def test_urgent_rule_matches_mixed_case(db):
    """The urgent rule should match keywords regardless of case."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = _make_email_event(subject="Action Required: Approve budget")
    actions = await engine.evaluate(event)

    notify_actions = [a for a in actions if a["type"] == "notify" and a["rule_name"] == "Notify on urgent emails"]
    assert len(notify_actions) == 1


@pytest.mark.asyncio
async def test_urgent_rule_matches_various_keywords(db):
    """Each urgent keyword in the list should trigger the rule."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    keywords = [
        "urgent", "action required", "action needed",
        "immediate", "asap", "time sensitive",
        "deadline", "past due", "overdue",
    ]
    for keyword in keywords:
        event = _make_email_event(subject=f"Re: {keyword} - project update")
        actions = await engine.evaluate(event)

        notify_actions = [a for a in actions if a["type"] == "notify" and a["rule_name"] == "Notify on urgent emails"]
        assert len(notify_actions) == 1, f"Failed to match keyword: {keyword}"


@pytest.mark.asyncio
async def test_urgent_rule_tags_as_urgent(db):
    """The urgent rule should also add a 'urgent' tag."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = _make_email_event(subject="ASAP: Need your signature")
    actions = await engine.evaluate(event)

    tag_actions = [a for a in actions if a["type"] == "tag" and a["rule_name"] == "Notify on urgent emails"]
    assert len(tag_actions) == 1
    assert tag_actions[0]["value"] == "urgent"


# ---------------------------------------------------------------
# Urgent email rule — non-matching
# ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_urgent_rule_does_not_match_normal_email(db):
    """A normal email without urgent keywords should NOT trigger the urgent rule."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = _make_email_event(subject="Weekly team sync notes")
    actions = await engine.evaluate(event)

    notify_actions = [a for a in actions if a["type"] == "notify" and a["rule_name"] == "Notify on urgent emails"]
    assert len(notify_actions) == 0


# ---------------------------------------------------------------
# Reply request rule — matching
# ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_reply_rule_matches_please_reply(db):
    """The reply-request rule should match 'please reply' in body."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = _make_email_event(body_plain="Hi, please reply with your availability.")
    actions = await engine.evaluate(event)

    notify_actions = [a for a in actions if a["type"] == "notify" and a["rule_name"] == "Notify on direct reply requests"]
    assert len(notify_actions) == 1
    assert notify_actions[0]["priority"] == "medium"


@pytest.mark.asyncio
async def test_reply_rule_matches_various_phrases(db):
    """Each reply-request phrase should trigger the rule."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    phrases = [
        "please reply", "please respond", "let me know",
        "can you confirm", "your thoughts",
        "waiting for your", "need your input",
        "rsvp", "please get back",
    ]
    for phrase in phrases:
        event = _make_email_event(body_plain=f"Hello, {phrase} when you can. Thanks!")
        actions = await engine.evaluate(event)

        notify_actions = [
            a for a in actions
            if a["type"] == "notify" and a["rule_name"] == "Notify on direct reply requests"
        ]
        assert len(notify_actions) == 1, f"Failed to match phrase: {phrase}"


@pytest.mark.asyncio
async def test_reply_rule_case_insensitive(db):
    """The reply-request rule should be case-insensitive."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = _make_email_event(body_plain="PLEASE RESPOND to this request ASAP.")
    actions = await engine.evaluate(event)

    notify_actions = [a for a in actions if a["type"] == "notify" and a["rule_name"] == "Notify on direct reply requests"]
    assert len(notify_actions) == 1


# ---------------------------------------------------------------
# Reply request rule — non-matching
# ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_reply_rule_does_not_match_normal_email(db):
    """A normal email without reply-request phrases should NOT trigger."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = _make_email_event(body_plain="Here are the meeting notes from today.")
    actions = await engine.evaluate(event)

    notify_actions = [
        a for a in actions
        if a["type"] == "notify" and a["rule_name"] == "Notify on direct reply requests"
    ]
    assert len(notify_actions) == 0


# ---------------------------------------------------------------
# Marketing email suppression
# ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_marketing_email_triggers_suppress_action(db):
    """Marketing emails should trigger the suppress action from the Archive rule.

    The suppress-before-notify ordering in main.py ensures that even if the
    urgent or reply-request rule also matches, the suppress action fires first
    and the notify handler checks event['_suppressed'] before creating a
    notification.  Here we verify the rules engine returns both suppress and
    notify actions — the ordering is enforced by main.py's action execution.
    """
    await install_default_rules(db)
    engine = RulesEngine(db)

    # Marketing email that ALSO has an urgent keyword
    event = _make_email_event(
        subject="URGENT: Last chance sale!",
        body_plain="Buy now! Click to unsubscribe from our mailing list.",
    )
    actions = await engine.evaluate(event)

    # Verify suppress action is present (from "Archive marketing emails")
    suppress_actions = [a for a in actions if a["type"] == "suppress"]
    assert len(suppress_actions) >= 1, "Marketing email should trigger suppress action"

    # The notify action from "Notify on urgent emails" will also be present,
    # but main.py executes suppress first, setting _suppressed=True, so the
    # notify handler will skip it.  We just confirm the suppress exists.


@pytest.mark.asyncio
async def test_marketing_reply_request_triggers_suppress(db):
    """Marketing emails with reply-request phrases should still be suppressed."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = _make_email_event(
        subject="Newsletter update",
        body_plain="Let me know if you want to opt out. Unsubscribe here.",
    )
    actions = await engine.evaluate(event)

    suppress_actions = [a for a in actions if a["type"] == "suppress"]
    assert len(suppress_actions) >= 1, "Marketing email should trigger suppress action"


# ---------------------------------------------------------------
# Action structure validation
# ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_urgent_notify_action_structure(db):
    """The urgent rule's notify action should have priority='high'."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = _make_email_event(subject="Deadline tomorrow")
    actions = await engine.evaluate(event)

    notify = [a for a in actions if a["type"] == "notify" and a["rule_name"] == "Notify on urgent emails"]
    assert len(notify) == 1
    assert notify[0]["priority"] == "high"
    assert "rule_id" in notify[0]
    assert notify[0]["rule_name"] == "Notify on urgent emails"


@pytest.mark.asyncio
async def test_reply_notify_action_structure(db):
    """The reply-request rule's notify action should have priority='medium'."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = _make_email_event(body_plain="Can you confirm receipt? Thanks.")
    actions = await engine.evaluate(event)

    notify = [a for a in actions if a["type"] == "notify" and a["rule_name"] == "Notify on direct reply requests"]
    assert len(notify) == 1
    assert notify[0]["priority"] == "medium"
    assert "rule_id" in notify[0]
    assert notify[0]["rule_name"] == "Notify on direct reply requests"
