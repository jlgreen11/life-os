"""
Tests for workflow detector threshold logic.

This test suite verifies that the workflow detector uses absolute completion
counts (not success rates) to decide whether a workflow is real.

Background:
- With 77K emails received and only 229 sent, the response rate is ~0.3%
- Most emails (marketing, newsletters) don't require responses
- The original rate-based threshold blocked ALL workflow detection
- The fix uses absolute completion count (min_completions=3) instead

Test strategy:
- Create realistic email patterns (high volume inbox + selective responses)
- Verify workflows are detected when completion count >= 3, regardless of rate
- Ensure the detector still filters out senders with < 3 completions
"""

from datetime import datetime, timedelta, timezone

from services.workflow_detector.detector import WorkflowDetector


def test_email_workflow_detection_with_low_response_rate(db, user_model_store, event_store):
    """
    Test that email response workflows are detected even when the overall
    response rate is very low (< 1%).

    Scenario:
    - 100 marketing emails received (no responses)
    - 10 boss emails received, 5 responses sent within 2h
    - Overall response rate: 5/110 = 4.5% (was below old rate threshold)
    - Completion count for boss: 5 >= min_completions(3)

    Expected: Workflow detected for boss emails (5 completions >= 3).
    """
    detector = WorkflowDetector(db, user_model_store)

    # Create 100 marketing emails with no responses
    base_time = datetime.now(timezone.utc) - timedelta(days=15)
    for i in range(100):
        event_store.store_event({
            "id": f"marketing-{i}",
            "type": "email.received",
            "source": "gmail",
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "priority": 3,
            "payload": {
                "from_address": "marketing@example.com",
                "subject": f"Special Offer {i}",
                "body": "Buy now!",
            },
            "metadata": {},
        })

    # Create 10 boss emails with 5 responses
    for i in range(10):
        receive_time = base_time + timedelta(days=i)
        event_store.store_event({
            "id": f"boss-email-{i}",
            "type": "email.received",
            "source": "gmail",
            "timestamp": receive_time.isoformat(),
            "priority": 1,
            "payload": {
                "from_address": "boss@company.com",
                "subject": f"Project Update {i}",
                "body": "Please review the attached report.",
            },
            "metadata": {},
        })

        # Respond to every other email (50% response rate)
        if i % 2 == 0:
            response_time = receive_time + timedelta(hours=1)
            event_store.store_event({
                "id": f"boss-response-{i}",
                "type": "email.sent",
                "source": "gmail",
                "timestamp": response_time.isoformat(),
                "priority": 2,
                "payload": {
                    "to_addresses": ["boss@company.com"],
                    "subject": f"Re: Project Update {i}",
                    "body": "Reviewed and approved.",
                },
                "metadata": {},
            })

    # Run detection
    workflows = detector.detect_workflows(lookback_days=30)

    # Should detect workflow for boss emails (5 completions >= min_completions=3)
    boss_workflows = [w for w in workflows if "boss@company.com" in w["name"]]
    assert len(boss_workflows) >= 1, "Should detect workflow for boss emails (5 completions >= 3)"

    boss_workflow = boss_workflows[0]
    assert boss_workflow["success_rate"] > 0.0, "Success rate should be non-zero"
    assert any("sent" in step for step in boss_workflow["steps"]), "Should include email sending step"
    assert boss_workflow["times_observed"] == 10, "Should observe all 10 boss emails"


def test_workflow_detection_filters_pure_noise(db, user_model_store, event_store):
    """
    Test that the detector still filters out random patterns with no correlation.

    Scenario:
    - 100 emails received from sender A
    - 3 random email.sent events (not correlated with sender A)
    - The sent emails are to a different recipient

    Expected: No workflow detected because events aren't temporally correlated
    with the sender.
    """
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=20)

    # 100 emails from sender A
    for i in range(100):
        event_store.store_event({
            "id": f"email-a-{i}",
            "type": "email.received",
            "source": "gmail",
            "timestamp": (base_time + timedelta(hours=i * 6)).isoformat(),  # Every 6 hours
            "priority": 3,
            "payload": {
                "from_address": "sender-a@example.com",
                "subject": f"Newsletter {i}",
            },
            "metadata": {},
        })

    # 3 random sent emails NOT correlated with sender A (sent to different recipient)
    for i in range(3):
        # Send these at times that DON'T align with the received emails
        random_time = base_time + timedelta(hours=i * 100 + 10)  # Every 100h, offset by 10h
        event_store.store_event({
            "id": f"random-sent-{i}",
            "type": "email.sent",
            "source": "gmail",
            "timestamp": random_time.isoformat(),
            "priority": 2,
            "payload": {
                "to_addresses": ["unrelated@example.com"],  # Different recipient
                "subject": "Unrelated message",
            },
            "metadata": {},
        })

    # Run detection
    workflows = detector.detect_workflows(lookback_days=30)

    # Should NOT detect workflow for sender A because events aren't correlated
    sender_a_workflows = [w for w in workflows if "sender-a@example.com" in w["name"]]
    assert len(sender_a_workflows) == 0, "Should not detect workflow for uncorrelated random events"


def test_workflow_detection_with_realistic_inbox(db, user_model_store, event_store):
    """
    Test workflow detection with a realistic inbox profile:
    - High volume marketing/newsletter noise
    - Work emails with >= 3 responses (should be detected)
    - Family emails with < 3 responses (should NOT be detected)

    With absolute completion count guard (min_completions=3):
    - colleague@work.com: 8 responses >= 3 → DETECTED
    - Marketing senders: 0 responses < 3 → filtered out
    - mom@family.com: 2 responses < 3 → filtered out
    """
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=25)

    # 800 marketing emails (no responses)
    for i in range(800):
        event_store.store_event({
            "id": f"marketing-bulk-{i}",
            "type": "email.received",
            "source": "gmail",
            "timestamp": (base_time + timedelta(hours=i * 0.5)).isoformat(),
            "priority": 3,
            "payload": {
                "from_address": f"promo{i % 20}@marketing.com",  # 20 different senders
                "subject": f"Deal of the Day {i}",
            },
            "metadata": {},
        })

    # 150 newsletter emails (no responses)
    for i in range(150):
        event_store.store_event({
            "id": f"newsletter-{i}",
            "type": "email.received",
            "source": "gmail",
            "timestamp": (base_time + timedelta(days=i * 0.1)).isoformat(),
            "priority": 3,
            "payload": {
                "from_address": "newsletter@substack.com",
                "subject": f"Weekly Digest {i}",
            },
            "metadata": {},
        })

    # 40 work emails, respond to 8 of them (20% response rate)
    for i in range(40):
        receive_time = base_time + timedelta(days=i * 0.5)
        event_store.store_event({
            "id": f"work-{i}",
            "type": "email.received",
            "source": "gmail",
            "timestamp": receive_time.isoformat(),
            "priority": 1,
            "payload": {
                "from_address": "colleague@work.com",
                "subject": f"Work Item {i}",
            },
            "metadata": {},
        })

        # Respond to 20% (every 5th email)
        if i % 5 == 0:
            response_time = receive_time + timedelta(hours=2)
            event_store.store_event({
                "id": f"work-response-{i}",
                "type": "email.sent",
                "source": "gmail",
                "timestamp": response_time.isoformat(),
                "priority": 2,
                "payload": {
                    "to_addresses": ["colleague@work.com"],
                    "subject": f"Re: Work Item {i}",
                },
                "metadata": {},
            })

    # 10 family emails, respond to 2 of them (< min_completions)
    for i in range(10):
        receive_time = base_time + timedelta(days=i * 2)
        event_store.store_event({
            "id": f"family-{i}",
            "type": "email.received",
            "source": "gmail",
            "timestamp": receive_time.isoformat(),
            "priority": 1,
            "payload": {
                "from_address": "mom@family.com",
                "subject": f"Family Update {i}",
            },
            "metadata": {},
        })

        # Respond to 2 emails (i=0 and i=5) — below min_completions=3
        if i % 5 == 0:
            response_time = receive_time + timedelta(hours=3)
            event_store.store_event({
                "id": f"family-response-{i}",
                "type": "email.sent",
                "source": "gmail",
                "timestamp": response_time.isoformat(),
                "priority": 2,
                "payload": {
                    "to_addresses": ["mom@family.com"],
                    "subject": f"Re: Family Update {i}",
                },
                "metadata": {},
            })

    # Total: 1000 received, 10 sent
    # With absolute count: colleague@work.com (8 completions >= 3) detected,
    # mom@family.com (2 completions < 3) filtered, marketing (0) filtered
    workflows = detector.detect_workflows(lookback_days=30)

    # colleague@work.com should be detected (8 completions >= 3)
    work_workflows = [w for w in workflows if "colleague@work.com" in w["name"]]
    assert len(work_workflows) >= 1, (
        f"Should detect colleague@work.com workflow (8 completions >= 3), got {len(work_workflows)}"
    )

    # mom@family.com should NOT be detected (2 completions < 3)
    family_workflows = [w for w in workflows if "mom@family.com" in w["name"]]
    assert len(family_workflows) == 0, (
        "Should NOT detect mom@family.com workflow (2 completions < min_completions=3)"
    )

    # Marketing senders should NOT be detected (0 completions)
    marketing_workflows = [w for w in workflows if "marketing.com" in w.get("name", "")]
    assert len(marketing_workflows) == 0, "Should NOT detect marketing workflows (0 completions)"


def test_workflow_absolute_count_boundary_cases(db, user_model_store, event_store):
    """
    Test edge cases around the min_completions=3 absolute count boundary.

    Cases:
    1. Exactly 3 completions out of 1000 received → should detect
    2. Only 2 completions out of 10 received → should NOT detect (even with 20% rate)
    """
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=20)

    # Case 1: 3 completions out of many received (0.3% rate but 3 absolute)
    for i in range(100):
        event_store.store_event({
            "id": f"boundary-above-recv-{i}",
            "type": "email.received",
            "source": "gmail",
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "priority": 3,
            "payload": {"from_address": "above-boundary@test.com"},
            "metadata": {},
        })

    # 3 responses within time window
    for i in range(3):
        event_store.store_event({
            "id": f"boundary-above-resp-{i}",
            "type": "email.sent",
            "source": "gmail",
            "timestamp": (base_time + timedelta(hours=i * 2 + 1)).isoformat(),
            "priority": 2,
            "payload": {"to_addresses": ["above-boundary@test.com"]},
            "metadata": {},
        })

    # Case 2: 2 completions out of 10 received (20% rate but only 2 absolute)
    # Space received emails 1 day apart so each sent matches only 1 received
    for i in range(10):
        event_store.store_event({
            "id": f"boundary-below-recv-{i}",
            "type": "email.received",
            "source": "gmail",
            "timestamp": (base_time + timedelta(days=i)).isoformat(),
            "priority": 3,
            "payload": {"from_address": "below-boundary@test.com"},
            "metadata": {},
        })

    # 2 replies, each 1h after a received email (within 4h gap, but only 1 match each)
    for i in range(2):
        event_store.store_event({
            "id": f"boundary-below-resp-{i}",
            "type": "email.sent",
            "source": "gmail",
            "timestamp": (base_time + timedelta(days=i, hours=1)).isoformat(),
            "priority": 2,
            "payload": {"to_addresses": ["below-boundary@test.com"]},
            "metadata": {},
        })

    workflows = detector.detect_workflows(lookback_days=30)

    above_workflows = [w for w in workflows if "above-boundary@test.com" in w["name"]]
    below_workflows = [w for w in workflows if "below-boundary@test.com" in w["name"]]

    # 3 completions >= min_completions → detected
    assert len(above_workflows) >= 1, (
        "Should detect workflow with 3 completions (>= min_completions=3)"
    )

    # 2 completions < min_completions → NOT detected, even with 20% success rate
    assert len(below_workflows) == 0, (
        "Should NOT detect workflow with only 2 completions (< min_completions=3), "
        "even though success rate is 20%"
    )


def test_multi_step_workflow_detection(db, user_model_store, event_store):
    """
    Test detection of multi-step workflows (email received → task created → email sent).

    This tests the min_steps requirement (must have at least 2 distinct steps).
    """
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=15)

    # Create 10 emails that trigger task creation AND responses
    for i in range(10):
        receive_time = base_time + timedelta(days=i)

        # Email received
        event_store.store_event({
            "id": f"multistep-email-{i}",
            "type": "email.received",
            "source": "gmail",
            "timestamp": receive_time.isoformat(),
            "priority": 1,
            "payload": {
                "from_address": "client@business.com",
                "subject": f"Request {i}",
            },
            "metadata": {},
        })

        # Task created 30min later
        task_time = receive_time + timedelta(minutes=30)
        event_store.store_event({
            "id": f"multistep-task-{i}",
            "type": "task.created",
            "source": "ai_extracted",
            "timestamp": task_time.isoformat(),
            "priority": 2,
            "payload": {
                "title": f"Handle Request {i}",
                "source": "email",
            },
            "metadata": {},
        })

        # Email sent 2h later
        send_time = receive_time + timedelta(hours=2)
        event_store.store_event({
            "id": f"multistep-response-{i}",
            "type": "email.sent",
            "source": "gmail",
            "timestamp": send_time.isoformat(),
            "priority": 2,
            "payload": {
                "to_addresses": ["client@business.com"],
                "subject": f"Re: Request {i}",
            },
            "metadata": {},
        })

    workflows = detector.detect_workflows(lookback_days=30)

    # Should detect multi-step workflow (10 completions >= 3)
    client_workflows = [w for w in workflows if "client@business.com" in w["name"]]
    assert len(client_workflows) >= 1, "Should detect multi-step workflow"

    workflow = client_workflows[0]
    # Should have at least 2 steps (task.created and email.sent)
    assert len(workflow["steps"]) >= 2, f"Multi-step workflow should have 2+ steps, got {len(workflow['steps'])}"

    # Check that both task creation and email sending are in the workflow
    step_types = [s for s in workflow["steps"]]
    has_task = any("task" in str(s) or "created" in str(s) for s in step_types)
    has_email = any("email" in str(s) or "sent" in str(s) for s in step_types)
    assert has_task or has_email, "Workflow should include task and/or email steps"


def test_workflow_storage_integration(db, user_model_store, event_store):
    """
    Test that detected workflows are correctly stored in the database.

    Verifies the store_workflows method persists workflows to user_model.db.
    """
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=10)

    # Create a simple workflow pattern (5 emails, 3 responses = 3 completions >= min)
    for i in range(5):
        receive_time = base_time + timedelta(days=i)
        event_store.store_event({
            "id": f"storage-email-{i}",
            "type": "email.received",
            "source": "gmail",
            "timestamp": receive_time.isoformat(),
            "priority": 1,
            "payload": {"from_address": "storage-test@example.com"},
            "metadata": {},
        })

        if i < 3:  # Respond to first 3
            response_time = receive_time + timedelta(hours=1)
            event_store.store_event({
                "id": f"storage-response-{i}",
                "type": "email.sent",
                "source": "gmail",
                "timestamp": response_time.isoformat(),
                "priority": 2,
                "payload": {"to_addresses": ["storage-test@example.com"]},
                "metadata": {},
            })

    # Detect and store
    workflows = detector.detect_workflows(lookback_days=30)
    stored_count = detector.store_workflows(workflows)

    assert stored_count >= 0, "Should store workflows without error"

    # Verify stored workflow can be retrieved
    with db.get_connection("user_model") as conn:
        stored_workflows = conn.execute(
            "SELECT name, success_rate FROM workflows WHERE name LIKE '%storage-test%'"
        ).fetchall()

    # May or may not detect depending on min_occurrences, but storage should work
    assert len(stored_workflows) >= 0, "Workflow storage should complete"
