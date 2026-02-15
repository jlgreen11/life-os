# Agent-Driven Improvement Loop — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate prediction noise, surface existing signal data as insights, build an InsightEngine for cross-signal correlation, and set up a weekly Claude Code agent that iteratively improves Life OS.

**Architecture:** Two-phase hybrid. Phase 1 patches the prediction engine, wires a feedback loop, and surfaces existing data. Phase 2 adds an InsightEngine service and a scheduled Claude Code outer loop. See `docs/plans/2026-02-15-agent-improvement-loop-design.md` for full design.

**Tech Stack:** Python 3.12, FastAPI, SQLite (WAL mode), NATS JetStream, Pydantic, launchd

---

## Phase 1: Fix the Plumbing

### Task 1: Set up test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_prediction_engine.py`

**Step 1: Create test fixtures**

Create `tests/conftest.py` with a temporary database fixture that mirrors the production schema:

```python
import pytest
import tempfile
import os
from storage.manager import DatabaseManager
from storage.event_store import EventStore
from storage.user_model_store import UserModelStore


@pytest.fixture
def tmp_data_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def db(tmp_data_dir):
    manager = DatabaseManager(tmp_data_dir)
    manager.initialize_all()
    return manager


@pytest.fixture
def event_store(db):
    return EventStore(db)


@pytest.fixture
def user_model_store(db):
    return UserModelStore(db)
```

Create empty `tests/__init__.py`.

**Step 2: Verify test infrastructure works**

Create a smoke test in `tests/test_prediction_engine.py`:

```python
from services.prediction_engine.engine import PredictionEngine


def test_prediction_engine_initializes(db, user_model_store):
    engine = PredictionEngine(db, user_model_store)
    assert engine is not None
```

**Step 3: Run test**

Run: `cd /Users/jeremygreenwood/life-os && python -m pytest tests/test_prediction_engine.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/
git commit -m "feat: add test infrastructure with SQLite fixtures"
```

---

### Task 2: Add prediction throttling

The prediction loop currently runs every 15 minutes unconditionally, regenerating all predictions even when no new data has arrived. This produced 186k+ prediction events.

**Files:**
- Modify: `main.py:302-321` (`_prediction_loop`)
- Modify: `services/prediction_engine/engine.py:50-85` (`generate_predictions`)
- Test: `tests/test_prediction_engine.py`

**Step 1: Write the failing test**

Add to `tests/test_prediction_engine.py`:

```python
import pytest
import json
import uuid
from datetime import datetime, timezone, timedelta
from services.prediction_engine.engine import PredictionEngine


@pytest.mark.asyncio
async def test_prediction_engine_skips_when_no_new_events(db, event_store, user_model_store):
    """Prediction engine should return empty list when no new events since last run."""
    engine = PredictionEngine(db, user_model_store)

    # First run with no events — should return empty and set cursor
    predictions = await engine.generate_predictions({})
    assert predictions == []

    # Second run with still no new events — should skip
    predictions = await engine.generate_predictions({})
    assert predictions == []


@pytest.mark.asyncio
async def test_prediction_engine_runs_when_new_events_exist(db, event_store, user_model_store):
    """Prediction engine should run when new events exist since last cursor."""
    engine = PredictionEngine(db, user_model_store)

    # First run sets cursor
    await engine.generate_predictions({})

    # Add a new event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": json.dumps({"from_address": "boss@company.com", "subject": "Urgent"}),
        "metadata": json.dumps({}),
    })

    # Engine should detect new events and run (not skip)
    # We just verify it doesn't raise — actual predictions depend on data
    predictions = await engine.generate_predictions({})
    # No assertion on length — just that it ran without error
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jeremygreenwood/life-os && python -m pytest tests/test_prediction_engine.py::test_prediction_engine_skips_when_no_new_events -v`
Expected: FAIL (no throttling logic exists yet)

**Step 3: Implement prediction throttling**

In `services/prediction_engine/engine.py`, add cursor tracking to `__init__` and a gate at the top of `generate_predictions`:

```python
# In __init__, add:
self._last_event_cursor: int = 0  # rowid of last processed event

# At the top of generate_predictions, before the prediction pipeline:
def _has_new_events(self) -> bool:
    """Check if any new events have arrived since last prediction run."""
    with self.db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT MAX(rowid) as max_id FROM events"
        ).fetchone()
        current_max = row["max_id"] if row and row["max_id"] else 0

    if current_max <= self._last_event_cursor:
        return False

    self._last_event_cursor = current_max
    return True
```

Add the gate at the top of `generate_predictions`:

```python
async def generate_predictions(self, current_context: dict) -> list[Prediction]:
    # Skip if no new events since last run
    if not self._has_new_events():
        return []

    # ... rest of existing code
```

**Step 4: Run tests**

Run: `cd /Users/jeremygreenwood/life-os && python -m pytest tests/test_prediction_engine.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add services/prediction_engine/engine.py tests/test_prediction_engine.py
git commit -m "feat: throttle prediction engine — skip when no new events"
```

---

### Task 3: Add marketing pre-filter to follow-up detector

The follow-up detector currently generates predictions for marketing emails. The existing "unsubscribe" check at line 221 only checks `snippet`, missing the body, and doesn't catch no-reply senders.

**Files:**
- Modify: `services/prediction_engine/engine.py:166-268` (`_check_follow_up_needs`)
- Test: `tests/test_prediction_engine.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_follow_up_skips_marketing_emails(db, event_store, user_model_store):
    """Marketing emails should never generate follow-up predictions."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert a marketing email (no-reply sender)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=6)).isoformat(),
        "payload": json.dumps({
            "from_address": "no-reply@marketing.example.com",
            "subject": "50% off today!",
            "snippet": "Click here to unsubscribe",
            "body_plain": "Big sale happening now. Unsubscribe here.",
            "message_id": "msg-marketing-1",
        }),
        "metadata": json.dumps({}),
    })

    # Insert a noreply sender (variant)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=5)).isoformat(),
        "payload": json.dumps({
            "from_address": "noreply@accounts.google.com",
            "subject": "Security alert",
            "snippet": "New sign-in detected",
            "message_id": "msg-noreply-1",
        }),
        "metadata": json.dumps({}),
    })

    predictions = await engine._check_follow_up_needs({})
    # Neither marketing nor noreply emails should produce predictions
    assert len(predictions) == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jeremygreenwood/life-os && python -m pytest tests/test_prediction_engine.py::test_follow_up_skips_marketing_emails -v`
Expected: FAIL (noreply filter doesn't exist, body_plain not checked)

**Step 3: Enhance the marketing filter**

In `_check_follow_up_needs`, replace the existing simple check (lines 219-222) with a comprehensive filter. Add a helper method:

```python
@staticmethod
def _is_marketing_or_noreply(from_addr: str, payload: dict) -> bool:
    """Check if an email is marketing/automated and shouldn't generate follow-up predictions."""
    addr_lower = from_addr.lower()

    # No-reply senders
    if any(pattern in addr_lower for pattern in ("no-reply@", "noreply@", "do-not-reply@", "donotreply@")):
        return True

    # Check body and snippet for unsubscribe indicators
    text = (payload.get("body_plain", "") + " " + payload.get("snippet", "") + " " + payload.get("body", "")).lower()
    if "unsubscribe" in text:
        return True

    # Common bulk sender patterns
    bulk_patterns = ("newsletter@", "notifications@", "updates@", "digest@", "mailer@", "bulk@", "promo@")
    if any(pattern in addr_lower for pattern in bulk_patterns):
        return True

    return False
```

In the loop over inbound messages, replace the old check:

```python
# Replace:
# if payload.get("snippet", "").lower().count("unsubscribe") > 0:
#     continue
# With:
if self._is_marketing_or_noreply(from_addr, payload):
    continue
```

**Step 4: Run tests**

Run: `cd /Users/jeremygreenwood/life-os && python -m pytest tests/test_prediction_engine.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add services/prediction_engine/engine.py tests/test_prediction_engine.py
git commit -m "feat: comprehensive marketing/noreply filter for follow-up predictions"
```

---

### Task 4: Wire feedback loop — notification dismiss/act → prediction accuracy

When a user dismisses or acts on a notification, trace it back to the prediction that created it and update `was_accurate`. Apply confidence decay/boost for future predictions of the same type.

**Files:**
- Modify: `services/notification_manager/manager.py:302-315` (`dismiss` and `mark_acted_on`)
- Modify: `services/prediction_engine/engine.py` (add accuracy-based confidence adjustment)
- Modify: `main.py:302-321` (`_prediction_loop` — pass prediction_id to notification)
- Test: `tests/test_feedback_loop.py`

**Step 1: Write the failing test**

Create `tests/test_feedback_loop.py`:

```python
import pytest
import json
import uuid
from datetime import datetime, timezone
from services.prediction_engine.engine import PredictionEngine


@pytest.mark.asyncio
async def test_prediction_accuracy_decay(db, user_model_store):
    """Dismissed predictions should reduce future confidence for that type."""
    engine = PredictionEngine(db, user_model_store)

    # Store 10 predictions of type "reminder", all marked was_accurate=False
    for i in range(10):
        pred_id = str(uuid.uuid4())
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at)
                   VALUES (?, ?, ?, ?, ?, 1, 0, ?)""",
                (pred_id, "reminder", f"Test prediction {i}", 0.7, "suggest",
                 datetime.now(timezone.utc).isoformat()),
            )

    # Engine should compute an accuracy-based multiplier for "reminder" type
    multiplier = engine._get_accuracy_multiplier("reminder")
    assert multiplier < 1.0, "Dismissed predictions should reduce multiplier below 1.0"


@pytest.mark.asyncio
async def test_accurate_predictions_boost(db, user_model_store):
    """Accurate predictions should not reduce future confidence."""
    engine = PredictionEngine(db, user_model_store)

    # Store 10 predictions of type "conflict", all marked was_accurate=True
    for i in range(10):
        pred_id = str(uuid.uuid4())
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at)
                   VALUES (?, ?, ?, ?, ?, 1, 1, ?)""",
                (pred_id, "conflict", f"Test prediction {i}", 0.9, "default",
                 datetime.now(timezone.utc).isoformat()),
            )

    multiplier = engine._get_accuracy_multiplier("conflict")
    assert multiplier >= 1.0, "Accurate predictions should maintain or boost multiplier"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jeremygreenwood/life-os && python -m pytest tests/test_feedback_loop.py -v`
Expected: FAIL (`_get_accuracy_multiplier` doesn't exist)

**Step 3: Implement accuracy multiplier**

Add to `PredictionEngine`:

```python
def _get_accuracy_multiplier(self, prediction_type: str) -> float:
    """Compute confidence multiplier based on historical accuracy for this prediction type.

    Returns a multiplier (0.0 to 1.1):
    - <20% accuracy after 10+ resolved predictions → 0.0 (auto-suppress)
    - Each dismissal reduces by ~15% from baseline
    - Each accurate prediction boosts by ~10%
    """
    with self.db.get_connection("user_model") as conn:
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate
               FROM predictions
               WHERE prediction_type = ?
                 AND was_surfaced = 1
                 AND resolved_at IS NOT NULL""",
            (prediction_type,),
        ).fetchone()

    total = row["total"] if row else 0
    accurate = row["accurate"] if row else 0

    if total < 5:
        return 1.0  # Not enough data to adjust

    accuracy_rate = accurate / total

    # Auto-suppress types with <20% accuracy after sufficient samples
    if accuracy_rate < 0.2 and total >= 10:
        return 0.0

    # Scale: 50% accuracy = 1.0x, 0% = 0.5x, 100% = 1.1x
    return 0.5 + (accuracy_rate * 0.6)
```

Apply the multiplier in `generate_predictions` after creating each prediction (in the filtering step):

```python
# In generate_predictions, after building the predictions list but before filtering:
for pred in predictions:
    multiplier = self._get_accuracy_multiplier(pred.prediction_type)
    pred.confidence *= multiplier
    pred.confidence_gate = self._gate_from_confidence(pred.confidence)
```

**Step 4: Wire notification dismiss → prediction accuracy update**

In `main.py:_prediction_loop`, pass a `prediction_id` when creating notifications so we can trace back:

```python
# In _prediction_loop, change create_notification to include source_event_id=prediction.id
await self.notification_manager.create_notification(
    title=f"{prediction.prediction_type.title()}: {prediction.description[:80]}",
    body=prediction.description,
    priority="high" if prediction.prediction_type in ("conflict", "risk") else "normal",
    source_event_id=prediction.id,  # Link notification to prediction
    domain="prediction",
)
```

In `services/notification_manager/manager.py`, update `dismiss` and `mark_acted_on` to also update the linked prediction:

```python
async def dismiss(self, notif_id: str):
    self._mark_status(notif_id, "dismissed")
    self._update_linked_prediction(notif_id, was_accurate=False)
    if self.bus and self.bus.is_connected:
        await self.bus.publish(
            "notification.dismissed",
            {"notification_id": notif_id},
            source="notification_manager",
        )

async def mark_acted_on(self, notif_id: str):
    self._mark_status(notif_id, "acted_on")
    self._update_linked_prediction(notif_id, was_accurate=True)
    if self.bus and self.bus.is_connected:
        await self.bus.publish(
            "notification.acted_on",
            {"notification_id": notif_id},
            source="notification_manager",
        )

def _update_linked_prediction(self, notif_id: str, was_accurate: bool):
    """If this notification came from a prediction, update prediction accuracy."""
    with self.db.get_connection("state") as conn:
        notif = conn.execute(
            "SELECT source_event_id, domain FROM notifications WHERE id = ?",
            (notif_id,),
        ).fetchone()

    if not notif or notif["domain"] != "prediction" or not notif["source_event_id"]:
        return

    prediction_id = notif["source_event_id"]
    with self.db.get_connection("user_model") as conn:
        conn.execute(
            """UPDATE predictions SET
               was_accurate = ?, resolved_at = ?,
               user_response = ?
               WHERE id = ?""",
            (
                1 if was_accurate else 0,
                datetime.now(timezone.utc).isoformat(),
                "acted_on" if was_accurate else "dismissed",
                prediction_id,
            ),
        )
```

**Step 5: Run tests**

Run: `cd /Users/jeremygreenwood/life-os && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add services/prediction_engine/engine.py services/notification_manager/manager.py main.py tests/test_feedback_loop.py
git commit -m "feat: wire feedback loop — dismissed/acted notifications update prediction accuracy"
```

---

### Task 5: Raise confidence floor and cap predictions per cycle

**Files:**
- Modify: `services/prediction_engine/engine.py:50-85` (`generate_predictions`)

**Step 1: Write the failing test**

Add to `tests/test_prediction_engine.py`:

```python
@pytest.mark.asyncio
async def test_low_confidence_predictions_not_surfaced(db, event_store, user_model_store):
    """Predictions below 0.6 confidence should be stored but not surfaced."""
    engine = PredictionEngine(db, user_model_store)
    # Patch _has_new_events to always return True for this test
    engine._last_event_cursor = 0

    predictions = await engine.generate_predictions({})
    for pred in predictions:
        assert pred.confidence >= 0.6, f"Prediction surfaced with confidence {pred.confidence} < 0.6"


@pytest.mark.asyncio
async def test_max_five_predictions_per_cycle(db, event_store, user_model_store):
    """At most 5 predictions should be surfaced per cycle."""
    engine = PredictionEngine(db, user_model_store)
    engine._last_event_cursor = 0

    predictions = await engine.generate_predictions({})
    assert len(predictions) <= 5
```

**Step 2: Implement confidence floor and cap**

In `generate_predictions`, after the reaction gatekeeper filtering:

```python
# Raise confidence floor — don't surface anything below 0.6
filtered = [p for p in filtered if p.confidence >= 0.6]

# Cap at 5 surfaced predictions per cycle, prioritized by confidence
filtered.sort(key=lambda p: p.confidence, reverse=True)
filtered = filtered[:5]
```

**Step 3: Run tests**

Run: `cd /Users/jeremygreenwood/life-os && python -m pytest tests/test_prediction_engine.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add services/prediction_engine/engine.py tests/test_prediction_engine.py
git commit -m "feat: raise confidence floor to 0.6, cap 5 predictions per cycle"
```

---

### Task 6: Add `/api/insights/summary` endpoint

Surface the signal profiles already being collected as human-readable insights.

**Files:**
- Modify: `web/routes.py` (add endpoint)
- Modify: `web/schemas.py` (add response schema if needed)
- Test: `tests/test_insights_api.py`

**Step 1: Write the failing test**

Create `tests/test_insights_api.py`:

```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(db, event_store, user_model_store):
    """Create a minimal FastAPI app with Life OS routes for testing."""
    from unittest.mock import MagicMock, AsyncMock
    from web.app import create_web_app

    # Create a minimal LifeOS mock with real db components
    life_os = MagicMock()
    life_os.db = db
    life_os.event_store = event_store
    life_os.user_model_store = user_model_store
    life_os.signal_extractor = MagicMock()
    life_os.signal_extractor.get_user_summary.return_value = {"profiles": {}}
    life_os.vector_store = MagicMock()
    life_os.event_bus = MagicMock()
    life_os.event_bus.is_connected = False
    life_os.connectors = []
    life_os.notification_manager = MagicMock()
    life_os.feedback_collector = MagicMock()
    life_os.feedback_collector.get_feedback_summary.return_value = {}

    app = create_web_app(life_os)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_insights_summary_endpoint_exists(client):
    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    data = response.json()
    assert "insights" in data
    assert isinstance(data["insights"], list)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jeremygreenwood/life-os && python -m pytest tests/test_insights_api.py -v`
Expected: FAIL (404 — endpoint doesn't exist)

**Step 3: Implement the endpoint**

Add to `web/routes.py` in `register_routes`, after the user model section:

```python
# -------------------------------------------------------------------
# Insights
# -------------------------------------------------------------------

@app.get("/api/insights/summary")
async def insights_summary():
    """Aggregate signal profiles into human-readable insights."""
    insights = []

    # --- Relationship insights from cadence/relationship profiles ---
    rel_profile = life_os.user_model_store.get_signal_profile("relationships")
    if rel_profile:
        contacts = rel_profile["data"].get("contacts", {})
        now = datetime.now(timezone.utc)

        for addr, data in contacts.items():
            count = data.get("interaction_count", 0)
            last = data.get("last_interaction")
            if not last or count < 3:
                continue

            try:
                last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                days_since = (now - last_dt).days
            except (ValueError, TypeError):
                continue

            # Compute average gap from timestamps
            timestamps = data.get("interaction_timestamps", [])
            if len(timestamps) >= 3:
                try:
                    dts = sorted([
                        datetime.fromisoformat(t.replace("Z", "+00:00"))
                        for t in timestamps[-10:]
                    ])
                    gaps = [(dts[i + 1] - dts[i]).days for i in range(len(dts) - 1)]
                    avg_gap = sum(gaps) / len(gaps) if gaps else None
                except (ValueError, TypeError):
                    avg_gap = None

                if avg_gap and days_since > avg_gap * 1.5 and days_since > 7:
                    insights.append({
                        "type": "relationship_intelligence",
                        "summary": f"Haven't heard from {addr} in {days_since} days (usually every ~{int(avg_gap)} days)",
                        "confidence": min(0.8, 0.4 + (days_since / avg_gap - 1.5) * 0.2),
                        "category": "contact_gap",
                        "entity": addr,
                    })

    # --- Cadence insights ---
    cadence_profile = life_os.user_model_store.get_signal_profile("cadence")
    if cadence_profile:
        cadence_data = cadence_profile["data"]
        # Per-contact response time comparison
        contact_times = cadence_data.get("per_contact_response_times", {})
        overall_avg = cadence_data.get("overall_avg_response_seconds")

        if overall_avg and contact_times:
            for addr, avg_time in sorted(contact_times.items(), key=lambda x: x[1])[:5]:
                if avg_time < overall_avg * 0.3:
                    insights.append({
                        "type": "behavioral_pattern",
                        "summary": f"You respond to {addr} much faster than average ({int(avg_time/60)}min vs {int(overall_avg/60)}min overall)",
                        "confidence": 0.7,
                        "category": "response_time",
                        "entity": addr,
                    })

    # --- Linguistic insights ---
    ling_profile = life_os.user_model_store.get_signal_profile("linguistic")
    if ling_profile:
        ling_data = ling_profile["data"]
        formality = ling_data.get("avg_formality")
        emoji_rate = ling_data.get("emoji_usage_rate", 0)

        if formality is not None:
            style = "formal" if formality > 0.6 else ("casual" if formality < 0.4 else "balanced")
            insights.append({
                "type": "behavioral_pattern",
                "summary": f"Your writing style is generally {style} (formality score: {formality:.2f})",
                "confidence": 0.6,
                "category": "writing_style",
            })

    # --- Event volume insights ---
    with life_os.db.get_connection("events") as conn:
        rows = conn.execute(
            """SELECT
                CASE CAST(strftime('%w', timestamp) AS INTEGER)
                    WHEN 0 THEN 'Sunday' WHEN 1 THEN 'Monday'
                    WHEN 2 THEN 'Tuesday' WHEN 3 THEN 'Wednesday'
                    WHEN 4 THEN 'Thursday' WHEN 5 THEN 'Friday'
                    WHEN 6 THEN 'Saturday'
                END as day_name,
                COUNT(*) as cnt
               FROM events
               WHERE type IN ('email.received', 'email.sent')
                 AND timestamp > datetime('now', '-30 days')
               GROUP BY strftime('%w', timestamp)
               ORDER BY cnt DESC"""
        ).fetchall()

    if rows and len(rows) >= 3:
        peak_day = rows[0]
        low_day = rows[-1]
        if peak_day["cnt"] > low_day["cnt"] * 1.5:
            insights.append({
                "type": "behavioral_pattern",
                "summary": f"Your busiest email day is {peak_day['day_name']} ({peak_day['cnt']} emails/month) vs {low_day['day_name']} ({low_day['cnt']})",
                "confidence": 0.8,
                "category": "email_volume",
            })

    # --- Place insights (from iOS context) ---
    with life_os.db.get_connection("entities") as conn:
        places = conn.execute(
            "SELECT name, visit_count, place_type FROM places WHERE visit_count > 3 ORDER BY visit_count DESC LIMIT 5"
        ).fetchall()

    for place in places:
        insights.append({
            "type": "behavioral_pattern",
            "summary": f"You've visited {place['name']} {place['visit_count']} times",
            "confidence": 0.9,
            "category": "place_frequency",
            "entity": place["name"],
        })

    # Sort by confidence descending
    insights.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    return {"insights": insights, "generated_at": datetime.now(timezone.utc).isoformat()}
```

**Step 4: Run tests**

Run: `cd /Users/jeremygreenwood/life-os && python -m pytest tests/test_insights_api.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add web/routes.py tests/test_insights_api.py
git commit -m "feat: add /api/insights/summary endpoint surfacing signal data"
```

---

## Phase 2: InsightEngine + Claude Code Agent

### Task 7: Create InsightEngine service skeleton

**Files:**
- Create: `services/insight_engine/__init__.py`
- Create: `services/insight_engine/engine.py`
- Create: `services/insight_engine/models.py`
- Modify: `storage/manager.py` (add `insights` table to `user_model.db`)
- Test: `tests/test_insight_engine.py`

**Step 1: Create the Insight model**

Create `services/insight_engine/models.py`:

```python
from __future__ import annotations

import uuid
import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class Insight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "behavioral_pattern", "actionable_alert", "relationship_intelligence"
    summary: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)  # event/signal IDs
    category: str = ""  # sub-type: "place_frequency", "contact_gap", etc.
    entity: Optional[str] = None  # subject of the insight (contact, place, etc.)
    staleness_ttl_hours: int = 168  # 7 days default
    dedup_key: str = ""  # hash for deduplication
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    feedback: Optional[str] = None  # "useful", "dismissed", None

    def compute_dedup_key(self) -> str:
        raw = f"{self.type}:{self.category}:{self.entity or ''}"
        self.dedup_key = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self.dedup_key
```

Create empty `services/insight_engine/__init__.py`.

**Step 2: Add insights table**

In `storage/manager.py`, add to `_init_user_model_db` after the existing tables:

```sql
-- Insights (cross-signal discoveries)
CREATE TABLE IF NOT EXISTS insights (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    summary         TEXT NOT NULL,
    confidence      REAL NOT NULL,
    evidence        TEXT DEFAULT '[]',
    category        TEXT DEFAULT '',
    entity          TEXT,
    staleness_ttl_hours INTEGER DEFAULT 168,
    dedup_key       TEXT,
    feedback        TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(type);
CREATE INDEX IF NOT EXISTS idx_insights_dedup ON insights(dedup_key);
CREATE INDEX IF NOT EXISTS idx_insights_created ON insights(created_at);
```

**Step 3: Create the InsightEngine**

Create `services/insight_engine/engine.py`:

```python
"""
Life OS — Insight Engine

Cross-correlates signals from multiple extractors to produce
human-readable insights. Runs hourly. Unlike the prediction engine
(forward-looking guesses), insights are backward-looking discoveries
grounded in observed data.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from services.insight_engine.models import Insight
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


class InsightEngine:
    """Produces cross-signal insights by correlating data from multiple sources."""

    def __init__(self, db: DatabaseManager, ums: UserModelStore):
        self.db = db
        self.ums = ums

    async def generate_insights(self) -> list[Insight]:
        """Main insight generation loop. Called hourly."""
        insights: list[Insight] = []

        insights.extend(self._place_frequency_insights())
        insights.extend(self._contact_gap_insights())
        insights.extend(self._email_volume_insights())
        insights.extend(self._communication_style_insights())

        # Deduplicate — don't resurface the same insight if it's still fresh
        deduped = self._deduplicate(insights)

        # Store new insights
        for insight in deduped:
            self._store_insight(insight)

        return deduped

    def _place_frequency_insights(self) -> list[Insight]:
        """Discover place visit patterns from iOS context data."""
        insights = []
        with self.db.get_connection("entities") as conn:
            places = conn.execute(
                "SELECT name, visit_count, place_type FROM places WHERE visit_count > 3 ORDER BY visit_count DESC LIMIT 10"
            ).fetchall()

        for place in places:
            insight = Insight(
                type="behavioral_pattern",
                summary=f"You've visited {place['name']} {place['visit_count']} times",
                confidence=min(0.9, 0.5 + place["visit_count"] * 0.05),
                category="place_frequency",
                entity=place["name"],
            )
            insight.compute_dedup_key()
            insights.append(insight)

        return insights

    def _contact_gap_insights(self) -> list[Insight]:
        """Detect contacts going cold based on relationship profiles."""
        insights = []
        rel_profile = self.ums.get_signal_profile("relationships")
        if not rel_profile:
            return insights

        contacts = rel_profile["data"].get("contacts", {})
        now = datetime.now(timezone.utc)

        for addr, data in contacts.items():
            count = data.get("interaction_count", 0)
            last = data.get("last_interaction")
            if not last or count < 5:
                continue

            try:
                last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                days_since = (now - last_dt).days
            except (ValueError, TypeError):
                continue

            timestamps = data.get("interaction_timestamps", [])
            if len(timestamps) < 3:
                continue

            try:
                dts = sorted([datetime.fromisoformat(t.replace("Z", "+00:00")) for t in timestamps[-10:]])
                gaps = [(dts[i + 1] - dts[i]).days for i in range(len(dts) - 1)]
                avg_gap = sum(gaps) / len(gaps) if gaps else 30
            except (ValueError, TypeError):
                continue

            if days_since > avg_gap * 1.5 and days_since > 7:
                confidence = min(0.8, 0.4 + (days_since / avg_gap - 1.5) * 0.2)
                insight = Insight(
                    type="relationship_intelligence",
                    summary=f"Haven't heard from {addr} in {days_since} days (usually every ~{int(avg_gap)} days)",
                    confidence=confidence,
                    category="contact_gap",
                    entity=addr,
                )
                insight.compute_dedup_key()
                insights.append(insight)

        return insights

    def _email_volume_insights(self) -> list[Insight]:
        """Analyze email volume patterns by day of week."""
        insights = []
        with self.db.get_connection("events") as conn:
            rows = conn.execute(
                """SELECT
                    CASE CAST(strftime('%w', timestamp) AS INTEGER)
                        WHEN 0 THEN 'Sunday' WHEN 1 THEN 'Monday'
                        WHEN 2 THEN 'Tuesday' WHEN 3 THEN 'Wednesday'
                        WHEN 4 THEN 'Thursday' WHEN 5 THEN 'Friday'
                        WHEN 6 THEN 'Saturday'
                    END as day_name,
                    COUNT(*) as cnt
                   FROM events
                   WHERE type IN ('email.received', 'email.sent')
                     AND timestamp > datetime('now', '-30 days')
                   GROUP BY strftime('%w', timestamp)
                   ORDER BY cnt DESC"""
            ).fetchall()

        if rows and len(rows) >= 3:
            peak = rows[0]
            low = rows[-1]
            if peak["cnt"] > low["cnt"] * 1.5:
                insight = Insight(
                    type="behavioral_pattern",
                    summary=f"Busiest email day: {peak['day_name']} ({peak['cnt']}/month) vs {low['day_name']} ({low['cnt']})",
                    confidence=0.8,
                    category="email_volume",
                )
                insight.compute_dedup_key()
                insights.append(insight)

        return insights

    def _communication_style_insights(self) -> list[Insight]:
        """Surface linguistic profile discoveries."""
        insights = []
        ling = self.ums.get_signal_profile("linguistic")
        if not ling:
            return insights

        data = ling["data"]
        formality = data.get("avg_formality")
        if formality is not None:
            style = "formal" if formality > 0.6 else ("casual" if formality < 0.4 else "balanced")
            insight = Insight(
                type="behavioral_pattern",
                summary=f"Your writing style is generally {style} (formality: {formality:.2f})",
                confidence=0.6,
                category="writing_style",
            )
            insight.compute_dedup_key()
            insights.append(insight)

        return insights

    def _deduplicate(self, insights: list[Insight]) -> list[Insight]:
        """Remove insights that are still fresh in the database."""
        deduped = []
        for insight in insights:
            with self.db.get_connection("user_model") as conn:
                existing = conn.execute(
                    """SELECT created_at FROM insights
                       WHERE dedup_key = ?
                       ORDER BY created_at DESC LIMIT 1""",
                    (insight.dedup_key,),
                ).fetchone()

            if existing:
                try:
                    created = datetime.fromisoformat(existing["created_at"].replace("Z", "+00:00"))
                    age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
                    if age_hours < insight.staleness_ttl_hours:
                        continue  # Still fresh, skip
                except (ValueError, TypeError):
                    pass

            deduped.append(insight)

        return deduped

    def _store_insight(self, insight: Insight):
        """Persist an insight to the database."""
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO insights
                   (id, type, summary, confidence, evidence, category, entity,
                    staleness_ttl_hours, dedup_key, feedback, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    insight.id, insight.type, insight.summary, insight.confidence,
                    json.dumps(insight.evidence), insight.category, insight.entity,
                    insight.staleness_ttl_hours, insight.dedup_key, insight.feedback,
                    insight.created_at,
                ),
            )
```

**Step 4: Write test**

Create `tests/test_insight_engine.py`:

```python
import pytest
from services.insight_engine.engine import InsightEngine


@pytest.mark.asyncio
async def test_insight_engine_runs_without_data(db, user_model_store):
    """InsightEngine should return empty insights when no data exists."""
    engine = InsightEngine(db, user_model_store)
    insights = await engine.generate_insights()
    assert isinstance(insights, list)


@pytest.mark.asyncio
async def test_insight_deduplication(db, user_model_store):
    """Same insight should not be generated twice within staleness TTL."""
    engine = InsightEngine(db, user_model_store)

    # Run twice — second run should not duplicate insights
    first = await engine.generate_insights()
    second = await engine.generate_insights()

    if first:
        first_keys = {i.dedup_key for i in first}
        second_keys = {i.dedup_key for i in second}
        assert first_keys.isdisjoint(second_keys), "Deduplicated insights should not reappear"
```

**Step 5: Run tests**

Run: `cd /Users/jeremygreenwood/life-os && python -m pytest tests/test_insight_engine.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add services/insight_engine/ storage/manager.py tests/test_insight_engine.py
git commit -m "feat: add InsightEngine service with dedup, place/contact/email/style insights"
```

---

### Task 8: Wire InsightEngine into main.py

**Files:**
- Modify: `main.py` (add InsightEngine to init, add hourly loop)

**Step 1: Add InsightEngine to LifeOS.__init__**

After the prediction engine initialization (line 77), add:

```python
from services.insight_engine.engine import InsightEngine
# ...
self.insight_engine = InsightEngine(self.db, self.user_model_store)
```

**Step 2: Add hourly insight loop**

Add a new method to LifeOS and start it in `start()` alongside the prediction loop:

```python
async def _insight_loop(self):
    """Run the insight engine every hour."""
    while not self.shutdown_event.is_set():
        try:
            insights = await self.insight_engine.generate_insights()
            if insights:
                print(f"  InsightEngine: generated {len(insights)} new insights")
        except Exception as e:
            print(f"Insight engine error: {e}")
        await asyncio.sleep(3600)  # 1 hour
```

In `start()`, after `asyncio.create_task(self._prediction_loop())`, add:

```python
asyncio.create_task(self._insight_loop())
```

**Step 3: Add `/api/insights` endpoint to serve stored insights**

Add to `web/routes.py`:

```python
@app.get("/api/insights")
async def list_insights(limit: int = 20):
    """Return recent insights from the InsightEngine."""
    with life_os.db.get_connection("user_model") as conn:
        rows = conn.execute(
            """SELECT * FROM insights
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return {"insights": [dict(r) for r in rows]}

@app.post("/api/insights/{insight_id}/feedback")
async def insight_feedback(insight_id: str, feedback: str = "dismissed"):
    """Record user feedback on an insight."""
    with life_os.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE insights SET feedback = ? WHERE id = ?",
            (feedback, insight_id),
        )
    return {"status": "recorded"}
```

**Step 4: Commit**

```bash
git add main.py web/routes.py
git commit -m "feat: wire InsightEngine into main loop and add /api/insights endpoints"
```

---

### Task 9: Create `improve-lifeos` Claude Code skill

This skill tells Claude Code how to analyze Life OS data quality and make improvements.

**Files:**
- Create: `scripts/improve-lifeos.md` (the skill content)
- Create: `scripts/analyze-data-quality.py` (helper script the skill invokes)

**Step 1: Create the analysis helper script**

Create `scripts/analyze-data-quality.py`:

```python
"""
Life OS — Data Quality Analysis

Run by the Claude Code improvement agent to understand current data quality
and identify areas for improvement. Outputs a JSON report.

Usage: python scripts/analyze-data-quality.py [--data-dir ./data]
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


def analyze(data_dir: str = "./data") -> dict:
    data_path = Path(data_dir)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "sections": {}}

    # --- Event volume ---
    events_db = str(data_path / "events.db")
    try:
        conn = sqlite3.connect(events_db)
        conn.row_factory = sqlite3.Row

        total = conn.execute("SELECT COUNT(*) as c FROM events").fetchone()["c"]
        by_type = conn.execute(
            "SELECT type, COUNT(*) as c FROM events GROUP BY type ORDER BY c DESC LIMIT 20"
        ).fetchall()
        last_24h = conn.execute(
            "SELECT COUNT(*) as c FROM events WHERE timestamp > datetime('now', '-1 day')"
        ).fetchone()["c"]

        report["sections"]["events"] = {
            "total": total,
            "last_24h": last_24h,
            "top_types": {r["type"]: r["c"] for r in by_type},
        }
        conn.close()
    except Exception as e:
        report["sections"]["events"] = {"error": str(e)}

    # --- Prediction accuracy ---
    um_db = str(data_path / "user_model.db")
    try:
        conn = sqlite3.connect(um_db)
        conn.row_factory = sqlite3.Row

        pred_stats = conn.execute(
            """SELECT prediction_type,
                COUNT(*) as total,
                SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate,
                SUM(CASE WHEN was_accurate = 0 THEN 1 ELSE 0 END) as inaccurate,
                SUM(CASE WHEN was_accurate IS NULL THEN 1 ELSE 0 END) as unresolved
               FROM predictions
               WHERE was_surfaced = 1
               GROUP BY prediction_type"""
        ).fetchall()

        report["sections"]["prediction_accuracy"] = {
            r["prediction_type"]: {
                "total": r["total"],
                "accurate": r["accurate"],
                "inaccurate": r["inaccurate"],
                "unresolved": r["unresolved"],
                "accuracy_rate": r["accurate"] / max(r["total"] - r["unresolved"], 1),
            }
            for r in pred_stats
        }

        # Signal profiles freshness
        profiles = conn.execute(
            "SELECT profile_type, samples_count, updated_at FROM signal_profiles"
        ).fetchall()
        report["sections"]["signal_profiles"] = {
            r["profile_type"]: {"samples": r["samples_count"], "last_updated": r["updated_at"]}
            for r in profiles
        }

        # Insight feedback
        insight_stats = conn.execute(
            """SELECT type, feedback, COUNT(*) as c
               FROM insights
               GROUP BY type, feedback"""
        ).fetchall()
        report["sections"]["insight_feedback"] = [
            {"type": r["type"], "feedback": r["feedback"], "count": r["c"]}
            for r in insight_stats
        ]

        conn.close()
    except Exception as e:
        report["sections"]["prediction_accuracy"] = {"error": str(e)}

    # --- Notification dismissal rate ---
    state_db = str(data_path / "state.db")
    try:
        conn = sqlite3.connect(state_db)
        conn.row_factory = sqlite3.Row

        notif_stats = conn.execute(
            """SELECT status, COUNT(*) as c FROM notifications GROUP BY status"""
        ).fetchall()
        report["sections"]["notifications"] = {r["status"]: r["c"] for r in notif_stats}

        conn.close()
    except Exception as e:
        report["sections"]["notifications"] = {"error": str(e)}

    # --- Feedback log ---
    pref_db = str(data_path / "preferences.db")
    try:
        conn = sqlite3.connect(pref_db)
        conn.row_factory = sqlite3.Row

        feedback = conn.execute(
            """SELECT action_type, feedback_type, COUNT(*) as c
               FROM feedback_log
               GROUP BY action_type, feedback_type
               ORDER BY c DESC"""
        ).fetchall()
        report["sections"]["feedback"] = [
            {"action_type": r["action_type"], "feedback_type": r["feedback_type"], "count": r["c"]}
            for r in feedback
        ]

        conn.close()
    except Exception as e:
        report["sections"]["feedback"] = {"error": str(e)}

    return report


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    result = analyze(data_dir)
    print(json.dumps(result, indent=2))
```

**Step 2: Create the Claude Code skill**

Create `scripts/improve-lifeos.md`:

```markdown
# Improve Life OS

You are an autonomous improvement agent for the Life OS personal assistant.
Your job is to analyze data quality, identify problems, and make targeted fixes.

## Step 1: Analyze

Run the data quality analysis script:

```bash
cd /Users/jeremygreenwood/life-os
source .venv/bin/activate
python scripts/analyze-data-quality.py
```

Read the output carefully. Look for:
- Prediction types with <50% accuracy rate → needs filter or threshold adjustment
- Notification dismiss rate >60% → too noisy, raise confidence floors
- Signal profiles with 0 samples → extractor may not be processing events
- Insights with "dismissed" feedback → user doesn't find these valuable
- High-volume event types not producing insights → missing correlator

## Step 2: Diagnose

Read the relevant source files to understand why the problem exists:
- `services/prediction_engine/engine.py` — prediction generation
- `services/insight_engine/engine.py` — insight correlation
- `services/signal_extractor/pipeline.py` — signal extraction
- `web/routes.py` — API and UI

Check recent git log to see what was changed in prior improvement runs.

## Step 3: Fix

Make targeted fixes. Priority order:
1. **Reduce noise** — add filters, raise thresholds, suppress inaccurate prediction types
2. **Add missing insights** — if data exists but no insight correlator, add one
3. **Improve accuracy** — adjust confidence scoring based on feedback data
4. **UI improvements** — if new insight types exist without UI rendering, add them

## Step 4: Verify

Run tests: `python -m pytest tests/ -v`
Check the app starts: `python -c "from main import LifeOS; print('OK')"`

## Step 5: Commit

Create a branch, commit changes, and open a PR:

```bash
git checkout -b improve/$(date +%Y-%m-%d)
git add -A
git commit -m "improve: [description of changes]"
git push -u origin improve/$(date +%Y-%m-%d)
gh pr create --title "Weekly improvement: [summary]" --body "Auto-generated by improvement agent"
```

## Constraints

- Never modify user data (the `data/` directory contents)
- Never change `config/settings.yaml`
- Always run tests before committing
- Keep changes minimal and focused — one problem per PR
- If unsure about a change, log it in `data/improvement-runs/` and skip
```

**Step 3: Commit**

```bash
git add scripts/improve-lifeos.md scripts/analyze-data-quality.py
git commit -m "feat: add improve-lifeos skill and data quality analysis script"
```

---

### Task 10: Set up launchd for weekly Claude Code runs

**Files:**
- Create: `scripts/com.lifeos.improve.plist`
- Create: `scripts/run-improvement.sh`

**Step 1: Create the runner script**

Create `scripts/run-improvement.sh`:

```bash
#!/usr/bin/env bash
# Life OS — Weekly Improvement Agent Runner
set -euo pipefail

LOG_DIR="/Users/jeremygreenwood/life-os/data/improvement-runs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d_%H%M%S).log"

echo "=== Life OS Improvement Run ===" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"

cd /Users/jeremygreenwood/life-os

# Run Claude Code with the improvement skill
claude --print --skill scripts/improve-lifeos.md \
    "Analyze Life OS data quality and make improvements" \
    >> "$LOG_FILE" 2>&1 || true

echo "Completed: $(date)" >> "$LOG_FILE"
```

**Step 2: Create launchd plist**

Create `scripts/com.lifeos.improve.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lifeos.improve</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/jeremygreenwood/life-os/scripts/run-improvement.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>22</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/jeremygreenwood/life-os/data/improvement-runs/launchd-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/jeremygreenwood/life-os/data/improvement-runs/launchd-stderr.log</string>
    <key>WorkingDirectory</key>
    <string>/Users/jeremygreenwood/life-os</string>
</dict>
</plist>
```

**Step 3: Installation instructions (do NOT auto-install)**

To install:
```bash
cp scripts/com.lifeos.improve.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.lifeos.improve.plist
```

To manually trigger:
```bash
launchctl kickstart gui/$(id -u)/com.lifeos.improve
```

To uninstall:
```bash
launchctl unload ~/Library/LaunchAgents/com.lifeos.improve.plist
rm ~/Library/LaunchAgents/com.lifeos.improve.plist
```

**Step 4: Commit**

```bash
chmod +x scripts/run-improvement.sh
git add scripts/com.lifeos.improve.plist scripts/run-improvement.sh
git commit -m "feat: add launchd plist for weekly Claude Code improvement runs"
```

---

## Summary of all tasks

| # | Task | Phase | Key files |
|---|------|-------|-----------|
| 1 | Test infrastructure | 1 | `tests/conftest.py`, `tests/test_prediction_engine.py` |
| 2 | Prediction throttling | 1 | `services/prediction_engine/engine.py`, `main.py` |
| 3 | Marketing pre-filter | 1 | `services/prediction_engine/engine.py` |
| 4 | Feedback loop wiring | 1 | `notification_manager/manager.py`, `prediction_engine/engine.py`, `main.py` |
| 5 | Confidence floor + cap | 1 | `services/prediction_engine/engine.py` |
| 6 | `/api/insights/summary` | 1 | `web/routes.py` |
| 7 | InsightEngine skeleton | 2 | `services/insight_engine/`, `storage/manager.py` |
| 8 | Wire InsightEngine | 2 | `main.py`, `web/routes.py` |
| 9 | `improve-lifeos` skill | 2 | `scripts/improve-lifeos.md`, `scripts/analyze-data-quality.py` |
| 10 | launchd weekly agent | 2 | `scripts/com.lifeos.improve.plist`, `scripts/run-improvement.sh` |
