"""
Life OS — Rules Engine

Deterministic automations that fire instantly without AI latency or cost.
Rules are evaluated against every event on the bus.

Rule format (stored in preferences.db):
    {
        "name": "Urgent email from boss",
        "trigger_event": "email.received",
        "conditions": [
            {"field": "payload.from_address", "op": "in", "value": ["boss@company.com"]},
            {"field": "payload.subject", "op": "contains_any", "value": ["urgent", "ASAP"]}
        ],
        "actions": [
            {"type": "notify", "priority": "high"},
            {"type": "tag", "value": "urgent-boss"}
        ]
    }

Supported operators: eq, neq, contains, contains_any, in, not_in, gt, lt, exists
Supported actions: notify, archive, tag, forward, auto_reply, create_task, suppress
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class RulesEngine:
    """Evaluates rules against events and executes matching actions."""

    def __init__(self, db: DatabaseManager, event_bus: Any = None, config: dict | None = None):
        self.db = db
        self.bus = event_bus
        # --- Rule caching ---
        # Rules are loaded from DB into memory to avoid per-event DB queries.
        # The cache is refreshed every 60 seconds (see evaluate()).
        self._rules_cache: list[dict] = []
        self._cache_loaded_at: Optional[datetime] = None

        # --- Per-rule telemetry throttle ---
        # Tracks {rule_id: last_published_monotonic_time} so we avoid
        # flooding the events table with system.rule.triggered events.
        # The rules table (last_triggered, times_triggered) is always
        # updated regardless of throttle state.
        self._telemetry_last_published: dict[str, float] = {}
        config = config or {}
        self._telemetry_throttle_seconds: float = float(
            config.get("telemetry_throttle_seconds", 300)
        )

    async def _publish_telemetry(self, event_type: str, payload: dict):
        """Publish a telemetry event if the event bus is available."""
        if self.bus and self.bus.is_connected:
            await self.bus.publish(event_type, payload, source="rules_engine")

    def load_rules(self):
        """
        Load active rules from the database into memory.

        Only active rules (is_active=1) are loaded. Conditions and actions
        are stored as JSON strings in the DB and deserialized here so the
        evaluation path works with native Python dicts.

        Rules with malformed JSON are skipped (logged as warnings) so that
        one bad rule doesn't prevent all other rules from loading.
        """
        with self.db.get_connection("preferences") as conn:
            rows = conn.execute(
                "SELECT * FROM rules WHERE is_active = 1 ORDER BY ROWID"
            ).fetchall()

            self._rules_cache = []
            skipped = 0
            for row in rows:
                rule = dict(row)
                try:
                    # Deserialize JSON-encoded conditions and actions
                    rule["conditions"] = json.loads(rule["conditions"])
                    rule["actions"] = json.loads(rule["actions"])
                    self._rules_cache.append(rule)
                except (json.JSONDecodeError, TypeError) as exc:
                    skipped += 1
                    logger.warning(
                        "Skipping rule %s (%s): malformed JSON — %s",
                        rule.get("id", "?"), rule.get("name", "?"), exc,
                    )

            self._cache_loaded_at = datetime.now(timezone.utc)
            if skipped:
                logger.info(
                    "Rules cache refreshed: %d active rules loaded (%d skipped due to errors)",
                    len(self._rules_cache), skipped,
                )
            else:
                logger.info("Rules cache refreshed: %d active rules loaded", len(self._rules_cache))

    async def evaluate(self, event: dict) -> list[dict]:
        """
        Evaluate all rules against an event.
        Returns a list of actions to execute.

        Rule evaluation flow (for each rule):
            1. Trigger matching  -> does the rule's trigger_event match this event type?
            2. Condition evaluation -> do ALL conditions pass? (AND logic)
            3. Action collection   -> if yes, collect the rule's actions for execution

        This runs on every event on the bus, so it must be fast. The
        in-memory cache avoids DB round-trips on the hot path.
        """
        # --- Periodic cache reload ---
        # Refresh the rules cache every 60 seconds to pick up changes
        # without requiring a full restart. The staleness window is
        # acceptable because rule changes are infrequent.
        if (
            not self._cache_loaded_at
            or (datetime.now(timezone.utc) - self._cache_loaded_at).seconds > 60
        ):
            self.load_rules()

        event_type = event.get("type", "")
        matching_actions = []

        matched = 0
        for rule in self._rules_cache:
            try:
                # Step 1: Trigger matching — does this rule care about this event type?
                # Supports exact match, wildcard ("*"), and glob patterns ("email.*").
                if not self._matches_trigger(rule["trigger_event"], event_type):
                    continue

                # Step 2: Condition evaluation — all conditions must pass (AND logic)
                if self._evaluate_conditions(rule["conditions"], event):
                    matched += 1
                    # Step 3: Action collection — attach rule metadata to each action
                    # so downstream handlers know which rule fired.
                    for action in rule["actions"]:
                        matching_actions.append({
                            "rule_id": rule["id"],
                            "rule_name": rule["name"],
                            **action,
                        })

                    # Record that this rule was triggered (for analytics and
                    # the "times_triggered" display in the rules management UI).
                    await self._record_trigger(rule["id"], rule["name"], event_type, event.get("id"))
            except Exception:
                logger.warning(
                    "Error evaluating rule %s (%s) against event type '%s', skipping",
                    rule.get("id", "?"), rule.get("name", "?"), event_type,
                    exc_info=True,
                )

        logger.debug(
            "Evaluated %d rules against '%s': %d matched",
            len(self._rules_cache), event_type, matched,
        )
        return matching_actions

    async def add_rule(self, name: str, trigger_event: str,
                       conditions: list[dict], actions: list[dict],
                       created_by: str = "user") -> str:
        """Add a new rule and reload the cache so it takes effect immediately."""
        rule_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection("preferences") as conn:
            conn.execute(
                """INSERT INTO rules (id, name, trigger_event, conditions, actions, created_by)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    rule_id, name, trigger_event,
                    json.dumps(conditions), json.dumps(actions), created_by,
                ),
            )

        # Reload cache
        self.load_rules()

        await self._publish_telemetry("system.rule.created", {
            "rule_id": rule_id,
            "name": name,
            "trigger_event": trigger_event,
            "conditions_count": len(conditions),
            "actions_count": len(actions),
            "action_types": [a.get("type") for a in actions],
            "created_by": created_by,
            "created_at": now,
        })

        return rule_id

    async def remove_rule(self, rule_id: str):
        """Deactivate a rule (soft delete — sets is_active=0, not a hard delete)."""
        # Fetch rule details for telemetry before deactivating
        rule_name = None
        with self.db.get_connection("preferences") as conn:
            row = conn.execute("SELECT name FROM rules WHERE id = ?", (rule_id,)).fetchone()
            if row:
                rule_name = row["name"]
            conn.execute("UPDATE rules SET is_active = 0 WHERE id = ?", (rule_id,))
        self.load_rules()

        await self._publish_telemetry("system.rule.deactivated", {
            "rule_id": rule_id,
            "name": rule_name,
            "deactivated_at": datetime.now(timezone.utc).isoformat(),
        })

    def get_all_rules(self) -> list[dict]:
        """Get all rules (active and inactive)."""
        with self.db.get_connection("preferences") as conn:
            rows = conn.execute("SELECT * FROM rules ORDER BY created_at DESC").fetchall()
            rules = []
            for row in rows:
                rule = dict(row)
                rule["conditions"] = json.loads(rule["conditions"])
                rule["actions"] = json.loads(rule["actions"])
                rules.append(rule)
            return rules

    # -------------------------------------------------------------------
    # Evaluation logic
    # -------------------------------------------------------------------

    def _matches_trigger(self, trigger: str, event_type: str) -> bool:
        """
        Check if a trigger pattern matches an event type.

        Supports three matching modes:
            "*"            -> matches every event (catch-all rule)
            "email.*"      -> glob-style wildcard (converted to regex)
            "email.received" -> exact string match
        """
        if trigger == "*":
            return True  # Wildcard: matches all event types
        if "*" in trigger:
            # Convert glob-style pattern to regex: escape dots, then
            # replace "*" with ".*" for regex wildcard matching.
            pattern = trigger.replace(".", r"\.").replace("*", ".*")
            return bool(re.match(pattern, event_type))
        return trigger == event_type  # Exact match

    def _evaluate_conditions(self, conditions: list[dict], event: dict) -> bool:
        """Evaluate all conditions against an event. All must be true (AND logic)."""
        for condition in conditions:
            if not self._evaluate_single_condition(condition, event):
                return False
        return True

    def _evaluate_single_condition(self, condition: dict, event: dict) -> bool:
        """
        Evaluate a single condition against an event.

        Each condition has three parts:
            field  - dot-notation path into the event dict (e.g., "payload.from_address")
            op     - comparison operator (see the operator system below)
            value  - the expected value to compare against

        The operator system supports:
            eq           - exact equality
            neq          - not equal
            contains     - case-insensitive substring match (field must be str)
            contains_any - field contains ANY of the values in the list
            in           - actual value is a member of the expected list
            not_in       - actual value is NOT in the expected list
            gt / lt      - greater than / less than (numeric comparison)
            gte / lte    - greater than or equal / less than or equal
            exists       - field is present and not None
            not_exists   - field is absent or None
            regex        - case-insensitive regex match (field must be str)
        """
        field_path = condition.get("field", "")
        op = condition.get("op", "eq")
        expected = condition.get("value")

        # Resolve the field value from the event using dot-notation traversal
        actual = self._resolve_field(field_path, event)

        # --- Apply the operator ---
        if op == "eq":
            return actual == expected
        elif op == "neq":
            return actual != expected
        elif op == "contains":
            # Case-insensitive substring match
            if not isinstance(expected, str):
                logger.debug("'contains' operator: expected is not a string (got %s), returning False", type(expected).__name__)
                return False
            return isinstance(actual, str) and expected.lower() in actual.lower()
        elif op == "contains_any":
            # True if the field string contains ANY of the expected values
            if not isinstance(actual, str):
                return False
            if not isinstance(expected, list):
                logger.debug("'contains_any' operator: expected is not a list (got %s), returning False", type(expected).__name__)
                return False
            actual_lower = actual.lower()
            return any(isinstance(v, str) and v.lower() in actual_lower for v in expected)
        elif op == "in":
            # Actual value is a member of the expected list
            return actual in expected
        elif op == "not_in":
            return actual not in expected
        elif op == "gt":
            return actual is not None and actual > expected
        elif op == "lt":
            return actual is not None and actual < expected
        elif op == "gte":
            return actual is not None and actual >= expected
        elif op == "lte":
            return actual is not None and actual <= expected
        elif op == "exists":
            return actual is not None
        elif op == "not_exists":
            return actual is None
        elif op == "regex":
            # Case-insensitive regex search within the field value
            return isinstance(actual, str) and bool(re.search(expected, actual, re.IGNORECASE))
        else:
            return False  # Unknown operator — fail closed

    def _resolve_field(self, field_path: str, obj: dict) -> Any:
        """
        Resolve a dotted field path like 'payload.from_address' from a dict.

        Dot-notation field resolution walks the nested dict structure one
        key at a time. For example, "payload.from_address" on the event
        {"type": "email.received", "payload": {"from_address": "a@b.com"}}
        would return "a@b.com". Returns None if any segment is missing.
        """
        parts = field_path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None  # Path broken — intermediate value isn't a dict
        return current

    def _should_publish_telemetry(self, rule_id: str) -> bool:
        """Check whether a system.rule.triggered event should be published for this rule.

        Uses a per-rule monotonic clock throttle so that high-frequency rules
        don't flood the events table.  A throttle of 0 disables throttling
        (every trigger publishes).  The rules table (last_triggered,
        times_triggered) is always updated regardless of this check.
        """
        if self._telemetry_throttle_seconds <= 0:
            return True  # Throttling disabled — publish every time
        now = time.monotonic()
        last = self._telemetry_last_published.get(rule_id, 0.0)
        if now - last >= self._telemetry_throttle_seconds:
            self._telemetry_last_published[rule_id] = now
            return True
        return False

    async def _record_trigger(self, rule_id: str, rule_name: Optional[str] = None,
                              event_type: Optional[str] = None,
                              event_id: Optional[str] = None):
        """Update the trigger count and last-triggered timestamp for a rule, and publish telemetry.

        A DB failure here should never prevent actions from executing,
        so errors are logged and swallowed.
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self.db.get_connection("preferences") as conn:
                conn.execute(
                    """UPDATE rules SET
                       times_triggered = times_triggered + 1,
                       last_triggered = ?
                       WHERE id = ?""",
                    (now, rule_id),
                )
        except Exception:
            logger.warning(
                "Failed to record trigger for rule %s (%s), continuing",
                rule_id, rule_name,
                exc_info=True,
            )

        # Only publish telemetry if the per-rule throttle allows it.
        # The DB record above (times_triggered, last_triggered) is always
        # updated — this only limits the event bus publication to avoid
        # flooding the events table with high-frequency rule telemetry.
        if self._should_publish_telemetry(rule_id):
            await self._publish_telemetry("system.rule.triggered", {
                "rule_id": rule_id,
                "rule_name": rule_name,
                "trigger_event_type": event_type,
                "trigger_event_id": event_id,
                "triggered_at": now,
            })


# -------------------------------------------------------------------
# Default Rules -- Installed on first run
# -------------------------------------------------------------------
# These rules ship with the system and provide sensible out-of-the-box
# automation. They can be deactivated or edited by the user at any time.
# Each rule follows the standard format: trigger_event, conditions, actions.

DEFAULT_RULES = [
    # Auto-tag and suppress marketing emails (detected by "unsubscribe" link)
    {
        "name": "Archive marketing emails",
        "trigger_event": "email.received",
        "conditions": [
            {"field": "payload.body_plain", "op": "contains_any",
             "value": ["unsubscribe", "opt out", "manage preferences"]},
        ],
        "actions": [
            {"type": "tag", "value": "marketing"},
            {"type": "suppress"},
        ],
    },
    # Tag emails with attachments so users can find them easily later
    {
        "name": "Flag emails with attachments",
        "trigger_event": "email.received",
        "conditions": [
            {"field": "payload.has_attachments", "op": "eq", "value": True},
        ],
        "actions": [
            {"type": "tag", "value": "has-attachments"},
        ],
    },
    # Immediately notify the user when a calendar conflict is detected
    {
        "name": "High priority: calendar conflict",
        "trigger_event": "calendar.conflict.detected",
        "conditions": [],
        "actions": [
            {"type": "notify", "priority": "high"},
        ],
    },
    # Alert on transactions over $500 — catches fraud and big purchases
    {
        "name": "Alert on large transactions",
        "trigger_event": "finance.transaction.new",
        "conditions": [
            {"field": "payload.amount", "op": "gt", "value": 500},
        ],
        "actions": [
            {"type": "notify", "priority": "high"},
            {"type": "tag", "value": "large-transaction"},
        ],
    },
]


async def install_default_rules(db: DatabaseManager, event_bus: Any = None):
    """
    Install the default rules on first run.

    Idempotent: checks existing rule names before inserting so running
    this multiple times won't create duplicates. Called during app
    initialization / database migration.
    """
    engine = RulesEngine(db, event_bus=event_bus)
    existing = engine.get_all_rules()
    existing_names = {r["name"] for r in existing}

    for rule in DEFAULT_RULES:
        # Only install if a rule with this name doesn't already exist
        if rule["name"] not in existing_names:
            await engine.add_rule(
                name=rule["name"],
                trigger_event=rule["trigger_event"],
                conditions=rule["conditions"],
                actions=rule["actions"],
                created_by="system",
            )
