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
import re
from datetime import datetime, timezone
from typing import Any, Optional

from storage.database import DatabaseManager


class RulesEngine:
    """Evaluates rules against events and executes matching actions."""

    def __init__(self, db: DatabaseManager):
        self.db = db
        self._rules_cache: list[dict] = []
        self._cache_loaded_at: Optional[datetime] = None

    def load_rules(self):
        """Load active rules from the database into memory."""
        with self.db.get_connection("preferences") as conn:
            rows = conn.execute(
                "SELECT * FROM rules WHERE is_active = 1 ORDER BY ROWID"
            ).fetchall()

            self._rules_cache = []
            for row in rows:
                rule = dict(row)
                rule["conditions"] = json.loads(rule["conditions"])
                rule["actions"] = json.loads(rule["actions"])
                self._rules_cache.append(rule)

            self._cache_loaded_at = datetime.now(timezone.utc)

    async def evaluate(self, event: dict) -> list[dict]:
        """
        Evaluate all rules against an event.
        Returns a list of actions to execute.
        """
        # Reload rules periodically (every 60 seconds)
        if (
            not self._cache_loaded_at
            or (datetime.now(timezone.utc) - self._cache_loaded_at).seconds > 60
        ):
            self.load_rules()

        event_type = event.get("type", "")
        matching_actions = []

        for rule in self._rules_cache:
            # Check if the rule's trigger matches this event type
            if not self._matches_trigger(rule["trigger_event"], event_type):
                continue

            # Evaluate all conditions
            if self._evaluate_conditions(rule["conditions"], event):
                # All conditions met — collect the actions
                for action in rule["actions"]:
                    matching_actions.append({
                        "rule_id": rule["id"],
                        "rule_name": rule["name"],
                        **action,
                    })

                # Update rule trigger count
                self._record_trigger(rule["id"])

        return matching_actions

    def add_rule(self, name: str, trigger_event: str,
                 conditions: list[dict], actions: list[dict],
                 created_by: str = "user") -> str:
        """Add a new rule."""
        import uuid
        rule_id = str(uuid.uuid4())

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
        return rule_id

    def remove_rule(self, rule_id: str):
        """Deactivate a rule."""
        with self.db.get_connection("preferences") as conn:
            conn.execute("UPDATE rules SET is_active = 0 WHERE id = ?", (rule_id,))
        self.load_rules()

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
        """Check if a trigger pattern matches an event type."""
        if trigger == "*":
            return True
        if "*" in trigger:
            pattern = trigger.replace(".", r"\.").replace("*", ".*")
            return bool(re.match(pattern, event_type))
        return trigger == event_type

    def _evaluate_conditions(self, conditions: list[dict], event: dict) -> bool:
        """Evaluate all conditions against an event. All must be true (AND)."""
        for condition in conditions:
            if not self._evaluate_single_condition(condition, event):
                return False
        return True

    def _evaluate_single_condition(self, condition: dict, event: dict) -> bool:
        """Evaluate a single condition against an event."""
        field_path = condition.get("field", "")
        op = condition.get("op", "eq")
        expected = condition.get("value")

        # Resolve the field value from the event using dot notation
        actual = self._resolve_field(field_path, event)

        # Apply the operator
        if op == "eq":
            return actual == expected
        elif op == "neq":
            return actual != expected
        elif op == "contains":
            return isinstance(actual, str) and expected.lower() in actual.lower()
        elif op == "contains_any":
            if not isinstance(actual, str):
                return False
            actual_lower = actual.lower()
            return any(v.lower() in actual_lower for v in expected)
        elif op == "in":
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
            return isinstance(actual, str) and bool(re.search(expected, actual, re.IGNORECASE))
        else:
            return False

    def _resolve_field(self, field_path: str, obj: dict) -> Any:
        """Resolve a dotted field path like 'payload.from_address' from a dict."""
        parts = field_path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _record_trigger(self, rule_id: str):
        """Update the trigger count for a rule."""
        with self.db.get_connection("preferences") as conn:
            conn.execute(
                """UPDATE rules SET 
                   times_triggered = times_triggered + 1,
                   last_triggered = ?
                   WHERE id = ?""",
                (datetime.now(timezone.utc).isoformat(), rule_id),
            )


# -------------------------------------------------------------------
# Default Rules — Installed on first run
# -------------------------------------------------------------------

DEFAULT_RULES = [
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
    {
        "name": "High priority: calendar conflict",
        "trigger_event": "calendar.conflict.detected",
        "conditions": [],
        "actions": [
            {"type": "notify", "priority": "high"},
        ],
    },
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


def install_default_rules(db: DatabaseManager):
    """Install the default rules on first run."""
    engine = RulesEngine(db)
    existing = engine.get_all_rules()
    existing_names = {r["name"] for r in existing}

    for rule in DEFAULT_RULES:
        if rule["name"] not in existing_names:
            engine.add_rule(
                name=rule["name"],
                trigger_event=rule["trigger_event"],
                conditions=rule["conditions"],
                actions=rule["actions"],
                created_by="system",
            )
