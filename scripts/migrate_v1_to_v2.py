"""v1 → v2 dry-run migration.

Reads the five v1 SQLite databases read-only and writes translated rows into a
fresh ``data/lifeos_v2_dryrun.db`` built from ``storage/schema.py``. Never
touches production data — opens sources with ``mode=ro`` and always emits the
target under a ``_dryrun`` filename.

Translations (per NEXT_TASKS Week 1):

- ``events.db.events`` → ``events``. Schema-compatible pass-through;
  TEXT ISO timestamps are coerced to INTEGER unix seconds. v1-only
  ``embedding_id`` column is dropped.
- ``entities.db.{contacts, places, subscriptions}`` → ``entities`` with
  ``kind`` in {contact, place, subscription}. Remaining v1 columns are
  packed into the v2 ``attributes`` JSON blob.
- ``state.db.tasks`` → synthetic Moments with ``source_insight_type='legacy_task'``,
  ``state='suggested'``, ``evidence='[]'``.
- ``user_model.db.signal_profiles`` → ``signal_profiles``. Profile types in
  {mood, decision, expertise, values} are dropped per CEO plan; the remaining
  six (cadence, relationship, temporal, spatial, comm_template, routine) are
  translated 1:1 with ``producer=profile_type`` and ``key='default'``.
- ``preferences.db.user_preferences`` → ``preferences`` (key/value pass-through;
  ``encrypted=0``).
- v1 notification feedback (``preferences.db.feedback_log``) → **SKIPPED**:
  the v2 schema does not yet define a ``feedback_events`` table. This is
  logged as a translation decision and surfaced in the report. See the note
  in ``NEXT_TASKS.md``.

Every translation decision is logged to stdout. Row-count invariants are
asserted per table after write. Returns a non-zero exit code if any invariant
fails.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from storage import schema as v2_schema  # noqa: E402

V1_DB_FILES: tuple[str, ...] = (
    "events.db",
    "entities.db",
    "state.db",
    "user_model.db",
    "preferences.db",
)

DEFAULT_OUTPUT_NAME = "lifeos_v2_dryrun.db"

DROPPED_PROFILE_TYPES: frozenset[str] = frozenset({"mood", "decision", "expertise", "values"})
KEPT_PROFILE_TYPES: frozenset[str] = frozenset(
    {"cadence", "relationship", "temporal", "spatial", "comm_template", "routine"}
)

LEGACY_MOMENT_EXPIRY_SECONDS = 30 * 24 * 3600


@dataclass
class TableCounts:
    """Row-count invariants observed during a single table translation."""

    source: int = 0
    translated: int = 0
    dropped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"source": self.source, "translated": self.translated, "dropped": self.dropped}


@dataclass
class MigrationReport:
    """Aggregate summary of the dry-run."""

    events: TableCounts = field(default_factory=TableCounts)
    entities: TableCounts = field(default_factory=TableCounts)
    moments_from_tasks: TableCounts = field(default_factory=TableCounts)
    signal_profiles: TableCounts = field(default_factory=TableCounts)
    preferences: TableCounts = field(default_factory=TableCounts)
    notification_feedback_skipped: int = 0
    missing_source_dbs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "events": self.events.as_dict(),
            "entities": self.entities.as_dict(),
            "moments_from_tasks": self.moments_from_tasks.as_dict(),
            "signal_profiles": self.signal_profiles.as_dict(),
            "preferences": self.preferences.as_dict(),
            "notification_feedback_skipped": self.notification_feedback_skipped,
            "missing_source_dbs": list(self.missing_source_dbs),
            "notes": list(self.notes),
        }


def _iso_to_unix(value: str | None) -> int | None:
    """Convert a v1 ISO-8601 timestamp string to unix seconds (UTC)."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return int(dt.datetime.fromisoformat(text).timestamp())
    except ValueError:
        return None


def _open_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    return any(row[1] == column for row in cur.fetchall())


def apply_v2_schema(conn: sqlite3.Connection) -> None:
    """Build the v2 schema on a freshly opened connection."""
    conn.execute("PRAGMA foreign_keys=ON")
    for ddl in v2_schema.get_all_ddl():
        conn.execute(ddl)
    conn.execute(
        "INSERT INTO schema_version (version) VALUES (?)",
        (v2_schema.SCHEMA_VERSION,),
    )
    conn.commit()


def _safe_json_loads(text: str | None, fallback: object) -> object:
    if not text:
        return fallback
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return fallback


def _pack_attributes(columns: dict[str, object]) -> str:
    """JSON-serialize the per-entity extra columns (skipping None)."""
    return json.dumps({k: v for k, v in columns.items() if v is not None}, sort_keys=True)


def migrate_events(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    log: logging.Logger,
) -> TableCounts:
    counts = TableCounts()
    if not _table_exists(src, "events"):
        log.info("events: source table missing; nothing to migrate")
        return counts

    rows = src.execute(
        "SELECT id, type, source, timestamp, priority, payload, metadata, created_at FROM events"
    ).fetchall()
    counts.source = len(rows)

    now_unix = int(time.time())
    with dst:
        for row in rows:
            event_id, type_, source_, ts, priority, payload, metadata, created_at = row
            ts_unix = _iso_to_unix(ts)
            if ts_unix is None:
                counts.dropped += 1
                log.warning("events: dropping id=%s (unparseable timestamp %r)", event_id, ts)
                continue
            created_unix = _iso_to_unix(created_at) or now_unix
            dst.execute(
                """
                INSERT INTO events
                    (id, type, source, timestamp, priority, payload, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    type_,
                    source_,
                    ts_unix,
                    (priority or "normal"),
                    (payload or "{}"),
                    (metadata or "{}"),
                    created_unix,
                ),
            )
            counts.translated += 1

    log.info(
        "events: read=%d translated=%d dropped=%d",
        counts.source,
        counts.translated,
        counts.dropped,
    )
    return counts


def migrate_entities(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    log: logging.Logger,
) -> TableCounts:
    counts = TableCounts()
    now_unix = int(time.time())

    def _insert(
        entity_id: str,
        kind: str,
        name: str,
        aliases_json: str,
        attributes: dict[str, object],
        created_at: str | None,
        updated_at: str | None,
    ) -> None:
        created_unix = _iso_to_unix(created_at) or now_unix
        updated_unix = _iso_to_unix(updated_at) or created_unix
        dst.execute(
            """
            INSERT INTO entities (id, kind, name, aliases, attributes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entity_id, kind, name, aliases_json, _pack_attributes(attributes), created_unix, updated_unix),
        )

    with dst:
        if _table_exists(src, "contacts"):
            rows = src.execute(
                """
                SELECT id, name, aliases, emails, phones, channels, relationship,
                       domains, is_priority, preferred_channel, always_surface,
                       typical_response_time, communication_style, last_contact,
                       contact_frequency_days, notes, created_at, updated_at
                FROM contacts
                """
            ).fetchall()
            counts.source += len(rows)
            for row in rows:
                (
                    cid,
                    name,
                    aliases,
                    emails,
                    phones,
                    channels,
                    relationship,
                    domains,
                    is_priority,
                    preferred_channel,
                    always_surface,
                    typical_response_time,
                    communication_style,
                    last_contact,
                    contact_frequency_days,
                    notes,
                    created_at,
                    updated_at,
                ) = row
                attrs = {
                    "emails": _safe_json_loads(emails, []),
                    "phones": _safe_json_loads(phones, []),
                    "channels": _safe_json_loads(channels, {}),
                    "relationship": relationship,
                    "domains": _safe_json_loads(domains, []),
                    "is_priority": bool(is_priority) if is_priority is not None else False,
                    "preferred_channel": preferred_channel,
                    "always_surface": bool(always_surface) if always_surface is not None else False,
                    "typical_response_time": typical_response_time,
                    "communication_style": communication_style,
                    "last_contact": last_contact,
                    "contact_frequency_days": contact_frequency_days,
                    "notes": _safe_json_loads(notes, []),
                }
                _insert(cid, "contact", name, aliases or "[]", attrs, created_at, updated_at)
                counts.translated += 1

        if _table_exists(src, "places"):
            rows = src.execute(
                """
                SELECT id, name, latitude, longitude, address, wifi_ssid, place_type,
                       domain, visit_count, avg_duration_minutes, associated_behaviors,
                       created_at, updated_at
                FROM places
                """
            ).fetchall()
            counts.source += len(rows)
            for row in rows:
                (
                    pid,
                    name,
                    latitude,
                    longitude,
                    address,
                    wifi_ssid,
                    place_type,
                    domain,
                    visit_count,
                    avg_duration_minutes,
                    associated_behaviors,
                    created_at,
                    updated_at,
                ) = row
                attrs = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "address": address,
                    "wifi_ssid": wifi_ssid,
                    "place_type": place_type,
                    "domain": domain,
                    "visit_count": visit_count,
                    "avg_duration_minutes": avg_duration_minutes,
                    "associated_behaviors": _safe_json_loads(associated_behaviors, {}),
                }
                _insert(pid, "place", name, "[]", attrs, created_at, updated_at)
                counts.translated += 1

        if _table_exists(src, "subscriptions"):
            rows = src.execute(
                """
                SELECT id, name, amount, currency, frequency, last_charge, next_charge,
                       category, last_used, usage_frequency, cancel_url, notes,
                       created_at, updated_at
                FROM subscriptions
                """
            ).fetchall()
            counts.source += len(rows)
            for row in rows:
                (
                    sid,
                    name,
                    amount,
                    currency,
                    frequency,
                    last_charge,
                    next_charge,
                    category,
                    last_used,
                    usage_frequency,
                    cancel_url,
                    notes,
                    created_at,
                    updated_at,
                ) = row
                attrs = {
                    "amount": amount,
                    "currency": currency,
                    "frequency": frequency,
                    "last_charge": last_charge,
                    "next_charge": next_charge,
                    "category": category,
                    "last_used": last_used,
                    "usage_frequency": usage_frequency,
                    "cancel_url": cancel_url,
                    "notes": notes,
                }
                _insert(sid, "subscription", name, "[]", attrs, created_at, updated_at)
                counts.translated += 1

    log.info(
        "entities: read=%d translated=%d dropped=%d",
        counts.source,
        counts.translated,
        counts.dropped,
    )
    return counts


def migrate_tasks_to_moments(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    log: logging.Logger,
) -> TableCounts:
    counts = TableCounts()
    if not _table_exists(src, "tasks"):
        log.info("moments_from_tasks: source state.db.tasks missing; nothing to migrate")
        return counts

    rows = src.execute(
        """
        SELECT id, title, description, source, source_event_id, due_date, reminder_at,
               priority, related_contacts, created_at
        FROM tasks
        """
    ).fetchall()
    counts.source = len(rows)
    now_unix = int(time.time())

    with dst:
        for row in rows:
            (
                task_id,
                title,
                description,
                source_,
                source_event_id,
                due_date,
                reminder_at,
                priority,
                related_contacts,
                created_at,
            ) = row
            created_unix = _iso_to_unix(created_at) or now_unix
            expires_at = created_unix + LEGACY_MOMENT_EXPIRY_SECONDS
            evidence_hash = hashlib.sha256(f"legacy_task::{task_id}".encode()).hexdigest()
            proposed_action = json.dumps(
                {
                    "kind": "NOTE_OBSERVATION",
                    "params": {
                        "legacy_task_id": task_id,
                        "title": title or "",
                        "description": description or "",
                        "due_date": due_date,
                        "reminder_at": reminder_at,
                        "priority": priority or "normal",
                        "source": source_,
                        "source_event_id": source_event_id,
                        "related_contacts": _safe_json_loads(related_contacts, []),
                    },
                },
                sort_keys=True,
            )
            dst.execute(
                """
                INSERT INTO moments (
                    id, created_at, scheduled_for, expires_at, context_trigger,
                    insight, evidence, evidence_hash, proposed_action, state,
                    snooze_until, confidence, feedback_weight, source_insight_type,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    created_unix,
                    None,
                    expires_at,
                    None,
                    title or "(legacy task)",
                    "[]",
                    evidence_hash,
                    proposed_action,
                    "suggested",
                    None,
                    0.0,
                    1.0,
                    "legacy_task",
                    created_unix,
                ),
            )
            dst.execute(
                """
                INSERT INTO moment_state_history (moment_id, from_state, to_state, ts, annotation)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task_id, None, "suggested", created_unix, "legacy_task_migration"),
            )
            counts.translated += 1

    log.info(
        "moments_from_tasks: read=%d translated=%d dropped=%d",
        counts.source,
        counts.translated,
        counts.dropped,
    )
    return counts


def migrate_signal_profiles(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    log: logging.Logger,
) -> TableCounts:
    counts = TableCounts()
    if not _table_exists(src, "signal_profiles"):
        log.info("signal_profiles: source table missing; nothing to migrate")
        return counts

    rows = src.execute("SELECT profile_type, data, samples_count, updated_at FROM signal_profiles").fetchall()
    counts.source = len(rows)
    now_unix = int(time.time())

    with dst:
        for profile_type, data, samples_count, updated_at in rows:
            if profile_type in DROPPED_PROFILE_TYPES:
                counts.dropped += 1
                log.info("signal_profiles: dropping type=%s (per CEO plan)", profile_type)
                continue
            if profile_type not in KEPT_PROFILE_TYPES:
                counts.dropped += 1
                log.warning(
                    "signal_profiles: dropping unknown type=%s (not in kept set)",
                    profile_type,
                )
                continue
            updated_unix = _iso_to_unix(updated_at) or now_unix
            profile_blob = data or "{}"
            if samples_count is not None:
                merged = _safe_json_loads(profile_blob, {})
                if isinstance(merged, dict) and "_samples_count" not in merged:
                    merged["_samples_count"] = samples_count
                    profile_blob = json.dumps(merged, sort_keys=True)
            dst.execute(
                """
                INSERT INTO signal_profiles (producer, key, profile, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (profile_type, "default", profile_blob, updated_unix),
            )
            counts.translated += 1

    log.info(
        "signal_profiles: read=%d translated=%d dropped=%d",
        counts.source,
        counts.translated,
        counts.dropped,
    )
    return counts


def migrate_preferences(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    log: logging.Logger,
) -> TableCounts:
    counts = TableCounts()
    if not _table_exists(src, "user_preferences"):
        log.info("preferences: source user_preferences missing; nothing to migrate")
        return counts

    rows = src.execute("SELECT key, value, updated_at FROM user_preferences").fetchall()
    counts.source = len(rows)
    now_unix = int(time.time())

    with dst:
        for key, value, updated_at in rows:
            updated_unix = _iso_to_unix(updated_at) or now_unix
            dst.execute(
                """
                INSERT INTO preferences (key, value, encrypted, updated_at)
                VALUES (?, ?, 0, ?)
                """,
                (key, value if value is not None else "", updated_unix),
            )
            counts.translated += 1

    log.info(
        "preferences: read=%d translated=%d dropped=%d",
        counts.source,
        counts.translated,
        counts.dropped,
    )
    return counts


def skip_notification_feedback(
    src: sqlite3.Connection,
    log: logging.Logger,
) -> int:
    """Log the skip decision and return the number of source rows that would
    need translation if a target table existed."""
    if not _table_exists(src, "feedback_log"):
        log.info("notification_feedback: source feedback_log missing; nothing to skip")
        return 0
    (n,) = src.execute("SELECT COUNT(*) FROM feedback_log").fetchone()
    log.warning(
        "notification_feedback: SKIPPED %d v1 feedback rows — no feedback_events table in v2 schema "
        "(see NEXT_TASKS Week 1 task; design decision pending)",
        n,
    )
    return int(n)


def _verify_invariants(
    dst: sqlite3.Connection,
    report: MigrationReport,
    log: logging.Logger,
) -> list[str]:
    """Return a list of invariant-violation messages (empty if all pass)."""
    problems: list[str] = []
    pairs: tuple[tuple[str, int], ...] = (
        ("events", report.events.translated),
        ("entities", report.entities.translated),
        ("moments", report.moments_from_tasks.translated),
        ("signal_profiles", report.signal_profiles.translated),
        ("preferences", report.preferences.translated),
    )
    for table, expected in pairs:
        (actual,) = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        if actual != expected:
            problems.append(f"{table}: expected {expected} rows, got {actual}")
        else:
            log.info("invariant OK: %s has %d rows", table, actual)

    # Confirm no dropped profile types leaked into output.
    if report.signal_profiles.translated > 0:
        leaks = dst.execute(
            "SELECT producer FROM signal_profiles WHERE producer IN (?,?,?,?)",
            tuple(DROPPED_PROFILE_TYPES),
        ).fetchall()
        if leaks:
            problems.append(f"signal_profiles: dropped types leaked: {sorted(r[0] for r in leaks)}")

    return problems


def run_migration(
    source_dir: Path,
    output_path: Path,
    *,
    log: logging.Logger | None = None,
) -> MigrationReport:
    """Run the dry-run migration and return a structured report.

    Raises ``FileExistsError`` if ``output_path`` already exists; caller is
    responsible for cleanup because we refuse to silently overwrite.
    """
    log = log or logging.getLogger(__name__)
    report = MigrationReport()

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing target {output_path}; delete it first.")

    for fname in V1_DB_FILES:
        if not (source_dir / fname).exists():
            report.missing_source_dbs.append(fname)
            log.info("source missing: %s", fname)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output_path) as dst:
        apply_v2_schema(dst)

        events_path = source_dir / "events.db"
        if events_path.exists():
            with _open_ro(events_path) as src:
                report.events = migrate_events(src, dst, log)

        entities_path = source_dir / "entities.db"
        if entities_path.exists():
            with _open_ro(entities_path) as src:
                report.entities = migrate_entities(src, dst, log)

        state_path = source_dir / "state.db"
        if state_path.exists():
            with _open_ro(state_path) as src:
                report.moments_from_tasks = migrate_tasks_to_moments(src, dst, log)

        user_model_path = source_dir / "user_model.db"
        if user_model_path.exists():
            with _open_ro(user_model_path) as src:
                report.signal_profiles = migrate_signal_profiles(src, dst, log)

        preferences_path = source_dir / "preferences.db"
        if preferences_path.exists():
            with _open_ro(preferences_path) as src:
                report.preferences = migrate_preferences(src, dst, log)
                report.notification_feedback_skipped = skip_notification_feedback(src, log)

        problems = _verify_invariants(dst, report, log)
        for msg in problems:
            report.notes.append(f"INVARIANT: {msg}")

    if report.notification_feedback_skipped:
        report.notes.append(
            f"notification_feedback: {report.notification_feedback_skipped} rows skipped — "
            "no feedback_events table in v2 schema"
        )

    return report


def _setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("migrate_v1_to_v2")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v1 → v2 dry-run migration")
    parser.add_argument(
        "--source-dir",
        default=str(REPO_ROOT / "data"),
        help="Directory containing v1 *.db files (default: ./data)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Path for the dry-run target DB (default: <source-dir>/{DEFAULT_OUTPUT_NAME})",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    log = _setup_logging(args.verbose)

    source_dir = Path(args.source_dir)
    output_path = Path(args.output) if args.output else source_dir / DEFAULT_OUTPUT_NAME

    if not source_dir.exists():
        log.error("source dir %s does not exist", source_dir)
        return 2

    if output_path.exists():
        log.error("target %s exists; delete it before running dry-run", output_path)
        return 2

    report = run_migration(source_dir, output_path, log=log)
    log.info("migration report: %s", json.dumps(report.as_dict(), indent=2, sort_keys=True))

    invariant_failures = [n for n in report.notes if n.startswith("INVARIANT:")]
    if invariant_failures:
        for msg in invariant_failures:
            log.error(msg)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
