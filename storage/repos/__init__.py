"""`storage.repos` — typed, single-purpose SQLite repositories for the v2 rewrite.

Each repository owns one table (or tightly coupled pair) from
``storage/schema.py`` and is the only write path for that data. Callers
inject a ``sqlite3.Connection`` per eng review §1a; there are no
module-level connections or globals.

Currently exported:

- :class:`storage.repos.moments.MomentRepository` — the Moment primitive
  store, including the ``moment_state_history`` append-only audit log.
- :class:`storage.repos.outbox.OutboxRepository` — transactional outbox
  for side-effecting events (send_message, create_calendar_entry, …).
- :class:`storage.repos.people.PeopleRepository` — read-only façade over
  the signal-profile tables for the You + People API payloads.

Later waves add: ``feedback_weights``, ``connector_state``, etc.
"""

from storage.repos.moments import MomentRepository
from storage.repos.outbox import OutboxEntry, OutboxRepository
from storage.repos.people import PeopleRepository

__all__ = [
    "MomentRepository",
    "OutboxEntry",
    "OutboxRepository",
    "PeopleRepository",
]
