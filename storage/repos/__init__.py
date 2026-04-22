"""`storage.repos` — typed, single-purpose SQLite repositories for the v2 rewrite.

Each repository owns one table (or tightly coupled pair) from
``storage/schema.py`` and is the only write path for that data. Callers
inject a ``sqlite3.Connection`` per eng review §1a; there are no
module-level connections or globals.

Currently exported:

- :class:`storage.repos.moments.MomentRepository` — the Moment primitive
  store, including the ``moment_state_history`` append-only audit log.

Later waves add: ``outbox``, ``feedback_weights``, ``signal_profiles``,
``connector_state``, etc.
"""

from storage.repos.moments import MomentRepository

__all__ = ["MomentRepository"]
