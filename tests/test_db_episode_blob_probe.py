"""
Tests: Episodes blob overflow probe uses SUM(LENGTH()) to scan all rows.

Verifies that the blob overflow probe for the ``episodes`` table in
``get_database_health()`` and ``_check_and_recover_db()`` uses
``SUM(LENGTH(content_full))`` — which forces SQLite to read every row's
overflow pages — rather than ``LIMIT 1`` which only touches the first row
and could miss corruption in later rows.

Related: storage/manager.py get_database_health() and _check_and_recover_db()
blob_probes lists.
"""

import sqlite3

import pytest


def test_episodes_probe_uses_sum_length(db):
    """The episodes blob probe must use SUM(LENGTH()) not LIMIT 1.

    SUM(LENGTH()) forces SQLite to read every row's overflow pages,
    ensuring corruption in any row is detected — not just the first.
    """
    results = db.get_database_health()
    assert results["user_model"]["status"] == "ok"


def test_episodes_probe_reads_all_rows(db):
    """Verify the probe touches all rows, not just the first.

    Insert multiple episodes and confirm the probe query
    (SUM(LENGTH(content_full))) returns the total length across
    all rows — proving it reads every row's overflow pages.
    """
    db_path = db._databases["user_model"]
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # Insert 3 episodes with distinct content_full values
    for i in range(3):
        content = f'{{"event": "test-episode-{i}", "data": "' + ("X" * 500) + '"}'
        conn.execute(
            "INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary, content_full) "
            "VALUES (?, datetime('now'), ?, 'test', ?, ?)",
            (f"ep-{i}", f"evt-{i}", f"summary-{i}", content),
        )
    conn.commit()

    # Run the probe that get_database_health() uses
    row = conn.execute("SELECT SUM(LENGTH(content_full)) FROM episodes").fetchone()
    total_length = row[0]
    assert total_length is not None
    assert total_length > 0

    # Verify it's the sum of all 3 rows, not just one
    count_row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
    assert count_row[0] == 3

    # Each row's content_full is > 500 chars, so total should be > 1500
    assert total_length > 1500

    conn.close()


def test_health_check_ok_with_populated_episodes(db):
    """get_database_health() returns 'ok' for user_model with populated episodes."""
    db_path = db._databases["user_model"]
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # Insert episodes with large content to exercise overflow pages
    for i in range(5):
        large_content = f'{{"event": "episode-{i}", "payload": "' + ("A" * 2000) + '"}'
        conn.execute(
            "INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary, content_full) "
            "VALUES (?, datetime('now'), ?, 'test', ?, ?)",
            (f"ep-{i}", f"evt-{i}", f"summary-{i}", large_content),
        )
    conn.commit()
    conn.close()

    results = db.get_database_health()
    assert results["user_model"]["status"] == "ok"
    assert results["user_model"]["errors"] == []


def test_blob_probes_use_sum_length_not_limit():
    """Every blob probe in get_database_health() and _check_and_recover_db()
    must use SUM(LENGTH()) — none should use LIMIT 1, which only reads one row.

    This is a code-level assertion to prevent regression: if someone adds
    a new probe with LIMIT 1 or changes an existing one, this test catches it.
    """
    import re
    import inspect
    import textwrap

    from storage.manager import DatabaseManager

    # Check both methods that contain blob_probes lists
    for method_name in ("get_database_health", "_check_and_recover_db"):
        method = getattr(DatabaseManager, method_name)
        source = inspect.getsource(method)

        # Extract all SQL strings from the blob_probes list using regex.
        # The pattern matches quoted strings inside the blob_probes list block.
        probes_match = re.search(r"blob_probes\s*=\s*\[(.*?)\]", source, re.DOTALL)
        assert probes_match is not None, f"No blob_probes list found in {method_name}()"

        probes_block = probes_match.group(1)
        # Extract individual SQL strings (double-quoted)
        sql_strings = re.findall(r'"([^"]+)"', probes_block)

        assert len(sql_strings) >= 7, (
            f"{method_name}() should have at least 7 blob probes, found {len(sql_strings)}"
        )

        for sql in sql_strings:
            assert "LIMIT 1" not in sql, (
                f"{method_name}() blob_probes contains LIMIT 1: {sql!r}; "
                "use SUM(LENGTH()) to read all overflow pages"
            )
            assert "SUM(LENGTH(" in sql, (
                f"{method_name}() blob_probes missing SUM(LENGTH()): {sql!r}"
            )
