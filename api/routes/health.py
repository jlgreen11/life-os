"""Health + metrics endpoints.

Two REST routes locked by engineering plan § "14-endpoint API contract":

- ``GET /api/health`` → deep-health multi-key payload
- ``GET /metrics``    → Prometheus text-format exposition

Both handlers pull their data off lightweight probes attached to the
LifeOS orchestrator:

- ``life_os.health_probe.snapshot() -> dict``
    Returns a dict in :class:`api.schemas.HealthOut` shape. When the
    probe is missing, ``/api/health`` still returns 200 with ``ok=False``
    and a diagnostic note — a 503 would mask the very outage an operator
    is hitting this endpoint to diagnose.
- ``life_os.metrics.snapshot() -> dict``
    Returns a dict with ``ts``, ``counters``, ``gauges``, ``histograms``
    — the same shape as :class:`api.schemas.MetricsOut`. When the probe
    is missing, ``/metrics`` returns a single ``# metrics unavailable``
    comment so Prometheus scrapers get a valid (empty) payload rather
    than a 503 that would pollute their health view.

Persistence
-----------
When ``life_os.config["metrics_dir"]`` is set, each ``/metrics`` scrape
appends one JSON line to ``{metrics_dir}/metrics-YYYYMMDD.jsonl`` (UTC
date). The ``lifeos-report`` CLI — engineering plan § "metrics
persistence" — ingests these daily dumps. When the config key is absent
(tests, local dev), the write is skipped quietly; ``/metrics`` still
returns the live text.

Prometheus exposition format
----------------------------
We emit the plain-text format (``# TYPE`` comments + ``name value``
lines) by hand rather than via ``prometheus-client``. The format is
stable and trivially parseable; keeping the dependency surface small
lets the v2 API import cleanly in environments where
``prometheus-client`` is not installed (same stance as the
fastapi-only minimum in ``requirements.txt``).
"""

from __future__ import annotations

import contextlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from api.schemas import HealthOut

router = APIRouter()


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def _health_probe(request: Request):
    """Return the health probe on ``app.state.life_os``, or ``None``."""
    life_os = getattr(request.app.state, "life_os", None)
    return getattr(life_os, "health_probe", None)


@router.get("/api/health", response_model=HealthOut)
def get_health(request: Request) -> HealthOut:
    """Return a multi-key deep-health payload.

    Unlike a naive ``/health`` that's green-or-red, this endpoint
    separates the failure modes the engineering plan calls out:

    - ``connectors``              — per-connector string status
    - ``db_last_write_ts``        — newest row across mutable tables
    - ``scheduler_heartbeat_ts``  — last scheduler tick
    - ``producer_activity``       — per-producer counter since last probe
    - ``pending_moments``         — outstanding SUGGESTED count

    ``ok`` is the AND of the component-level checks the probe performs.

    When no probe is wired, returns ``ok=False`` with a diagnostic note.
    We deliberately do NOT return 503 in that case: an operator hitting
    ``/api/health`` during a partial-boot outage should see exactly the
    outage, not a blanket service-unavailable.
    """
    probe = _health_probe(request)
    if probe is None:
        return HealthOut(
            ok=False,
            ts=int(time.time()),
            notes=["health_probe not wired on life_os"],
        )
    return HealthOut(**probe.snapshot())


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def _metrics_probe(request: Request):
    """Return the metrics probe on ``app.state.life_os``, or ``None``."""
    life_os = getattr(request.app.state, "life_os", None)
    return getattr(life_os, "metrics", None)


def _metrics_dir(request: Request) -> Path | None:
    """Read the configured JSONL dump dir from ``life_os.config``.

    Returns ``None`` when the key is missing, empty, or the config
    attribute is not a dict — tests and ephemeral environments skip the
    persistent dump without extra branching in the route body.
    """
    life_os = getattr(request.app.state, "life_os", None)
    config = getattr(life_os, "config", None)
    if not isinstance(config, dict):
        return None
    raw = config.get("metrics_dir")
    if not raw:
        return None
    return Path(raw)


def _escape_label_value(value: str) -> str:
    """Escape a Prometheus label value per the exposition format spec.

    Backslash first, then double-quote, then newline — matches the
    ``prometheus-client`` implementation so downstream scrapers parse
    our output identically.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_prometheus(snapshot: dict[str, Any]) -> str:
    """Render a metrics snapshot as Prometheus text exposition format.

    Layout, repeated per metric:

    .. code-block:: text

        # TYPE metric_name counter
        metric_name 42

    Histograms are emitted as one line per bucket with a ``le=""`` label
    (Prometheus convention). Extra top-level keys on the snapshot are
    ignored so the probe can evolve without breaking scrapers.
    """
    lines: list[str] = []

    for name, value in (snapshot.get("counters") or {}).items():
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")

    for name, value in (snapshot.get("gauges") or {}).items():
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")

    for name, buckets in (snapshot.get("histograms") or {}).items():
        lines.append(f"# TYPE {name} histogram")
        if isinstance(buckets, dict):
            for bucket, value in buckets.items():
                label = _escape_label_value(str(bucket))
                lines.append(f'{name}{{le="{label}"}} {value}')

    return "\n".join(lines) + "\n"


def _append_jsonl(metrics_dir: Path, snapshot: dict[str, Any]) -> None:
    """Append ``snapshot`` as one JSON line to today's dump file.

    The directory is created lazily. The dump is append-only and
    crash-safe for a single writer — we do not hold the file open across
    scrapes. Filename is UTC-dated so a Mac Mini that sleeps overnight
    still rolls cleanly at midnight UTC.
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).strftime("%Y%m%d")
    file_path = metrics_dir / f"metrics-{today}.jsonl"
    with file_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot, sort_keys=True) + "\n")


@router.get("/metrics", response_class=PlainTextResponse)
def get_metrics(request: Request) -> str:
    """Return the metrics snapshot in Prometheus text format.

    Side effect: when ``life_os.config["metrics_dir"]`` is configured,
    each scrape also appends one JSON line to the day's dump file for
    the ``lifeos-report`` CLI. Failures in the dump path are suppressed
    — the scraper must always get a response.

    When no metrics probe is wired, returns a single ``# metrics
    unavailable`` comment. Prometheus scrapers treat an all-comment
    response as zero series, not an error, which is the behaviour an
    operator wants during partial-boot.
    """
    probe = _metrics_probe(request)
    if probe is None:
        return "# metrics unavailable\n"

    snapshot = probe.snapshot()

    metrics_dir = _metrics_dir(request)
    if metrics_dir is not None:
        # Never block the scrape on a dump failure — the structured log
        # emitted by the probe layer is the signal an operator will
        # actually act on.
        with contextlib.suppress(OSError):
            _append_jsonl(metrics_dir, snapshot)

    return _format_prometheus(snapshot)


__all__ = ["router"]
