"""Tests for :mod:`api.routes.health` — ``/api/health`` + ``/metrics``.

Coverage (per NEXT_TASKS.md Week 8 acceptance):

- ``GET /api/health`` returns a multi-key response (connectors, DB
  last-write, scheduler heartbeat, producer activity, pending moments).
- When ``health_probe`` is not wired, the endpoint still returns 200
  with ``ok=False`` and a diagnostic note — an operator hitting this
  endpoint to diagnose a partial boot must not see a blanket 503.
- ``GET /metrics`` returns Prometheus text-format exposition that round-
  trips through a simple parser (one ``name value`` line per series).
- ``GET /metrics`` appends one JSON line per scrape to the configured
  ``metrics_dir`` for the ``lifeos-report`` CLI.
- ``GET /metrics`` returns a safe empty payload when no probe is wired.
- Histograms emit one bucketed line per entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from api.app import create_app


class StubHealthProbe:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def snapshot(self) -> dict[str, Any]:
        self.calls += 1
        return self.payload


class StubMetrics:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def snapshot(self) -> dict[str, Any]:
        self.calls += 1
        return self.payload


class DummyLifeOS:
    def __init__(
        self,
        health_probe: Any = None,
        metrics: Any = None,
        metrics_dir: Path | None = None,
    ) -> None:
        self.config: dict[str, Any] = {}
        if metrics_dir is not None:
            self.config["metrics_dir"] = str(metrics_dir)
        self.health_probe = health_probe
        self.metrics = metrics


def _client(life_os: Any) -> TestClient:
    return TestClient(create_app(life_os))


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------


def test_health_returns_multi_key_payload():
    probe = StubHealthProbe(
        {
            "ok": True,
            "ts": 1_777_204_800,
            "connectors": {"proton_mail": "ok", "signal": "degraded"},
            "db_last_write_ts": 1_777_204_700,
            "scheduler_heartbeat_ts": 1_777_204_795,
            "producer_activity": {"routine": 3, "comm_template": 1},
            "pending_moments": 7,
            "notes": [],
        }
    )
    resp = _client(DummyLifeOS(health_probe=probe)).get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["connectors"] == {"proton_mail": "ok", "signal": "degraded"}
    assert body["db_last_write_ts"] == 1_777_204_700
    assert body["scheduler_heartbeat_ts"] == 1_777_204_795
    assert body["producer_activity"] == {"routine": 3, "comm_template": 1}
    assert body["pending_moments"] == 7
    assert probe.calls == 1


def test_health_returns_diagnostic_when_probe_not_wired():
    resp = _client(DummyLifeOS(health_probe=None)).get("/api/health")
    # Intentionally 200 — an operator hitting this to diagnose a partial
    # boot must see the outage, not a blanket 503.
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["connectors"] == {}
    assert body["pending_moments"] == 0
    assert any("health_probe" in note for note in body["notes"])


def test_health_surfaces_probe_ok_false():
    """When the probe returns ``ok=False``, the route preserves it verbatim."""
    probe = StubHealthProbe(
        {
            "ok": False,
            "ts": 1_777_204_800,
            "connectors": {"signal": "error"},
            "db_last_write_ts": None,
            "scheduler_heartbeat_ts": None,
            "producer_activity": {},
            "pending_moments": 0,
            "notes": ["scheduler stopped", "signal connector unauthenticated"],
        }
    )
    resp = _client(DummyLifeOS(health_probe=probe)).get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["notes"] == [
        "scheduler stopped",
        "signal connector unauthenticated",
    ]


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


def _parse_prometheus_text(text: str) -> tuple[dict[str, str], dict[str, float]]:
    """Simple parser for the ``name value`` + ``# TYPE`` dialect we emit.

    Returns ``(types_by_name, values_by_series)`` — good enough to assert
    both the TYPE comments and the values without pulling in
    ``prometheus-client``. Histogram lines with labels are accepted
    verbatim as the series name (``metric{le="0.1"}``).
    """
    types: dict[str, str] = {}
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# TYPE "):
            _, _, rest = line.partition("# TYPE ")
            name, kind = rest.split()
            types[name] = kind
            continue
        if line.startswith("#"):
            continue
        # Split on the last space so label-bearing series parse too.
        name, _, value = line.rpartition(" ")
        values[name] = float(value)
    return types, values


def test_metrics_returns_prometheus_exposition_format():
    metrics = StubMetrics(
        {
            "ts": 1_777_204_800,
            "counters": {
                "moments_created_total": 42,
                "moments_dismissed_total": 5,
            },
            "gauges": {"pending_moments": 7},
            "histograms": {},
        }
    )
    resp = _client(DummyLifeOS(metrics=metrics)).get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    types, values = _parse_prometheus_text(resp.text)
    assert types == {
        "moments_created_total": "counter",
        "moments_dismissed_total": "counter",
        "pending_moments": "gauge",
    }
    assert values == {
        "moments_created_total": 42.0,
        "moments_dismissed_total": 5.0,
        "pending_moments": 7.0,
    }


def test_metrics_emits_histogram_buckets():
    metrics = StubMetrics(
        {
            "ts": 0,
            "counters": {},
            "gauges": {},
            "histograms": {
                "ollama_latency_seconds": {"0.1": 1, "0.5": 3, "+Inf": 5},
            },
        }
    )
    text = _client(DummyLifeOS(metrics=metrics)).get("/metrics").text
    assert "# TYPE ollama_latency_seconds histogram" in text
    assert 'ollama_latency_seconds{le="0.1"} 1' in text
    assert 'ollama_latency_seconds{le="0.5"} 3' in text
    assert 'ollama_latency_seconds{le="+Inf"} 5' in text


def test_metrics_writes_jsonl_when_metrics_dir_configured(tmp_path: Path):
    metrics = StubMetrics(
        {
            "ts": 1_777_204_800,
            "counters": {"x": 1},
            "gauges": {"y": 2.5},
            "histograms": {},
        }
    )
    life_os = DummyLifeOS(metrics=metrics, metrics_dir=tmp_path)
    resp = _client(life_os).get("/metrics")
    assert resp.status_code == 200
    files = sorted(tmp_path.glob("metrics-*.jsonl"))
    assert len(files) == 1
    record = json.loads(files[0].read_text().splitlines()[0])
    assert record["counters"] == {"x": 1}
    assert record["gauges"] == {"y": 2.5}
    assert record["ts"] == 1_777_204_800


def test_metrics_appends_one_line_per_scrape(tmp_path: Path):
    metrics = StubMetrics({"ts": 0, "counters": {"n": 1}, "gauges": {}, "histograms": {}})
    life_os = DummyLifeOS(metrics=metrics, metrics_dir=tmp_path)
    client = _client(life_os)
    for _ in range(3):
        assert client.get("/metrics").status_code == 200
    (dump_file,) = tmp_path.glob("metrics-*.jsonl")
    lines = dump_file.read_text().splitlines()
    assert len(lines) == 3
    assert metrics.calls == 3


def test_metrics_skips_jsonl_when_metrics_dir_unset(tmp_path: Path):
    """With no ``metrics_dir``, the scrape succeeds but writes nothing."""
    metrics = StubMetrics({"ts": 0, "counters": {"n": 1}, "gauges": {}, "histograms": {}})
    life_os = DummyLifeOS(metrics=metrics)  # no metrics_dir
    resp = _client(life_os).get("/metrics")
    assert resp.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_metrics_returns_placeholder_when_probe_not_wired():
    resp = _client(DummyLifeOS(metrics=None)).get("/metrics")
    assert resp.status_code == 200
    assert "# metrics unavailable" in resp.text
