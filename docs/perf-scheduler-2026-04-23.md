# Scheduler performance profile — 2026-04-23

- Generated: 2026-04-23T00:42:56+00:00
- Script: `scripts/profile_scheduler.py` (rng_seed=42)
- Simulated reference now: 2026-04-26T12:00:00+00:00

## Setup

- Fleet size: **10,000** synthetic Moments
  - 70% SUGGESTED scheduled, 20% SNOOZED, 10% context-trigger-only
  - `scheduled_for` uniform across next 24h from reference now
- Ticks: **120** × `tick_seconds=30` = 3,600s of simulated wall time
- Batch limit per tick: 1,000
- Storage: tempfile SQLite, WAL + synchronous=NORMAL (production PRAGMAs)

## Results

| Metric | Value |
|---|---:|
| p50 tick latency | 10.91 ms |
| p95 tick latency | 26.67 ms |
| p99 tick latency | **28.51 ms** |
| max tick latency | 29.75 ms |
| Total fires | 19,564 |
| Mean fires / tick | 163.03 |
| Throughput | 5.43 fires / simulated second |

## Acceptance

- Budget: p99 tick latency < **500 ms** at 10,000-row fleet
- Result: **PASS** (p99 = 28.51 ms)

## Reproduce

```
python scripts/profile_scheduler.py --count 10000 \
    --ticks 120 --tick-seconds 30 \
    --batch-limit 1000 --seed 42
```

