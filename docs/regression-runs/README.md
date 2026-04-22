# Regression runs

Output directory for `tests/regression/test_golden_30day.py`. Each run writes
`{YYYY-MM-DD}.md` summarising the four assertions:

- (a) v2 Moment count within ±10% of v1 prediction count for the same window.
- (b) every high-signal v1 prediction has a thematically matching v2 Moment
  (Jaccard token overlap above 0.20).
- (c) zero duplicate `(source_insight_type, evidence_hash)` pairs.
- (d) Ollama latency budget — currently delegated to
  `scripts/measure_ollama_budget.py` and the baseline at
  `docs/plans/2026-04-22-ollama-baseline.md`.

## Running the harness

```bash
python -m pytest tests/regression/test_golden_30day.py -v
```

The test is skipped if no v1 snapshot is available at `data/v1-snapshot/`
(or the directory pointed at by `LIFEOS_V1_SNAPSHOT_DIR`). To run it on the
Mac Mini that holds production data:

1. Stop Life OS so the SQLite WAL can checkpoint cleanly.
2. Copy the five v1 databases into `data/v1-snapshot/`:
   - `events.db` (required)
   - `user_model.db` (required — supplies `predictions` for assertion b)
   - `entities.db` (optional — improves producer signal)
   - `state.db` (optional)
   - `preferences.db` (optional)
3. Restart Life OS.
4. Run the command above. The report lands here as `<today>.md`.

Reports are committed so successive runs can be diffed; the snapshot under
`data/v1-snapshot/` is **not** committed (covered by `data/`'s gitignore).
