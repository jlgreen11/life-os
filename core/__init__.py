"""`core/` — v2 domain primitives.

Houses pure domain types and business logic that are free of I/O, framework,
and transport concerns. Anything that has to know about SQLite, NATS, HTTP,
or Ollama belongs in `storage/`, `api/`, or `ai/` — not here.

See `docs/plans/2026-04-21-v2-rewrite-plan.md` for the module layout locked
in by the engineering review.
"""
