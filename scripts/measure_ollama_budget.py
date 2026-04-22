#!/usr/bin/env python3
"""Measure Ollama latency + token budget for Life OS v2 operations.

Runs N=10 iterations of each of the five budgeted operations from the v2 CEO
plan and reports p50/p95/p99 latency plus token counts per operation. Emits a
markdown baseline report at ``docs/plans/2026-04-22-ollama-baseline.md`` (path
configurable via ``--output``).

The five operations:

1. briefing_synthesis — 12-context-block synthesis prompt (the morning brief).
2. task_extraction — 5 realistic email bodies fed one-by-one.
3. priority_classification — the same 5 emails, asked for PRIORITY only.
4. draft_reply — 3 contact profiles, draft a short reply for each.
5. semantic_search — embed a query + cosine similarity against 100 precomputed
   embeddings, returning the top-5 nearest.

Each iteration records wall-clock latency. Token counts come from Ollama's
``prompt_eval_count`` + ``eval_count`` fields when available.

Usage::

    python scripts/measure_ollama_budget.py \\
        --url http://localhost:11434 \\
        --chat-model mistral \\
        --embed-model nomic-embed-text \\
        --iterations 10 \\
        --output docs/plans/2026-04-22-ollama-baseline.md

If Ollama is unreachable the script prints a warning, writes a stub baseline
report noting the skip, and exits 0 — so CI / agents can still record that the
script shipped without requiring a running server. Measurement is expected to
run on the Mac Mini that hosts the models.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx


# ---------------------------------------------------------------------------
# Fixtures — realistic-looking prompts for each operation.
# ---------------------------------------------------------------------------

BRIEFING_CONTEXT_BLOCKS: list[str] = [
    "[email] From: Sarah (mom). Re: Thanksgiving flight. She wants to know if you can pick her up from SFO Wed afternoon.",
    "[calendar] 10:00-11:00 Today: '1:1 with Priya' (recurring, last declined 2 weeks ago).",
    "[calendar] 14:00 Today: 'Architecture review — v2 rewrite' (newly added by Max).",
    "[task] Overdue by 2 days: 'Send Q3 investor update to LP list' (priority: high).",
    "[task] Snoozed to today: 'Reply to Anna re: piano lessons for Emma'.",
    "[signal] Mood profile: stress trending up over the last 3 days (linguistic indicators).",
    "[signal] Cadence: you typically reply to Sarah within 4 hours; 27 hours since last reply.",
    "[imessage] Max (coworker) sent 'heads up, shipping the Stripe migration today' at 08:14.",
    "[finance] Amazon charge $482.11 yesterday — larger than 95th percentile of recent charges.",
    "[location] You arrived at the office 45 min earlier than your 30-day average.",
    "[habit] Running streak: 6 days. Last run: yesterday 07:12.",
    "[news] Fed announcement at 11:00 ET today — you've asked to be reminded before market-moving events.",
]

EMAIL_BODIES: list[str] = [
    (
        "Hey — quick one. Can you pull together the Q3 investor update by EOD "
        "Thursday? I want to get it to the LP list before our board meeting "
        "on Monday. Let me know if that timeline is realistic; if not, Friday "
        "morning is the hard deadline. Thanks. — Jen"
    ),
    (
        "Hi team, confirming tomorrow's 9am design review. Agenda attached. "
        "Please review the Figma link (https://figma.example/abc) before the "
        "meeting and bring one concrete feedback item each. The CEO will be "
        "joining for the first 15 min. Jonathan — can you kick us off with "
        "the discovery-mode mockup?"
    ),
    (
        "Just a heads-up that United Flight UA2873 (LAX → SFO, Wednesday) has "
        "been delayed by 90 minutes. New arrival time 4:45pm local. No action "
        "needed on your end, but I'll update calendar invites for anyone "
        "meeting you at SFO."
    ),
    (
        "Dr. Chen's office here — confirming your annual physical appointment "
        "on May 3 at 10:30am. Please arrive 15 minutes early to update "
        "paperwork and bring a photo ID + insurance card. Reply CONFIRM to "
        "this email or call us at 415-555-0142 to reschedule."
    ),
    (
        "Hi love, quick note: Emma's piano recital is moved from Saturday to "
        "Sunday at 3pm. Same venue. Your mom said she can come but needs a "
        "ride from the train station. Also — Whole Foods is out of the "
        "steel-cut oats again, picked up bob's red mill instead. xo"
    ),
]

CONTACT_PROFILES: list[str] = [
    (
        "Contact: Sarah (mom). Relationship: family, very close. "
        "Communication style: warm, detail-seeking, often asks follow-up "
        "questions. Preferred tone: affectionate, specific about logistics. "
        "Recent pattern: you reply within 4 hours, 2-5 sentences. "
        "Message to reply to: 'Can you pick me up from SFO Wednesday?'"
    ),
    (
        "Contact: Max (coworker, senior engineer). Relationship: peer, "
        "frequent collaboration on v2 rewrite. Communication style: terse, "
        "bullet-points, minimal pleasantries. Preferred tone: direct, "
        "technical. Recent pattern: you reply within 30 min, 1-2 sentences. "
        "Message to reply to: 'Heads up, shipping the Stripe migration "
        "today - breaking change in the webhook signer.'"
    ),
    (
        "Contact: Anna (parent of Emma's friend). Relationship: friendly "
        "acquaintance. Communication style: polite, long-form, cautious. "
        "Preferred tone: warm but concise; avoid commitments without "
        "checking calendar. Recent pattern: you reply within 24 hours, "
        "3-6 sentences. Message to reply to: 'Would Emma want to join us "
        "for piano lessons on Tuesdays? Teacher has one spot left.'"
    ),
]

SEMANTIC_SEARCH_QUERY = "when was the last time Sarah and I talked about Thanksgiving travel?"


# ---------------------------------------------------------------------------
# Measurement data model.
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """One (operation, iteration) sample."""

    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class OperationResult:
    """Aggregated statistics across all iterations of one operation."""

    name: str
    samples: list[Sample] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def latencies(self) -> list[float]:
        return [s.latency_seconds for s in self.samples]

    def quantile(self, q: float) -> float | None:
        """Return the q-th quantile of latencies (q in [0,1]), or None if no samples.

        Uses statistics.quantiles with n=100 for p50/p95/p99 granularity. With
        small N (e.g. 10) the result is linearly interpolated and approximate —
        this matches what production baselining expects.
        """
        lats = self.latencies()
        if not lats:
            return None
        if len(lats) == 1:
            return lats[0]
        idx = round(q * 100) - 1
        idx = max(0, min(98, idx))
        return statistics.quantiles(lats, n=100)[idx]

    def mean_tokens(self, field_name: str) -> float | None:
        vals = [getattr(s, field_name) for s in self.samples if getattr(s, field_name) is not None]
        if not vals:
            return None
        return statistics.fmean(vals)


# ---------------------------------------------------------------------------
# Ollama HTTP client helpers.
# ---------------------------------------------------------------------------


async def ollama_available(client: httpx.AsyncClient, url: str) -> bool:
    """Probe /api/tags; return True iff Ollama responds."""
    import httpx  # local import keeps the module importable without httpx.

    try:
        resp = await client.get(f"{url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        return True
    except (httpx.HTTPError, httpx.ConnectError):
        return False


async def ollama_chat(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    system: str,
    user: str,
) -> tuple[float, dict[str, Any]]:
    """Send one chat completion; return (latency_seconds, raw_response)."""
    start = time.perf_counter()
    resp = await client.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    latency = time.perf_counter() - start
    return latency, resp.json()


async def ollama_embed(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    text: str,
) -> tuple[float, list[float]]:
    """Embed one string; return (latency_seconds, vector)."""
    start = time.perf_counter()
    resp = await client.post(
        f"{url}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60.0,
    )
    resp.raise_for_status()
    latency = time.perf_counter() - start
    vec = resp.json().get("embedding", [])
    return latency, vec


def sample_from_chat_response(latency: float, body: dict[str, Any]) -> Sample:
    prompt_tokens = body.get("prompt_eval_count")
    completion_tokens = body.get("eval_count")
    total = None
    if prompt_tokens is not None and completion_tokens is not None:
        total = prompt_tokens + completion_tokens
    return Sample(
        latency_seconds=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total,
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    # Length-guarded above; index-based to avoid zip(strict=) version issues.
    dot = sum(a[i] * b[i] for i in range(len(a)))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Per-operation runners.
# ---------------------------------------------------------------------------


async def run_briefing_synthesis(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    iterations: int,
) -> OperationResult:
    result = OperationResult(name="briefing_synthesis")
    system = (
        "You are a terse assistant producing a 3-sentence morning briefing from "
        "the provided context blocks. Ground every claim in a block. No padding."
    )
    user = "Context blocks:\n" + "\n".join(f"- {b}" for b in BRIEFING_CONTEXT_BLOCKS)
    for _ in range(iterations):
        try:
            latency, body = await ollama_chat(client, url, model, system, user)
            result.samples.append(sample_from_chat_response(latency, body))
        except Exception as exc:
            result.errors.append(str(exc))
    return result


async def run_per_email_operation(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    iterations: int,
    op_name: str,
    system: str,
) -> OperationResult:
    """Run an operation that feeds one email body per chat request.

    Each iteration feeds all 5 emails in sequence; the aggregated samples
    represent per-email latency (so N effective samples = iterations * 5).
    """
    result = OperationResult(name=op_name)
    for _ in range(iterations):
        for body_text in EMAIL_BODIES:
            try:
                latency, body = await ollama_chat(client, url, model, system, body_text)
                result.samples.append(sample_from_chat_response(latency, body))
            except Exception as exc:
                result.errors.append(str(exc))
    return result


async def run_draft_reply(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    iterations: int,
) -> OperationResult:
    result = OperationResult(name="draft_reply")
    system = (
        "You are drafting a single short reply. Match the contact's preferred "
        "tone and length exactly. Output ONLY the reply body — no preamble, "
        "no sign-off unless the profile indicates one."
    )
    for _ in range(iterations):
        for profile in CONTACT_PROFILES:
            try:
                latency, body = await ollama_chat(client, url, model, system, profile)
                result.samples.append(sample_from_chat_response(latency, body))
            except Exception as exc:
                result.errors.append(str(exc))
    return result


async def run_semantic_search(
    client: httpx.AsyncClient,
    url: str,
    embed_model: str,
    iterations: int,
    corpus_size: int = 100,
) -> OperationResult:
    """Embed a query, cosine-similarity against a prebuilt corpus, return top 5.

    The corpus is generated once (deterministic seed) so iteration-to-iteration
    variance comes from the embedding call and the similarity math, not corpus
    construction.
    """
    result = OperationResult(name="semantic_search")
    rng = random.Random(42)
    corpus_texts = [
        f"Historical event #{i}: {rng.choice(['meeting', 'email', 'message', 'task', 'note'])} "
        f"about {rng.choice(['travel', 'work', 'family', 'finances', 'health'])}."
        for i in range(corpus_size)
    ]
    corpus_vectors: list[list[float]] = []
    try:
        for t in corpus_texts:
            _, v = await ollama_embed(client, url, embed_model, t)
            corpus_vectors.append(v)
    except Exception as exc:
        result.errors.append(f"corpus build failed: {exc}")
        return result

    for _ in range(iterations):
        try:
            total_start = time.perf_counter()
            _, qvec = await ollama_embed(client, url, embed_model, SEMANTIC_SEARCH_QUERY)
            scored = sorted(
                ((cosine_similarity(qvec, cv), i) for i, cv in enumerate(corpus_vectors)),
                reverse=True,
            )[:5]
            latency = time.perf_counter() - total_start
            _ = scored  # top-5 consumed; not recorded per-iteration
            result.samples.append(Sample(latency_seconds=latency))
        except Exception as exc:
            result.errors.append(str(exc))
    return result


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------


def format_latency(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 1000:.1f} ms"


def format_tokens(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f}"


def render_report(
    results: list[OperationResult],
    chat_model: str,
    embed_model: str,
    iterations: int,
    host_note: str,
) -> str:
    lines = [
        "# Ollama Baseline — Life OS v2",
        "",
        f"- Chat model: `{chat_model}`",
        f"- Embed model: `{embed_model}`",
        f"- Iterations per op: {iterations}",
        f"- Host: {host_note}",
        "",
        "## Per-operation latency (wall-clock)",
        "",
        "| Operation | N | p50 | p95 | p99 | mean prompt tok | mean output tok |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        n = len(r.samples)
        p50 = format_latency(r.quantile(0.50))
        p95 = format_latency(r.quantile(0.95))
        p99 = format_latency(r.quantile(0.99))
        pt = format_tokens(r.mean_tokens("prompt_tokens"))
        ot = format_tokens(r.mean_tokens("completion_tokens"))
        lines.append(f"| {r.name} | {n} | {p50} | {p95} | {p99} | {pt} | {ot} |")

    lines.append("")
    lines.append("## Errors")
    any_errors = any(r.errors for r in results)
    if not any_errors:
        lines.append("")
        lines.append("_No errors recorded._")
    else:
        for r in results:
            if r.errors:
                lines.append("")
                lines.append(f"### {r.name}")
                for e in r.errors[:10]:
                    lines.append(f"- `{e}`")
                if len(r.errors) > 10:
                    lines.append(f"- … ({len(r.errors) - 10} more)")
    lines.append("")
    lines.append("## Budget reference (from CEO plan, 2026-04-21)")
    lines.append("")
    lines.append("- `task_extraction` ≤ 2s foreground")
    lines.append("- `briefing_synthesis` ≤ 20s async")
    lines.append("- `priority_classification` ≤ 1s foreground (inferred from UX)")
    lines.append("- `draft_reply` ≤ 5s foreground")
    lines.append("- `semantic_search` ≤ 500ms foreground")
    lines.append("")
    return "\n".join(lines)


def render_skip_report(url: str, reason: str) -> str:
    return "\n".join(
        [
            "# Ollama Baseline — Life OS v2 (NOT RUN)",
            "",
            f"- Attempted URL: {url}",
            f"- Reason: {reason}",
            "",
            "> NOTE: measurement must be re-run on the Mac Mini where Ollama",
            "> is actually installed. This file is a placeholder so the v2",
            "> plan references the artifact path; it will be overwritten when",
            "> the measurement runs.",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--chat-model", default="mistral")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--output",
        default="docs/plans/2026-04-22-ollama-baseline.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--host-note",
        default="local (from measure_ollama_budget.py)",
        help="Freeform host identifier recorded in the report.",
    )
    return parser.parse_args(argv)


async def measure(args: argparse.Namespace) -> int:
    import httpx  # imported here so the module is usable without httpx installed.

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        if not await ollama_available(client, args.url):
            reason = f"Ollama not reachable at {args.url} (connection refused or 404)."
            print(f"warning: {reason}", file=sys.stderr)
            print("warning: emitting skip-report; measurement deferred to Mac Mini.", file=sys.stderr)
            output.write_text(render_skip_report(args.url, reason))
            print(f"wrote {output}")
            return 0

        print(f"measuring against {args.url} (chat={args.chat_model}, embed={args.embed_model})")

        results: list[OperationResult] = []

        print("  → briefing_synthesis")
        results.append(await run_briefing_synthesis(client, args.url, args.chat_model, args.iterations))

        print("  → task_extraction")
        results.append(
            await run_per_email_operation(
                client,
                args.url,
                args.chat_model,
                args.iterations,
                op_name="task_extraction",
                system=(
                    "Extract all actionable tasks from the email. Return JSON: "
                    "{tasks: [{description, priority, deadline_iso}]}. No prose."
                ),
            )
        )

        print("  → priority_classification")
        results.append(
            await run_per_email_operation(
                client,
                args.url,
                args.chat_model,
                args.iterations,
                op_name="priority_classification",
                system=(
                    "Classify the email priority. Output exactly one token: "
                    "URGENT, HIGH, NORMAL, LOW, or ARCHIVE. No explanation."
                ),
            )
        )

        print("  → draft_reply")
        results.append(await run_draft_reply(client, args.url, args.chat_model, args.iterations))

        print("  → semantic_search")
        results.append(await run_semantic_search(client, args.url, args.embed_model, args.iterations))

    report = render_report(
        results,
        chat_model=args.chat_model,
        embed_model=args.embed_model,
        iterations=args.iterations,
        host_note=args.host_note,
    )
    output.write_text(report)
    print(f"wrote {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(measure(args))


if __name__ == "__main__":
    sys.exit(main())
