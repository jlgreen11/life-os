# Agent-Driven Improvement Loop — Design

## Problem

Life OS collects rich data (34k emails, iOS location/device context every 5 min, signal profiles) but produces noise instead of insights. The prediction engine generates 186k+ events per cycle, mostly false follow-up alerts on marketing emails. Signal extractors build profiles that are never surfaced. No feedback loop exists to downweight bad predictions.

## Solution: Two-Phase Hybrid

### Phase 1 — Fix the Plumbing

Fix root causes of noise, surface existing data, and wire a feedback loop that Phase 2 agents depend on.

#### Foundation Fixes

**Prediction throttling:** Only run prediction engine when `events` table has new rows since last run. Store a `last_processed_event_id` cursor in connector state.

**Marketing pre-filter:** Before the follow-up detector checks "should we remind about this?", check if the sender matches marketing patterns: contains "unsubscribe" in body, "no-reply@" or "noreply@" in sender, bulk sender heuristics. Skip entirely — don't generate a prediction.

**Feedback wiring:** When a user dismisses a notification, trace back to the source prediction and mark `was_accurate = false`. When they act on it, mark `true`. Exponential decay: each dismissal reduces future confidence by 15% for that prediction type. Acted-on predictions boost by 10%.

**Confidence floor:** Raise minimum surfacing threshold from 0.4 to 0.6. Below that, store for pattern analysis but don't notify.

#### Surface Existing Data

**`/api/insights/summary` endpoint:** Aggregates top patterns from each signal extractor into structured JSON:
- Response time comparisons per contact vs overall average
- Email volume by day-of-week
- Contacts overdue relative to their usual interaction interval
- Linguistic profile highlights (formality variance by contact)

**Dashboard "Insights" panel:** Renders the summary endpoint in the web UI. Shows behavioral patterns, relationship health, and actionable alerts.

#### Feedback Loop

**Prediction accuracy tracking:** Prediction engine loads accuracy history per type/source before generating. Applies decay multiplier to future confidence. Predictions with <20% accuracy after 10+ samples get auto-suppressed.

**Throttle and cap:** Max 5 surfaced predictions per cycle. Batch low-priority predictions into daily digest.

### Phase 2 — InsightEngine + Scheduled Claude Code Agent

#### InsightEngine Service

New service at `services/insight_engine/`. Runs hourly. Cross-correlates signals that individual extractors can't.

**Input:** Signal profiles, events, predictions, iOS context (location/device/time).

**Output:** `Insight` objects:
```python
{
    "id": "uuid",
    "type": "behavioral_pattern | actionable_alert | relationship_intelligence",
    "confidence": 0.82,
    "summary": "You've been at Starbucks every Tuesday 9-11am for 4 weeks",
    "evidence": ["event_id_1", "event_id_2", ...],
    "staleness_ttl": "7d",
    "created_at": "2026-02-15T...",
    "feedback": null
}
```

**Insight types:**
- **Behavioral patterns:** Place frequency, time-of-day habits, communication volume trends. Sourced from iOS location + temporal signals + cadence extractor.
- **Actionable alerts:** Unreplied priority contacts, spending anomalies, calendar conflicts. Higher bar than predictions — requires cross-signal confirmation.
- **Relationship intelligence:** Contact frequency changes, interaction pattern shifts, network clustering.

**Deduplication:** Same pattern doesn't resurface until underlying evidence changes. Keyed on (type, subject_entity, pattern_hash).

**Confidence model:** Inherits from source signals. Boosted/decayed by Phase 1 feedback loop. Insights with multiple supporting signals get a correlation bonus.

**Key distinction from predictions:** Predictions are forward-looking guesses. Insights are backward-looking discoveries grounded in observed data.

#### Claude Code Outer Loop

A `launchd` plist runs Claude Code weekly. Also triggerable manually. Uses a custom skill (`improve-lifeos`).

**Agent workflow per run:**

1. **Query database** for:
   - Dismissed insights (what's annoying the user?)
   - Low-accuracy prediction types (what's broken?)
   - Unclassified high-volume event sources (what's being ignored?)
   - Feedback log trends (what's improving/degrading?)
   - Data quality metrics (signal coverage, null rates, stale profiles)

2. **Generate diagnosis:**
   - "Follow-up predictions from mailing lists still have 80% dismissal rate"
   - "No insights generated from iOS location data — place learning not implemented"
   - "Relationship alerts have 90% engagement rate — consider lowering threshold"

3. **Apply fixes:**
   - Write new rules to the rules engine
   - Adjust confidence thresholds
   - Add event source filters
   - Improve insight templates and correlation logic
   - Add missing UI components for new insight types

4. **Commit to branch, open PR** for review. Optionally auto-merge if configured.

5. **Self-referential improvement:** Each run reads the previous run's changes (git log) + resulting data quality metrics. Builds on prior work iteratively.

**Example improvement trajectory:**
- Week 1: Fixes marketing noise, raises confidence thresholds
- Week 2: Notices location patterns aren't surfacing, adds place frequency aggregation
- Week 3: Sees relationship alerts have high engagement, lowers their threshold to surface more
- Week 4: Detects that Tuesday email volume insight is stale, adjusts staleness TTL

#### launchd Configuration

Plist at `~/Library/LaunchAgents/com.lifeos.improve.plist`:
- Runs weekly (Sunday night)
- Invokes Claude Code with the `improve-lifeos` skill
- Logs to `data/improvement-runs/`
- Manual trigger: `launchctl kickstart gui/$(id -u)/com.lifeos.improve`

## Data Flow

```
Connectors → NATS Event Bus → master_event_handler
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Event Store    Signal Extractors    Rules Engine
                    │               │               │
                    │               ▼               │
                    │        Signal Profiles        │
                    │               │               │
                    ▼               ▼               ▼
              ┌─────────────────────────────────────────┐
              │         InsightEngine (hourly)          │
              │  Cross-correlates signals + events +    │
              │  location + predictions                 │
              └──────────────┬──────────────────────────┘
                             │
                             ▼
                     Insight Objects
                             │
                    ┌────────┼────────┐
                    ▼        ▼        ▼
              Dashboard  Briefing  Notifications
                    │
                    ▼
              User Feedback
                    │
              ┌─────┴──────┐
              ▼             ▼
        Inner Loop      Outer Loop
    (accuracy decay,  (weekly Claude Code
     threshold adj)    agent via launchd)
```

## Build Sequence

1. Foundation fixes (prediction throttle, marketing filter, feedback wiring, confidence floor)
2. `/api/insights/summary` endpoint + dashboard panel
3. InsightEngine service skeleton (hourly runner, insight schema, dedup)
4. Behavioral pattern agent (place frequency, time habits)
5. Relationship intelligence agent (contact frequency, interaction changes)
6. Actionable alert agent (cross-signal confirmation)
7. `improve-lifeos` Claude Code skill
8. launchd plist + improvement run logging
9. Manual trigger command
