# Life OS Unused Capability Audit

**Date:** 2026-02-15 (updated 2026-04-14)
**Scope:** Full codebase review for stubbed, dead, or underutilized features

---

## Critical: Out-of-Box Features Broken

These are features referenced by default rules or core pipeline stages that ship as
no-ops, meaning the system's advertised automation is partially non-functional.

### 1. Event Tagging (FIXED)

- **File:** `main.py:247-249`
- **Symptom:** `_execute_rule_action()` receives `"tag"` actions but the handler is
  `pass`. Three of four default rules emit tag actions (`"marketing"`,
  `"has-attachments"`, `"large-transaction"`), so none of them actually tag anything.
- **Root cause:** No `event_tags` table existed and `EventStore` had no tagging API.
- **Fix:** Added `event_tags` table to `events.db` schema, added `EventStore.add_tag()`
  / `get_tags()` / `has_tag()` methods, and wired the tag action handler in `main.py`.

### 2. Suppress Flag (FIXED)

- **File:** `main.py:250-255`
- **Symptom:** The "Archive marketing emails" default rule fires a `"suppress"` action,
  but the handler is `pass`. Marketing emails are never actually suppressed.
- **Root cause:** No persistence or in-memory flagging mechanism existed.
- **Fix:** Suppress now (a) sets `event["_suppressed"] = True` on the in-memory dict so
  downstream pipeline stages skip the event, (b) persists a `system:suppressed` tag in
  `event_tags` for audit/query purposes, and (c) suppress actions are processed before
  notify actions so they can cancel notifications from other rules.

### 3. Response Time Calculation (FIXED)

- **File:** `services/signal_extractor/cadence.py:92-103`
- **Symptom:** `_calculate_response_time()` always returns `None`. The cadence extractor
  detects replies correctly (checks `is_reply` + `in_reply_to`) but never computes the
  actual delta, so per-contact response-time signals are never emitted. This breaks
  priority contact detection in the prediction engine.
- **Root cause:** Missing event-store query to look up the original inbound message.
- **Fix:** Implemented the lookup using `json_extract(payload, '$.message_id')` against
  the events table, with an expression index for performance. Returns the delta in
  seconds between original and reply timestamps.

---

## Major: Defined but Never Populated

These features have complete schemas and model definitions but no code path that
writes data into them. They are architectural scaffolding awaiting implementation.

### 4. CalDAV Conflict Detection

- **File:** `connectors/caldav/connector.py` (the `detect_conflicts()` method)
- **Impact:** Calendar overlap alerts never fire. The default rule
  `"High priority: calendar conflict"` depends on `calendar.conflict.detected` events
  that are never published.
- **Recommendation:** Implement conflict detection in the CalDAV connector's sync loop:
  after fetching events, compare start/end times for overlaps and publish
  `calendar.conflict.detected` events to the bus.

### 5. User Model Profiles (Composite)

- **File:** `models/user_model.py`
- **Classes:** `TemporalProfile`, `DecisionProfile`, `SpatialProfile`, and the
  composite `UserModel` class that aggregates all profiles.
- **Impact:** The `UserModel` class is never instantiated. Individual signal profiles
  (linguistic, cadence, mood, etc.) are stored via `signal_profiles` table but the
  higher-level composite model is unused.
- **Recommendation:** Either wire `UserModel` into the orchestrator as the unified
  access point for all profile data, or remove the composite class and keep the
  per-signal-profile approach (which works well today).

### 6. Communication Templates

- **File:** `storage/manager.py:432-448` (schema), `storage/user_model_store.py:300+`
  (store methods)
- **Impact:** The `communication_templates` table exists with full schema (greeting,
  closing, formality, tone_notes, etc.) and `UserModelStore` has
  `store_communication_template()`, but no extractor or service ever calls it.
- **Recommendation:** Add template extraction to `LinguisticExtractor` or
  `RelationshipExtractor` — analyze outbound messages grouped by contact/channel to
  populate greeting, closing, and formality fields.

---

## Dead Code / Over-Engineered

These components have full implementations but are unreachable from any active code path.

### 7. Episodic Memory Layer

- **File:** `storage/manager.py:378-398` (schema), `storage/user_model_store.py`
  (store/query methods)
- **Status:** Full DB schema (`episodes` table with 15 columns), store methods, and
  query methods exist. Zero writes anywhere in the codebase.
- **Recommendation:** Wire episode creation into `master_event_handler` or the signal
  extractor pipeline. Each processed event should create an episode summarizing the
  interaction.

### 8. Unused DB Columns

The following columns are defined in schemas but never written to or queried:

| Table | Column | DB |
|-------|--------|----|
| `semantic_facts` | `is_user_corrected` | user_model.db |
| `semantic_facts` | `times_confirmed` | user_model.db |
| `semantic_facts` | `source_episodes` | user_model.db |

These columns support the confidence-growth loop described in the architecture doc
(+0.05 per confirmation) but the increment logic is not implemented.

### 9. LinguisticProfile Fields

- **File:** `models/user_model.py` (LinguisticProfile class)
- **Status:** ~60% of fields are defined but never set by the `LinguisticExtractor`.
  The extractor computes basic metrics (word count, vocabulary richness) but skips
  advanced fields like `hedge_words_ratio`, `question_ratio`, `emoji_density`, etc.
- **Recommendation:** Implement incrementally — each field is a straightforward text
  analysis that can be added to `LinguisticExtractor.extract()` one at a time.

---

---

## Added 2026-04-14 — Insight Engine Population Layer

### 11. `insight_outcomes` Write Path (FIXED 2026-04-14)

- **Files:** `services/insight_engine/engine.py`
- **Symptom:** The `insight_outcomes` table had a complete schema and a reader
  (`_insight_effectiveness_insights`) but zero writes anywhere. The effectiveness
  correlator always returned `[]`.
- **Fix:** `_record_outcome_baseline()` writes a `days_after=0` baseline row whenever
  an insight with `metric_key` in its evidence is stored. `measure_pending_outcomes()`
  fills 7/14/30-day post rows with signed delta (positive = improvement). Called from
  `_insight_loop` each cycle via `asyncio.to_thread`. The effectiveness correlator
  now has real data after ~7 days of operation.

### 12. `generate_insights` Blocked the Event Loop (FIXED 2026-04-14)

- **File:** `services/insight_engine/engine.py`
- **Symptom:** All 28 correlators ran synchronously on the event loop thread. During
  the multi-second DB scan, WebSocket pushes and HTTP responses were delayed.
- **Fix:** Extracted correlator loop into `_run_correlators_sync()`, called via
  `asyncio.to_thread` from `generate_insights`. The event loop is now free during
  correlation scans.

### 13. `contribution_queue` — Never Written To

- **Files:** `storage/manager.py`, `storage/population_baselines.py`
- **Symptom:** The table and `update_from_peer()` method exist but nothing writes to
  `contribution_queue`. There is no peer network, no aggregation server, and no
  differential-privacy noise addition. Pure scaffolding.
- **Recommendation:** Either remove until a real peer network exists, or implement a
  local simulation so the differential-privacy mechanics can be tested in isolation.

### 14. Cohort Key Computed but Never Used

- **File:** `services/insight_engine/cohort.py`
- **Symptom:** `CohortProfiler` computes a SHA-256 cohort key and stores 6 dimension
  buckets, but nothing reads the key back. Population baselines are all seeded as
  `general` or `knowledge_worker` — there are no cohort-specific baseline variants
  for the key to select between.
- **Recommendation:** Add cohort-specific baseline variants (e.g., by stress profile
  or communication volume), then wire `get_cohort_key()` into the comparative
  correlator to select the most specific matching baseline.

### 15. `email.avg_response_minutes` Baseline Has No Materializer

- **Files:** `services/insight_engine/metric_materializer.py`, `storage/population_baselines.py`
- **Symptom:** `email.avg_response_minutes` is seeded in `population_baselines` but
  `MetricMaterializer` never writes it to `metric_time_series`. The cadence signal
  extractor computes per-contact response times but not a daily aggregate.
- **Recommendation:** Add `_materialize_response_time_metrics()` to `MetricMaterializer`
  that reads from the cadence signal profile and computes a daily average response time.

---

## Intentionally Disabled

### 10. Browser Automation

- **Files:** `connectors/browser/whatsapp.py`, `connectors/browser/youtube.py`,
  `connectors/browser/reddit.py`, `connectors/browser/orchestrator.py`
- **Status:** Fully working Playwright-based automation for WhatsApp, YouTube, and
  Reddit. Disabled by config flag (`browser_automation.enabled: false` in settings).
- **Reason:** Intentionally gated — browser automation is resource-intensive and
  requires user opt-in. This is correct behavior, not a bug.
