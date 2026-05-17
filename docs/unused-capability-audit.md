# Life OS Unused Capability Audit

**Date:** 2026-02-15
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

### 4. CalDAV Conflict Detection (RESOLVED 2026-05-17)

- **File:** `connectors/caldav/connector.py` (the `detect_conflicts()` method)
- **Impact:** Calendar overlap alerts never fire. The default rule
  `"High priority: calendar conflict"` depends on `calendar.conflict.detected` events
  that are never published.
- **Recommendation:** Implement conflict detection in the CalDAV connector's sync loop:
  after fetching events, compare start/end times for overlaps and publish
  `calendar.conflict.detected` events to the bus.
- **Resolution:** Implemented in `connectors/caldav/connector.py:308-490` as
  `_detect_conflicts()`. The method queries calendar events in the upcoming 48h
  window, runs a sweep-line overlap detection, dedupes pairs via `frozenset`, and
  publishes `calendar.conflict.detected` events at `high` priority for each new
  overlap. Persisted pair tracking (`_published_conflicts`) prevents duplicate
  alerts across sync cycles. Iteration 169 widened the query window from "events
  synced in last 24h" to "events with start_time in next 48h" so it catches
  conflicts regardless of when the underlying events were synced.

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

### 6. Communication Templates (IMPLEMENTED 2026-05-17)

- **File:** `storage/manager.py:432-448` (schema), `storage/user_model_store.py:300+`
  (store methods)
- **Impact:** The `communication_templates` table exists with full schema (greeting,
  closing, formality, tone_notes, etc.) and `UserModelStore` has
  `store_communication_template()`, but no extractor or service ever calls it.
- **Recommendation:** Add template extraction to `LinguisticExtractor` or
  `RelationshipExtractor` — analyze outbound messages grouped by contact/channel to
  populate greeting, closing, and formality fields.
- **Resolution:** `LinguisticExtractor._store_per_contact_templates()` in
  `services/signal_extractor/linguistic.py:948` now writes templates by calling
  `self.ums.store_communication_template()` (line 1032). For each contact with
  at least `_MIN_TEMPLATE_SAMPLES` outbound messages, it derives the most-common
  greeting/closing, average word count, dominant channel, formality/emoji-rate
  averages, and tone notes (e.g. "tends to hedge", "asks many questions"). The
  template ID is a deterministic SHA-256 of `linguistic:{contact}:outbound` so
  repeated calls produce idempotent upserts. Existing phrase data from
  `RelationshipExtractor` is preserved via `_get_existing_phrase_data()` to avoid
  overwriting `common_phrases` / `avoids_phrases`.

---

## Dead Code / Over-Engineered

These components have full implementations but are unreachable from any active code path.

### 7. Episodic Memory Layer (IMPLEMENTED 2026-05-17)

- **File:** `storage/manager.py:378-398` (schema), `storage/user_model_store.py`
  (store/query methods)
- **Status:** Full DB schema (`episodes` table with 15 columns), store methods, and
  query methods exist. Zero writes anywhere in the codebase.
- **Recommendation:** Wire episode creation into `master_event_handler` or the signal
  extractor pipeline. Each processed event should create an episode summarizing the
  interaction.
- **Resolution:** `master_event_handler` in `main.py:2607` now calls
  `self._create_episode(event)` after signal extraction, and the implementation at
  `main.py:3027` (with `store_episode` at `main.py:3223`) persists every processed
  event as an episode with interaction-type classification, topics, entities, and
  contact linkage. As of 2026-05-17 the `episodes` table contains ~42,078 rows
  (confirmed via the data quality analyzer). Downstream consumers — the semantic
  fact inferrer, routine detector, and topic queries — read from this table in
  production. The "topics from most recent episode" path in
  `services/insight_engine/engine.py:986` is one example of an active reader.

### 8. Unused DB Columns (IMPLEMENTED 2026-05-17)

The following columns are defined in schemas but never written to or queried:

| Table | Column | DB |
|-------|--------|----|
| `semantic_facts` | `is_user_corrected` | user_model.db |
| `semantic_facts` | `times_confirmed` | user_model.db |
| `semantic_facts` | `source_episodes` | user_model.db |

These columns support the confidence-growth loop described in the architecture doc
(+0.05 per confirmation) but the increment logic is not implemented.

- **Resolution:** All three columns are now load-bearing in
  `storage/user_model_store.py`. `store_semantic_fact()` (around line 294)
  accumulates `source_episodes` as a JSON array of supporting episode IDs and
  increments `times_confirmed` on every upsert of an existing fact. The
  semantic fact inferrer at `services/semantic_fact_inferrer/inferrer.py:882-983`
  uses `is_user_corrected = 0` as a safety filter so user-corrected facts are
  never auto-deleted during reconciliation. `get_high_confidence_facts()` filters
  by `times_confirmed` to surface well-established facts. The +0.05-per-
  confirmation confidence-growth loop runs through these columns end-to-end.

### 9. LinguisticProfile Fields

- **File:** `models/user_model.py` (LinguisticProfile class)
- **Status:** ~60% of fields are defined but never set by the `LinguisticExtractor`.
  The extractor computes basic metrics (word count, vocabulary richness) but skips
  advanced fields like `hedge_words_ratio`, `question_ratio`, `emoji_density`, etc.
- **Recommendation:** Implement incrementally — each field is a straightforward text
  analysis that can be added to `LinguisticExtractor.extract()` one at a time.

---

## Intentionally Disabled

### 10. Browser Automation

- **Files:** `connectors/browser/whatsapp.py`, `connectors/browser/youtube.py`,
  `connectors/browser/reddit.py`, `connectors/browser/orchestrator.py`
- **Status:** Fully working Playwright-based automation for WhatsApp, YouTube, and
  Reddit. Disabled by config flag (`browser_automation.enabled: false` in settings).
- **Reason:** Intentionally gated — browser automation is resource-intensive and
  requires user opt-in. This is correct behavior, not a bug.

---

## Revision history

- **2026-05-17** — Marked sections 4, 6, 7, and 8 as resolved/implemented after
  verifying via grep + line-range citations:
  - §4 CalDAV Conflict Detection: `_detect_conflicts()` at
    `connectors/caldav/connector.py:308-490` publishes `calendar.conflict.detected`.
  - §6 Communication Templates: `_store_per_contact_templates()` at
    `services/signal_extractor/linguistic.py:948` calls
    `UserModelStore.store_communication_template()` (line 1032).
  - §7 Episodic Memory Layer: episode creation wired into `master_event_handler`
    via `_create_episode()` at `main.py:3027` (`store_episode` at `main.py:3223`).
    Current row count ~42,078.
  - §8 Unused DB Columns: `is_user_corrected`, `times_confirmed`, and
    `source_episodes` are all active in `storage/user_model_store.py` and
    `services/semantic_fact_inferrer/inferrer.py`.
- **2026-02-15** — Initial audit.
