# Insight Engine — Database & Population Layer

**Date:** 2026-04-14
**Status:** Phase 1 and 2 complete (see git log). Phase 3 open.

---

## What Was Built

### Phase 1 — Cross-Database Correlators
Five new correlators that join data across the five SQLite databases instead of
scanning each one in isolation:

| Correlator | What it finds |
|---|---|
| `_mood_finance_correlation_insights` | Higher spending on high-stress days |
| `_stress_trigger_insights` | Event types that precede stress spikes |
| `_weekly_mood_cycle_insights` | Consistent worst/best day of the week |
| `_prediction_accuracy_insights` | Prediction types with <40% or >80% accuracy |
| `_episode_satisfaction_insights` | Contacts/interaction types with outlier satisfaction |

### Phase 2 — Population Insight Layer (Schema v5)
Six new tables in `user_model.db`:

| Table | Purpose |
|---|---|
| `metric_time_series` | Pre-aggregated daily metrics (16 keys: email, mood, tasks, finance, social) |
| `population_baselines` | 20 research-seeded percentile distributions (McKinsey, APA, BLS, etc.) |
| `metric_correlations` | Significant pairwise Pearson correlations (\|r\| >= 0.4) |
| `cohort_profiles` | Anonymous 6-dimension user classification |
| `insight_outcomes` | Tracks whether insights led to behavior change |
| `contribution_queue` | Scaffolding for future differential-privacy peer network |

Three new modules:
- `storage/population_baselines.py` — `PopulationBaselineStore`
- `services/insight_engine/metric_materializer.py` — `MetricMaterializer`
- `services/insight_engine/cohort.py` — `CohortProfiler`

Three new correlators:
- `_personal_trend_insights` — 14-day recent vs user's own 90-day baseline (>20% fires)
- `_comparative_population_insights` — user metric vs research-seeded population percentile
- `_cross_metric_correlation_insights` — surfaces discovered cross-domain relationships
- `_insight_effectiveness_insights` — which insight categories drive behavior change

### Phase 2b — Three Concrete Fixes
1. **Event loop unblocked** — correlator loop extracted to `_run_correlators_sync`, called
   via `asyncio.to_thread`. 28 correlators no longer block WebSocket/HTTP during scans.
2. **Personal baselines** — `_personal_trend_insights` answers "is this unusual for me?"
   before the population comparison answers "is this unusual in general?"
3. **`insight_outcomes` write path wired** — `_record_outcome_baseline` writes baseline at
   insight storage time; `measure_pending_outcomes` fills 7/14/30-day post values with
   signed delta (positive = improvement via per-metric direction map).

---

## What Still Needs Work

### High Priority

#### 1. `contribution_queue` goes nowhere
**File:** `storage/population_baselines.py`, `storage/manager.py`
**Problem:** The table and `update_from_peer()` method are wired, but nothing ever reads
from or writes to `contribution_queue`. There is no peer network, no aggregation server,
no differential-privacy noise addition. It is placeholder code.
**Options:**
- Remove the table and method until a real peer network exists, OR
- Implement a local simulation: use the contribution_queue to track what *would* be sent,
  so the diff-privacy mechanics are testable when the network is eventually built.

#### 2. Cohort key goes nowhere
**File:** `services/insight_engine/cohort.py`
**Problem:** `CohortProfiler.compute_and_store()` computes an 8-char SHA-256 cohort key
and logs it, but nothing reads the stored key back. The `population_baselines` table has
a `cohort` column that supports cohort-specific baselines, but no baseline is seeded per
cohort — all baselines are `general` or `knowledge_worker`.
**Fix:** Either:
- Use the cohort key to select between baseline cohorts once cohort-specific baselines
  exist, OR
- Add cohort-specific baseline variants (e.g., `knowledge_worker` sub-cohorts by stress
  level) so the classification actually changes which percentile you compare against.

#### 3. `_insight_effectiveness_insights` needs >= 3 months of data
**File:** `services/insight_engine/engine.py`
**Problem:** The correlator requires `insight_outcomes` rows with `days_after >= 7`.
With the write path now wired, the first meaningful effectiveness insights will not
fire until insights are ~7 days old. The system effectively has a 7-day cold start.
**Not a bug** — this is inherent to the design. Document it in the data sufficiency
report output so operators understand why it shows "no_data" on fresh installs.

### Medium Priority

#### 4. `metric_time_series` has no pruning TTL
**File:** `services/insight_engine/metric_materializer.py`
**Problem:** Rows accumulate indefinitely. At 16 metrics × 365 days = 5,840 rows/year
this is fine for storage, but `metric_correlations` rows that drop below |r| >= 0.4
are now pruned (7-day stale cleanup added), while `metric_time_series` never is.
**Fix:** Add a lookback ceiling: delete rows older than `_lookback_days` (currently 90)
during `materialize()` to bound table size.

#### 5. Population baselines are static research data from 2012–2023
**Problem:** McKinsey 2012 email stats, APA 2023 stress surveys. These are broad
population samples, not the user's actual peer group.
**Mitigation already in place:** `_personal_trend_insights` now runs first, so the
user always sees their own deviation before the population comparison.
**Long-term fix:** Allow the user to tag their own role/industry in settings, and
seed role-specific baselines that better match their actual peer group.

#### 6. `email.avg_response_minutes` baseline exists but materializer never writes it
**File:** `services/insight_engine/metric_materializer.py`, `storage/population_baselines.py`
**Problem:** `email.avg_response_minutes` is in `_SEED_BASELINES` but `_materialize_event_counts`
only counts email volume, not response times. The cadence signal extractor computes
response times per contact but does not write to `metric_time_series`.
**Fix:** Add a `_materialize_response_time_metrics()` step that reads from the cadence
signal profile's `avg_response_time_by_contact` and computes a daily global average.

### Low Priority

#### 7. `_approximate_p_value` uses normal approximation for small n
**File:** `services/insight_engine/metric_materializer.py`
**Problem:** For n=14 (min_samples), the normal CDF approximation of the t-distribution
underestimates p-values (0.046 normal vs 0.069 t-dist at df=12). The code filters
on |r| >= 0.4 not on p-value, so this is cosmetic — displayed p-values are wrong.
**Fix:** Implement the regularized incomplete beta function for the t-distribution, or
add a note that p-values are approximate for n < 30.

---

## Test Coverage

| File | Tests | What's covered |
|---|---|---|
| `tests/test_cross_database_insight_correlators.py` | 25 | All 5 cross-DB correlators |
| `tests/test_population_insight_layer.py` | 44 | PBS, materializer, cohort, schema, wiring |
| `tests/test_solid_insight_fixes.py` | 17 | Personal trend, outcome write path, threading |

**Total: 86 tests**, all passing.

---

## Architecture Decisions Made

**Personal baselines before population baselines.** The question order is:
1. "Is this unusual for *me*?" (personal_trend, 20% threshold)
2. "Is this unusual for *my cohort*?" (comparative_population, 80th/20th pct)
3. "What metrics are correlated?" (cross_metric_correlation)

**Signed delta for outcome tracking.** `insight_outcomes.delta` is always positive
when the metric improved (per `higher_is_better` map in `measure_pending_outcomes`),
so `_insight_effectiveness_insights` can count "improved" without knowing each
metric's direction.

**No numpy dependency.** Pearson r and p-value implemented with pure Python `math`
module to keep the dependency footprint small for a Mac Mini deployment.
