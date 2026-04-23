# Cutover runbook — v1 → v2

> Operator guide for the live Phase 1 cutover from v1 (master, NATS + 5 SQLite
> DBs + `main.py`) to v2 (`v2-rewrite`, asyncio + one SQLite DB + LanceDB index,
> Moment-primitive engine, 14-endpoint FastAPI surface).
>
> **RTO target: ≤ 30 minutes** (fixed, per CEO plan § "Data migration").
> A longer cutover triggers the rollback decision tree in § 7.
>
> **Pre-requisites:** three clean dry-runs completed via
> `scripts/cutover_rehearsal.py` with reports filed to
> `docs/cutover-rehearsals/`. Do not proceed without them.

## 0 · At-a-glance timeline

| Phase                                     | Wall-clock | Cumulative |
|-------------------------------------------|:---------:|:----------:|
| 1. Pre-flight (backup, disk, snapshot)    |   5 min   |     5      |
| 2. Stop v1                                |   2 min   |     7      |
| 3. Run migration                          |   5 min*  |    12      |
| 4. Bring up v2                            |   3 min   |    15      |
| 5. Verify (`/api/health`, sample Moment)  |   5 min   |    20      |
| 6. Begin 24-hour watch window             |   0 min   |    20      |
| **Total cutover window**                  |           | **~20 min**|

\* 5 min is the observed scale rehearsal wall-clock at 10K events /
500 entities / 200 signal_profiles; worst-case at 5× that volume is ≤ 25 min.
If phase 3 exceeds **15 min wall-clock**, follow § 7 rollback.

---

## 1 · Pre-flight (target 5 min)

All commands run from the repo root, Mac Mini, as the `lifeos` user.

### 1.1 Snapshot disk state

```bash
df -h ./data
du -sh ./data/*.db ./data/*.lance/
```

**Required:** ≥ 2× the combined size of `events.db + entities.db + state.db +
user_model.db + preferences.db + lance/` free on the filesystem hosting `./data`.
Migration writes a fresh target DB alongside the sources; LanceDB is untouched.

If free space is below 2×, **stop** and make room before proceeding.

### 1.2 Snapshot v1 health (for post-cutover comparison)

```bash
curl -s http://localhost:8080/health > /tmp/v1-health-pre-cutover.json
sqlite3 ./data/events.db "SELECT COUNT(*) AS events, MAX(timestamp) AS newest FROM events;" \
  > /tmp/v1-rowcount-pre-cutover.txt
sqlite3 ./data/state.db    "SELECT COUNT(*) FROM tasks;"            >> /tmp/v1-rowcount-pre-cutover.txt
sqlite3 ./data/entities.db "SELECT COUNT(*) FROM contacts;"         >> /tmp/v1-rowcount-pre-cutover.txt
```

Keep both files. You will diff them against v2 in § 5 and reference them
if rollback triggers during the watch window.

### 1.3 Timestamped full backup

```bash
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p ./data/backup-${TS}
cp -a ./data/*.db ./data/*.lance/ ./data/backup-${TS}/
ls -la ./data/backup-${TS}/
```

**Verify:** the backup directory lists every v1 SQLite file and the LanceDB
directory. This is the artifact the rollback step restores if v2 verification
fails. Do not skip.

### 1.4 Secrets key location

Fernet-encrypted credentials in v1's `preferences.db` are copied intact during
migration (see `scripts/migrate_v1_to_v2.py::migrate_preferences`). The
encryption key lives at:

```text
~/.config/life_os/fernet.key
```

**Do not copy or move the key during cutover.** v2 uses the same key at the
same path. If the key file is missing, stop — decrypting encrypted connector
secrets is impossible without it and v2 will fail to initialize the connector
repo.

---

## 2 · Stop v1 (target 2 min)

### 2.1 Disable the autonomous-improvement launchd jobs

```bash
launchctl unload ~/Library/LaunchAgents/com.lifeos.continuous-improve.plist 2>/dev/null || true
launchctl unload ~/Library/LaunchAgents/com.lifeos.improve.plist           2>/dev/null || true
```

These must stay unloaded until v2 stabilizes (end of watch window). They only
touched v1; re-enabling them against v2 is out of scope for this runbook.

### 2.2 Stop v1 core services

```bash
docker compose stop lifeos       # v1 FastAPI + LifeOS orchestrator
# NATS + Ollama stay up; we tear down NATS in § 4 once v2 is verified
```

### 2.3 Confirm v1 stopped

```bash
curl -sf http://localhost:8080/health && echo "STILL RUNNING — STOP" || echo "v1 down"
docker compose ps lifeos | grep -q 'Exited\|stopped' && echo "container stopped"
```

Expected: "v1 down" + "container stopped". If `/health` still responds, a
second FastAPI process is running outside Docker — find and kill it before
continuing.

---

## 3 · Run migration (target 5 min, hard limit 15 min)

### 3.1 Delete stale dry-run target (if any)

```bash
rm -f ./data/lifeos_v2_dryrun.db
```

The migrator refuses to overwrite; a leftover from a rehearsal will block it.

### 3.2 Run the migration

```bash
time python scripts/migrate_v1_to_v2.py \
  --source-dir ./data \
  --output     ./data/lifeos.db \
  --verbose \
  2>&1 | tee ./data/migrate-${TS}.log
```

**Expected output shape** (trimmed):

```text
2026-MM-DD HH:MM:SS INFO events: read=N translated=N dropped=0
2026-MM-DD HH:MM:SS INFO entities: (contacts/places/subscriptions) translated=M
2026-MM-DD HH:MM:SS INFO signal_profiles: read=P kept=6 dropped=4 (mood/decision/expertise/values)
2026-MM-DD HH:MM:SS INFO preferences: read=K translated=K
2026-MM-DD HH:MM:SS INFO feedback_events: read=F translated=F (source='v1_migration')
2026-MM-DD HH:MM:SS INFO migration report: { "events": {"source": N, "translated": N, "dropped": 0}, ... }
```

### 3.3 Hard-fail checklist

- Exit code **MUST be 0**. Any non-zero exit → § 7 rollback.
- `"dropped": 0` on **every** table. Non-zero drops → § 7 rollback (unless the
  drops are the 4 killed profile types, which are expected: `mood`, `decision`,
  `expertise`, `values`).
- No `INVARIANT:` lines in the report. Any invariant failure → § 7 rollback.

### 3.4 Post-migration FK integrity

```bash
sqlite3 ./data/lifeos.db "PRAGMA foreign_key_check;"
```

Expected: empty output. Any rows printed → § 7 rollback.

---

## 4 · Bring up v2 (target 3 min)

### 4.1 Tear down NATS (v2 does not use it)

```bash
docker compose stop nats
# Ollama stays up — v2 still uses it for local LLM calls.
```

### 4.2 Start v2

```bash
source .venv/bin/activate
python -m life_os &  # serves on :8080, backgrounds into a supervised loop
V2_PID=$!
echo $V2_PID > ./data/v2.pid
```

Wait ~15 seconds for schema checks + connector cold-start. Tail the log:

```bash
tail -f ./data/v2-runs/current.log  # ^C once you see "scheduler: loop started"
```

---

## 5 · Verify (target 5 min)

### 5.1 `/api/health` shape + values

```bash
curl -s http://localhost:8080/api/health | python3 -m json.tool
```

**Required** fields and values (see `api/schemas.py::HealthOut`):

```json
{
  "ok": true,
  "ts": <unix_seconds_within_last_60s>,
  "connectors": { "proton_mail": "ready", "imessage": "ready", "caldav": "ready", "ios_context": "ready" },
  "db_last_write_ts": <unix_seconds_within_last_60s>,
  "scheduler_heartbeat_ts": <unix_seconds_within_last_60s>,
  "producer_activity": { "cadence": 0, "relationship": 0, "temporal": 0, "spatial": 0, "comm_template": 0, "routine": 0 },
  "pending_moments": <int>,
  "notes": []
}
```

**Fail triggers:**

- `ok: false` → § 7 rollback
- Any connector status other than `ready` → § 6 watch; if still not `ready`
  after 5 min wall-clock → § 7 rollback
- `scheduler_heartbeat_ts` stale by > 30 s → § 7 rollback
- `notes` non-empty → read them; most are cosmetic, but any mentioning
  "probe not wired" or "schema" → § 7 rollback

### 5.2 Sample Moment fires end-to-end

Confirm the Moment engine is producing new Moments from live events (not just
the migrated `legacy_task` Moments):

```bash
curl -s 'http://localhost:8080/api/now?limit=20' | python3 -m json.tool | head -80
```

**Expected:**
- At least one Moment in the response.
- If all Moments have `source_insight_type: "legacy_task"` and `state: "suggested"`,
  that is fine at minute 20 — it means v1 tasks migrated cleanly but the engine
  has not yet run its first producer cycle (producers fire every 60-120 s).
  Re-poll after 3 min; expect at least one Moment with a non-`legacy_task`
  `source_insight_type`.

### 5.3 iOS compat shim responds

```bash
curl -s -X POST http://localhost:8080/api/context/event \
  -H 'Content-Type: application/json' \
  -d '{"event_type":"location","timestamp":'"$(date +%s)"',"data":{"latitude":37.7749,"longitude":-122.4194}}' \
  | python3 -m json.tool
```

**Expected:** 200 + JSON acknowledgment (see `api/routes/context.py::post_event`).
A 404 or 503 here means the shim didn't load; check startup logs. The legacy
iOS app (v1 build still installed on the operator's phone during transition)
depends on this route.

### 5.4 Row-count diff against v1

```bash
sqlite3 ./data/lifeos.db "SELECT COUNT(*) FROM events;" > /tmp/v2-rowcount-post-cutover.txt
sqlite3 ./data/lifeos.db "SELECT COUNT(*) FROM moments WHERE source_insight_type='legacy_task';" >> /tmp/v2-rowcount-post-cutover.txt
sqlite3 ./data/lifeos.db "SELECT COUNT(*) FROM entities WHERE kind='contact';" >> /tmp/v2-rowcount-post-cutover.txt
diff /tmp/v1-rowcount-pre-cutover.txt /tmp/v2-rowcount-post-cutover.txt
```

**Expected diff:** numbers match exactly for `events` and `contacts`.
`legacy_task` count should equal v1 `state.tasks` count. Any mismatch → § 7 rollback.

---

## 6 · 24-hour watch window

Once § 5 passes, the cutover window ends and the 24-hour watch window begins.
`scripts/cutover_monitor.py` runs in the background (starts automatically when
`python -m life_os` is up; run manually as a fallback):

```bash
python scripts/cutover_monitor.py \
  --healthy-for-minutes 1440 \
  --alerts-log ./data/cutover-alerts-${TS}.jsonl \
  &
```

### 6.1 What to watch (alert thresholds)

| Signal                                    | Alert threshold                    | Source                       |
|-------------------------------------------|------------------------------------|------------------------------|
| `/api/health → ok`                        | `false` at any scrape              | `/api/health`                |
| Connector offline                         | status != `ready` for > 5 min      | `/api/health.connectors`     |
| DB last-write lag                         | `now - db_last_write_ts > 30 s`    | `/api/health`                |
| Scheduler heartbeat missing               | `now - scheduler_heartbeat_ts > 2 min` | `/api/health`            |
| Pending-Moment backlog grows unbounded    | `pending_moments` rising with no accept/dismiss for > 15 min | `/api/health` + `/api/now` |
| Outbox backlog                            | Any events in `outbox` with `attempts > 5` | `SELECT * FROM outbox WHERE attempts > 5` |
| Silent producer                           | `producer_activity[type] == 0` for > 4 h during active hours | `/api/health` (polled; watch cadence + relationship first) |
| Error log                                 | Any `ERROR` or `CRITICAL` in `./data/v2-runs/current.log` | tail |

### 6.2 Minute 0 + 30 min + 2 h + 6 h + 24 h spot-checks

At each checkpoint, the operator:

1. Curls `/api/health` — `ok: true`.
2. Curls `/api/now` — at least one non-`legacy_task` Moment since cutover.
3. Scans `./data/cutover-alerts-${TS}.jsonl` — zero new alerts.
4. Tails `./data/v2-runs/current.log` — no `ERROR`/`CRITICAL` since last check.

If any checkpoint reveals an alert that was not auto-resolved within 5 min,
evaluate § 7 rollback trigger criteria.

### 6.3 End of watch window

At **hour 24**:
- Stop `cutover_monitor.py`.
- Archive `./data/migrate-${TS}.log`, `./data/cutover-alerts-${TS}.jsonl`, and
  the `./data/backup-${TS}/` directory to cold storage.
- Re-enable the continuous-improvement launchd job **only** if:
  - `ok: true` sustained for the full 24 h
  - Zero rollback triggers fired
  - At least one Moment of each non-empty producer type fired and was either
    accepted or dismissed (not just backlogged)

---

## 7 · Rollback

### 7.1 Trigger criteria (any one → rollback)

Rollback is cheap (RTO ≤ 30 min via § 7.3); the bar to use it is deliberately low.

- Migration exit code non-zero, or any table shows `dropped > 0` outside the
  expected 4 killed profile types, or any `INVARIANT:` line in the report.
- `PRAGMA foreign_key_check` returns any rows on `./data/lifeos.db`.
- v2 `/api/health` returns `ok: false` and the cause cannot be fixed within
  10 min wall-clock.
- Row-count diff in § 5.4 mismatches.
- iOS compat shim (§ 5.3) returns non-200.
- During watch window: pending-Moment backlog grows without accept/dismiss for
  > 1 h **and** a connector is reporting `error` **and** the root cause is not
  a single bad credential.
- Discovered data loss: any query a human operator runs against v2 returns
  fewer rows than the equivalent query against the v1 backup.

### 7.2 Rollback decision

Rollback is the operator's call. When in doubt: **rollback**. The v1 system is
preserved in full in `./data/backup-${TS}/`; a second cutover attempt is cheap.
A half-broken v2 in production is expensive.

### 7.3 Rollback procedure (target ≤ 30 min)

Prefer the scripted path (Phase 2 will harden this further):

```bash
python scripts/cutover_rollback.py \
  --snapshot ./data/backup-${TS} \
  --v1-service lifeos \
  --v2-pid     $(cat ./data/v2.pid) \
  --dry-run     # first pass — audit the plan
python scripts/cutover_rollback.py \
  --snapshot ./data/backup-${TS} \
  --v1-service lifeos \
  --v2-pid     $(cat ./data/v2.pid)
```

Manual fallback if the script is unavailable:

```bash
# 1. Stop v2
kill $(cat ./data/v2.pid); rm -f ./data/v2.pid
# 2. Move v2 DB aside (keep for forensics; don't delete)
mv ./data/lifeos.db ./data/lifeos.db.failed-cutover-${TS}
# 3. Restore v1 DBs + lance from the backup
cp -a ./data/backup-${TS}/*.db ./data/backup-${TS}/*.lance/ ./data/
# 4. Restart NATS, then v1
docker compose start nats
docker compose start lifeos
# 5. Verify v1 up
curl -sf http://localhost:8080/health >/dev/null && echo "v1 back up" || echo "v1 DID NOT RESTART — escalate"
# 6. Reconcile: any events that v2 accepted during its runtime window get
#    replayed into v1 on the next (future) migration attempt via the
#    append-only events table. No action needed now.
```

### 7.4 Post-rollback

- Leave the continuous-improvement launchd jobs **unloaded** until a
  postmortem and a second cutover plan are in place.
- File a rollback-incident note under `docs/cutover-rollbacks/${TS}.md` with:
  trigger, root-cause hypothesis, actual wall-clock time to recover,
  any v1 data integrity check results.
- The next cutover attempt starts at § 1 with a fresh timestamp.

---

## 8 · References

- CEO plan: `~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md`
  § "Data migration (expanded — this is the highest-risk operational step)".
- Engineering plan: `docs/plans/2026-04-21-v2-rewrite-plan.md` § Week 12.
- Migration dry-run: `scripts/migrate_v1_to_v2.py`.
- Cutover rehearsal: `scripts/cutover_rehearsal.py`.
- Cutover monitor: `scripts/cutover_monitor.py` (Category C task).
- Cutover rollback: `scripts/cutover_rollback.py` (Category C task).
- Health schema: `api/schemas.py::HealthOut`.
- ADRs: `docs/adr/2026-04-22-single-sqlite-db.md`,
  `docs/adr/2026-04-22-asyncio-outbox-over-nats.md`,
  `docs/adr/2026-04-22-feedback-events-disposition.md`.
