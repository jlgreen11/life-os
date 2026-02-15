# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Life OS is a local-first, AI-powered personal life management system running on a Mac Mini server. It ingests data from email, messaging, calendar, finance, and smart home connectors, builds a cognitive user model through passive observation, and provides briefings, predictions, and automation.

## Running the System

```bash
# First-time setup
bash scripts/setup.sh

# Start all services (NATS, Ollama, Life OS)
docker compose up -d

# View logs
docker compose logs -f lifeos

# Restart after config changes
docker compose restart lifeos

# Local dev (without Docker)
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Ports:** Web UI at :8080, NATS monitoring at :8222, Ollama at :11434

**Admin tools:** `/admin` (connector management), `/admin/db` (database browser), `/health` (system status)

## Architecture

### Event-Driven Pipeline

All data flows through NATS JetStream (stream: `LIFEOS`, subjects: `lifeos.*`). When an event arrives, `main.py:master_event_handler` processes it through a fixed pipeline:

1. **Store** → append-only `events` table
2. **Signal extraction** → update user model (linguistic, cadence, mood, relationships, topics)
3. **Rules evaluation** → deterministic automation (notify, tag, suppress, create_task, etc.)
4. **Task extraction** → AI-identified action items
5. **Vector embedding** → semantic search index (LanceDB with all-MiniLM-L6-v2, 384 dims)

### Dependency Injection

No global singletons. Every service receives dependencies via constructor args. The `LifeOS` class in `main.py` orchestrates initialization order:

Database → Event Bus → Vector Store → Services → Connectors → Web Server

### 5-Database Storage (`storage/manager.py`)

SQLite with WAL mode, foreign keys enabled. Separated by concern:
- **events.db** — Immutable event log
- **entities.db** — Contacts, places, subscriptions
- **state.db** — Tasks, notifications, connector state
- **user_model.db** — Episodes, facts, routines, signal profiles
- **preferences.db** — Settings, rules, feedback

### AI Engine (`services/ai_engine/engine.py`)

Dual-model: local Ollama (mistral, handles 90% of tasks) + optional Anthropic Claude (complex reasoning, PII-stripped via `PIIShield`). Cloud requires `ai.use_cloud: true` in settings.

### Four-Layer User Model (`models/user_model.py`)

1. **Episodic** — Individual interactions with full context
2. **Semantic** — Facts, explicit/implicit/anti-preferences, expertise, values
3. **Procedural** — Routines, workflows, communication templates
4. **Predictive** — Forward-looking models with accuracy tracking

Confidence grows incrementally (+0.05 per confirmation). Mood state is tracked but NEVER shared externally.

### Prediction Confidence Gates (`services/prediction_engine/engine.py`)

- < 0.3: OBSERVE (silent)
- 0.3–0.6: SUGGEST
- 0.6–0.8: DEFAULT (do it, allow undo)
- \> 0.8: AUTONOMOUS

Reaction scoring adjusts for stress, recent dismissals, time of day, and quiet hours.

## Key Directories

- `connectors/` — External service integrations. Each extends `connectors/base/connector.py` with `authenticate()`, `sync()`, `execute()`, `health_check()`. Registry in `connectors/registry.py` defines config schemas.
- `services/` — Core services: `ai_engine`, `signal_extractor`, `prediction_engine`, `rules_engine`, `task_manager`, `notification_manager`, `event_bus`
- `models/` — Data models: `core.py` (81 event types, enums), `user_model.py` (cognitive model)
- `storage/` — `manager.py` (DatabaseManager, EventStore, UserModelStore), `vector_store.py` (LanceDB + NumPy fallback)
- `web/` — FastAPI app: `app.py` (factory), `routes.py` (all endpoints), `templates/` (Jinja2)
- `config/` — `settings.example.yaml` (template). Active config at `config/settings.yaml` (gitignored)
- `ios/` — Swift/SwiftUI companion app sending context events (location, device proximity)

## Conventions

- **Fail-open:** Errors in one service never crash others. Every service wraps its processing in try/except.
- **Append-only events:** The events table is an immutable log. Never update or delete event rows.
- **Encrypted credentials:** Connector passwords/API keys are Fernet-encrypted via `ConfigEncryptor`. Masked as `********` in API responses.
- **Idempotent upserts:** Use `INSERT OR REPLACE` with `COALESCE` for counter preservation.
- **Event envelope format:** All events published to NATS follow `{id, type, source, timestamp, priority, payload, metadata}`.
- **Config is YAML:** `config/settings.yaml` drives all behavior. Environment variables `NATS_URL` and `OLLAMA_URL` override in Docker.

## Testing

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

Fixtures in `tests/conftest.py` provide real `DatabaseManager` instances with temporary SQLite databases. Manual verification also available via `/health`, `/admin/db`, and connector test endpoints.
