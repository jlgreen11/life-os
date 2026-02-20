# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Life OS is a local-first, AI-powered personal life management system running on a Mac Mini server. It ingests data from email, messaging, calendar, finance, browser, and smart home connectors, builds a cognitive user model through passive observation, and provides briefings, predictions, and automation. The system is designed around privacy — all data stays local by default, with optional PII-stripped cloud AI for complex reasoning.

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

**Ports:** Web UI at `:8080`, NATS client at `:4222`, NATS monitoring at `:8222`, Ollama at `:11434`

**Admin tools:** `/admin` (connector management), `/admin/db` (database browser), `/health` (system status)

## Testing

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

- **77 test files** in `tests/` covering storage, services, connectors, web routes, signal extractors, and regression fixes
- Pytest with `asyncio_mode = "auto"` (configured in `pyproject.toml`)
- Fixtures in `tests/conftest.py` provide real `DatabaseManager` instances with temporary SQLite databases — no mocking of the storage layer
- Manual verification also available via `/health`, `/admin/db`, and connector test endpoints

## Linting and Formatting

**Ruff** handles both linting and formatting (replaces flake8, black, isort). Pre-commit hooks are configured for automatic enforcement.

```bash
# Install pre-commit hooks (one-time)
pre-commit install

# Run manually
pre-commit run --all-files

# Or run ruff directly
ruff check . --fix    # lint with auto-fix
ruff format .         # format
```

**Ruff rules** (from `pyproject.toml`): E, W, F, I, UP, B, SIM, RUF. Line length is 120. Target Python 3.12+.

**Ignored rules:** `E501` (formatter handles line length), `B008` (FastAPI uses function calls in defaults), `SIM108` (ternary not always clearer).

**Per-file ignores:** B and SIM rules relaxed in `tests/*`.

**Import ordering:** First-party packages are `connectors`, `models`, `services`, `storage`, `web`.

## Architecture

### Event-Driven Pipeline

All data flows through NATS JetStream (stream: `LIFEOS`, subjects: `lifeos.*`). When an event arrives, `main.py:master_event_handler` processes it through a fixed pipeline:

1. **Store** → append-only `events` table
2. **Feedback processing** → handle notification feedback (acted_on, dismissed)
3. **Source weight tracking** → classify event source, adjust weights
4. **Signal extraction** → update user model (linguistic, cadence, mood, relationships, topics)
5. **Rules evaluation** → deterministic automation (notify, tag, suppress, create_task, etc.)
6. **Task extraction** → AI-identified action items
7. **Vector embedding** → semantic search index (LanceDB with all-MiniLM-L6-v2, 384 dims)
8. **Episode creation** → link related events for episodic memory
9. **Notification generation** → create user-facing alerts

### Dependency Injection

No global singletons. Every service receives dependencies via constructor args. The `LifeOS` class in `main.py` orchestrates initialization order:

Database → Event Bus → Vector Store → Services → Connectors → Web Server

For testing, dependencies can be injected directly via keyword args to `LifeOS.__init__()`.

### 5-Database Storage (`storage/database.py`)

SQLite with WAL mode, foreign keys enabled. Separated by concern:
- **events.db** — Immutable event log (append-only, never update/delete)
- **entities.db** — Contacts, places, subscriptions
- **state.db** — Tasks, notifications, connector state (mutable)
- **user_model.db** — Episodes, facts, routines, signal profiles
- **preferences.db** — Settings, rules, feedback

Key classes: `DatabaseManager` (connection management, schema migrations), `EventStore` (event log interface), `UserModelStore` (4-layer model persistence).

### AI Engine (`services/ai_engine/engine.py`)

Dual-model: local Ollama (mistral, handles 90% of tasks) + optional Anthropic Claude (complex reasoning, PII-stripped via `PIIShield` in `services/ai_engine/pii.py`). Cloud requires `ai.use_cloud: true` in settings. Context assembly in `services/ai_engine/context.py` pulls from episodic, semantic, and procedural memory.

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

Prediction types: NEED, CONFLICT, OPPORTUNITY, RISK, REMINDER. Reaction scoring adjusts for stress, recent dismissals, time of day, and quiet hours.

### Background Loops

`main.py` starts several background `asyncio` loops:
- `_prediction_loop()` — periodic prediction generation
- `_insight_loop()` — behavioral insight computation
- `_semantic_inference_loop()` — high-level fact extraction
- `_routine_detection_loop()` — pattern detection from episodes
- `_behavioral_accuracy_loop()` — prediction calibration tracking

## Key Directories

```
life-os/
├── main.py                 # Application entry point and orchestrator (LifeOS class)
├── config/                 # settings.example.yaml (template); settings.yaml is gitignored
├── connectors/             # External service integrations
│   ├── base/               # BaseConnector abstract class
│   ├── browser/            # Playwright-based browser automation connectors
│   │   ├── engine.py       # Browser automation engine (stealth, state management)
│   │   ├── orchestrator.py # Browser connector orchestration
│   │   ├── whatsapp.py     # WhatsApp via browser
│   │   ├── youtube.py      # YouTube subscriptions via browser
│   │   ├── reddit.py       # Reddit feeds via browser
│   │   └── generic.py      # Generic CSS-selector-based scraping
│   ├── proton_mail/        # Email via Proton Bridge IMAP/SMTP
│   ├── signal_msg/         # Signal Messenger via signal-cli
│   ├── imessage/           # macOS iMessage via chat.db & AppleScript
│   ├── caldav/             # Calendar sync via CalDAV
│   ├── finance/            # Bank transactions via Plaid
│   ├── google/             # Gmail, Calendar, Contacts via OAuth2
│   ├── home_assistant/     # Home automation integration
│   ├── registry.py         # Connector type registry and config schemas
│   └── crypto.py           # Fernet credential encryption
├── services/               # Core business logic
│   ├── ai_engine/          # LLM orchestration (Ollama + Claude), PII shield, context assembly
│   ├── signal_extractor/   # Behavioral signals: linguistic, cadence, mood, relationship, topic
│   ├── prediction_engine/  # Forward-looking intelligence with confidence gates
│   ├── rules_engine/       # Deterministic automation rules
│   ├── task_manager/       # Action item extraction and tracking
│   ├── notification_manager/ # Alert routing and lifecycle
│   ├── event_bus/          # NATS JetStream pub/sub
│   ├── feedback_collector/ # Implicit/explicit user feedback
│   ├── insight_engine/     # Behavioral insights, summaries, source weight management
│   ├── semantic_fact_inferrer/ # High-level fact extraction from episodes
│   ├── routine_detector/   # Pattern detection from behavioral data
│   ├── behavioral_accuracy_tracker/ # Prediction calibration
│   └── onboarding/         # First-run setup wizard
├── models/                 # Data models
│   ├── core.py             # 81 event types, enums (Priority, ConfidenceGate, etc.)
│   └── user_model.py       # 4-layer cognitive model, signal profiles
├── storage/                # Persistence layer
│   ├── database.py         # DatabaseManager, EventStore, UserModelStore
│   └── vector_store.py     # LanceDB semantic search (NumPy fallback)
├── web/                    # FastAPI web server
│   ├── app.py              # Application factory with CORS config
│   ├── routes.py           # 50+ API endpoints
│   ├── schemas.py          # Pydantic request/response schemas
│   ├── websocket.py        # WebSocket manager for real-time push
│   ├── template.py         # Main dashboard HTML (Jinja2)
│   ├── admin_template.py   # Connector management UI
│   ├── db_template.py      # Database browser UI
│   └── setup_template.py   # First-run setup wizard
├── ios/                    # Swift/SwiftUI companion app
│   └── LifeOS/             # Location, device proximity, context events
├── tests/                  # 77 test files (pytest)
├── scripts/                # Setup, backfill, improvement scripts
├── docs/                   # Documentation and plans
├── docker-compose.yaml     # NATS + Ollama + Life OS orchestration
├── Dockerfile              # Python 3.12, Playwright, embedding model
├── pyproject.toml          # Ruff config, pytest config
├── requirements.txt        # Python dependencies
└── .pre-commit-config.yaml # Pre-commit hooks (ruff, formatting, YAML checks)
```

## Connectors

Each connector extends `connectors/base/connector.py` and implements:
- `authenticate()` — Establish connection to external service
- `sync()` — Pull new data, publish events to NATS
- `execute(action, params)` — Perform outbound actions (send message, create event, etc.)
- `health_check()` — Verify connection status

**API connectors:** proton_mail, signal, imessage, caldav, finance, google, home_assistant

**Browser connectors** (Playwright-based, in `connectors/browser/`): whatsapp, youtube, reddit, plus a generic CSS-selector scraper. These use stealth mode and human-speed simulation to avoid detection.

Registry in `connectors/registry.py` defines config schemas for each connector type. Credentials are Fernet-encrypted via `connectors/crypto.py`.

## Web API

50+ endpoints in `web/routes.py`. Key route groups:

- `GET /health` — System health check
- `POST /api/command` — NLP command bar routing
- `GET /api/briefing` — Morning briefing
- `POST /api/search` — Semantic vector search
- `/api/tasks` — CRUD for tasks
- `/api/notifications` — Notification lifecycle (read, dismiss, act)
- `/api/rules` — Automation rule management
- `/api/user-model` — User profile, facts, mood
- `/api/insights` — Behavioral insights and summaries
- `/api/source-weights` — Source weight management
- `/api/preferences` — User preference CRUD
- `/api/feedback` — Feedback recording
- `/api/context/*` — iOS context events (location, device proximity)
- `/api/admin/connectors/*` — Connector management (config, test, enable/disable)
- `GET /ws` — WebSocket for real-time event streaming

Templates are inline Python strings (Jinja2) in `web/template.py`, `admin_template.py`, `db_template.py`, and `setup_template.py`.

## iOS Companion App

Native Swift/SwiftUI app in `ios/LifeOS/` (~3,000 lines). Sends context events to the backend:
- **ContextEngine** — Aggregates location, device proximity, time, activity
- **LocationManager** — Location updates and geofencing
- **DeviceDiscovery** — Nearby Bluetooth/WiFi device detection
- **BackgroundTaskManager** — iOS background sync
- **WebSocketManager** — Real-time server connection
- Views: Dashboard, Chat (AI interface), Context details, Settings

## Conventions

- **Fail-open:** Errors in one service never crash others. Every service wraps its processing in try/except.
- **Append-only events:** The events table is an immutable log. Never update or delete event rows.
- **Encrypted credentials:** Connector passwords/API keys are Fernet-encrypted via `ConfigEncryptor`. Masked as `********` in API responses.
- **Idempotent upserts:** Use `INSERT OR REPLACE` with `COALESCE` for counter preservation.
- **Event envelope format:** All events published to NATS follow `{id, type, source, timestamp, priority, payload, metadata}`.
- **Config is YAML:** `config/settings.yaml` drives all behavior. Environment variables `NATS_URL` and `OLLAMA_URL` override in Docker.
- **No global singletons:** All dependencies are injected via constructor args. Never import shared mutable state.
- **Mood privacy:** Mood state is tracked internally but never exposed to external APIs or shared outside the system.
- **Confidence incrementing:** User model confidence grows by +0.05 per confirmation, never jumps.

## Configuration

Copy `config/settings.example.yaml` to `config/settings.yaml` and fill in credentials. The active config file is gitignored.

Key sections:
- `data_dir` — Path to SQLite databases and browser state (default: `./data`)
- `nats_url` — NATS server URL
- `web_host` / `web_port` — Web server binding
- `cors.allowed_origins` — CORS whitelist
- `embedding_model` — Vector embedding model (`all-MiniLM-L6-v2` or `nomic-embed-text`)
- `ai` — Ollama model, optional cloud API key and model
- `connectors` — Per-connector config blocks (all commented out by default)
- `browser` — Browser automation settings (headless mode, rate limits, credential source)
- `defaults` — User preferences (verbosity, tone, proactivity, autonomy, quiet hours)

## Scripts

- `scripts/setup.sh` — First-time environment setup
- `scripts/analyze-data-quality.py` — Data quality analysis
- `scripts/backfill_reminder_contacts.py` — Backfill contact info for reminder predictions
- `scripts/backfill_task_extraction.py` — Backfill historical task extraction
- `scripts/cleanup_prediction_backlog.py` — Clean up old predictions
- `scripts/run-improvement.sh` / `run-continuous-improvement.sh` — Improvement loop scripts
- `scripts/improve-lifeos.md` / `improvement-agent.md` — AI improvement agent instructions
