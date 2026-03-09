# How Life OS Works

Life OS is a local-first, AI-powered personal assistant that runs on a Mac Mini server. It passively observes your digital life — email, messages, calendar, smart home, finances — builds a cognitive model of who you are, and uses that model to surface briefings, predictions, and automations. All data stays on-device by default.

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────┐
│                        Web Dashboard                         │
│         3-column layout: Topics │ Feed │ AI Sidebar           │
│              Command bar, WebSocket push                     │
└────────────────────────┬─────────────────────────────────────┘
                         │ FastAPI (:8080)
┌────────────────────────┴─────────────────────────────────────┐
│                      LifeOS Core                             │
│                                                              │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐  ┌───────────┐  │
│  │ Signal  │  │Prediction│  │   Rules    │  │    AI     │  │
│  │Extractor│  │  Engine  │  │  Engine    │  │  Engine   │  │
│  │Pipeline │  │          │  │            │  │(Ollama +  │  │
│  │(8 types)│  │(5 types) │  │(triggers + │  │ Claude)   │  │
│  │         │  │          │  │ actions)   │  │           │  │
│  └────┬────┘  └────┬─────┘  └─────┬──────┘  └─────┬─────┘  │
│       │            │              │                │         │
│  ┌────┴────────────┴──────────────┴────────────────┴─────┐  │
│  │                    NATS JetStream                      │  │
│  │              Event Bus (lifeos.* subjects)             │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────┴───────────────────────────────┐  │
│  │                     Connectors                         │  │
│  │  Gmail │ ProtonMail │ Signal │ iMessage │ CalDAV │ ... │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    Storage Layer                        │  │
│  │  events.db │ entities.db │ state.db │ user_model.db    │  │
│  │  preferences.db │ LanceDB vectors                      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Key principle:** No global singletons. Every service receives its dependencies via constructor injection. The `LifeOS` class in `main.py` orchestrates initialization order.

---

## Startup

Startup is split into two phases so the web UI is available within seconds.

### Phase A: Fast Bootstrap (~5 seconds)

1. **Databases** — 5 SQLite databases created/opened in WAL mode:
   - `events.db` — Append-only immutable event log (never updated or deleted)
   - `entities.db` — Contacts, places, subscriptions
   - `state.db` — Tasks, notifications, connector state
   - `user_model.db` — Episodic memory, semantic facts, routines, signal profiles
   - `preferences.db` — Settings, rules, feedback

2. **Vector Store** — LanceDB initialized with `all-MiniLM-L6-v2` embeddings (384 dimensions) for semantic search

3. **Event Bus** — Connects to NATS JetStream

4. **Services** — All services instantiated in dependency order: SignalExtractorPipeline, AIEngine, RulesEngine, PredictionEngine, InsightEngine, SemanticFactInferrer, RoutineDetector, TaskManager, NotificationManager, etc.

5. **Background Loops** — 13 async loops started with automatic crash recovery:
   - Prediction generation (every 60–120s)
   - Behavioral insight computation
   - Semantic fact inference
   - Routine detection
   - Prediction accuracy calibration
   - Task completion detection & overdue monitoring
   - Digest delivery (morning/weekly briefings)
   - Database health & daily backup
   - Calendar conflict detection
   - Connector health monitoring
   - NATS reconnection
   - Notification expiry cleanup

6. **Web Server** — FastAPI starts on port 8080. Dashboard is immediately accessible.

### Phase B: Background Initialization

While the web UI is live, heavier work runs in the background:

- User model integrity checks (detects and repairs SQLite corruption)
- Episode backfill from events.db
- Signal profile backfill (relationship, temporal, topic, linguistic, cadence, mood, spatial, decision — run concurrently)
- Semantic fact re-inference if the fact store is empty
- Communication template extraction
- **Connectors started** — All enabled connectors begin syncing
- Startup state transitions: `starting` → `ready` → `backfilling` → `running`

---

## Event Pipeline

Every piece of data — an email, a message, a calendar event, a smart home sensor reading — flows through the same 6-stage processing pipeline.

### The Journey of an Email

```
ProtonMail connector polls IMAP
    │
    ▼
New email found → parse headers, body, attachments
    │
    ▼
Publish to NATS: subject "lifeos.email.received"
    │
    ▼
master_event_handler() picks it up
    │
    ├── Stage 1: PERSIST
    │   Store event in events.db (append-only, never modified)
    │
    ├── Stage 1b: FEEDBACK
    │   Process any notification feedback (acted_on, dismissed)
    │
    ├── Stage 1c: SOURCE WEIGHTS
    │   Classify event source, track engagement patterns
    │
    │   [Guard: skip system events like notification.*, task.*]
    │
    ├── Stage 2: LEARN
    │   Signal extractor pipeline updates 8 behavioral profiles:
    │   linguistic, cadence, mood, relationship, topic,
    │   temporal, spatial, decision
    │
    ├── Stage 3: REACT
    │   Rules engine evaluates user-defined automations
    │   (e.g., "if from:boss and subject:urgent → notify immediately")
    │
    ├── Stage 4: EXTRACT
    │   Task manager uses AI to identify action items in the email
    │   ("Please send the report by Friday" → new task)
    │
    ├── Stage 5: EMBED
    │   Generate 384-dim vector embedding, store in LanceDB
    │   (enables semantic search: "What did Mike say about Denver?")
    │
    └── Stage 6: EPISODIC MEMORY
        Create an episode linking to the event with full context:
        location, mood, domain, contacts, topics, outcome
```

Later, background loops process episodes for higher-order intelligence: routine detection, semantic fact inference, prediction generation.

### Fail-Open Design

Each stage is wrapped in try/except. If signal extraction fails, the event still gets embedded and stored as an episode. No single failure crashes the pipeline.

---

## The User Model

Life OS builds a 4-layer cognitive model entirely through passive observation — no manual input required.

### Layer 1: Episodic Memory

Individual interactions with full context. Every content-bearing event becomes an episode stored in `user_model.db`:

- **What happened:** Content summary and full text
- **When:** Timestamp from the original source (not sync time)
- **Where:** Location context (if available from iOS companion or spatial signals)
- **Mood:** Inferred emotional state at the time (energy, stress, social battery, cognitive load, valence)
- **Domain:** Work, personal, social, health, finance
- **Who:** Contacts involved
- **Outcome:** Did the user act on it, ignore it, or did the AI handle it?

Use case: "What happened in my meeting yesterday?" or "When did I last talk to Mike?"

### Layer 2: Semantic Memory

Distilled, persistent knowledge extracted from episodes by the SemanticFactInferrer:

- **Facts:** "shellfish allergy", "prefers aisle seats", "born in Denver"
- **Explicit preferences:** Things the user stated ("no reply-all emails")
- **Implicit preferences:** Observed but never stated (prefers text over calls)
- **Anti-preferences:** Things they dislike
- **Expertise map:** `{"python": 0.9, "cooking": 0.4, "tax_law": 0.1}`
- **Values:** `{"privacy": 0.95, "family_time": 0.9, "career_growth": 0.7}`

Each fact tracks confidence (starts at 0.5, grows +0.05 per confirmation), source episodes, and whether the user has corrected it.

Use case: Personalize tone, detect allergies for restaurant suggestions, adapt to values.

### Layer 3: Procedural Memory

Learned sequences and behavioral patterns:

- **Routines:** Recurring patterns like "morning routine" (check email → review calendar → respond to urgent items). Tracks consistency score and known variations.
- **Workflows:** Multi-step goal-driven processes like "responding to boss" or "planning a trip". Records tools used and success rate.
- **Communication Templates:** Per-contact writing style — greeting, closing, formality level, typical length, emoji usage, common phrases, tone. Used by the AI engine to draft messages in the user's voice.

Use case: AI drafts an email matching the user's style for that specific contact.

### Layer 4: Predictive Models

Forward-looking intelligence with 5 prediction types:

| Type | Example |
|------|---------|
| NEED | "You'll probably want to follow up with Mike — it's been 14 days" |
| CONFLICT | "Meeting overlap detected: standup and dentist at 10am Tuesday" |
| OPPORTUNITY | "Good time for deep work — your calendar is clear and energy is high" |
| RISK | "Tax documents due in 3 days, no progress detected" |
| REMINDER | "You haven't contacted your brother in 30 days" |

Each prediction carries a confidence score that determines how it's surfaced:

| Confidence | Gate | Behavior |
|------------|------|----------|
| < 0.3 | OBSERVE | Silent learning, no notification |
| 0.3 – 0.6 | SUGGEST | "Would you like me to..." |
| 0.6 – 0.8 | DEFAULT | Do it, allow undo |
| > 0.8 | AUTONOMOUS | Handle it automatically |

**Reaction scoring** adjusts confidence based on user behavior: acting on a prediction boosts it, dismissing it lowers it. The system also accounts for stress levels, quiet hours, and recent dismissal history.

---

## Signal Extraction

Every event triggers up to 8 behavioral analyzers that update signal profiles. These profiles power predictions, insights, and AI personalization.

| Signal | What It Tracks |
|--------|---------------|
| **Linguistic** | Vocabulary complexity, formality, emoji usage, punctuation habits, personality markers (hedging, assertiveness), per-contact voice variations |
| **Cadence** | Response times by contact/channel, activity windows, conversation initiation patterns, read-but-not-replied indicators |
| **Temporal** | Chronotype (early bird vs night owl), energy curve by hour, weekly rhythm, deadline behavior, procrastination patterns |
| **Relationship** | Interaction frequency per contact, reciprocity ratios, channel diversity, importance scoring, conflict indicators |
| **Topic** | Interest domains, expertise levels, discussion frequency, content consumption patterns |
| **Mood** | Energy level, stress, social battery, cognitive load, emotional valence — inferred from typing patterns, message frequency, tone shifts |
| **Spatial** | Place-behavior associations, typical transitions, notification preferences by location |
| **Decision** | Decision speed by domain, research depth, risk tolerance, delegation comfort, decision fatigue patterns |

All profiles are stored as JSON in the `signal_profiles` table and updated on every relevant event.

---

## AI Engine

Life OS uses a dual-model architecture.

### Local Model (Ollama — 90% of tasks)

- Model: Mistral, running locally on the Mac Mini with Metal acceleration
- Used for: Briefing synthesis, action item extraction, priority classification, semantic search, triage
- Latency: <2 seconds (no network round-trip)
- Privacy: All data stays on-device

### Cloud Model (Anthropic Claude — complex tasks)

- Optional: requires `ai.use_cloud: true` in config
- Used for: High-quality email drafts, nuanced behavioral insights
- **PII Shield:** Before sending anything to the cloud, `PIIShield` replaces names, emails, phone numbers, and addresses with tokens (`PII_REDACTED_0`, etc.). After the cloud response, real values are reinserted. The cloud model never sees actual personal information.

### Key AI Operations

| Operation | Description | Model |
|-----------|-------------|-------|
| **Briefing** | Synthesizes calendar, tasks, unread messages, mood, and patterns into a personalized prose summary | Local |
| **Draft Reply** | Generates a reply matching the user's writing style for a specific contact (formality, length, greeting, tone) | Cloud preferred, local fallback |
| **Action Items** | Scans email/message bodies and extracts tasks with due dates, priority, and completion status | Local |
| **Priority Classification** | Quick triage of events into critical/high/normal/low | Local |
| **Semantic Search** | Natural language search across all events via vector similarity ("What did Mike say about Denver?") | Local |

### Context Assembly

The AI engine assembles rich context windows before each LLM call:

- **Briefing context:** 12 sections — calendar, tasks, unread messages, completions, predictions, episodes, mood, facts, insights, routines, habits, preferences
- **Draft context:** 5 priority layers — communication template for this contact (highest), per-contact outbound metrics, global linguistic profile, contact's inbound register, recent conversation history
- **Search context:** Query + intent framing for the LLM

---

## Connectors

Connectors integrate with external services. Each extends `BaseConnector` and implements four methods:

```python
async def authenticate()   # Establish connection
async def sync()           # Pull new data → publish events to NATS
async def execute(action)  # Outbound actions (send email, create event)
async def health_check()   # Verify connection status
```

### API Connectors

| Connector | Protocol | Data |
|-----------|----------|------|
| **Gmail** | Google OAuth2 | Email, Calendar, Contacts |
| **Proton Mail** | IMAP/SMTP via Proton Bridge | Email |
| **Signal** | signal-cli daemon socket | Messages |
| **iMessage** | macOS chat.db + AppleScript | Messages |
| **CalDAV** | CalDAV protocol | Calendar events |
| **Finance** | Plaid API | Bank transactions |
| **Home Assistant** | REST API | Smart home state |

### Browser Connectors (Playwright)

For services without APIs, Life OS uses headless browser automation with stealth mode and human-speed simulation:

| Connector | Target |
|-----------|--------|
| **WhatsApp** | WhatsApp Web |
| **YouTube** | Subscription feeds |
| **Reddit** | Feed content |
| **Generic** | Any site via CSS selectors |

### Connector Lifecycle

1. Config loaded from `settings.yaml` or the admin UI
2. Credentials encrypted at rest via Fernet (`ConfigEncryptor`)
3. `authenticate()` establishes the connection
4. `sync()` runs on a configurable interval (e.g., every 30 seconds for email)
5. Each new item is transformed into a normalized event and published to NATS
6. The event enters the 6-stage pipeline described above

---

## Web Dashboard

The dashboard is a single-page application served at `/`. All HTML, CSS, and JavaScript are embedded in a single Python template — zero static-file dependencies.

### Layout

```
┌──────────────────────────────────────────────────────┐
│  Logo    [Command Bar _______________]   Mood   Nav  │
├────────┬─────────────────────────┬───────────────────┤
│        │                         │                   │
│ Topics │      Main Feed          │   AI Sidebar      │
│        │                         │                   │
│ Inbox  │  ┌─────────────────┐   │  Briefing         │
│ Msgs   │  │ Email from Mike │   │  ──────────       │
│ Email  │  │ Subject: Denver │   │  Good morning...  │
│ Cal    │  │ Preview text... │   │                   │
│ Tasks  │  └─────────────────┘   │  Predictions      │
│ Insight│  ┌─────────────────┐   │  ──────────       │
│ System │  │ Task: Send rpt  │   │  Follow up with   │
│        │  │ Due: Friday     │   │  Mike (87%)       │
│        │  └─────────────────┘   │                   │
│        │                         │  People Radar     │
│        │                         │  ──────────       │
│        │                         │  Mike - 3d ago    │
│        │                         │  Sarah - 1d ago   │
│        │                         │                   │
│ 160px  │      flexible           │  Mood Snapshot    │
│        │                         │  280px            │
├────────┴─────────────────────────┴───────────────────┤
│  Status: Connected │ Events: 12,847 │ Last sync: 30s │
└──────────────────────────────────────────────────────┘
```

### Features

- **Topic Navigation** — Switch between Inbox, Messages, Email, Calendar, Tasks, Insights, System views
- **Command Bar** — Natural language input routed to `/api/command` ("Email Mike about Denver", "Add milk to tasks")
- **AI Sidebar** — Briefing, predictions, people radar, mood snapshot — each refreshes independently
- **Card Actions** — Reply, Complete, Dismiss, Create Task, Forward
- **Real-time Updates** — WebSocket connection to `/ws` shows "New items" banner (user clicks to refresh — no auto-reload)
- **Dark Theme** — CSS custom properties system, priority-colored borders (red = critical, orange = high)
- **Responsive** — Sidebar hides at 900px, navigation hides at 600px with mobile tab bar
- **Security** — All dynamic content passes through `escHtml()` before DOM insertion

### Key API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/briefing` | Morning briefing |
| `POST /api/command` | NLP command routing |
| `POST /api/search` | Semantic vector search |
| `GET/POST /api/tasks` | Task management |
| `GET /api/predictions` | Active predictions |
| `GET/POST /api/notifications` | Notification lifecycle |
| `GET/POST /api/rules` | Automation rules |
| `GET /api/user-model` | User profile and facts |
| `GET /api/insights` | Behavioral insights |
| `GET /health` | System health check |
| `GET /ws` | WebSocket for real-time push |
| `/admin` | Connector management UI |
| `/admin/db` | Database browser |

---

## Rules Engine

The rules engine provides deterministic automation — user-defined triggers and actions that evaluate on every event.

**Triggers:** Sender filter, subject regex, body keywords, time of day, calendar state, task status, mood state

**Actions:** `notify`, `suppress`, `tag`, `create_task`, `send_template_reply`, `forward`, `archive`

Example: "If an email arrives from my boss with 'urgent' in the subject, send a high-priority notification immediately."

Suppress actions run first to prevent competing notifications.

---

## Privacy Model

Life OS is designed around privacy:

1. **Local-first:** All data stored in local SQLite databases. No cloud sync by default.
2. **Encrypted credentials:** Connector passwords and API keys are Fernet-encrypted at rest.
3. **PII Shield:** When the optional cloud AI model is used, all personally identifiable information is tokenized before leaving the device and restored after the response.
4. **Mood privacy:** Mood state is tracked internally for tone calibration and stress-sensitive autonomy, but is never exposed to external APIs or shared outside the system.
5. **On-device AI:** Ollama runs locally with Metal acceleration. The LLM never sends data over the network.

---

## iOS Companion App

A native Swift/SwiftUI app (~3,000 lines) sends context events to the backend:

- **Location updates** and geofencing
- **Device proximity** via Bluetooth/WiFi discovery
- **Activity recognition** (walking, driving, stationary)
- **Background sync** via iOS background tasks
- **WebSocket connection** for real-time server communication
- **Dashboard view** for on-the-go access

---

## Deployment

Life OS runs as a set of services on the Mac Mini:

| Service | Runtime | Port |
|---------|---------|------|
| **Life OS** | Python/FastAPI via launchd | 8080 |
| **NATS** | Docker container | 4222, 8222 |
| **Ollama** | brew services | 11434 |
| **Caddy** | brew services (reverse proxy) | 80 |

Access via Tailscale VPN from any device on the tailnet. Caddy proxies port 80 to 8080. WireGuard encryption means HTTP is fine for local traffic.

### Configuration

All behavior is driven by `config/settings.yaml`:

```yaml
data_dir: ./data
nats_url: nats://localhost:4222
web_port: 8080
embedding_model: all-MiniLM-L6-v2
timezone: America/Los_Angeles

ai:
  ollama_url: http://localhost:11434
  ollama_model: mistral
  use_cloud: false              # Optional Anthropic Claude
  cloud_api_key: sk-xxx

connectors:
  proton_mail:
    imap_host: 127.0.0.1
    imap_port: 1143
    username: you@proton.me
    password: bridge_password
    sync_interval: 30
  google:
    credentials_path: config/google_credentials.json
    sync_interval: 60
  # ... other connectors
```

### Monitoring

- `/health` — System health check with connector status
- `/admin` — Connector management UI
- `/admin/db` — Database browser for all 5 databases
- Application logs: `~/life-os/data/lifeos.log`

### Resilience

- Background loops auto-restart on crash (with restart counting)
- NATS reconnection loop handles event bus outages
- Database corruption detection with automatic rebuild
- Daily backup of `user_model.db`
- Connector health monitoring with status reporting
