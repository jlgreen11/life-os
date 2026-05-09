# Life OS — Your Private Command Center

A local-first, AI-powered personal life management system that connects
everything in your digital life through a single private intelligence layer.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLIENT LAYER                       │
│        (PWA / CLI / Voice via Whisper)                │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              API GATEWAY (FastAPI)                    │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│            EVENT BUS (NATS JetStream)                │
└──────────────────────┬──────────────────────────────┘
                       │
  ┌────────┬───────────┼───────────┬──────────┐
  ▼        ▼           ▼           ▼          ▼
Signal   AI        Rules       Task      Notification
Extractor Engine   Engine     Manager    Manager
  │        │           │           │          │
  ▼        ▼           ▼           ▼          ▼
┌─────────────────────────────────────────────────────┐
│               USER MODEL STORE                       │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────┐  │
│  │ Episodic │ │ Semantic │ │Procedural│ │Predict-│  │
│  │ Memory   │ │ Memory   │ │ Memory  │ │  ive   │  │
│  └──────────┘ └──────────┘ └─────────┘ └────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────────┐  │
│  │ SQLite   │ │ LanceDB  │ │ File Store          │  │
│  └──────────┘ └──────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and configure
cp config/settings.example.yaml config/settings.yaml
# Edit settings.yaml with your credentials

# 2. Start everything
docker compose up -d

# 3. Open the web UI
open http://localhost:8080
```

## Build Order

1. **Foundation**: Event bus, database schemas, user model store
2. **First Connector**: Proton Mail via Bridge (IMAP)
3. **Signal Extractor**: Begin building the user model from email
4. **AI Engine**: Ollama + morning briefing
5. **Add Connectors**: Signal, CalDAV, finance
6. **Prediction Engine**: Proactive suggestions
7. **Web UI**: Briefing, inbox, command bar
8. **Refinement**: Feedback loop, rules engine, tuning

## Project Structure

```
life-os/
├── config/                  # Configuration files
├── connectors/              # Service connectors (one per external service)
│   ├── base/                # Base connector framework
│   ├── proton_mail/         # Proton Mail via Bridge
│   ├── signal_msg/          # Signal via signal-cli
│   ├── caldav/              # CalDAV calendars
│   ├── finance/             # Plaid/bank connectors
│   └── home_assistant/      # Smart home
├── models/                  # Data models and schemas
├── services/                # Core services
│   ├── event_bus/           # NATS pub/sub wrapper
│   ├── ai_engine/           # LLM orchestration
│   ├── signal_extractor/    # Passive signal collection
│   ├── prediction_engine/   # Forward-looking intelligence
│   ├── feedback_collector/  # Implicit/explicit feedback
│   ├── rules_engine/        # Deterministic automations
│   ├── task_manager/        # Task tracking
│   └── notification_manager/# Alert routing
├── storage/                 # Database and vector store management
├── web/                     # Web UI (FastAPI + templates)
├── scripts/                 # Utility scripts
├── docker-compose.yaml
└── requirements.txt
```
