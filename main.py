"""
Life OS — Main Application Entry Point

Boots the entire system:
    1. Initialize databases
    2. Connect to NATS event bus
    3. Start all enabled connectors
    4. Start the signal extractor pipeline
    5. Start the prediction engine
    6. Start the rules engine
    7. Start the feedback collector
    8. Launch the web API
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

import uvicorn
import yaml

from services.event_bus.bus import EventBus
from services.signal_extractor.pipeline import SignalExtractorPipeline
from services.prediction_engine.engine import PredictionEngine
from services.feedback_collector.collector import FeedbackCollector
from services.rules_engine.engine import RulesEngine, install_default_rules
from services.ai_engine.engine import AIEngine
from services.notification_manager.manager import NotificationManager
from services.task_manager.manager import TaskManager
from storage.database import DatabaseManager, EventStore, UserModelStore
from storage.vector_store import VectorStore
from connectors.browser.orchestrator import BrowserOrchestrator


class LifeOS:
    """The main application orchestrator."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        # Used to signal all background tasks (prediction loop, etc.) to stop
        self.shutdown_event = asyncio.Event()

        # --- Core infrastructure ---
        # Initialization order matters: DB must be created first because almost
        # every other component receives it via constructor injection (the
        # dependency-injection pattern used throughout Life OS — each service
        # declares its dependencies as constructor args rather than importing
        # global singletons).
        self.db = DatabaseManager(self.config.get("data_dir", "./data"))
        self.event_store = EventStore(self.db)
        self.user_model_store = UserModelStore(self.db)
        self.vector_store = VectorStore(
            db_path=str(Path(self.config.get("data_dir", "./data")) / "vectors"),
            model_name=self.config.get("embedding_model", "all-MiniLM-L6-v2"),
        )
        # EventBus wraps NATS; actual connection happens later in start()
        self.event_bus = EventBus(self.config.get("nats_url", "nats://localhost:4222"))

        # --- Services ---
        # Each service receives only the dependencies it needs (db,
        # user_model_store, event_bus, config) — keeping coupling explicit and
        # making unit-testing straightforward via mock injection.
        self.signal_extractor = SignalExtractorPipeline(self.db, self.user_model_store)
        self.ai_engine = AIEngine(self.db, self.user_model_store, self.config.get("ai", {}))
        self.rules_engine = RulesEngine(self.db)
        self.feedback_collector = FeedbackCollector(self.db, self.user_model_store)
        self.prediction_engine = PredictionEngine(
            self.db, self.user_model_store
        )
        # NotificationManager needs the event_bus so it can publish notification events
        self.notification_manager = NotificationManager(self.db, self.event_bus, self.config)
        self.task_manager = TaskManager(self.db)

        # --- Browser automation layer ---
        # Wraps Playwright and manages browser-based connectors separately
        self.browser_orchestrator = BrowserOrchestrator(self.event_bus, self.db, self.config)

        # Connectors are loaded dynamically in _start_connectors() based on
        # which keys are present in the config; empty list until then.
        self.connectors = []

    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            # Fallback: if the YAML config file is missing, return a complete
            # set of sensible defaults so the app can still boot (useful for
            # first-run / development).  See _default_config() for the values.
            print(f"Config not found at {path}, using defaults.")
            return self._default_config()

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _default_config(self) -> dict:
        return {
            "data_dir": "./data",
            "nats_url": "nats://localhost:4222",
            "web_port": 8080,
            "web_host": "0.0.0.0",
            "embedding_model": "all-MiniLM-L6-v2",
            "ai": {
                "ollama_url": "http://localhost:11434",
                "ollama_model": "mistral",
                "use_cloud": False,
            },
            "connectors": {},
        }

    async def start(self):
        """Boot the entire system."""
        print("=" * 60)
        print("  Life OS — Starting Up")
        print("=" * 60)

        # 1. Initialize databases
        print("[1/7] Initializing databases...")
        self.db.initialize_all()
        install_default_rules(self.db)

        # 2. Initialize vector store
        print("[2/7] Initializing vector store...")
        self.vector_store.initialize()

        # 3. Connect to NATS
        print("[3/7] Connecting to event bus...")
        try:
            await self.event_bus.connect()
            print("       Connected to NATS.")
        except Exception as e:
            print(f"       NATS connection failed: {e}")
            # Degraded mode: the app continues to run without the event bus.
            # In this mode, connectors cannot publish events, the signal
            # extractor / rules engine won't fire, and real-time notifications
            # are unavailable.  The web UI, DB, and prediction engine still
            # work — so users can still browse history and manage tasks.
            print("       Running in degraded mode (no event bus).")

        # 4. Register core event handlers
        print("[4/7] Registering event handlers...")
        # Only register handlers when the bus is actually connected;
        # in degraded mode this block is skipped entirely.
        if self.event_bus.is_connected:
            await self._register_event_handlers()

        # 5. Start connectors
        print("[5/7] Starting connectors...")
        await self._start_connectors()

        # 6. Start prediction engine background loop
        print("[6/7] Starting prediction engine...")
        asyncio.create_task(self._prediction_loop())

        # 7. Launch web server
        print("[7/7] Starting web server...")
        print()
        print(f"  → Web UI:  http://localhost:{self.config.get('web_port', 8080)}")
        print(f"  → API:     http://localhost:{self.config.get('web_port', 8080)}/api")
        print(f"  → Health:  http://localhost:{self.config.get('web_port', 8080)}/health")
        print()
        print("  Life OS is running. Press Ctrl+C to stop.")
        print("=" * 60)

    async def _register_event_handlers(self):
        """Wire up the core event processing pipeline."""

        # master_event_handler is the core event-processing pipeline.  ALL
        # events from every connector flow through this single handler, which
        # fans them out to each processing stage in order.  Each stage is
        # wrapped in its own try/except so that a failure in one stage never
        # blocks the remaining stages from executing.
        async def master_event_handler(event: dict):
            """Every event flows through this pipeline."""

            # Stage 1 — Persist: store the raw event in the relational DB
            # so we always have a durable audit trail.
            try:
                self.event_store.store_event(event)
            except Exception as e:
                print(f"Event store error: {e}")

            # Stage 2 — Learn: the signal extractor passively analyses the
            # event to update the user model (patterns, preferences, etc.).
            try:
                await self.signal_extractor.process_event(event)
            except Exception as e:
                print(f"Signal extractor error: {e}")

            # Stage 3 — React: the rules engine evaluates deterministic,
            # user-defined rules and returns a list of actions to execute
            # (notify, tag, suppress, create_task, etc.).
            try:
                actions = await self.rules_engine.evaluate(event)
                for action in actions:
                    await self._execute_rule_action(action, event)
            except Exception as e:
                print(f"Rules engine error: {e}")

            # Stage 4 — Extract tasks: scan the event payload (e.g. email
            # body, chat message) for actionable items and create tasks.
            try:
                await self.task_manager.process_event(event)
            except Exception as e:
                print(f"Task manager error: {e}")

            # Stage 5 — Embed: generate a vector embedding of the event
            # content so it can be retrieved via semantic search later.
            try:
                await self._embed_event(event)
            except Exception as e:
                print(f"Embedding error: {e}")

        # Subscribe with a wildcard so every subject published on the bus
        # is routed to master_event_handler.
        await self.event_bus.subscribe_all(master_event_handler)

    async def _execute_rule_action(self, action: dict, event: dict):
        """Execute an action triggered by the rules engine."""
        action_type = action.get("type")

        if action_type == "notify":
            await self.notification_manager.create_notification(
                title=f"Rule: {action.get('rule_name', 'Unknown')}",
                body=event.get("payload", {}).get("snippet", ""),
                priority=action.get("priority", "normal"),
                source_event_id=event.get("id"),
                domain=event.get("metadata", {}).get("domain"),
            )
        elif action_type == "tag":
            # Store tag on the event
            pass  # TODO [FLAGGED]: implement event tagging
        elif action_type == "suppress":
            # Mark event as suppressed so downstream stages (especially
            # NotificationManager) know not to surface it to the user.
            # Currently a no-op; the suppress flag should be persisted on the
            # event record once event-tagging (above) is implemented.
            pass
        elif action_type == "create_task":
            await self.task_manager.create_task(
                title=action.get("title", "Auto-created task"),
                source="rule",
                source_event_id=event.get("id"),
                priority=action.get("priority", "normal"),
            )

    async def _embed_event(self, event: dict):
        """Embed event content for vector search."""
        payload = event.get("payload", {})

        # Text-extraction strategy: pull from multiple payload fields to
        # build the richest possible text representation of the event.
        # Priority order:
        #   1. "subject"    — email subjects, message titles (always short)
        #   2. "body_plain" — plain-text body (preferred over HTML)
        #      "body"       — HTML/rich body, used as a fallback when no
        #                     plain-text variant exists
        #   3. "title"      — calendar event titles, task titles, etc.
        # Bodies are truncated to 2 000 chars to keep embeddings focused on
        # the most relevant content and to limit memory / compute cost.
        text_parts = []

        if payload.get("subject"):
            text_parts.append(payload["subject"])
        if payload.get("body_plain"):
            text_parts.append(payload["body_plain"][:2000])
        elif payload.get("body"):
            text_parts.append(payload["body"][:2000])
        if payload.get("title"):
            text_parts.append(payload["title"])

        text = " ".join(text_parts).strip()
        # Skip events with very little textual content (< 20 chars).  Short
        # strings produce low-quality embeddings that pollute search results.
        if len(text) < 20:
            return

        self.vector_store.add_document(
            doc_id=event["id"],
            text=text,
            metadata={
                "type": event.get("type"),
                "source": event.get("source"),
                "timestamp": event.get("timestamp"),
            },
        )

    async def _prediction_loop(self):
        """Run the prediction engine every 15 minutes."""
        while not self.shutdown_event.is_set():
            try:
                predictions = await self.prediction_engine.run_predictions()
                for prediction in predictions:
                    # Only surface predictions the engine deems worth
                    # interrupting the user for (high-confidence / high-value).
                    if prediction.get("should_surface"):
                        await self.notification_manager.create_notification(
                            title=prediction.get("title", "Suggestion"),
                            body=prediction.get("description", ""),
                            priority=prediction.get("priority", "low"),
                        )
            except Exception as e:
                print(f"Prediction engine error: {e}")

            # 900 seconds = 15 minutes.  This interval balances freshness
            # against compute cost; predictions depend on aggregated patterns,
            # so sub-minute granularity is unnecessary.
            await asyncio.sleep(900)  # 15 minutes

    async def _start_connectors(self):
        """Initialize and start all configured connectors."""
        connector_configs = self.config.get("connectors") or {}

        # Dynamic / lazy import pattern: connector modules are imported
        # inside their respective ``if`` blocks rather than at the top of
        # the file.  This avoids importing heavy dependencies (e.g.
        # Playwright, protonmail-bridge, signalc) when the connector is
        # not enabled in the config, which speeds up startup and lets the
        # app run on machines where those optional libraries are not
        # installed.

        if "proton_mail" in connector_configs:
            from connectors.proton_mail.connector import ProtonMailConnector
            connector = ProtonMailConnector(
                self.event_bus, self.db, connector_configs["proton_mail"]
            )
            self.connectors.append(connector)

        if "signal" in connector_configs:
            from connectors.signal_msg.connector import SignalConnector
            connector = SignalConnector(
                self.event_bus, self.db, connector_configs["signal"]
            )
            self.connectors.append(connector)

        if "caldav" in connector_configs:
            from connectors.caldav.connector import CalDAVConnector
            connector = CalDAVConnector(
                self.event_bus, self.db, connector_configs["caldav"]
            )
            self.connectors.append(connector)

        if "finance" in connector_configs:
            from connectors.finance.connector import FinanceConnector
            connector = FinanceConnector(
                self.event_bus, self.db, connector_configs["finance"]
            )
            self.connectors.append(connector)

        if "home_assistant" in connector_configs:
            from connectors.home_assistant.connector import HomeAssistantConnector
            connector = HomeAssistantConnector(
                self.event_bus, self.db, connector_configs["home_assistant"]
            )
            self.connectors.append(connector)

        # Attempt to start each instantiated connector.  Failures are logged
        # but do not prevent other connectors from starting.
        for connector in self.connectors:
            try:
                await connector.start()
                print(f"       ✓ {connector.DISPLAY_NAME}")
            except Exception as e:
                print(f"       ✗ {connector.DISPLAY_NAME}: {e}")

        # Start browser automation layer (shared engine + browser connectors)
        if self.browser_orchestrator.is_enabled:
            await self.browser_orchestrator.start()
            await self.browser_orchestrator.start_connectors()
            self.connectors.extend(self.browser_orchestrator.connectors)

    async def stop(self):
        """Graceful shutdown."""
        print("\nShutting down Life OS...")
        # Signal all background loops (_prediction_loop, etc.) to exit
        self.shutdown_event.set()

        # Shutdown order is the reverse of startup:
        # 1. Stop connectors first so no new events are produced.
        for connector in self.connectors:
            try:
                await connector.stop()
            except Exception:
                pass

        # 2. Tear down the browser automation layer (closes Playwright
        #    browser contexts and the shared engine).
        if self.browser_orchestrator.is_enabled:
            await self.browser_orchestrator.stop()

        # 3. Disconnect from NATS last — this ensures that any in-flight
        #    events published during connector shutdown can still be delivered
        #    before the bus goes away.
        if self.event_bus.is_connected:
            await self.event_bus.disconnect()

        print("Goodbye.")


# ---------------------------------------------------------------------------
# Dual entry-point pattern
# ---------------------------------------------------------------------------
# Life OS can be started in two ways:
#
#   1. ASGI factory  — ``create_app()`` returns a FastAPI instance that an
#      external ASGI server (e.g. ``uvicorn main:create_app --factory``) can
#      serve.  This is the recommended approach for production deployments.
#
#   2. Direct run    — ``python main.py`` calls ``main()`` which boots the
#      full system (DB, connectors, event bus, prediction loop) AND starts
#      an embedded Uvicorn server.  Convenient for local development.
# ---------------------------------------------------------------------------


def create_app():
    """Create the FastAPI application (imported by web module).

    ASGI factory entry point — used when an external server manages the
    process (e.g. ``uvicorn main:create_app --factory``).
    """
    from web.app import create_web_app
    life_os = LifeOS()
    return create_web_app(life_os)


async def main():
    """Main entry point for direct execution (``python main.py``)."""
    life_os = LifeOS()

    # Handle Ctrl+C and SIGTERM so the app shuts down gracefully
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(life_os.stop()))

    # Boot all subsystems (DB, connectors, event bus, prediction loop, etc.)
    await life_os.start()

    # Run the embedded web server — blocks until shutdown
    from web.app import create_web_app
    app = create_web_app(life_os)

    config = uvicorn.Config(
        app,
        host=life_os.config.get("web_host", "0.0.0.0"),
        port=life_os.config.get("web_port", 8080),
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
