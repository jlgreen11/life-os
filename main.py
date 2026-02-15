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
from services.signal_extractor.extractor import SignalExtractorPipeline
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
        self.shutdown_event = asyncio.Event()

        # Core infrastructure
        self.db = DatabaseManager(self.config.get("data_dir", "./data"))
        self.event_store = EventStore(self.db)
        self.user_model_store = UserModelStore(self.db)
        self.vector_store = VectorStore(
            db_path=str(Path(self.config.get("data_dir", "./data")) / "vectors"),
            model_name=self.config.get("embedding_model", "all-MiniLM-L6-v2"),
        )
        self.event_bus = EventBus(self.config.get("nats_url", "nats://localhost:4222"))

        # Services
        self.signal_extractor = SignalExtractorPipeline(self.db, self.user_model_store)
        self.ai_engine = AIEngine(self.db, self.user_model_store, self.config.get("ai", {}))
        self.rules_engine = RulesEngine(self.db)
        self.feedback_collector = FeedbackCollector(self.db, self.user_model_store)
        self.prediction_engine = PredictionEngine(
            self.db, self.user_model_store, self.ai_engine
        )
        self.notification_manager = NotificationManager(self.db, self.event_bus, self.config)
        self.task_manager = TaskManager(self.db, self.ai_engine)

        # Browser automation layer
        self.browser_orchestrator = BrowserOrchestrator(self.event_bus, self.db, self.config)

        # Connectors (loaded dynamically based on config)
        self.connectors = []

    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
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
            print("       Running in degraded mode (no event bus).")

        # 4. Register core event handlers
        print("[4/7] Registering event handlers...")
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

        async def master_event_handler(event: dict):
            """Every event flows through this pipeline."""
            # Store the raw event
            try:
                self.event_store.store_event(event)
            except Exception as e:
                print(f"Event store error: {e}")

            # Run through signal extractor (passive learning)
            try:
                await self.signal_extractor.process_event(event)
            except Exception as e:
                print(f"Signal extractor error: {e}")

            # Run through rules engine (deterministic actions)
            try:
                actions = await self.rules_engine.evaluate(event)
                for action in actions:
                    await self._execute_rule_action(action, event)
            except Exception as e:
                print(f"Rules engine error: {e}")

            # Run through task manager (extract tasks from messages)
            try:
                await self.task_manager.process_event(event)
            except Exception as e:
                print(f"Task manager error: {e}")

            # Embed for vector search
            try:
                await self._embed_event(event)
            except Exception as e:
                print(f"Embedding error: {e}")

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
            pass  # TODO: implement event tagging
        elif action_type == "suppress":
            # Mark event as suppressed (don't notify)
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
                    if prediction.get("should_surface"):
                        await self.notification_manager.create_notification(
                            title=prediction.get("title", "Suggestion"),
                            body=prediction.get("description", ""),
                            priority=prediction.get("priority", "low"),
                        )
            except Exception as e:
                print(f"Prediction engine error: {e}")

            await asyncio.sleep(900)  # 15 minutes

    async def _start_connectors(self):
        """Initialize and start all configured connectors."""
        connector_configs = self.config.get("connectors", {})

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
        self.shutdown_event.set()

        for connector in self.connectors:
            try:
                await connector.stop()
            except Exception:
                pass

        if self.browser_orchestrator.is_enabled:
            await self.browser_orchestrator.stop()

        if self.event_bus.is_connected:
            await self.event_bus.disconnect()

        print("Goodbye.")


def create_app():
    """Create the FastAPI application (imported by web module)."""
    from web.app import create_web_app
    life_os = LifeOS()
    return create_web_app(life_os)


async def main():
    """Main entry point."""
    life_os = LifeOS()

    # Handle Ctrl+C
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(life_os.stop()))

    await life_os.start()

    # Run web server
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
