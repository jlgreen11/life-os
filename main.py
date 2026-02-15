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
import json
import signal
import sys
import uuid
from datetime import datetime, timezone
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
from connectors.registry import CONNECTOR_REGISTRY, get_connector_class
from connectors.crypto import ConfigEncryptor
from services.onboarding.manager import OnboardingManager
from services.insight_engine.engine import InsightEngine
from services.contact_classifier import classify_contact_type
from services.email_classifier import is_marketing_email


class LifeOS:
    """The main application orchestrator."""

    def __init__(self, config_path: str = "config/settings.yaml",
                 db=None, event_bus=None, event_store=None,
                 user_model_store=None, config=None):
        """Initialize Life OS.

        For testing, dependencies can be injected directly via keyword args.
        For production, pass only config_path and all dependencies will be
        initialized from the config file.
        """
        # Allow config override for testing
        if config is not None:
            self.config = config
        else:
            self.config = self._load_config(config_path)

        # Used to signal all background tasks (prediction loop, etc.) to stop
        self.shutdown_event = asyncio.Event()

        # --- Core infrastructure ---
        # Initialization order matters: DB must be created first because almost
        # every other component receives it via constructor injection (the
        # dependency-injection pattern used throughout Life OS — each service
        # declares its dependencies as constructor args rather than importing
        # global singletons).
        data_dir = self.config.get("data_dir", "./data")

        # Allow dependency injection for testing
        self.db = db if db is not None else DatabaseManager(data_dir)
        self.event_store = event_store if event_store is not None else EventStore(self.db)
        self.event_bus = event_bus if event_bus is not None else EventBus(self.config.get("nats_url", "nats://localhost:4222"))
        self.user_model_store = user_model_store if user_model_store is not None else UserModelStore(self.db, event_bus=self.event_bus)
        self.vector_store = VectorStore(
            db_path=str(Path(data_dir) / "vectors"),
            model_name=self.config.get("embedding_model", "all-MiniLM-L6-v2"),
        )

        # --- Services ---
        # Each service receives only the dependencies it needs (db,
        # user_model_store, event_bus, config) — keeping coupling explicit and
        # making unit-testing straightforward via mock injection.
        # All services are wired to the event bus for data creation telemetry.
        self.signal_extractor = SignalExtractorPipeline(self.db, self.user_model_store)
        self.ai_engine = AIEngine(self.db, self.user_model_store, self.config.get("ai", {}),
                                   vector_store=self.vector_store)
        self.rules_engine = RulesEngine(self.db, event_bus=self.event_bus)
        self.feedback_collector = FeedbackCollector(self.db, self.user_model_store, event_bus=self.event_bus)
        self.prediction_engine = PredictionEngine(
            self.db, self.user_model_store
        )
        self.insight_engine = InsightEngine(self.db, self.user_model_store)
        # NotificationManager needs the event_bus so it can publish notification events
        self.notification_manager = NotificationManager(self.db, self.event_bus, self.config)
        self.task_manager = TaskManager(self.db, event_bus=self.event_bus)

        # --- Browser automation layer ---
        # Wraps Playwright and manages browser-based connectors separately
        self.browser_orchestrator = BrowserOrchestrator(self.event_bus, self.db, self.config)

        # Onboarding
        self.onboarding = OnboardingManager(self.db)

        # Connector management
        self.config_encryptor = ConfigEncryptor(data_dir)
        self.connector_map: dict[str, object] = {}  # connector_id -> BaseConnector
        self.connectors = []  # flat list for backward compat

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

        # Install default rules (after event bus is available for telemetry)
        await install_default_rules(self.db, event_bus=self.event_bus)

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
        asyncio.create_task(self._insight_loop())

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
            # so we always have a durable audit trail.  store_event() uses
            # INSERT OR IGNORE and returns False when the event ID already
            # exists (duplicate from connector re-sync or NATS redelivery).
            # In that case we skip all downstream processing to avoid
            # double-counting signals, creating duplicate tasks, etc.
            try:
                is_new = self.event_store.store_event(event)
                if not is_new:
                    return  # Duplicate event — already processed
            except Exception as e:
                print(f"Event store error: {e}")

            # Stage 1.1 — Marketing Classification: identify marketing and
            # automated emails early, BEFORE they generate noise through
            # episodic memory, signal extraction, task extraction, and
            # notifications.  Uses sender patterns, domain heuristics, subject
            # analysis, and body indicators (unsubscribe links, opt-out text).
            # Suppressed emails are still stored (Stage 1) and embedded
            # (Stage 5) for completeness, but skip the noisy middle stages.
            try:
                event_type = event.get("type", "")
                if event_type == "email.received":
                    payload = event.get("payload", {})
                    from_addr = payload.get("from_address", "")
                    if is_marketing_email(from_addr, payload):
                        event["_suppressed"] = True
                        self.event_store.add_tag(event["id"], "marketing")
                        self.event_store.add_tag(event["id"], "system:suppressed")
            except Exception as e:
                print(f"Marketing classification error: {e}")

            # Stage 1.15 — Contact Type Classification: classify the sender
            # as "person" or "business" so downstream stages (predictions,
            # notifications, relationship maintenance) can prioritize people.
            # Only runs for email events since messaging contacts (iMessage,
            # Signal) are classified during connector contact sync.
            try:
                event_type = event.get("type", "")
                if event_type in ("email.received", "email.sent"):
                    payload = event.get("payload", {})
                    addresses = []
                    if event_type == "email.received":
                        addresses = [payload.get("from_address", "")]
                    else:
                        addresses = payload.get("to_addresses", [])

                    for addr in addresses:
                        if not addr:
                            continue
                        self._ensure_contact_type(addr)
            except Exception as e:
                print(f"Contact classification error: {e}")

            # Stage 1.2 — Feedback Loop: process notification feedback events
            # (acted_on, dismissed) to close the learning loop. This enables
            # the system to learn which notifications are useful vs annoying.
            try:
                event_type = event.get("type", "")
                if event_type == "notification.acted_on":
                    # User acted on a notification - strong positive signal
                    notif_id = event.get("payload", {}).get("notification_id")
                    if notif_id:
                        await self.feedback_collector.process_notification_response(
                            notification_id=notif_id,
                            response_type="engaged",
                            response_time_seconds=0,  # We don't track timing yet
                        )
                elif event_type == "notification.dismissed":
                    # User dismissed a notification - negative signal
                    notif_id = event.get("payload", {}).get("notification_id")
                    if notif_id:
                        await self.feedback_collector.process_notification_response(
                            notification_id=notif_id,
                            response_type="dismissed",
                            response_time_seconds=0,
                        )
            except Exception as e:
                print(f"Feedback collector error: {e}")

            # Stage 1.5 — Episodic Memory: convert each event into a memory
            # episode for the user model's Layer 1 (Episodic) storage.
            # This provides the raw interaction history that feeds semantic
            # fact extraction and enables the system to answer "when did I
            # last talk to X" or "what happened in my meeting yesterday".
            # Skip for suppressed events — marketing emails shouldn't pollute
            # the user's episodic memory with promotional content.
            if not event.get("_suppressed"):
                try:
                    await self._create_episode(event)
                except Exception as e:
                    print(f"Episode creation error: {e}")

            # Stage 2 — Learn: the signal extractor passively analyses the
            # event to update the user model (patterns, preferences, etc.).
            # Skip for suppressed events — marketing emails shouldn't skew
            # topic profiles or communication cadence patterns.
            if not event.get("_suppressed"):
                try:
                    await self.signal_extractor.process_event(event)
                except Exception as e:
                    print(f"Signal extractor error: {e}")

            # Stage 3 — React: the rules engine evaluates deterministic,
            # user-defined rules and returns a list of actions to execute
            # (notify, tag, suppress, create_task, etc.).
            #
            # Suppress actions are executed first so they can set the
            # _suppressed flag before any notify actions fire.  This
            # prevents a race where a notify action from one rule runs
            # before the suppress action from another rule.
            try:
                actions = await self.rules_engine.evaluate(event)
                suppress_actions = [a for a in actions if a["type"] == "suppress"]
                other_actions = [a for a in actions if a["type"] != "suppress"]
                for action in suppress_actions + other_actions:
                    await self._execute_rule_action(action, event)
            except Exception as e:
                print(f"Rules engine error: {e}")

            # Stage 4 — Extract tasks: scan the event payload (e.g. email
            # body, chat message) for actionable items and create tasks.
            # Skip for suppressed events — marketing emails should never
            # generate tasks ("Buy now!" is not an action item).
            if not event.get("_suppressed"):
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

    def _infer_domain_from_event_type(self, event_type: str) -> str:
        """Infer notification domain from event type.

        Extracts the primary domain from an event type string by taking
        the first segment before a dot. This enables the feedback loop to
        work for all notification types, not just predictions.

        Examples:
            email.received → email
            calendar.event.created → calendar
            message.sent → message
            usermodel.prediction.generated → usermodel

        Args:
            event_type: The event type string (e.g., "email.received")

        Returns:
            The inferred domain (e.g., "email")
        """
        if not event_type or "." not in event_type:
            return "system"  # Fallback for malformed event types
        return event_type.split(".")[0]

    async def _execute_rule_action(self, action: dict, event: dict):
        """Execute an action triggered by the rules engine.

        Supported action types:
            notify      — create a user-visible notification (skipped if suppressed)
            tag         — attach a label to the event in event_tags
            suppress    — flag the event so it is hidden from notifications
            create_task — auto-create a task linked to the source event
        """
        action_type = action.get("type")

        if action_type == "notify":
            # Respect the suppress flag set by earlier suppress actions — if
            # the event was suppressed (by this or another rule), skip the
            # notification entirely.
            if event.get("_suppressed"):
                return

            # Infer domain from event type if not explicitly provided in metadata.
            # This enables feedback loop tracking for all notification types.
            domain = event.get("metadata", {}).get("domain")
            if not domain:
                domain = self._infer_domain_from_event_type(event.get("type", ""))

            await self.notification_manager.create_notification(
                title=f"Rule: {action.get('rule_name', 'Unknown')}",
                body=event.get("payload", {}).get("snippet", ""),
                priority=action.get("priority", "normal"),
                source_event_id=event.get("id"),
                domain=domain,
            )
        elif action_type == "tag":
            # Persist the tag in the event_tags table (separate from the
            # append-only events table to preserve its immutability).
            self.event_store.add_tag(
                event_id=event["id"],
                tag=action.get("value", ""),
                rule_id=action.get("rule_id"),
            )
        elif action_type == "suppress":
            # Two-part suppression:
            #   1. In-memory flag so downstream pipeline stages (task extraction,
            #      embedding, other rule actions) can check event["_suppressed"].
            #   2. Persistent tag so the web UI and search can filter out suppressed
            #      events and the audit trail is preserved.
            event["_suppressed"] = True
            self.event_store.add_tag(
                event_id=event["id"],
                tag="system:suppressed",
                rule_id=action.get("rule_id"),
            )
        elif action_type == "create_task":
            await self.task_manager.create_task(
                title=action.get("title", "Auto-created task"),
                source="rule",
                source_event_id=event.get("id"),
                priority=action.get("priority", "normal"),
            )

    async def _create_episode(self, event: dict):
        """Create an episodic memory from an event.

        Episodic memory (Layer 1 of the user model) stores individual
        interactions with full context — who was involved, what was discussed,
        what the user's mood was at the time. This provides the foundation
        for semantic fact extraction and enables queries like "when did I last
        talk to X" or "what happened in my meeting yesterday".

        Episodes are created for events that involve meaningful interaction:
        - Communication (emails, messages, calls)
        - Calendar events (meetings, appointments)
        - Financial transactions (spending decisions)
        - Task completions (work outcomes)
        - Location changes (context shifts)

        System-internal events (connector syncs, rule triggers, predictions)
        are NOT converted to episodes — they're metadata, not memories.
        """
        event_type = event.get("type", "")
        payload = event.get("payload", {})

        # Filter: Only create episodes for user-facing interactions.
        # System events (connector syncs, internal state changes) are
        # excluded because they're not part of the user's lived experience.
        episodic_event_types = {
            "email.received", "email.sent",
            "message.received", "message.sent",
            "call.received", "call.missed",
            "calendar.event.created", "calendar.event.updated",
            "finance.transaction.new",
            "task.created", "task.completed",
            "location.changed", "location.arrived", "location.departed",
            "context.location", "context.activity",
            "system.user.command",
        }

        if event_type not in episodic_event_types:
            return  # Skip non-episodic events

        # Determine interaction type — maps fine-grained event types to
        # coarse episodic categories for easier querying.
        if "email" in event_type or "message" in event_type or "call" in event_type:
            interaction_type = "communication"
        elif "calendar" in event_type:
            interaction_type = "calendar"
        elif "finance" in event_type:
            interaction_type = "financial"
        elif "task" in event_type:
            interaction_type = "task"
        elif "location" in event_type or "context" in event_type:
            interaction_type = "context"
        elif event_type == "system.user.command":
            interaction_type = "command"
        else:
            interaction_type = "other"

        # Extract contacts involved from the event payload.
        # For inbound communication, the contact is the sender.
        # For outbound communication, the contacts are the recipients.
        contacts_involved = []
        if payload.get("from_address"):
            contacts_involved.append(payload["from_address"])
        if payload.get("to_addresses"):
            contacts_involved.extend(payload["to_addresses"])

        # Extract topics from existing event metadata if the signal extractor
        # or AI engine has already processed this event. Topics help with
        # semantic search ("show me all episodes about the renovation").
        topics = payload.get("topics", [])

        # Generate content summary: a concise (< 200 char) description of
        # what happened, suitable for display in timeline UIs.
        content_summary = self._generate_episode_summary(event)

        # Store the full event payload as content_full for later retrieval
        # when the user wants complete details ("show me that email again").
        content_full = json.dumps(payload)

        # Retrieve current mood from the user model if available — this
        # provides emotional context that helps the system understand why
        # the user made certain decisions at certain times.
        inferred_mood = None
        try:
            mood_profile = self.user_model_store.get_signal_profile("mood_signals")
            if mood_profile and mood_profile.get("data"):
                # Use the most recent mood reading as the inferred state.
                recent_moods = mood_profile["data"].get("samples", [])
                if recent_moods:
                    latest_mood = recent_moods[-1]
                    inferred_mood = {
                        "energy_level": latest_mood.get("energy_level", 0.5),
                        "stress_level": latest_mood.get("stress_level", 0.3),
                        "emotional_valence": latest_mood.get("emotional_valence", 0.5),
                    }
        except Exception:
            pass  # Mood inference is optional — episode creation should not fail

        # Determine the active domain (work, personal, health, finance) from
        # event metadata if available. This helps segment episodes by life area.
        active_domain = event.get("metadata", {}).get("domain", "personal")

        # Build the episode dict matching the schema in user_model.db.
        episode = {
            "id": str(uuid.uuid4()),
            "timestamp": event.get("timestamp"),
            "event_id": event["id"],
            "location": payload.get("location"),
            "inferred_mood": inferred_mood,
            "active_domain": active_domain,
            "energy_level": inferred_mood.get("energy_level") if inferred_mood else None,
            "interaction_type": interaction_type,
            "content_summary": content_summary,
            "content_full": content_full,
            "contacts_involved": contacts_involved,
            "topics": topics,
            "entities": payload.get("entities", []),
            "outcome": None,  # Will be populated later if task is completed
            "user_satisfaction": None,  # Will be populated from explicit feedback
            "embedding_id": None,  # Could link to vector store entry if needed
        }

        # Persist to the episodes table via UserModelStore.
        self.user_model_store.store_episode(episode)

    def _generate_episode_summary(self, event: dict) -> str:
        """Generate a concise (< 200 char) summary for an episode.

        The summary is human-readable and suitable for timeline displays.
        It captures the essence of what happened without overwhelming detail.

        Examples:
        - "Email from john@company.com: Project update"
        - "Meeting: Q1 Planning Session"
        - "Task completed: Fix login bug"
        - "Transaction: $45.23 at Whole Foods"
        """
        event_type = event.get("type", "")
        payload = event.get("payload", {})

        # Communication events: show direction + sender/recipient + subject
        if event_type == "email.received":
            from_addr = payload.get("from_address", "unknown")
            subject = payload.get("subject", "No subject")
            return f"Email from {from_addr}: {subject}"[:200]
        elif event_type == "email.sent":
            to_addrs = payload.get("to_addresses", [])
            to_str = ", ".join(to_addrs[:2]) if to_addrs else "unknown"
            subject = payload.get("subject", "No subject")
            return f"Email to {to_str}: {subject}"[:200]
        elif event_type == "message.received":
            from_addr = payload.get("from_address", "unknown")
            snippet = payload.get("snippet", payload.get("body_plain", ""))[:50]
            return f"Message from {from_addr}: {snippet}"[:200]
        elif event_type == "message.sent":
            to_addrs = payload.get("to_addresses", [])
            to_str = ", ".join(to_addrs[:2]) if to_addrs else "unknown"
            snippet = payload.get("snippet", payload.get("body_plain", ""))[:50]
            return f"Message to {to_str}: {snippet}"[:200]
        elif "call" in event_type:
            from_addr = payload.get("from_address", "unknown")
            return f"Call from {from_addr}"[:200]

        # Calendar events: show title + time
        elif "calendar" in event_type:
            title = payload.get("title", "Untitled event")
            start_time = payload.get("start_time", "")
            return f"Meeting: {title} at {start_time}"[:200]

        # Tasks: show title + status
        elif "task" in event_type:
            title = payload.get("title", "Untitled task")
            status = "completed" if event_type == "task.completed" else "created"
            return f"Task {status}: {title}"[:200]

        # Financial: show amount + merchant
        elif "finance.transaction" in event_type:
            amount = payload.get("amount", 0)
            merchant = payload.get("merchant", "Unknown")
            return f"Transaction: ${amount:.2f} at {merchant}"[:200]

        # Location: show location name or coordinates
        elif "location" in event_type:
            location = payload.get("location", "Unknown location")
            action = "arrived at" if "arrived" in event_type else "departed from" if "departed" in event_type else "changed to"
            return f"Location {action} {location}"[:200]

        # User commands: show the command text
        elif event_type == "system.user.command":
            command = payload.get("command", "Unknown command")
            return f"Command: {command}"[:200]

        # Fallback: generic summary
        else:
            snippet = payload.get("snippet", payload.get("subject", payload.get("title", "")))
            return f"{event_type}: {snippet}"[:200]

    def _ensure_contact_type(self, address: str):
        """Ensure a contact has a contact_type classification.

        Looks up the contact by address via the identifier index. If the
        contact exists but has no contact_type, classifies it using the
        heuristic classifier and persists the result.

        This runs on every email event to gradually backfill contact types
        for contacts that were created before the classification feature
        was added. The DB write only happens when contact_type is NULL,
        so it's a one-time cost per contact.
        """
        try:
            with self.db.get_connection("entities") as conn:
                row = conn.execute(
                    """SELECT c.id, c.contact_type, c.is_priority, c.relationship,
                              c.phones, c.channels, c.name
                       FROM contact_identifiers ci
                       JOIN contacts c ON ci.contact_id = c.id
                       WHERE ci.identifier = ?
                       LIMIT 1""",
                    (address,),
                ).fetchone()

                if not row:
                    return  # No contact record for this address

                if row["contact_type"]:
                    return  # Already classified

                # Classify using available signals
                phones = json.loads(row["phones"]) if row["phones"] else []
                channels = json.loads(row["channels"]) if row["channels"] else {}
                contact_type = classify_contact_type(
                    email_address=address,
                    name=row["name"],
                    relationship=row["relationship"],
                    phones=phones,
                    channels=channels,
                    is_priority=bool(row["is_priority"]),
                )

                # Persist the classification
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "UPDATE contacts SET contact_type = ?, updated_at = ? WHERE id = ?",
                    (contact_type, now, row["id"]),
                )
        except Exception as e:
            print(f"Contact type classification error for {address}: {e}")

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
                # Generate new predictions based on current context and patterns
                predictions = await self.prediction_engine.generate_predictions({})
                for prediction in predictions:
                    await self.notification_manager.create_notification(
                        title=f"{prediction.prediction_type.title()}: {prediction.description[:80]}",
                        body=prediction.description,
                        priority="high" if prediction.prediction_type in ("conflict", "risk") else "normal",
                        source_event_id=prediction.id,
                        domain="prediction",
                    )

                # Auto-resolve stale prediction notifications to close the feedback loop.
                # Predictions that users ignore for 24+ hours are marked inaccurate so
                # the prediction engine can learn which types to suppress.
                resolved = await self.notification_manager.auto_resolve_stale_predictions(timeout_hours=24)
                if resolved > 0:
                    print(f"  Auto-resolved {resolved} stale prediction(s)")

                # Auto-resolve filtered predictions to prevent database bloat.
                # Predictions with was_surfaced=0 were filtered by confidence gates
                # or reaction prediction and never shown to the user. After 1 hour,
                # these are no longer relevant and should be cleaned up.
                filtered_resolved = self.notification_manager.auto_resolve_filtered_predictions(timeout_hours=1)
                if filtered_resolved > 0:
                    print(f"  Auto-resolved {filtered_resolved} filtered prediction(s)")

            except Exception as e:
                print(f"Prediction engine error: {e}")

            # 900 seconds = 15 minutes.  This interval balances freshness
            # against compute cost; predictions depend on aggregated patterns,
            # so sub-minute granularity is unnecessary.
            await asyncio.sleep(900)  # 15 minutes

    async def _insight_loop(self):
        """Run the insight engine every hour."""
        while not self.shutdown_event.is_set():
            try:
                insights = await self.insight_engine.generate_insights()
                if insights:
                    print(f"  InsightEngine: generated {len(insights)} new insights")
            except Exception as e:
                print(f"Insight engine error: {e}")
            await asyncio.sleep(3600)  # 1 hour

    async def _start_connectors(self):
        """Initialize and start all configured connectors.

        Config priority: DB config > YAML config > connector defaults.
        Starts connectors that have DB config with status 'active', plus
        any YAML-configured connectors not yet in the DB.
        """
        yaml_configs = self.config.get("connectors") or {}

        # Build set of connectors to start: YAML-configured + DB-enabled
        to_start: dict[str, dict] = {}

        # YAML-configured API connectors
        for cid in ("proton_mail", "signal", "caldav", "finance", "home_assistant", "google"):
            if cid in yaml_configs:
                to_start[cid] = yaml_configs[cid]

        # Check DB for previously-enabled connectors (admin UI configs).
        # Uses the `enabled` flag which persists across restarts — unlike
        # `status` which gets overwritten to 'inactive' during shutdown.
        with self.db.get_connection("state") as conn:
            rows = conn.execute(
                "SELECT connector_id, config FROM connector_state WHERE enabled = 1"
            ).fetchall()
            for row in rows:
                cid = row["connector_id"]
                if cid in CONNECTOR_REGISTRY and cid not in to_start:
                    import json
                    db_config = json.loads(row["config"]) if row["config"] else {}
                    if db_config:
                        to_start[cid] = db_config

        # Start API connectors via registry
        for cid, raw_config in to_start.items():
            typedef = CONNECTOR_REGISTRY.get(cid)
            if not typedef or typedef.category != "api":
                continue
            try:
                config = self._resolve_connector_config(cid, raw_config)
                cls = get_connector_class(cid)
                connector = cls(self.event_bus, self.db, config)
                await connector.start()
                self.connector_map[cid] = connector
                self.connectors.append(connector)
                print(f"       ✓ {connector.DISPLAY_NAME}")
            except Exception as e:
                print(f"       ✗ {typedef.display_name}: {e}")

        # Start browser automation layer (shared engine + browser connectors)
        if self.browser_orchestrator.is_enabled:
            await self.browser_orchestrator.start()
            await self.browser_orchestrator.start_connectors()
            for c in self.browser_orchestrator.connectors:
                self.connector_map[c.CONNECTOR_ID] = c
                self.connectors.append(c)

    # -------------------------------------------------------------------
    # Runtime connector management (used by admin API)
    # -------------------------------------------------------------------

    def _get_sensitive_fields(self, connector_id: str) -> set[str]:
        """Return set of sensitive field names for a connector type."""
        typedef = CONNECTOR_REGISTRY.get(connector_id)
        if not typedef:
            return set()
        return {f.name for f in typedef.config_fields if f.sensitive}

    def _resolve_connector_config(self, connector_id: str,
                                  override: dict | None = None) -> dict:
        """Merge and decrypt config: DB > override/YAML > defaults."""
        import json

        typedef = CONNECTOR_REGISTRY.get(connector_id)
        if not typedef:
            raise ValueError(f"Unknown connector: {connector_id}")

        # Start with field defaults from registry
        config = {}
        for f in typedef.config_fields:
            if f.default is not None:
                config[f.name] = f.default

        # Layer YAML config
        yaml_configs = self.config.get("connectors") or {}
        if connector_id in yaml_configs:
            config.update(yaml_configs[connector_id])

        # Layer DB config (highest priority)
        with self.db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT config FROM connector_state WHERE connector_id = ?",
                (connector_id,),
            ).fetchone()
            if row and row["config"]:
                db_config = json.loads(row["config"])
                if db_config:
                    config.update(db_config)

        # Apply override if provided
        if override:
            config.update(override)

        # Decrypt sensitive fields
        sensitive = self._get_sensitive_fields(connector_id)
        config = self.config_encryptor.decrypt_config(config, sensitive)

        return config

    def get_connector_status(self, connector_id: str) -> dict:
        """Get full status for a connector including DB state and runtime health."""
        import json

        result = {
            "connector_id": connector_id,
            "running": connector_id in self.connector_map,
            "status": "unconfigured",
            "last_sync": None,
            "error_count": 0,
            "last_error": None,
        }

        with self.db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT status, last_sync, error_count, last_error, config, updated_at "
                "FROM connector_state WHERE connector_id = ?",
                (connector_id,),
            ).fetchone()
            if row:
                result["status"] = row["status"]
                result["last_sync"] = row["last_sync"]
                result["error_count"] = row["error_count"] or 0
                result["last_error"] = row["last_error"]
                result["updated_at"] = row["updated_at"]
                db_config = json.loads(row["config"]) if row["config"] else {}
                result["has_config"] = bool(db_config)
            else:
                # Check if YAML has config
                yaml_configs = self.config.get("connectors") or {}
                result["has_config"] = connector_id in yaml_configs

        if not result["has_config"] and result["status"] == "inactive":
            result["status"] = "unconfigured"

        return result

    def get_connector_config(self, connector_id: str) -> dict:
        """Get the merged config for a connector, masked for API response."""
        config = self._resolve_connector_config(connector_id)
        sensitive = self._get_sensitive_fields(connector_id)
        return self.config_encryptor.mask_config(config, sensitive)

    def save_connector_config(self, connector_id: str, new_config: dict):
        """Validate, encrypt, and store connector config in the DB."""
        import json

        typedef = CONNECTOR_REGISTRY.get(connector_id)
        if not typedef:
            raise ValueError(f"Unknown connector: {connector_id}")

        sensitive = self._get_sensitive_fields(connector_id)

        # Preserve existing encrypted values for unchanged password fields
        existing = {}
        with self.db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT config FROM connector_state WHERE connector_id = ?",
                (connector_id,),
            ).fetchone()
            if row and row["config"]:
                existing = json.loads(row["config"])

        # If a sensitive field is "********", keep the existing encrypted value
        final = {}
        for key, value in new_config.items():
            if key in sensitive and value == "********" and key in existing:
                final[key] = existing[key]  # keep existing encrypted value
            else:
                final[key] = value

        # Encrypt sensitive fields
        encrypted = self.config_encryptor.encrypt_config(final, sensitive)

        # Store in DB
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO connector_state (connector_id, config, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(connector_id) DO UPDATE SET
                       config = ?, updated_at = ?""",
                (connector_id, json.dumps(encrypted), now,
                 json.dumps(encrypted), now),
            )

    async def enable_connector(self, connector_id: str) -> dict:
        """Instantiate, start, and register a connector at runtime.

        Persists status='active' in the DB so the connector auto-starts
        on the next service restart.
        """
        if connector_id in self.connector_map:
            return {"status": "already_running"}

        typedef = CONNECTOR_REGISTRY.get(connector_id)
        if not typedef:
            raise ValueError(f"Unknown connector: {connector_id}")

        if typedef.category == "browser":
            return {"status": "error",
                    "detail": "Browser connectors are managed via the browser orchestrator"}

        config = self._resolve_connector_config(connector_id)
        cls = get_connector_class(connector_id)
        connector = cls(self.event_bus, self.db, config)
        await connector.start()

        self.connector_map[connector_id] = connector
        self.connectors.append(connector)

        # Persist enabled flag so connector auto-starts on next boot.
        # The `enabled` column is never touched by _update_state(), so it
        # survives the shutdown race where stop() writes status='inactive'.
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection("state") as conn:
            conn.execute(
                "UPDATE connector_state SET enabled = 1, updated_at = ? WHERE connector_id = ?",
                (now, connector_id),
            )

        return {"status": "started"}

    async def disable_connector(self, connector_id: str) -> dict:
        """Stop and unregister a running connector.

        Persists status='inactive' so the connector stays off on restart.
        """
        connector = self.connector_map.pop(connector_id, None)
        if not connector:
            return {"status": "not_running"}

        await connector.stop()
        if connector in self.connectors:
            self.connectors.remove(connector)

        # Persist disabled flag so connector stays off on restart
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection("state") as conn:
            conn.execute(
                "UPDATE connector_state SET enabled = 0, updated_at = ? WHERE connector_id = ?",
                (now, connector_id),
            )

        return {"status": "stopped"}

    async def test_connector(self, connector_id: str,
                             config: dict | None = None) -> dict:
        """Create a temporary connector instance and test authentication."""
        typedef = CONNECTOR_REGISTRY.get(connector_id)
        if not typedef:
            raise ValueError(f"Unknown connector: {connector_id}")

        if typedef.category == "browser":
            return {"success": False,
                    "detail": "Browser connectors require the browser engine to test"}

        resolved = self._resolve_connector_config(connector_id, override=config)
        cls = get_connector_class(connector_id)
        tmp = cls(self.event_bus, self.db, resolved)

        try:
            success = await tmp.authenticate()
            return {"success": success,
                    "detail": "Authentication successful" if success else "Authentication failed"}
        except Exception as e:
            return {"success": False, "detail": str(e)}

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
