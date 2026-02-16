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
from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer
from services.routine_detector.detector import RoutineDetector
from services.workflow_detector.detector import WorkflowDetector
from services.insight_engine.source_weights import SourceWeightManager
from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker
from services.task_completion_detector.detector import TaskCompletionDetector


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
        # AIEngine receives the vector_store for semantic search capabilities.
        # The vector store enables intelligent search queries like "What did Mike
        # say about the Denver project?" via embedding-based similarity matching.
        self.ai_engine = AIEngine(self.db, self.user_model_store, self.config.get("ai", {}),
                                  vector_store=self.vector_store)
        self.rules_engine = RulesEngine(self.db, event_bus=self.event_bus)
        self.feedback_collector = FeedbackCollector(self.db, self.user_model_store, event_bus=self.event_bus)
        self.prediction_engine = PredictionEngine(
            self.db, self.user_model_store
        )
        self.source_weight_manager = SourceWeightManager(self.db)
        self.insight_engine = InsightEngine(
            self.db, self.user_model_store,
            source_weight_manager=self.source_weight_manager,
        )
        # SemanticFactInferrer derives high-level facts from signal profiles
        self.semantic_fact_inferrer = SemanticFactInferrer(self.user_model_store)
        # RoutineDetector analyzes episodic memory to find recurring behavioral patterns
        self.routine_detector = RoutineDetector(self.db, self.user_model_store)
        # WorkflowDetector analyzes event sequences to find goal-driven multi-step processes
        self.workflow_detector = WorkflowDetector(self.db, self.user_model_store)
        # BehavioralAccuracyTracker infers prediction accuracy from user behavior
        self.behavioral_tracker = BehavioralAccuracyTracker(self.db)
        # NotificationManager needs the event_bus so it can publish notification events
        self.notification_manager = NotificationManager(self.db, self.event_bus, self.config)
        # TaskManager needs the ai_engine to extract action items from events
        self.task_manager = TaskManager(self.db, event_bus=self.event_bus, ai_engine=self.ai_engine)
        # TaskCompletionDetector infers task completion from behavioral signals
        # (emails sent, inactivity, etc.) to enable workflow detection
        self.task_completion_detector = TaskCompletionDetector(
            self.db, self.task_manager, self.event_bus
        )

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

        # 1.5 — Seed default source weights (no-op if already populated)
        self.source_weight_manager.seed_defaults()

        # 1.6 — Backfill episode classification if needed
        # This ensures routine and workflow detection have the granular
        # interaction types they need. Without this, all old episodes would
        # remain classified as generic "communication" and the detectors
        # would have no signal to work with.
        await self._backfill_episode_classification_if_needed()

        # 1.7 — Backfill task completion if needed
        # This marks historical tasks as completed based on behavioral signals
        # (sent emails/messages that reference the task). Without this, workflow
        # detection cannot operate because it needs task.completed events to
        # identify multi-step task-completion patterns. The backfill is critical
        # for bootstrapping Layer 3 procedural memory on systems with historical
        # task data but no completion events.
        await self._backfill_task_completion_if_needed()

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

        # 6. Start background loops (prediction, insight, semantic inference)
        print("[6/7] Starting background services...")
        asyncio.create_task(self._prediction_loop())
        asyncio.create_task(self._insight_loop())
        asyncio.create_task(self._semantic_inference_loop())
        asyncio.create_task(self._routine_detection_loop())
        asyncio.create_task(self._behavioral_accuracy_loop())
        asyncio.create_task(self._task_completion_loop())

        # 7. Launch web server
        print("[7/7] Starting web server...")
        print()
        print(f"  → Web UI:  http://localhost:{self.config.get('web_port', 8080)}")
        print(f"  → API:     http://localhost:{self.config.get('web_port', 8080)}/api")
        print(f"  → Health:  http://localhost:{self.config.get('web_port', 8080)}/health")
        print()
        print("  Life OS is running. Press Ctrl+C to stop.")
        print("=" * 60)

    async def _backfill_episode_classification_if_needed(self):
        """Reclassify old episodes with granular interaction types if needed.

        Checks for episodes with the old generic "communication" interaction_type
        and reclassifies them using the granular classification logic. This is
        critical for enabling routine and workflow detection, which rely on seeing
        diverse interaction types (email_received, email_sent, meeting_scheduled, etc.)
        rather than everything collapsing into one generic type.

        This migration runs automatically on startup, making the system self-healing
        after deployments that add new classification logic.
        """
        try:
            # Count episodes with the old generic classification
            with self.db.get_connection("user_model") as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM episodes
                    WHERE interaction_type = 'communication'
                """)
                stale_count = cursor.fetchone()[0]

            # If there are no stale episodes, skip the backfill
            if stale_count == 0:
                return

            print(f"       → Backfilling {stale_count} episode classifications...")

            # Run the backfill: fetch each stale episode, get its original event,
            # reclassify it, and update the database
            with self.db.get_connection("user_model") as user_model_conn, \
                 self.db.get_connection("events") as events_conn:

                # Fetch all stale episodes
                cursor = user_model_conn.execute("""
                    SELECT id, event_id FROM episodes
                    WHERE interaction_type = 'communication'
                """)
                stale_episodes = cursor.fetchall()

                reclassified = 0
                for episode_id, event_id in stale_episodes:
                    # Fetch the original event to get its type and payload
                    event_cursor = events_conn.execute("""
                        SELECT type, payload FROM events
                        WHERE id = ?
                    """, (event_id,))
                    event_row = event_cursor.fetchone()

                    if not event_row:
                        continue  # Event was deleted; skip this episode

                    event_type = event_row[0]
                    payload = json.loads(event_row[1])

                    # Reclassify using the current granular logic
                    new_interaction_type = self._classify_interaction_type(event_type, payload)

                    # Update the episode
                    user_model_conn.execute("""
                        UPDATE episodes
                        SET interaction_type = ?
                        WHERE id = ?
                    """, (new_interaction_type, episode_id))

                    reclassified += 1

            print(f"       → Reclassified {reclassified} episodes")

        except Exception as e:
            # Backfill errors should not crash startup
            print(f"       ⚠ Episode classification backfill failed: {e}")

    async def _backfill_task_completion_if_needed(self):
        """Mark historical tasks as completed based on behavioral signals.

        Searches for pending tasks that should already be marked complete
        based on sent emails/messages that reference the task + contain
        completion keywords (done, finished, sent, etc.).

        This is critical for bootstrapping workflow detection (Layer 3
        procedural memory) on systems with historical task data. Without
        task.completed events, the workflow detector cannot identify
        multi-step task-completion patterns like:
          - Receive request → research → draft → send confirmation
          - Get assignment → complete work → submit deliverable → follow up

        The backfill runs automatically on startup and is idempotent (safe
        to run multiple times). It only marks tasks complete if there's
        strong evidence (keyword overlap >= 2.0 + completion keywords).

        This migration is self-healing: if new tasks are extracted from
        historical emails via backfill_task_extraction.py, this will
        automatically detect which ones were already completed before the
        extraction happened.
        """
        try:
            # Count pending tasks that might need completion detection
            with self.db.get_connection("state") as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM tasks
                    WHERE status = 'pending'
                """)
                pending_count = cursor.fetchone()[0]

            # If there are no pending tasks, skip the backfill
            if pending_count == 0:
                return

            # Only run backfill if we have a significant number of pending tasks
            # (10+). This avoids unnecessary work on fresh systems and focuses on
            # systems where historical data needs cleanup.
            if pending_count < 10:
                return

            print(f"       → Backfilling task completion for {pending_count} pending tasks...")

            # Import the backfill script inline to avoid circular dependencies
            # and keep the main.py imports clean
            import re
            from datetime import datetime, timezone

            # Completion signal keywords to look for in sent email/message content
            completion_keywords = {
                'done', 'finished', 'completed', 'sent', 'submitted',
                'delivered', 'shipped', 'resolved', 'closed', 'fixed',
                'merged', 'deployed', 'published', 'launched', 'ready'
            }

            # Stop words to filter out when extracting task keywords
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'with', 'from', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'between', 'under', 'again', 'further',
                'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                'all', 'each', 'other', 'some', 'such', 'only', 'own', 'same',
                'than', 'too', 'very', 'just', 'should', 'would', 'could', 'will'
            }

            # Get all pending tasks
            with self.db.get_connection("state") as state_conn:
                cursor = state_conn.execute("""
                    SELECT id, title, description, created_at, source
                    FROM tasks
                    WHERE status = 'pending'
                    ORDER BY created_at ASC
                """)
                pending_tasks = [dict(row) for row in cursor.fetchall()]

            completed_count = 0

            # Check each task for completion signals
            for task in pending_tasks:
                task_id = task['id']
                task_title = task['title'].lower() if task['title'] else ''
                task_desc = task['description'].lower() if task['description'] else ''
                created_at = task['created_at']

                # Extract meaningful keywords from task title for matching
                title_words = {
                    word for word in re.findall(r'\w+', task_title)
                    if len(word) > 3 and word not in stop_words
                }
                title_stems = {word[:4] for word in title_words if len(word) >= 4}

                if not title_words and not title_stems:
                    # No meaningful keywords to match against
                    continue

                # Search for sent emails/messages after task creation
                with self.db.get_connection("events") as events_conn:
                    cursor = events_conn.execute("""
                        SELECT id, type, payload, timestamp
                        FROM events
                        WHERE type IN ('email.sent', 'message.sent')
                          AND timestamp >= ?
                        ORDER BY timestamp ASC
                        LIMIT 100
                    """, (created_at,))
                    sent_events = cursor.fetchall()

                # Check each sent event for task reference + completion keywords
                task_completed = False
                for event_row in sent_events:
                    event_id, event_type, payload_json, timestamp = event_row

                    try:
                        payload = json.loads(payload_json)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    # Extract text content from the payload
                    text_parts = []
                    if payload.get('subject'):
                        text_parts.append(payload['subject'])
                    if payload.get('body_plain'):
                        text_parts.append(payload['body_plain'])
                    if payload.get('snippet'):
                        text_parts.append(payload['snippet'])

                    text_content = ' '.join(text_parts).lower()

                    if not text_content:
                        continue

                    # Count keyword matches
                    text_words = set(re.findall(r'\w+', text_content))
                    text_stems = {word[:4] for word in text_words if len(word) >= 4}

                    exact_matches = len(title_words & text_words)
                    stem_matches = len(title_stems & text_stems)
                    keyword_overlap = exact_matches + (stem_matches * 0.5)

                    # Check for completion signal keywords
                    has_completion_keyword = any(
                        keyword in text_content for keyword in completion_keywords
                    )

                    # If we have both keyword overlap AND completion signals, mark complete
                    if keyword_overlap >= 2.0 and has_completion_keyword:
                        # Update task status to completed
                        with self.db.get_connection("state") as state_conn:
                            state_conn.execute("""
                                UPDATE tasks
                                SET status = 'completed',
                                    completed_at = ?
                                WHERE id = ?
                            """, (datetime.now(timezone.utc).isoformat(), task_id))

                        # Publish task.completed event for workflow detection
                        event = {
                            'id': f"{task_id}-completion",
                            'type': 'task.completed',
                            'source': 'backfill.task_completion',
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'priority': 'normal',
                            'payload': {
                                'task_id': task_id,
                                'title': task['title'],
                                'source': task.get('source', 'unknown'),
                                'backfill': True
                            },
                            'metadata': {
                                'backfill_run': True,
                                'detection_method': 'behavioral_signal'
                            }
                        }

                        # Store the event
                        with self.db.get_connection("events") as events_conn:
                            events_conn.execute("""
                                INSERT OR IGNORE INTO events
                                (id, type, source, timestamp, priority, payload, metadata)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                event['id'],
                                event['type'],
                                event['source'],
                                event['timestamp'],
                                event['priority'],
                                json.dumps(event['payload']),
                                json.dumps(event['metadata'])
                            ))

                        completed_count += 1
                        task_completed = True
                        break  # Found completion signal, no need to check more events

            if completed_count > 0:
                print(f"       → Marked {completed_count} tasks as completed")

        except Exception as e:
            # Backfill errors should not crash startup
            print(f"       ⚠ Task completion backfill failed: {e}")

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

            # Stage 1.3 — Source Weight Tracking: classify the event into a
            # source_key and increment its interaction counter.  This builds
            # the data the AI drift algorithm needs to learn which sources
            # the user cares about.
            try:
                source_key = self.source_weight_manager.classify_event(event)
                self.source_weight_manager.record_interaction(source_key)
            except Exception as e:
                print(f"Source weight tracking error: {e}")

            # Stage 1.5 — Episodic Memory: convert each event into a memory
            # episode for the user model's Layer 1 (Episodic) storage.
            # This provides the raw interaction history that feeds semantic
            # fact extraction and enables the system to answer "when did I
            # last talk to X" or "what happened in my meeting yesterday".
            try:
                await self._create_episode(event)
            except Exception as e:
                print(f"Episode creation error: {e}")

            # Stage 2 — Learn: the signal extractor passively analyses the
            # event to update the user model (patterns, preferences, etc.).
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
        # granular action categories that enable routine detection.
        # The routine detector needs specific action types (e.g., "email_received"
        # vs "email_sent") to identify recurring behavioral patterns. Using
        # coarse categories like "communication" provides no signal for pattern
        # detection because all emails/messages collapse into the same type.
        interaction_type = self._classify_interaction_type(event_type, payload)

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

    def _classify_interaction_type(self, event_type: str, payload: dict) -> str:
        """Classify event into a granular interaction type for routine detection.

        The routine detector relies on seeing specific, granular action types to
        identify recurring behavioral patterns. For example:
        - "email_received" vs "email_sent" reveals inbox-checking vs correspondence routines
        - "meeting_attended" vs "calendar_reviewed" distinguishes participation from planning
        - "task_created" vs "task_completed" shows work initiation vs completion patterns

        If all events collapse into "communication", the detector has no signal to work with.

        Args:
            event_type: The fine-grained event type (e.g., "email.received")
            payload: Event payload containing additional context

        Returns:
            Granular interaction type suitable for routine detection (15+ distinct types)
        """
        # Email interactions — distinguish inbound (inbox checking) from outbound (correspondence)
        if event_type == "email.received":
            return "email_received"
        elif event_type == "email.sent":
            return "email_sent"

        # Messaging interactions — distinguish chat/IM from email
        elif event_type == "message.received":
            return "message_received"
        elif event_type == "message.sent":
            return "message_sent"

        # Call interactions — distinguish answered, missed, initiated
        elif event_type == "call.received":
            return "call_answered"
        elif event_type == "call.missed":
            return "call_missed"

        # Calendar interactions — distinguish meeting participation from calendar management
        elif event_type == "calendar.event.created":
            # If the event has participants, it's a meeting; otherwise it's a personal event
            if payload.get("participants") or payload.get("attendees"):
                return "meeting_scheduled"
            else:
                return "calendar_blocked"
        elif event_type == "calendar.event.updated":
            return "calendar_reviewed"

        # Financial interactions — distinguish spending from income/transfers
        elif event_type == "finance.transaction.new":
            amount = payload.get("amount", 0)
            if amount < 0:
                return "spending"
            else:
                return "income"

        # Task interactions — distinguish creation (work planning) from completion (execution)
        elif event_type == "task.created":
            return "task_created"
        elif event_type == "task.completed":
            return "task_completed"

        # Location interactions — distinguish arrivals (entering contexts) from departures
        elif event_type == "location.arrived":
            return "location_arrived"
        elif event_type == "location.departed":
            return "location_departed"
        elif event_type == "location.changed":
            return "location_changed"

        # Context interactions — device/activity state changes
        elif event_type == "context.location":
            return "context_location"
        elif event_type == "context.activity":
            return "context_activity"

        # User commands — explicit user interactions with the system
        elif event_type == "system.user.command":
            return "user_command"

        # Fallback for any unmapped event types — should be rare
        else:
            # Try to extract a meaningful type from the event_type string
            # e.g., "system.rule.triggered" -> "rule_triggered"
            if "." in event_type:
                return event_type.split(".")[-1]
            return "other"

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
                filtered_resolved = await asyncio.to_thread(
                    self.notification_manager.auto_resolve_filtered_predictions, 1
                )
                if filtered_resolved > 0:
                    print(f"  Auto-resolved {filtered_resolved} filtered prediction(s)")

            except Exception as e:
                print(f"Prediction engine error: {e}")

            # 900 seconds = 15 minutes.  This interval balances freshness
            # against compute cost; predictions depend on aggregated patterns,
            # so sub-minute granularity is unnecessary.
            await asyncio.sleep(900)  # 15 minutes

    async def _insight_loop(self):
        """Run the insight engine every hour.

        Also runs the source weight bulk drift recalculation once per
        cycle so that AI drift adjusts based on aggregate engagement
        patterns, not just individual feedback events.
        """
        while not self.shutdown_event.is_set():
            try:
                insights = await self.insight_engine.generate_insights()
                if insights:
                    print(f"  InsightEngine: generated {len(insights)} new insights")
            except Exception as e:
                print(f"Insight engine error: {e}")

            # Recalculate AI drift based on engagement ratios
            try:
                await asyncio.to_thread(self.source_weight_manager.bulk_recalculate_drift)
            except Exception as e:
                print(f"Source weight drift recalc error: {e}")

            await asyncio.sleep(3600)  # 1 hour

    async def _semantic_inference_loop(self):
        """Run semantic fact inference every 6 hours.

        The semantic fact inferrer analyzes accumulated signal profiles
        (linguistic, relationship, topic, cadence, mood) and derives high-level
        facts about the user's preferences, expertise, and values.

        This bridges Layer 0/1 (raw signals and episodes) to Layer 2 (semantic
        memory). It runs less frequently than prediction generation because
        semantic facts are long-term patterns that don't change minute-to-minute.

        Examples of derived facts:
          - "User prefers casual communication" (from linguistic formality < 0.3)
          - "User has expertise in Python" (from topic frequency + depth)
          - "Contact X is high priority" (from response time consistently < 1hr)
          - "User values work-life boundaries" (from cadence showing no work emails after 6pm)

        Interval: 6 hours (21600 seconds)
          - Balances freshness against compute cost
          - Allows sufficient new data to accumulate between runs
          - Semantic patterns are stable, so frequent re-inference is unnecessary
        """
        while not self.shutdown_event.is_set():
            try:
                # Run inference across all signal profiles to extract semantic facts.
                # Offloaded to a thread to avoid blocking the async event loop —
                # run_all_inference is fully synchronous (SQLite queries + computation).
                await asyncio.to_thread(self.semantic_fact_inferrer.run_all_inference)
                print("  SemanticFactInferrer: completed inference cycle")
            except Exception as e:
                print(f"Semantic fact inferrer error: {e}")

            # 21600 seconds = 6 hours. Semantic facts are long-term patterns,
            # so this interval provides a good balance between staying current
            # and avoiding unnecessary compute on stable patterns.
            await asyncio.sleep(21600)  # 6 hours

    async def _routine_detection_loop(self):
        """Run routine detection with adaptive retry intervals.

        The routine detector analyzes episodic memory to identify recurring
        behavioral patterns (Layer 3: Procedural Memory). Examples:
          - Morning routine: check email → review calendar → coffee
          - Arrive at work: open IDE → check Slack → review tasks
          - End of day: inbox zero → update task list → plan tomorrow

        Routines are detected by finding sequences of actions that:
          1. Occur at similar times (temporal routines)
          2. Follow location changes (location-based routines)
          3. Follow specific events (event-triggered routines)

        These patterns power prediction features like:
          - "You usually check your calendar now"
          - "Time for your end-of-day review"
          - "You typically update your task list after meetings"

        Adaptive retry interval:
          - If 0 routines detected: retry in 1 hour (connectors may still be syncing)
          - If 1-2 routines detected: retry in 3 hours (partial data, check again soon)
          - If 3+ routines detected: retry in 12 hours (stable patterns, normal cadence)

        This adaptive approach ensures Layer 3 is populated quickly after startup
        (when connectors are still syncing and episodes are sparse) while settling
        into an efficient 12-hour rhythm once patterns are established.

        Startup behavior:
          - Runs immediately after 60-second delay on startup to check for
            existing routines in episodic history
          - Retries frequently (1-3 hours) until patterns stabilize
          - Settles into 12-hour rhythm once 3+ routines are detected
        """
        # Wait 60 seconds on startup to allow episodic memory to populate from
        # any ongoing connector syncs, then run immediately to check for routines
        # from existing episodes.
        await asyncio.sleep(60)

        while not self.shutdown_event.is_set():
            try:
                # Detect routines from last 30 days of episodic memory.
                # Offloaded to threads — these are synchronous SQLite + computation.
                routines = await asyncio.to_thread(self.routine_detector.detect_routines, 30)

                # Store detected routines to database
                stored_count = await asyncio.to_thread(self.routine_detector.store_routines, routines)

                print(f"  RoutineDetector: detected {len(routines)} routines, stored {stored_count}")

                # Detect workflows from last 30 days of event sequences
                workflows = await asyncio.to_thread(self.workflow_detector.detect_workflows, 30)

                # Store detected workflows to database
                workflow_stored = await asyncio.to_thread(self.workflow_detector.store_workflows, workflows)

                print(f"  WorkflowDetector: detected {len(workflows)} workflows, stored {workflow_stored}")

                # Adaptive retry interval based on detection success:
                # - 0 patterns: retry in 1 hour (cold start, connectors may be syncing)
                # - 1-2 patterns: retry in 3 hours (partial data, check again soon)
                # - 3+ patterns: retry in 12 hours (stable patterns, normal cadence)
                total_patterns = len(routines) + len(workflows)
                if total_patterns == 0:
                    retry_seconds = 3600  # 1 hour
                    retry_desc = "1 hour (no patterns yet, will retry soon)"
                elif total_patterns <= 2:
                    retry_seconds = 10800  # 3 hours
                    retry_desc = "3 hours (partial patterns, will check again soon)"
                else:
                    retry_seconds = 43200  # 12 hours
                    retry_desc = "12 hours (stable patterns detected)"

                print(f"  Next detection cycle in {retry_desc}")

            except Exception as e:
                print(f"Routine/workflow detector error: {e}")
                # On error, retry in 1 hour to avoid tight error loops
                retry_seconds = 3600
                print(f"  Next detection cycle in 1 hour (after error)")

            await asyncio.sleep(retry_seconds)

    async def _behavioral_accuracy_loop(self):
        """Run behavioral accuracy inference every 15 minutes.

        The behavioral accuracy tracker infers prediction accuracy from user
        behavior patterns, closing the feedback loop without requiring explicit
        user interaction with notifications. This dramatically accelerates the
        learning loop for new systems.

        Examples of behavioral signals:
          - Prediction: "Reply to Alice about dinner plans"
            Behavior: User sends a message to Alice within 6 hours
            → Mark prediction as ACCURATE

          - Prediction: "Calendar conflict: Team sync overlaps with dentist"
            Behavior: User reschedules one of the events within 24 hours
            → Mark prediction as ACCURATE

          - Prediction: "Follow up with Bob about the project"
            Behavior: 48 hours pass, no message sent to Bob
            → Mark prediction as INACCURATE

        This allows the system to bootstrap its learning from observed behavior
        instead of waiting for explicit feedback, dramatically accelerating the
        calibration loop.

        Interval: 15 minutes (900 seconds)
          - Same cadence as prediction engine for responsive feedback
          - Predictions can be validated within 1-2 cycles of user action
          - Balances responsiveness against compute cost
        """
        while not self.shutdown_event.is_set():
            try:
                # Run inference cycle over all unresolved predictions
                stats = await self.behavioral_tracker.run_inference_cycle()

                if stats['marked_accurate'] + stats['marked_inaccurate'] > 0:
                    print(f"  BehavioralAccuracyTracker: inferred accuracy for "
                          f"{stats['marked_accurate']} accurate, "
                          f"{stats['marked_inaccurate']} inaccurate predictions")
            except Exception as e:
                print(f"Behavioral accuracy tracker error: {e}")

            # 900 seconds = 15 minutes. Same interval as prediction engine to
            # ensure predictions are validated shortly after user behavior occurs.
            await asyncio.sleep(900)  # 15 minutes

    async def _task_completion_loop(self):
        """Run task completion detection every 30 minutes.

        The task completion detector infers when tasks have been completed based
        on behavioral signals, enabling workflow detection (Layer 3 procedural
        memory). Without this loop, tasks remain in "pending" state forever because
        users often complete work without explicitly marking tasks done in the UI.

        Detection strategies:
          1. Activity-based: User sends email/message referencing the task with
             completion keywords ("done", "finished", "sent", "shipped")
          2. Inactivity-based: Task has no related activity for 7+ days (likely
             completed or abandoned)
          3. Stale task cleanup: Tasks older than 30 days are auto-archived

        Publishing task.completed events enables the workflow detector to learn
        multi-step task completion patterns:
          - Email received → research → execute → confirm completion
          - Task assigned → clarify → work → report back
          - Meeting scheduled → prepare → attend → follow up

        Interval: 30 minutes (1800 seconds)
          - Slower than prediction/behavioral loops (those need quick response)
          - Task completion is inherently delayed (hours/days between create/complete)
          - 30min ensures timely workflow detection without excessive compute
        """
        while not self.shutdown_event.is_set():
            try:
                # Run detection cycle over all pending tasks
                completed_count = await self.task_completion_detector.detect_completions()

                if completed_count > 0:
                    print(f"  TaskCompletionDetector: auto-completed {completed_count} tasks "
                          f"from behavioral signals")
            except Exception as e:
                print(f"Task completion detector error: {e}")

            # 1800 seconds = 30 minutes. Tasks aren't typically completed within
            # minutes (unlike prediction validation), so we can run less frequently
            # while still maintaining responsive workflow detection.
            await asyncio.sleep(1800)  # 30 minutes

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
