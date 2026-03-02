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
import logging
import os
import signal
import sys
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

logger = logging.getLogger(__name__)

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


def _prediction_priority(prediction) -> str:
    """Determine notification priority for a prediction.

    Conflict and risk predictions always get 'high' priority regardless of
    confidence.  For other types, predictions in the DEFAULT or AUTONOMOUS
    confidence gate (>= 0.6) also get 'high' so they reach users in minimal
    notification mode.  Lower-confidence predictions stay 'normal'.
    """
    if prediction.prediction_type in ("conflict", "risk"):
        return "high"
    if prediction.confidence >= 0.6:
        return "high"
    return "normal"


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
        self.background_tasks: dict[str, asyncio.Task] = {}  # Track background loops for exception monitoring

        # --- Core infrastructure ---
        # Initialization order matters: DB must be created first because almost
        # every other component receives it via constructor injection (the
        # dependency-injection pattern used throughout Life OS — each service
        # declares its dependencies as constructor args rather than importing
        # global singletons).
        data_dir = self.config.get("data_dir", "./data")
        self.user_tz = self.config.get("timezone", "America/Los_Angeles")

        # Allow dependency injection for testing
        self.db = db if db is not None else DatabaseManager(data_dir)
        self.event_store = event_store if event_store is not None else EventStore(self.db)
        nats_url = os.environ.get("NATS_URL") or self.config.get("nats_url", "nats://localhost:4222")
        self.event_bus = event_bus if event_bus is not None else EventBus(nats_url)
        self.user_model_store = user_model_store if user_model_store is not None else UserModelStore(
            self.db,
            event_bus=self.event_bus,
            event_store=self.event_store
        )
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
            self.db, self.user_model_store, timezone=self.user_tz
        )
        self.source_weight_manager = SourceWeightManager(self.db)
        self.insight_engine = InsightEngine(
            self.db, self.user_model_store,
            source_weight_manager=self.source_weight_manager,
            timezone=self.user_tz,
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
        self.notification_manager = NotificationManager(self.db, self.event_bus, self.config, timezone=self.user_tz)
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
            logger.warning("Config not found at %s, using defaults.", path)
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

    def _start_background_task(self, name: str, coro):
        """
        Start and monitor a background task with exception tracking.

        Background tasks (prediction loop, insight loop, etc.) run indefinitely
        in the background via asyncio.create_task(). If these tasks crash due to
        an unhandled exception, asyncio silently swallows the error and the task
        dies — leaving the system in a degraded state with no indication of failure.

        This helper:
        1. Creates the task and stores it by name for tracking
        2. Adds a done callback that logs exceptions if the task crashes
        3. Enables monitoring via /health endpoint or runtime inspection

        Without this, background loop failures are invisible. The system continues
        running but critical functionality (predictions, insights, routine detection)
        silently stops working.

        Args:
            name: Human-readable task name for logging and monitoring
            coro: The async coroutine to run as a background task

        Example:
            self._start_background_task("prediction_loop", self._prediction_loop())
        """
        task = asyncio.create_task(coro)
        self.background_tasks[name] = task

        def handle_task_exception(task: asyncio.Task):
            """Log exception if background task crashes."""
            try:
                # Accessing result() will re-raise any exception that occurred
                task.result()
            except asyncio.CancelledError:
                # Normal shutdown, not an error
                pass
            except Exception as e:
                # Background task crashed — log the full traceback so we can diagnose
                logger.critical(
                    "Background task '%s' crashed: %s — system is running in degraded mode",
                    name, e, exc_info=True,
                )

        task.add_done_callback(handle_task_exception)

    async def start(self):
        """Boot the entire system."""
        logger.info("=" * 60)
        logger.info("  Life OS — Starting Up")
        logger.info("=" * 60)

        # 1. Initialize databases
        logger.info("[1/7] Initializing databases...")
        self.db.initialize_all()

        # 1.5 — Seed default source weights (no-op if already populated)
        self.source_weight_manager.seed_defaults()

        # 1.54 — Detect and rebuild user_model.db when deep B-tree corruption
        # affects the episodes table (e.g. content_full overflow pages).  This
        # is a more comprehensive repair than the signal_profiles-only fix below:
        # it rebuilds the entire user_model.db from all readable columns, so
        # ALL signal profile backfills and semantic inference recover correctly.
        # Must run BEFORE _repair_signal_profiles_if_corrupted so the latter
        # operates on a healthy (or freshly rebuilt) database.
        await self._rebuild_user_model_db_if_corrupted()

        # 1.55 — Detect and repair signal_profiles table corruption
        # SQLite B-tree pages for the signal_profiles.data column can become
        # corrupted (manifests as "database disk image is malformed" on SELECT).
        # This repair runs before any backfills so that the existing backfill
        # guards (which check whether profiles are empty) correctly see an empty
        # table and trigger full repopulation.
        await self._repair_signal_profiles_if_corrupted()

        # 1.58 — Backfill episodes from events.db if user_model.db is empty
        # After a user_model.db rebuild (step 1.54 above), the episodes table is empty
        # but events.db still has the full event history.  Episodes are the foundation
        # of the cognitive pipeline: routine detection, semantic fact inference, and
        # prediction accuracy all depend on episodes existing.  This must run BEFORE
        # episode classification (1.6) since classification operates on existing episodes.
        await self._backfill_episodes_from_events_if_needed()

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

        # 1.5. Backfill communication templates (Layer 3 procedural memory)
        # Template extraction was added in PR #130 but only processes new events
        # going forward. This backfill populates writing-style templates from all
        # historical communication events so the system immediately has rich
        # communication patterns for every contact.
        await self._backfill_communication_templates_if_needed()

        # 1.8. Backfill relationship signal profile if empty after migration
        # The relationship backfill script (scripts/backfill_relationship_profile.py)
        # was added in PR #313 but had no auto-trigger in main.py.  After Migration
        # 0→1 wipes signal_profiles, the connector must re-sync to refill it — but if
        # the Google connector is stale, the table stays empty forever.  This trigger
        # replays all historical email events through RelationshipExtractor on startup
        # so the profile is always populated regardless of connector health.
        await self._backfill_relationship_profile_if_needed()

        # 1.9. Backfill temporal signal profile if empty after migration
        # Same problem: backfill_temporal_profile.py exists but was never auto-triggered.
        # The temporal profile drives time-of-day patterns, energy peak detection, and
        # preparation-needs predictions. Without it, predictions based on calendar events
        # or task scheduling patterns never fire.
        await self._backfill_temporal_profile_if_needed()

        # 1.10. Clean marketing contacts from relationships profile
        # The relationship extractor was filtering marketing emails at extraction time
        # (PR #143) but this left 469+ marketing contacts from historical data that
        # were tracked before the filter existed. This cleanup removes them to enable
        # relationship maintenance predictions (which currently don't generate because
        # 57% of tracked "contacts" are marketing automations, not humans).
        await self._clean_relationship_profile_if_needed()

        # 1.11. Backfill topic signal profile if empty after migration
        # The topic profile drives the semantic inferrer's expertise/interest fact
        # generation (infer_from_topic_profile() produces expertise_<topic> and
        # interest_<topic> semantic facts).  After Migration 0→1 wipes signal_profiles,
        # the topic profile stays empty and no new expertise facts are generated
        # until the connector re-syncs.  This trigger replays all historical email
        # events through TopicExtractor to immediately re-populate the profile.
        await self._backfill_topic_profile_if_needed()

        # 1.12. Backfill linguistic signal profile if empty after migration
        # The linguistic profile drives the semantic inferrer's communication style
        # fact generation (infer_from_linguistic_profile() produces facts about
        # formality, directness, enthusiasm, and emoji usage).  These facts feed
        # the communication template system and tone-matching in draft replies.
        # Without this backfill the profile stays empty after a DB reset even though
        # historical outbound emails are available in events.db.
        await self._backfill_linguistic_profile_if_needed()

        # 1.13. Backfill cadence signal profile if empty after migration/rebuild
        # The cadence profile tracks response times (per-contact and per-channel) and
        # activity-window heatmaps.  It drives response-time priority contact detection
        # in the prediction engine and peak/quiet hours enforcement.  Without this
        # backfill, these features are broken after a DB rebuild until enough new
        # live events accumulate (typically weeks of usage).
        await self._backfill_cadence_profile_if_needed()

        # 1.14. Backfill mood_signals profile if empty after migration/rebuild
        # The mood_signals profile powers compute_current_mood() for the dashboard
        # mood widget and provides energy-level data for episode creation.  After a
        # DB rebuild, the mood widget shows no data and all episodes have NULL
        # energy_level until enough new events arrive.  This replays historical
        # events through the MoodInferenceEngine to immediately re-populate.
        await self._backfill_mood_signals_profile_if_needed()

        # 1.15. Backfill spatial profile if empty after migration/rebuild
        # The spatial profile tracks place-based behavior patterns (visit frequency,
        # duration, dominant domain per location).  It drives the semantic inferrer's
        # infer_from_spatial_profile() which produces facts like primary_work_location
        # and frequent_location_{place}.  Without this backfill, the spatial inference
        # path is permanently blocked (requires >= 10 samples) after a DB rebuild.
        await self._backfill_spatial_profile_if_needed()

        # 1.16. Backfill decision profile if empty after migration/rebuild
        # The decision profile tracks decision-making patterns (speed by domain,
        # delegation comfort, risk tolerance).  It drives the semantic inferrer's
        # infer_from_decision_profile() which produces facts like decision_speed_{domain}
        # and risk_tolerance_{domain}.  Without this backfill, the decision inference
        # path is permanently blocked (requires >= 20 samples) after a DB rebuild.
        await self._backfill_decision_profile_if_needed()

        # 2. Initialize vector store
        logger.info("[2/7] Initializing vector store...")
        self.vector_store.initialize()

        # 3. Connect to NATS
        logger.info("[3/7] Connecting to event bus...")
        try:
            await self.event_bus.connect()
            logger.info("       Connected to NATS.")
        except Exception as e:
            logger.error("       NATS connection failed: %s", e)
            # Degraded mode: the app continues to run without the event bus.
            # In this mode, connectors cannot publish events, the signal
            # extractor / rules engine won't fire, and real-time notifications
            # are unavailable.  The web UI, DB, and prediction engine still
            # work — so users can still browse history and manage tasks.
            logger.warning("       Running in degraded mode (no event bus).")

        # Install default rules (after event bus is available for telemetry)
        await install_default_rules(self.db, event_bus=self.event_bus)

        # 4. Register core event handlers
        logger.info("[4/7] Registering event handlers...")
        # Only register handlers when the bus is actually connected;
        # in degraded mode this block is skipped entirely.
        if self.event_bus.is_connected:
            await self._register_event_handlers()

        # 5. Start connectors
        logger.info("[5/7] Starting connectors...")
        await self._start_connectors()

        # 6. Start background loops (prediction, insight, semantic inference)
        logger.info("[6/7] Starting background services...")
        self._start_background_task("prediction_loop", self._prediction_loop())
        self._start_background_task("insight_loop", self._insight_loop())
        self._start_background_task("semantic_inference_loop", self._semantic_inference_loop())
        self._start_background_task("routine_detection_loop", self._routine_detection_loop())
        self._start_background_task("behavioral_accuracy_loop", self._behavioral_accuracy_loop())
        self._start_background_task("task_completion_loop", self._task_completion_loop())
        self._start_background_task("digest_delivery_loop", self._digest_delivery_loop())

        # 7. Launch web server
        logger.info("[7/7] Starting web server...")
        port = self.config.get('web_port', 8080)
        logger.info("  → Web UI:  http://localhost:%s", port)
        logger.info("  → API:     http://localhost:%s/api", port)
        logger.info("  → Health:  http://localhost:%s/health", port)
        logger.info("  Life OS is running. Press Ctrl+C to stop.")
        logger.info("=" * 60)

    async def _backfill_episodes_from_events_if_needed(self):
        """Create episodic memory from events.db when user_model.db has no episodes.

        Episodes are the foundation of the entire cognitive pipeline:
        - Routine detection queries episodes to find recurring patterns
        - Semantic fact inference uses episodes to derive expertise, interests, etc.
        - Prediction accuracy tracking needs episodes for calibration
        - The dashboard timeline and episodic memory views are empty without them

        After a user_model.db rebuild (e.g., due to corruption repair in step 1.54),
        the episodes table is wiped but events.db retains the full event history.
        This method detects the empty-episodes condition and backfills from events.db
        using the same logic as ``scripts/backfill_episodes_from_events.py``.

        The backfill is idempotent: it checks for existing episodes per event_id
        and uses INSERT OR IGNORE, so running it multiple times is safe.
        """
        try:
            # Check if episodes already exist — if so, nothing to do
            with self.db.get_connection("user_model") as conn:
                episode_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

            if episode_count > 0:
                return

            # Check if there are episodic events in events.db to backfill from
            from scripts.backfill_episodes_from_events import EPISODIC_EVENT_TYPES

            placeholders = ", ".join("?" for _ in EPISODIC_EVENT_TYPES)
            with self.db.get_connection("events") as conn:
                event_count = conn.execute(
                    f"SELECT COUNT(*) FROM events WHERE type IN ({placeholders})",
                    list(EPISODIC_EVENT_TYPES),
                ).fetchone()[0]

            if event_count == 0:
                return

            logger.info("       → Backfilling episodes from %d events...", event_count)

            # Run in a thread so the async event loop isn't blocked during the
            # O(n_events) SQLite iteration that can take several seconds on large datasets.
            def _run_backfill():
                from scripts.backfill_episodes_from_events import backfill_episodes

                return backfill_episodes(self.db)

            stats = await asyncio.to_thread(_run_backfill)

            logger.info(
                "       ✓ Episodes backfilled: %d created, %d skipped, %d errors",
                stats["episodes_created"],
                stats["episodes_skipped_existing"],
                stats["errors"],
            )

        except Exception as e:
            # Fail-open: backfill errors must never crash the startup sequence.
            logger.warning("       ⚠ Episode backfill from events failed (non-fatal): %s", e)

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

            logger.info("       → Backfilling %d episode classifications...", stale_count)

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

            logger.info("       → Reclassified %d episodes", reclassified)

        except Exception as e:
            # Backfill errors should not crash startup
            logger.warning("       ⚠ Episode classification backfill failed: %s", e)

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

            logger.info("       → Backfilling task completion for %d pending tasks...", pending_count)

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
                logger.info("       → Marked %d tasks as completed", completed_count)

        except Exception as e:
            # Backfill errors should not crash startup
            logger.warning("       ⚠ Task completion backfill failed: %s", e)

    async def _rebuild_user_model_db_if_corrupted(self):
        """Detect and rebuild user_model.db when deep B-tree corruption affects episodes.

        The ``episodes.content_full`` column stores large email/message bodies as
        TEXT blobs.  When these blobs span many SQLite B-tree overflow pages, a
        single unclean shutdown or disk I/O error can corrupt those pages, making
        any column read that touches the same page-chain fail with:
            sqlite3.DatabaseError: database disk image is malformed

        The prior ``_repair_signal_profiles_if_corrupted`` method only fixes the
        ``signal_profiles`` table.  But if the *episodes* table is also corrupted
        (which happens when ``content_full`` shares overflow pages with neighbouring
        columns like ``contacts_involved``, ``topics``, ``entities``, and
        ``created_at``), the backfill methods that read episodes will also fail,
        leaving signal profiles permanently empty.

        Recovery strategy:
          1. Probe: attempt ``SELECT content_full FROM episodes LIMIT 1``.
             If this succeeds, the DB is healthy and we return immediately.
          2. Dump: read all readable episode columns (skip the corrupted large-blob
             ones: content_full, contacts_involved, topics, entities, created_at).
             Also dump semantic_facts, routines, predictions, insights.
          3. Rebuild: create a fresh user_model.db in a temp directory using
             DatabaseManager.initialize_all(), which runs the full migration stack
             and produces a schema-correct database.
          4. Restore: insert the recovered data into the fresh DB.
          5. Swap: atomically rename the temp DB to user_model.db, archiving the
             corrupted file as user_model.db.corrupted.
          6. Log a clear WARNING so the user knows a rebuild occurred.

        Columns preserved from episodes (all behavioral intelligence is retained):
          id, timestamp, event_id, location, inferred_mood, active_domain,
          energy_level, interaction_type, content_summary, outcome,
          user_satisfaction, embedding_id

        Columns lost (were already unreadable due to corruption):
          content_full, contacts_involved, topics, entities, created_at
          (content_full is the root-cause column; the others share its overflow pages)

        After the rebuild:
          - signal_profiles is empty → backfill guards repopulate on startup
          - episodes are fully accessible → routine/workflow detection works
          - semantic inference runs without the "database disk image is malformed" error

        This method is idempotent: if the DB is healthy it returns after a single
        cheap probe query and touches nothing.

        Example::

            await life_os._rebuild_user_model_db_if_corrupted()
            # → "user_model.db is healthy" if OK
            # → "Rebuilt corrupted user_model.db — recovered N episodes" if fixed
        """
        import shutil
        import tempfile

        # ---------------------------------------------------------------
        # Step 1: Probe for corruption.
        # Reading episodes.content_full is the canary: it spans the same
        # overflow-page chains as contacts_involved, topics, entities.
        # A successful read means the DB is healthy.
        # ---------------------------------------------------------------
        try:
            with self.db.get_connection("user_model") as conn:
                try:
                    conn.execute("SELECT content_full FROM episodes LIMIT 1").fetchone()
                    # Also check signal_profiles data (the other common corruption point)
                    conn.execute("SELECT SUM(LENGTH(data)) FROM signal_profiles").fetchone()
                    logger.debug("       ✓ user_model.db is healthy — no rebuild needed")
                    return
                except Exception as probe_err:
                    logger.warning(
                        "       ⚠ user_model.db corruption detected (%s) — rebuilding from readable columns...",
                        probe_err,
                    )
        except Exception:
            # Can't even connect — let initialize_all() deal with it
            return

        # ---------------------------------------------------------------
        # Step 2: Dump all recoverable data from the corrupted DB.
        # We read columns individually so that one corrupted column
        # does not block the recovery of all other columns.
        # ---------------------------------------------------------------
        # These episode columns are always readable even when content_full is not,
        # because they use inline record storage (not overflow pages).
        SAFE_EPISODE_COLS = (
            "id, timestamp, event_id, location, inferred_mood, active_domain, "
            "energy_level, interaction_type, content_summary, outcome, "
            "user_satisfaction, embedding_id"
        )

        recovered_episodes = []
        recovered_tables: dict[str, list] = {
            "semantic_facts": [],
            "routines": [],
            "predictions": [],
            "insights": [],
        }

        try:
            with self.db.get_connection("user_model") as src:
                # Episodes: read only the safe columns
                try:
                    rows = src.execute(
                        f"SELECT {SAFE_EPISODE_COLS} FROM episodes"
                    ).fetchall()
                    recovered_episodes = [tuple(r) for r in rows]
                    logger.info(
                        "       ✓ Recovered %d episodes (safe columns only)", len(rows)
                    )
                except Exception as e:
                    logger.warning("       ⚠ Could not recover episodes: %s", e)

                # Other tables: read everything (these are small and corruption-free)
                for table in recovered_tables:
                    try:
                        rows = src.execute(f"SELECT * FROM {table}").fetchall()
                        recovered_tables[table] = [tuple(r) for r in rows]
                        logger.info(
                            "       ✓ Recovered %d rows from %s", len(rows), table
                        )
                    except Exception as e:
                        logger.warning("       ⚠ Could not recover %s: %s", table, e)

        except Exception as dump_err:
            logger.error("       ✗ Failed to dump recoverable data: %s", dump_err)
            return

        # ---------------------------------------------------------------
        # Step 3: Build a fresh user_model.db in a temp directory.
        # Using DatabaseManager.initialize_all() ensures the full migration
        # stack runs and the schema matches what the rest of the code expects.
        # ---------------------------------------------------------------
        db_path = self.db.data_dir / "user_model.db"
        temp_dir = None
        try:
            # Create a sibling temp file so the eventual rename is atomic
            # (same filesystem, no cross-device move required).
            temp_dir = tempfile.mkdtemp(dir=str(self.db.data_dir), prefix=".rebuild_")
            from storage.manager import DatabaseManager as _DBManager

            new_db = _DBManager(temp_dir)
            new_db.initialize_all()  # Creates proper schema with all tables

            # ---------------------------------------------------------------
            # Step 4: Insert recovered data into the fresh DB.
            # ---------------------------------------------------------------
            with new_db.get_connection("user_model") as dst:
                # Episodes: insert with safe columns only; other columns get defaults
                if recovered_episodes:
                    safe_col_count = len(SAFE_EPISODE_COLS.split(","))
                    placeholders = ", ".join(["?"] * safe_col_count)
                    dst.executemany(
                        f"INSERT OR IGNORE INTO episodes ({SAFE_EPISODE_COLS}) "
                        f"VALUES ({placeholders})",
                        recovered_episodes,
                    )

                # Other tables: insert all columns
                for table, rows in recovered_tables.items():
                    if not rows:
                        continue
                    # Determine column count from first row
                    placeholders = ", ".join(["?"] * len(rows[0]))
                    try:
                        dst.executemany(
                            f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})",
                            rows,
                        )
                    except Exception as insert_err:
                        logger.warning(
                            "       ⚠ Could not restore %s: %s", table, insert_err
                        )

            logger.info(
                "       ✓ Rebuilt fresh user_model.db with %d episodes, "
                "%d semantic_facts, %d insights",
                len(recovered_episodes),
                len(recovered_tables["semantic_facts"]),
                len(recovered_tables["insights"]),
            )

            # ---------------------------------------------------------------
            # Step 4.5: Checkpoint the WAL before moving the new DB.
            #
            # SQLite WAL (Write-Ahead Log) mode writes new data to a
            # separate WAL file rather than directly to the main DB file.
            # If we move only the main DB without first checkpointing, the
            # WAL file stays in temp_dir and is deleted by shutil.rmtree in
            # the finally block.  The main DB then has no WAL data baked in
            # and reads as "database disk image is malformed" — the exact
            # corruption symptom we were trying to heal.
            #
            # PRAGMA wal_checkpoint(TRUNCATE):
            #   1. Copies all WAL frames into the main DB (checkpoint).
            #   2. Resets the WAL file to zero bytes (truncate).
            # After this call the main DB is self-contained and can be moved
            # safely without its WAL companion.
            # ---------------------------------------------------------------
            with new_db.get_connection("user_model") as chk_conn:
                chk_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.debug("       ✓ WAL checkpoint complete — new DB is self-contained")

            # ---------------------------------------------------------------
            # Step 5: Atomic swap — archive the corrupted file and replace.
            # ---------------------------------------------------------------
            corrupted_archive = db_path.with_suffix(".db.corrupted")
            # Remove old archive if present to avoid rename collision
            if corrupted_archive.exists():
                corrupted_archive.unlink()

            # Remove any WAL/SHM associated with the OLD (corrupted) DB
            # BEFORE the swap so they cannot be erroneously applied to the
            # fresh DB after it is moved into place.
            for suffix in ["-wal", "-shm"]:
                old_wal = db_path.with_name(db_path.name + suffix)
                if old_wal.exists():
                    old_wal.unlink(missing_ok=True)

            new_db_path = Path(temp_dir) / "user_model.db"
            # Archive the corrupted DB
            shutil.move(str(db_path), str(corrupted_archive))
            # Move the fresh DB into place (WAL already checkpointed → empty)
            shutil.move(str(new_db_path), str(db_path))

            logger.warning(
                "       ✓ Rebuilt corrupted user_model.db — "
                "recovered %d episodes, signal_profiles will be repopulated by backfills. "
                "Corrupted DB archived at %s",
                len(recovered_episodes),
                corrupted_archive,
            )

        except Exception as rebuild_err:
            logger.error(
                "       ✗ Failed to rebuild user_model.db: %s — "
                "system will continue with the existing (possibly corrupted) database",
                rebuild_err,
            )
        finally:
            # Clean up temp directory regardless of success/failure
            if temp_dir:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

    async def _repair_signal_profiles_if_corrupted(self):
        """Detect and repair SQLite B-tree corruption in the signal_profiles table.

        The ``signal_profiles`` table stores large JSON blobs in its ``data``
        column. These blobs can span multiple B-tree overflow pages in the
        SQLite file. When these pages become corrupted (e.g. after an unclean
        shutdown or disk I/O error), any SELECT that touches the ``data`` column
        raises ``sqlite3.DatabaseError: database disk image is malformed``.

        Effect of undetected corruption:
          - ``UserModelStore.get_signal_profile()`` silently returns ``None``
          - All semantic fact inference stops (semantic inferrer skips profiles
            that return ``None``)
          - Relationship maintenance predictions stop firing
          - Preparation-needs predictions stop firing
          - The data quality report always shows ``signal_profiles: {}``

        Recovery strategy:
          1. Force SQLite to read ALL data blobs by computing SUM(LENGTH(data)).
             This is the only reliable way to detect partial corruption: a
             ``LIMIT 1`` check only exercises the first row's overflow pages and
             will succeed even when all other rows are corrupted — a false
             negative that leaves the system silently broken.
          2. If that aggregate fails with "malformed" or any error, DROP the
             table and immediately re-create it using the same schema as the
             original CREATE TABLE statement in the migration.
          3. Log a WARNING so the user knows a repair occurred.
          4. The subsequent backfill guards in ``_backfill_relationship_profile_if_needed()``,
             ``_backfill_topic_profile_if_needed()``, etc. will see an empty table
             and automatically trigger full repopulation.

        Why SUM(LENGTH(data)) instead of LIMIT 1:
          SQLite computes aggregate functions by scanning every row.  Each scan
          step reads the data column, which forces every overflow page to be
          loaded.  A corrupted page raises immediately, regardless of which row
          owns it.  By contrast, ``LIMIT 1`` fetches only the first row —
          typically stored in the earliest allocated B-tree page — and returns
          successfully even when later pages (holding other rows) are corrupted.

        This method is idempotent: if the table is healthy, it returns quickly
        after a single cheap aggregate query.

        Example::

            await life_os._repair_signal_profiles_if_corrupted()
            # → "signal_profiles table is healthy" if OK
            # → "Repaired corrupted signal_profiles table" if fixed
        """
        try:
            # Step 1: Force SQLite to read ALL overflow pages by summing the
            # lengths of every data blob in the table.  This is the minimum
            # query that exercises every row's data column; a LIMIT 1 check
            # only reads the first row and misses corruption in later rows.
            with self.db.get_connection("user_model") as conn:
                conn.execute(
                    "SELECT SUM(LENGTH(data)) FROM signal_profiles"
                ).fetchone()
            # Table is healthy — nothing to do.
            logger.debug("       ✓ signal_profiles table is healthy")

        except Exception as e:
            error_msg = str(e).lower()
            # Catch both "malformed" corruption and any other read failure that
            # would leave profiles permanently unreadable.
            logger.warning(
                "       ⚠ signal_profiles table is corrupted (%s) — rebuilding…", e
            )

            try:
                # Step 2: Drop the corrupted table and re-create it.
                # The schema mirrors the definition in
                # DatabaseManager._create_user_model_schema() (Migration 1).
                with self.db.get_connection("user_model") as conn:
                    conn.execute("DROP TABLE IF EXISTS signal_profiles")
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS signal_profiles (
                            profile_type  TEXT PRIMARY KEY,
                            data          TEXT NOT NULL DEFAULT '{}',
                            samples_count INT  NOT NULL DEFAULT 0,
                            updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                        )
                    """)

                logger.info(
                    "       ✓ Repaired corrupted signal_profiles table — "
                    "backfill methods will repopulate profiles on startup"
                )

            except Exception as repair_err:
                # Repair itself failed — log and continue. The system will
                # run without signal profiles rather than crash on startup.
                logger.error(
                    "       ✗ Failed to repair signal_profiles table: %s — "
                    "system will run without signal profiles", repair_err
                )

    async def _backfill_communication_templates_if_needed(self):
        """Extract communication templates from all historical communication events.

        Communication template extraction was added in PR #130 but only processes
        new events going forward. This backfill populates Layer 3 procedural memory
        with writing-style templates learned from all historical emails and messages.

        Templates capture per-contact, per-channel writing patterns:
        - Greeting and closing phrases (e.g., "Hi" vs "Dear" vs none)
        - Formality level (0.0 = casual, 1.0 = formal)
        - Typical message length
        - Emoji usage patterns
        - Common phrases and tone indicators

        This enables:
        1. Style-matching when drafting AI-generated replies
        2. Detecting formality mismatches in communication
        3. Learning relationship-specific patterns
        4. Better prediction of incoming message characteristics

        The backfill is idempotent and safe to run multiple times. It processes
        events in chronological order so templates evolve naturally as writing
        style changes over time.
        """
        try:
            # Check if we already have templates (skip if > 100)
            with self.db.get_connection("user_model") as conn:
                template_count = conn.execute(
                    "SELECT COUNT(*) FROM communication_templates"
                ).fetchone()[0]

            # If we already have a significant number of templates, skip backfill
            if template_count >= 100:
                return

            # Count communication events available for template extraction
            with self.db.get_connection("events") as conn:
                event_count = conn.execute("""
                    SELECT COUNT(*) FROM events
                    WHERE type IN ('email.sent', 'email.received', 'message.sent', 'message.received')
                      AND (LENGTH(json_extract(payload, '$.body_plain')) > 10
                           OR LENGTH(json_extract(payload, '$.body')) > 10)
                """).fetchone()[0]

            # Only run backfill if we have sufficient communication events (50+)
            if event_count < 50:
                return

            logger.info("       → Backfilling communication templates from %s historical events...", f"{event_count:,}")

            # Run the backfill in a thread to avoid blocking startup
            def _run_backfill():
                from scripts.backfill_communication_templates import backfill_communication_templates
                stats = backfill_communication_templates(
                    data_dir=self.db.data_dir,
                    batch_size=5000,
                )
                return stats

            stats = await asyncio.to_thread(_run_backfill)

            logger.info(
                "       ✓ Created %s templates from %s events (%.1fs)",
                f"{stats['templates_created']:,}",
                f"{stats['events_processed']:,}",
                stats['elapsed_seconds'],
            )

        except Exception as e:
            logger.warning("       ⚠ Communication template backfill failed (non-fatal): %s", e)

    async def _backfill_relationship_profile_if_needed(self):
        """Auto-trigger relationship signal profile backfill when the profile is missing.

        The relationship signal profile is the foundation of Life OS's social intelligence:
        it powers relationship maintenance predictions, the People Radar in the dashboard,
        and semantic fact inference about high-priority contacts.

        After Migration 0→1 wiped ``signal_profiles``, this profile is empty.  The live
        pipeline refills it only when new email events arrive, but if the connector is
        stale (e.g., OAuth token expired), it stays empty indefinitely.

        This method mirrors ``_backfill_communication_templates_if_needed()``:
        - Checks whether the profile already has data (idempotent guard)
        - Counts eligible events to confirm there's history to learn from
        - Delegates the heavy lifting to ``scripts/backfill_relationship_profile.py``
          in a background thread so startup is not blocked

        The backfill processes ``email.received``, ``email.sent``, ``message.received``,
        and ``message.sent`` events through the live RelationshipExtractor, building the
        same per-contact interaction graph that real-time processing would produce.

        Effects once populated:
        - Semantic inferrer generates ``relationship_priority_*`` facts for frequent contacts
        - Prediction engine's ``_check_relationship_maintenance()`` fires alerts for overdue contacts
        - People Radar shows real contacts with last-contact timestamps and frequency dots
        - Draft Reply endpoint has relationship context for tone-matching
        """
        try:
            # Guard: skip if the profile already has meaningful data.
            # Threshold of 10 samples allows for the case where a very small backfill
            # ran before and we don't want to re-run it needlessly.
            profile = self.user_model_store.get_signal_profile("relationships")
            if profile and profile.get("samples_count", 0) >= 10:
                return

            # Check that we have enough historical communication events to learn from.
            # Running the backfill on <10 events wouldn't produce useful relationship data.
            with self.db.get_connection("events") as conn:
                event_count = conn.execute(
                    """SELECT COUNT(*) FROM events
                       WHERE type IN (
                           'email.received', 'email.sent',
                           'message.received', 'message.sent'
                       )"""
                ).fetchone()[0]

            if event_count < 10:
                return

            logger.info(
                "       → Backfilling relationship signal profile from %s communication events...",
                f"{event_count:,}",
            )

            # Run in a thread so the async event loop isn't blocked during the
            # O(n_emails) SQLite iteration that can take several seconds on large datasets.
            def _run_backfill():
                from scripts.backfill_relationship_profile import backfill_relationship_profile

                return backfill_relationship_profile(
                    data_dir=self.db.data_dir,
                    batch_size=1000,  # Reasonable batch size for progress reporting
                )

            stats = await asyncio.to_thread(_run_backfill)

            logger.info(
                "       ✓ Relationship profile: %s contacts discovered from %s events (%.1fs)",
                f"{stats['contacts_discovered']:,}",
                f"{stats['events_processed']:,}",
                stats["elapsed_seconds"],
            )

        except Exception as e:
            # Fail-open: backfill errors must never crash the startup sequence.
            logger.warning("       ⚠ Relationship profile backfill failed (non-fatal): %s", e)

    async def _backfill_temporal_profile_if_needed(self):
        """Auto-trigger temporal signal profile backfill when the profile is missing.

        The temporal profile captures the user's activity patterns across hours and days:
        peak energy times, preferred meeting slots, planning horizon behavior, and the
        ratio of reactive vs. scheduled work.  It feeds:

        - ``_check_preparation_needs()`` in the prediction engine (e.g., "You usually
          prepare meeting agendas the day before; one is due tomorrow")
        - Routine detection (time-based routines require temporal data to normalize by
          time-of-day and day-of-week)
        - Insights tab's behavioral pattern cards

        The temporal backfill processes user-initiated events:
        ``email.sent``, ``message.sent``, ``calendar.event.created``, ``task.created``,
        ``task.completed``, and ``system.user.command``.

        Uses the same idempotent guard and background-thread pattern as the other
        backfill methods in this startup sequence.
        """
        try:
            # Guard: skip if temporal profile already populated.
            profile = self.user_model_store.get_signal_profile("temporal")
            if profile and profile.get("samples_count", 0) >= 5:
                return

            # Confirm there are user-initiated events to analyze.
            with self.db.get_connection("events") as conn:
                event_count = conn.execute(
                    """SELECT COUNT(*) FROM events
                       WHERE type IN (
                           'email.sent', 'message.sent',
                           'calendar.event.created', 'calendar.event.updated',
                           'task.created', 'task.completed', 'task.updated',
                           'system.user.command'
                       )"""
                ).fetchone()[0]

            if event_count < 5:
                return

            logger.info(
                "       → Backfilling temporal signal profile from %s user-initiated events...",
                f"{event_count:,}",
            )

            def _run_backfill():
                from scripts.backfill_temporal_profile import backfill_temporal_profile

                return backfill_temporal_profile(
                    data_dir=self.db.data_dir,
                    batch_size=1000,
                )

            stats = await asyncio.to_thread(_run_backfill)

            logger.info(
                "       ✓ Temporal profile: %s signals from %s events (%.1fs)",
                f"{stats['signals_extracted']:,}",
                f"{stats['events_processed']:,}",
                stats["elapsed_seconds"],
            )

        except Exception as e:
            logger.warning("       ⚠ Temporal profile backfill failed (non-fatal): %s", e)

    async def _backfill_topic_profile_if_needed(self):
        """Auto-trigger topic signal profile backfill when the profile is missing.

        The topic signal profile is the data foundation for the semantic inferrer's
        expertise and interest fact generation.  ``infer_from_topic_profile()`` reads
        this profile and produces ``expertise_<topic>`` and ``interest_<topic>`` semantic
        facts — the richest user model data that powers the "AI that knows you" experience.

        After Migration 0→1 wiped ``signal_profiles``, this profile is empty.  The live
        pipeline refills it only when new email events arrive, but if the connector is
        stale the table stays empty indefinitely.  This trigger replays all historical
        communication events through TopicExtractor on startup so the profile is always
        populated regardless of connector health.

        Effects once populated:
        - Semantic inferrer generates ``expertise_<topic>`` facts for frequently-discussed
          technical/professional topics (e.g., expertise_python, expertise_machine_learning)
        - Semantic inferrer generates ``interest_<topic>`` facts for common discussion areas
        - Topic distribution shown in the Insights tab behavioral pattern cards
        - Prediction engine has topic context for preparation-need detection
        """
        try:
            # Guard: skip if the topic profile already has meaningful data.
            # A threshold of 30 samples matches the inferrer's minimum requirement.
            profile = self.user_model_store.get_signal_profile("topics")
            if profile and profile.get("samples_count", 0) >= 30:
                return

            # Check that we have enough historical communication events to learn from.
            with self.db.get_connection("events") as conn:
                event_count = conn.execute(
                    """SELECT COUNT(*) FROM events
                       WHERE type IN (
                           'email.received', 'email.sent',
                           'message.received', 'message.sent',
                           'system.user.command'
                       )"""
                ).fetchone()[0]

            if event_count < 30:
                return

            logger.info(
                "       → Backfilling topic signal profile from %s communication events...",
                f"{event_count:,}",
            )

            # Run in a thread so the async event loop isn't blocked during the
            # O(n_emails) SQLite iteration that can take several seconds on large datasets.
            def _run_backfill():
                from scripts.backfill_topic_profile import backfill_topic_profile

                return backfill_topic_profile(
                    data_dir=self.db.data_dir,
                    batch_size=1000,
                )

            stats = await asyncio.to_thread(_run_backfill)

            logger.info(
                "       ✓ Topic profile: %s topics discovered from %s events (%.1fs)",
                f"{stats['topics_discovered']:,}",
                f"{stats['events_processed']:,}",
                stats["elapsed_seconds"],
            )

        except Exception as e:
            # Fail-open: backfill errors must never crash the startup sequence.
            logger.warning("       ⚠ Topic profile backfill failed (non-fatal): %s", e)

    async def _backfill_linguistic_profile_if_needed(self):
        """Auto-trigger linguistic signal profile backfill when the profile is missing.

        The linguistic signal profile captures the user's writing-style fingerprint:
        vocabulary complexity, formality level, sentence structure preferences, hedge
        word usage, exclamation rate, and emoji density.  The semantic inferrer reads
        this profile via ``infer_from_linguistic_profile()`` to produce facts about
        communication style preferences (formal vs. casual, direct vs. tentative, etc.).

        These facts in turn feed:
        - Communication template extraction (the templates learn the user's preferred
          tone per contact, derived from their linguistic baseline)
        - Draft reply generation (AI-drafted replies match the user's observed style)
        - Insights tab linguistic profile card

        After Migration 0→1 wiped ``signal_profiles``, this profile is empty.  This
        trigger replays all historical communication events through LinguisticExtractor
        so the profile is always populated on startup.
        """
        try:
            # Guard: skip if the linguistic profile already has meaningful data.
            # Threshold of 1 sample matches the inferrer's minimum requirement.
            profile = self.user_model_store.get_signal_profile("linguistic")
            if profile and profile.get("samples_count", 0) >= 1:
                return

            # Check that we have outbound communication events to learn from.
            # The linguistic profile is most informative from email.sent events
            # (the user's actual writing), so we require at least 1 sent email.
            with self.db.get_connection("events") as conn:
                event_count = conn.execute(
                    """SELECT COUNT(*) FROM events
                       WHERE type IN (
                           'email.received', 'email.sent',
                           'message.received', 'message.sent',
                           'system.user.command'
                       )"""
                ).fetchone()[0]

            if event_count < 1:
                return

            logger.info(
                "       → Backfilling linguistic signal profile from %s communication events...",
                f"{event_count:,}",
            )

            def _run_backfill():
                from scripts.backfill_linguistic_profile import backfill_linguistic_profile

                return backfill_linguistic_profile(
                    data_dir=self.db.data_dir,
                    batch_size=1000,
                )

            stats = await asyncio.to_thread(_run_backfill)

            logger.info(
                "       ✓ Linguistic profile: %s samples from %s events (%.1fs)",
                f"{stats['final_samples']:,}",
                f"{stats['events_processed']:,}",
                stats["elapsed_seconds"],
            )

        except Exception as e:
            # Fail-open: backfill errors must never crash the startup sequence.
            logger.warning("       ⚠ Linguistic profile backfill failed (non-fatal): %s", e)

    async def _backfill_cadence_profile_if_needed(self):
        """Auto-trigger cadence signal profile backfill when the profile is missing.

        The cadence signal profile tracks response times (per-contact and per-channel)
        and activity-window heatmaps (hourly/daily histograms).  It drives:

        - Response-time priority contact detection in the prediction engine
          (``_check_relationship_maintenance`` uses per-contact reply latency to
          identify high-priority contacts the user hasn't responded to)
        - Peak-hours and quiet-hours detection (requires 50+ activity samples)
        - Per-domain response-time breakdowns in the Insights tab

        After a DB migration or rebuild, the cadence profile is wiped.  Without this
        backfill, response-time priority detection is permanently broken until enough
        new live events accumulate (typically weeks of usage).  This trigger replays
        all historical email and message events through CadenceExtractor on startup.
        """
        try:
            # Guard: skip if the cadence profile already has meaningful data.
            profile = self.user_model_store.get_signal_profile("cadence")
            if profile and profile.get("samples_count", 0) >= 10:
                return

            # Check that we have enough historical communication events to learn from.
            with self.db.get_connection("events") as conn:
                event_count = conn.execute(
                    """SELECT COUNT(*) FROM events
                       WHERE type IN (
                           'email.received', 'email.sent',
                           'message.received', 'message.sent'
                       )"""
                ).fetchone()[0]

            if event_count < 10:
                return

            logger.info(
                "       → Backfilling cadence signal profile from %s communication events...",
                f"{event_count:,}",
            )

            # Run in a thread so the async event loop isn't blocked during the
            # O(n_events) SQLite iteration that can take several seconds on large datasets.
            def _run_backfill():
                from scripts.backfill_cadence_profile import backfill_cadence_profile

                return backfill_cadence_profile(
                    data_dir=self.db.data_dir,
                    batch_size=1000,
                )

            stats = await asyncio.to_thread(_run_backfill)

            logger.info(
                "       ✓ Cadence profile: %s contacts tracked from %s events (%.1fs)",
                f"{stats['contacts_tracked']:,}",
                f"{stats['events_processed']:,}",
                stats["elapsed_seconds"],
            )

        except Exception as e:
            # Fail-open: backfill errors must never crash the startup sequence.
            logger.warning("       ⚠ Cadence profile backfill failed (non-fatal): %s", e)

    async def _backfill_mood_signals_profile_if_needed(self):
        """Auto-trigger mood_signals profile backfill when the profile is missing.

        The mood_signals profile stores a ring buffer of recent mood-relevant signals
        that ``compute_current_mood()`` reads to produce the ``MoodState`` displayed
        on the dashboard mood widget.  It also provides energy-level data for episode
        creation and mood trend computation.

        The MoodInferenceEngine casts a wide net — it processes 10 event types:
        email (sent/received), message (sent/received), health metrics, sleep data,
        calendar events, financial transactions, location changes, and user commands.

        After a DB migration or rebuild, the mood_signals profile is wiped.  Without
        this backfill, compute_current_mood() always returns a neutral MoodState with
        0.0 confidence until enough new live events accumulate.  This trigger replays
        all qualifying historical events through MoodInferenceEngine on startup.
        """
        try:
            # Guard: skip if the mood_signals profile already has meaningful data.
            profile = self.user_model_store.get_signal_profile("mood_signals")
            if profile and profile.get("samples_count", 0) >= 10:
                return

            # Check that we have enough historical events to learn from.
            with self.db.get_connection("events") as conn:
                event_count = conn.execute(
                    """SELECT COUNT(*) FROM events
                       WHERE type IN (
                           'email.sent', 'email.received',
                           'message.sent', 'message.received',
                           'health.metric.updated', 'sleep.recorded',
                           'calendar.event.created', 'transaction.new',
                           'location.changed', 'system.user.command'
                       )"""
                ).fetchone()[0]

            if event_count < 10:
                return

            logger.info(
                "       → Backfilling mood_signals profile from %s events...",
                f"{event_count:,}",
            )

            def _run_backfill():
                from scripts.backfill_mood_profile import backfill_mood_profile

                return backfill_mood_profile(
                    data_dir=self.db.data_dir,
                    batch_size=1000,
                )

            stats = await asyncio.to_thread(_run_backfill)

            logger.info(
                "       ✓ Mood signals profile: %s samples from %s events (%.1fs)",
                f"{stats['final_samples']:,}",
                f"{stats['events_processed']:,}",
                stats["elapsed_seconds"],
            )

        except Exception as e:
            # Fail-open: backfill errors must never crash the startup sequence.
            logger.warning("       ⚠ Mood signals profile backfill failed (non-fatal): %s", e)

    async def _backfill_spatial_profile_if_needed(self):
        """Auto-trigger spatial profile backfill when the profile is missing.

        The spatial profile tracks place-based behavior patterns (visit frequency,
        duration, dominant domain per location) used by the SpatialExtractor.  It
        powers the semantic fact inferrer's ``infer_from_spatial_profile()`` which
        derives facts like primary work location, work location type, and frequent
        location patterns.  That inference requires >= 10 samples.

        The SpatialExtractor processes three event types:
        - calendar.event.created (with non-empty location field)
        - ios.context.update (with location or device_proximity)
        - system.user.location_update

        After a DB migration or rebuild the spatial profile is wiped.  Without this
        backfill, no spatial semantic facts are ever generated until enough new live
        events accumulate (typically weeks of usage).
        """
        try:
            # Guard: skip if the spatial profile already has meaningful data.
            profile = self.user_model_store.get_signal_profile("spatial")
            if profile and profile.get("samples_count", 0) >= 10:
                return

            # Check that we have enough historical events to learn from.
            # Use a broader query — the SpatialExtractor.can_process() will filter
            # further (e.g. calendar events without a location field are skipped).
            with self.db.get_connection("events") as conn:
                event_count = conn.execute(
                    """SELECT COUNT(*) FROM events
                       WHERE type IN (
                           'calendar.event.created',
                           'ios.context.update',
                           'system.user.location_update'
                       )"""
                ).fetchone()[0]

            if event_count < 10:
                return

            logger.info(
                "       → Backfilling spatial profile from %s events...",
                f"{event_count:,}",
            )

            def _run_backfill():
                from scripts.backfill_spatial_profile import backfill_spatial_profile

                return backfill_spatial_profile(
                    data_dir=self.db.data_dir,
                    batch_size=1000,
                )

            stats = await asyncio.to_thread(_run_backfill)

            logger.info(
                "       ✓ Spatial profile: %s samples from %s events (%.1fs)",
                f"{stats['final_samples']:,}",
                f"{stats['events_processed']:,}",
                stats["elapsed_seconds"],
            )

        except Exception as e:
            # Fail-open: backfill errors must never crash the startup sequence.
            logger.warning("       ⚠ Spatial profile backfill failed (non-fatal): %s", e)

    async def _backfill_decision_profile_if_needed(self):
        """Auto-trigger decision profile backfill when the profile is missing.

        The decision profile tracks decision-making patterns (speed by domain,
        delegation comfort, risk tolerance, fatigue indicators) used by the
        DecisionExtractor.  It powers the semantic fact inferrer's
        ``infer_from_decision_profile()`` which derives facts like decision speed,
        risk tolerance, and delegation preferences.  That inference requires >= 20
        samples.

        The DecisionExtractor processes five event types:
        - task.completed (decision execution)
        - task.created (commitment decisions)
        - email.sent (decision communication)
        - message.sent (decision communication)
        - calendar.event.created (commitment decisions)

        After a DB migration or rebuild the decision profile is wiped.  Without
        this backfill, no decision-making semantic facts are ever generated until
        enough new live events accumulate (typically weeks of usage).
        """
        try:
            # Guard: skip if the decision profile already has meaningful data.
            profile = self.user_model_store.get_signal_profile("decision")
            if profile and profile.get("samples_count", 0) >= 20:
                return

            # Check that we have enough historical events to learn from.
            with self.db.get_connection("events") as conn:
                event_count = conn.execute(
                    """SELECT COUNT(*) FROM events
                       WHERE type IN (
                           'task.completed', 'task.created',
                           'email.sent', 'message.sent',
                           'calendar.event.created'
                       )"""
                ).fetchone()[0]

            if event_count < 20:
                return

            logger.info(
                "       → Backfilling decision profile from %s events...",
                f"{event_count:,}",
            )

            def _run_backfill():
                from scripts.backfill_decision_profile import backfill_decision_profile

                return backfill_decision_profile(
                    data_dir=self.db.data_dir,
                    batch_size=1000,
                )

            stats = await asyncio.to_thread(_run_backfill)

            logger.info(
                "       ✓ Decision profile: %s samples from %s events (%.1fs)",
                f"{stats['final_samples']:,}",
                f"{stats['events_processed']:,}",
                stats["elapsed_seconds"],
            )

        except Exception as e:
            # Fail-open: backfill errors must never crash the startup sequence.
            logger.warning("       ⚠ Decision profile backfill failed (non-fatal): %s", e)

    async def _clean_relationship_profile_if_needed(self):
        """Remove marketing contacts from the relationships signal profile.

        PROBLEM:
        The relationship extractor added marketing email filtering in PR #143, but
        this left 469+ marketing contacts from historical data tracked before the
        filter existed. These pollute the relationships profile with non-human
        "contacts" that waste storage and break relationship maintenance predictions.

        SOLUTION:
        Run the cleanup logic once on startup to purge marketing contacts (no-reply@,
        newsletter@, @comms., @email., etc.) while preserving real human contacts.
        This is idempotent and safe to run multiple times.

        IMPACT:
        - Enables relationship maintenance predictions (currently 0 generated)
        - Improves prediction engine performance (no longer loops through 469+ marketing
          contacts every 15 minutes)
        - Cleans up storage bloat in signal_profiles table
        - Reduces relationships profile from 820 contacts to ~350 real humans
        """
        try:
            # Check if we have a relationships profile that needs cleaning
            profile = self.user_model_store.get_signal_profile("relationships")
            if not profile:
                return

            contacts = profile["data"].get("contacts", {})
            if len(contacts) == 0:
                return

            # Apply marketing detection to every contact address
            from scripts.clean_relationship_profile_marketing import is_marketing_or_noreply

            marketing_count = sum(
                1 for addr in contacts.keys()
                if is_marketing_or_noreply(addr)
            )

            # If < 10% of contacts are marketing, profile is already clean
            if marketing_count / len(contacts) < 0.1:
                return

            logger.info("       → Cleaning %s marketing contacts from relationships profile...", f"{marketing_count:,}")

            # Run the cleanup in a thread to avoid blocking startup
            def _run_cleanup():
                from scripts.clean_relationship_profile_marketing import clean_relationship_profile
                stats = clean_relationship_profile(
                    db=self.db,
                    dry_run=False,  # Actually modify the database
                    verbose=False,  # Suppress output to prevent asyncio.to_thread() hangs
                )
                return stats

            stats = await asyncio.to_thread(_run_cleanup)

            logger.info(
                "       ✓ Removed %s marketing contacts, %s human contacts remaining",
                f"{stats['removed']:,}",
                f"{stats['remaining']:,}",
            )

        except Exception as e:
            logger.warning("       ⚠ Relationship profile cleanup failed (non-fatal): %s", e)

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
                logger.error("Event store error (event_id=%s, type=%s): %s",
                             event.get("id"), event.get("type"), e)

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
                logger.error("Feedback collector error (event_id=%s): %s", event.get("id"), e)

            # Stage 1.3 — Source Weight Tracking: classify the event into a
            # source_key and increment its interaction counter.  This builds
            # the data the AI drift algorithm needs to learn which sources
            # the user cares about.
            try:
                source_key = self.source_weight_manager.classify_event(event)
                self.source_weight_manager.record_interaction(source_key)
            except Exception as e:
                logger.error("Source weight tracking error (event_id=%s): %s", event.get("id"), e)

            # Stage 1.5 — Episodic Memory: convert each event into a memory
            # episode for the user model's Layer 1 (Episodic) storage.
            # This provides the raw interaction history that feeds semantic
            # fact extraction and enables the system to answer "when did I
            # last talk to X" or "what happened in my meeting yesterday".
            try:
                await self._create_episode(event)
            except Exception as e:
                logger.error("Episode creation error (event_id=%s, type=%s): %s",
                             event.get("id"), event.get("type"), e)

            # Stage 2 — Learn: the signal extractor passively analyses the
            # event to update the user model (patterns, preferences, etc.).
            try:
                await self.signal_extractor.process_event(event)
            except Exception as e:
                logger.error("Signal extractor error (event_id=%s, type=%s): %s",
                             event.get("id"), event.get("type"), e)

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
                logger.error("Rules engine error (event_id=%s, type=%s): %s",
                             event.get("id"), event.get("type"), e)

            # Stage 4 — Extract tasks: scan the event payload (e.g. email
            # body, chat message) for actionable items and create tasks.
            try:
                await self.task_manager.process_event(event)
            except Exception as e:
                logger.error("Task manager error (event_id=%s, type=%s): %s",
                             event.get("id"), event.get("type"), e)

            # Stage 5 — Embed: generate a vector embedding of the event
            # content so it can be retrieved via semantic search later.
            try:
                await self._embed_event(event)
            except Exception as e:
                logger.error("Embedding error (event_id=%s, type=%s): %s",
                             event.get("id"), event.get("type"), e)

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

    def _build_notification_content(self, event: dict, action: dict) -> tuple[str, str]:
        """Build a meaningful title and body for a rule-triggered notification.

        Most event types don't have a ``snippet`` payload field, so the old
        hard-coded ``payload.get("snippet", "")`` produced empty notification
        bodies for email, calendar, finance, and message events.  This helper
        inspects the payload for the most informative fields available.

        Args:
            event: The event dict with ``payload`` and ``type`` fields.
            action: The rule action dict (used for ``rule_name`` fallback).

        Returns:
            A (title, body) tuple with the notification content.
        """
        payload = event.get("payload", {})

        # --- Title: pick the most descriptive field available ---
        title = (
            payload.get("subject")
            or payload.get("summary")
            or payload.get("merchant_name")
            or f"Rule: {action.get('rule_name', 'Unknown')}"
        )

        # For finance events, append the amount to the merchant name title
        if payload.get("merchant_name") and payload.get("amount") is not None:
            title = f"{payload['merchant_name']} — ${payload['amount']}"

        # --- Body: try payload fields in order of usefulness ---
        body = (
            payload.get("snippet")
            or payload.get("body_plain")
            or payload.get("body")
            or payload.get("description")
            or payload.get("content")
            or ""
        )

        # For email events, prepend the sender so the user knows who it's from
        from_address = payload.get("from_address") or payload.get("from")
        if from_address and event.get("type", "").startswith("email"):
            body = f"From: {from_address}\n{body}" if body else f"From: {from_address}"

        # Truncate to 200 characters to keep notifications concise
        if len(body) > 200:
            body = body[:197] + "..."

        return title, body

    async def _execute_rule_action(self, action: dict, event: dict):
        """Execute an action triggered by the rules engine.

        Supported action types:
            notify      — create a user-visible notification (skipped if suppressed)
            tag         — attach a label to the event in event_tags
            suppress    — flag the event so it is hidden from notifications
            create_task — auto-create a task linked to the source event
            archive     — suppress + tag as archived (like email archive)
            forward     — forward the event content via the source connector
            auto_reply  — send an automatic reply via the source connector
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

            title, body = self._build_notification_content(event, action)
            await self.notification_manager.create_notification(
                title=title,
                body=body,
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
        elif action_type == "archive":
            # Archive = suppress (hide from notifications) + tag for filtering.
            # This mirrors how email archive works: remove from inbox but keep
            # in storage. The event remains in the append-only log.
            event["_suppressed"] = True
            self.event_store.add_tag(
                event_id=event["id"],
                tag="system:archived",
                rule_id=action.get("rule_id"),
            )
        elif action_type == "forward":
            # Forward the event content via the originating connector's
            # execute() interface. Connectors that don't support 'forward'
            # will raise NotImplementedError, which we catch and log.
            forward_to = action.get("value") or action.get("to")
            if not forward_to:
                logger.warning(
                    "forward action missing target address (rule_id=%r)",
                    action.get("rule_id"),
                )
                return
            source = event.get("source", "")
            connector = self.connector_map.get(source)
            if not connector:
                logger.warning(
                    "forward action: no connector for source %r (rule_id=%r)",
                    source,
                    action.get("rule_id"),
                )
                return
            try:
                await connector.execute("forward", {
                    "to": forward_to,
                    "event_id": event.get("id"),
                    "subject": event.get("payload", {}).get("subject", ""),
                    "body": event.get("payload", {}).get(
                        "body", event.get("payload", {}).get("snippet", "")
                    ),
                })
            except Exception as e:
                logger.error(
                    "forward action failed (rule_id=%r, source=%r): %s",
                    action.get("rule_id"),
                    source,
                    e,
                )
        elif action_type == "auto_reply":
            # Send an automatic reply via the originating connector's
            # execute() interface. Connectors that don't support 'reply'
            # will raise NotImplementedError, which we catch and log.
            reply_text = action.get("value") or action.get("message", "")
            if not reply_text:
                logger.warning(
                    "auto_reply action missing message text (rule_id=%r)",
                    action.get("rule_id"),
                )
                return
            source = event.get("source", "")
            connector = self.connector_map.get(source)
            if not connector:
                logger.warning(
                    "auto_reply action: no connector for source %r (rule_id=%r)",
                    source,
                    action.get("rule_id"),
                )
                return
            try:
                await connector.execute("reply", {
                    "to": event.get("payload", {}).get(
                        "sender", event.get("payload", {}).get("from", "")
                    ),
                    "in_reply_to": event.get("payload", {}).get(
                        "message_id", event.get("id")
                    ),
                    "subject": "Re: " + (
                        event.get("payload", {}).get("subject", "") or ""
                    ),
                    "body": reply_text,
                })
            except Exception as e:
                logger.error(
                    "auto_reply action failed (rule_id=%r, source=%r): %s",
                    action.get("rule_id"),
                    source,
                    e,
                )
        else:
            # Log unrecognised action types so operators can catch misconfigured
            # rules. Previously these were silently dropped, making it impossible
            # to diagnose rules that specified unsupported types.
            logger.warning(
                "Rule action type %r is not implemented; action dropped "
                "(rule_id=%r, event_type=%r)",
                action_type,
                action.get("rule_id"),
                event.get("type"),
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

        # Store a COMPACT version of the payload as content_full.
        #
        # Root-cause fix (iteration 43): storing the raw payload caused
        # user_model.db to balloon to 7.4 GB because email bodies (often
        # 50–200 KB of HTML) were duplicated into every episode row.  With
        # 55 K+ email episodes the overflow-page chains that SQLite uses for
        # large BLOBs became corrupted, breaking signal_profiles reads and
        # semantic-fact inference.
        #
        # The full event payload is already in events.db (immutable log, keyed
        # on event_id).  Episodes only need the metadata fields needed for
        # timeline display and relationship analysis; the heavy body text is
        # never read from content_full by any service or query.
        #
        # Strategy:
        #  1. Drop fields that are large by definition (body, html, raw mime).
        #  2. Truncate any remaining string value that exceeds 500 characters
        #     to a concise snippet (adds "…" suffix).
        #  3. Cap the total JSON at 4 000 chars as a hard backstop.
        _LARGE_FIELDS = frozenset(
            {"body", "html_body", "raw", "raw_mime", "text_body", "html", "content"}
        )
        _SNIPPET_CHARS = 500
        _MAX_TOTAL_CHARS = 4_000

        compact_payload: dict = {}
        for k, v in payload.items():
            if k in _LARGE_FIELDS:
                # Replace bulky body fields with a short snippet so the field
                # still exists (useful for debugging) but takes minimal space.
                if isinstance(v, str) and v:
                    compact_payload[k] = v[:_SNIPPET_CHARS] + ("…" if len(v) > _SNIPPET_CHARS else "")
                # Non-string body fields (rare) are simply dropped.
            elif isinstance(v, str) and len(v) > _SNIPPET_CHARS:
                compact_payload[k] = v[:_SNIPPET_CHARS] + "…"
            else:
                compact_payload[k] = v

        content_full_json = json.dumps(compact_payload)
        # Hard cap: if the compacted JSON is somehow still large, truncate
        # at a safe boundary (this should never happen in practice, but
        # prevents any future schema additions from reintroducing bloat).
        content_full = (
            content_full_json[:_MAX_TOTAL_CHARS] if len(content_full_json) > _MAX_TOTAL_CHARS
            else content_full_json
        )

        # Retrieve current mood from the user model if available — this
        # provides emotional context that helps the system understand why
        # the user made certain decisions at certain times.
        #
        # CRITICAL FIX (iteration 131):
        # Previously this code looked for mood_profile["data"]["samples"],
        # but the MoodInferenceEngine stores raw signals in "recent_signals"
        # and never populates a "samples" array. The correct approach is to
        # call compute_current_mood() which aggregates recent_signals into
        # a MoodState with energy_level, stress_level, and emotional_valence.
        #
        # CRITICAL FIX (iteration 134):
        # The bare except was swallowing exceptions silently, causing ALL
        # episodes to be created with mood=null despite 27K+ mood signals
        # being available. Now we log the exception so we can diagnose and
        # fix the root cause instead of silently degrading.
        inferred_mood = None
        try:
            mood_state = self.signal_extractor.get_current_mood()
            # Only include mood if we have enough signals for confidence > 0
            # (MoodState.confidence scales with signal count, 0 means no data)
            if mood_state and mood_state.confidence > 0:
                inferred_mood = {
                    "energy_level": mood_state.energy_level,
                    "stress_level": mood_state.stress_level,
                    "emotional_valence": mood_state.emotional_valence,
                }
        except Exception as e:
            # Log the exception so we can diagnose mood retrieval failures.
            # Mood inference is optional (episodes should still be created),
            # but complete silence on failures prevents us from fixing bugs.
            logger.warning("Mood retrieval failed in episode creation: %s", e, exc_info=True)

        # Determine the active domain (work, personal, health, finance) from
        # event metadata if available. This helps segment episodes by life area.
        active_domain = event.get("metadata", {}).get("domain", "personal")

        # Build the episode dict matching the schema in user_model.db.
        #
        # CRITICAL FIX: Use the event's actual interaction timestamp, not the sync
        # timestamp. Without this fix, all email episodes collapse to the sync date
        # (e.g., 2026-02-22 when the Google connector first ran), making it impossible
        # for the routine detector to see multi-day patterns — the detector requires
        # activities on 3+ distinct days to call something a routine.
        #
        # Connector-specific field names for the actual event timestamp:
        #   - email_date   : Google/Proton mail connectors (from RFC 2822 Date header)
        #   - sent_at      : iMessage and Signal connectors (when message was sent)
        #   - received_at  : Some message connectors (when message arrived)
        #   - date         : Generic connectors that use the bare "date" key
        #   - start_time   : CalDAV and Google Calendar (meeting start time)
        #
        # The relationship extractor uses an identical priority chain; keep in sync.
        actual_timestamp = (
            payload.get("email_date")   # Google/Proton mail — actual Date header
            or payload.get("sent_at")   # iMessage, Signal — message send time
            or payload.get("received_at")  # some connectors — arrival time
            or payload.get("date")      # generic fallback for older connectors
            or payload.get("start_time")  # Calendar: actual event start
            or event.get("timestamp")   # Last resort: sync timestamp
        )

        episode = {
            "id": str(uuid.uuid4()),
            "timestamp": actual_timestamp,
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
                        priority=_prediction_priority(prediction),
                        source_event_id=prediction.id,
                        domain="prediction",
                    )

                # Auto-resolve stale prediction notifications to close the feedback loop.
                # Predictions that users ignore for 24+ hours are marked inaccurate so
                # the prediction engine can learn which types to suppress.
                resolved = await self.notification_manager.auto_resolve_stale_predictions(timeout_hours=24)
                if resolved > 0:
                    logger.info("  Auto-resolved %d stale prediction(s)", resolved)

                # Auto-resolve filtered predictions to prevent database bloat.
                # Predictions with was_surfaced=0 were filtered by confidence gates
                # or reaction prediction and never shown to the user. After 1 hour,
                # these are no longer relevant and should be cleaned up.
                filtered_resolved = await asyncio.to_thread(
                    self.notification_manager.auto_resolve_filtered_predictions, 1
                )
                if filtered_resolved > 0:
                    logger.info("  Auto-resolved %d filtered prediction(s)", filtered_resolved)

            except Exception as e:
                logger.error("Prediction engine error: %s", e)

            # 900 seconds = 15 minutes.  This interval balances freshness
            # against compute cost; predictions depend on aggregated patterns,
            # so sub-minute granularity is unnecessary.
            await asyncio.sleep(900)  # 15 minutes

    async def _insight_loop(self):
        """Run the insight engine every 15 minutes.

        Also runs the source weight bulk drift recalculation once per
        cycle so that AI drift adjusts based on aggregate engagement
        patterns, not just individual feedback events.
        """
        while not self.shutdown_event.is_set():
            try:
                insights = await self.insight_engine.generate_insights()
                if insights:
                    logger.info("  InsightEngine: generated %d new insights", len(insights))
            except Exception as e:
                logger.error("Insight engine error: %s", e)

            # Recalculate AI drift based on engagement ratios
            try:
                await asyncio.to_thread(self.source_weight_manager.bulk_recalculate_drift)
            except Exception as e:
                logger.error("Source weight drift recalc error: %s", e)

            await asyncio.sleep(900)  # 15 minutes

    async def _semantic_inference_loop(self):
        """Run semantic fact inference every hour.

        The semantic fact inferrer analyzes accumulated signal profiles
        (linguistic, relationship, topic, cadence, mood) and derives high-level
        facts about the user's preferences, expertise, and values.

        This bridges Layer 0/1 (raw signals and episodes) to Layer 2 (semantic
        memory). It runs more frequently than the default to keep the user model
        current and ensure the My Profile feedback loop has fresh data.

        Examples of derived facts:
          - "User prefers casual communication" (from linguistic formality < 0.3)
          - "User has expertise in Python" (from topic frequency + depth)
          - "Contact X is high priority" (from response time consistently < 1hr)
          - "User values work-life boundaries" (from cadence showing no work emails after 6pm)

        Interval: 1 hour (3600 seconds)
          - Keeps the user model current with recent activity
          - Ensures My Profile tab has fresh data for the feedback loop
        """
        while not self.shutdown_event.is_set():
            try:
                # Run inference across all signal profiles to extract semantic facts.
                # Offloaded to a thread to avoid blocking the async event loop —
                # run_all_inference is fully synchronous (SQLite queries + computation).
                await asyncio.to_thread(self.semantic_fact_inferrer.run_all_inference)
                logger.info("  SemanticFactInferrer: completed inference cycle")
            except Exception as e:
                logger.error("Semantic fact inferrer error: %s", e)

            await asyncio.sleep(3600)  # 1 hour

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

                logger.info("  RoutineDetector: detected %d routines, stored %d", len(routines), stored_count)

                # Publish routine detection events from async context (not from thread).
                # UserModelStore._emit_telemetry() fails silently when called from threads
                # because there's no running event loop, so we publish here instead.
                #
                # CRITICAL: We must publish events for EVERY stored routine, not just new ones.
                # Routines use UPSERT logic (INSERT OR REPLACE), so every detection cycle
                # updates existing routines and creates events for observability/auditing.
                if self.event_bus and self.event_bus.is_connected:
                    for routine in routines:
                        try:
                            await self.event_bus.publish(
                                "usermodel.routine.updated",
                                {
                                    "routine_name": routine["name"],
                                    "trigger": routine["trigger"],
                                    "steps_count": len(routine.get("steps", [])),
                                    "consistency_score": routine.get("consistency_score", 0.5),
                                    "times_observed": routine.get("times_observed", 0),
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                },
                                source="routine_detector",
                            )
                        except Exception as e:
                            logger.warning("  ⚠ Failed to publish routine event for '%s': %s", routine.get('name'), e)
                else:
                    # Event bus not connected — log the skip for visibility
                    if routines:
                        logger.warning("  ⚠ Skipping %d routine event publications (event bus not connected)", len(routines))

                # Detect workflows from last 30 days of event sequences
                workflows = await asyncio.to_thread(self.workflow_detector.detect_workflows, 30)

                # Store detected workflows to database
                workflow_stored = await asyncio.to_thread(self.workflow_detector.store_workflows, workflows)

                logger.info("  WorkflowDetector: detected %d workflows, stored %d", len(workflows), workflow_stored)

                # Publish workflow detection events from async context (not from thread).
                # UserModelStore._emit_telemetry() fails silently when called from threads
                # because there's no running event loop, so we publish here instead.
                #
                # CRITICAL: We must publish events for EVERY stored workflow, not just new ones.
                # Workflows use UPSERT logic (INSERT OR REPLACE), so every detection cycle
                # updates existing workflows and creates events for observability/auditing.
                if self.event_bus and self.event_bus.is_connected:
                    for workflow in workflows:
                        try:
                            await self.event_bus.publish(
                                "usermodel.workflow.updated",
                                {
                                    "workflow_name": workflow["name"],
                                "trigger_conditions_count": len(workflow.get("trigger_conditions", [])),
                                "steps_count": len(workflow.get("steps", [])),
                                "tools_count": len(workflow.get("tools_used", [])),
                                "success_rate": workflow.get("success_rate", 0.5),
                                "times_observed": workflow.get("times_observed", 0),
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                            },
                            source="workflow_detector",
                        )
                        except Exception as e:
                            logger.warning("  ⚠ Failed to publish workflow event for '%s': %s", workflow.get('name'), e)
                else:
                    # Event bus not connected — log the skip for visibility
                    if workflows:
                        logger.warning("  ⚠ Skipping %d workflow event publications (event bus not connected)", len(workflows))

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

                logger.info("  Next detection cycle in %s", retry_desc)

            except Exception as e:
                logger.error("Routine/workflow detector error: %s", e)
                # On error, retry in 1 hour to avoid tight error loops
                retry_seconds = 3600
                logger.info("  Next detection cycle in 1 hour (after error)")

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
                    logger.info(
                        "  BehavioralAccuracyTracker: inferred accuracy for %d accurate, %d inaccurate predictions",
                        stats['marked_accurate'],
                        stats['marked_inaccurate'],
                    )
            except Exception as e:
                logger.error("Behavioral accuracy tracker error: %s", e)

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
                    logger.info("  TaskCompletionDetector: auto-completed %d tasks from behavioral signals",
                                completed_count)
            except Exception as e:
                logger.error("Task completion detector error: %s", e)

            # 1800 seconds = 30 minutes. Tasks aren't typically completed within
            # minutes (unlike prediction validation), so we can run less frequently
            # while still maintaining responsive workflow detection.
            await asyncio.sleep(1800)  # 30 minutes

    async def _digest_delivery_loop(self):
        """Deliver batched notifications on a scheduled cadence.

        The notification manager accumulates low-priority notifications in an
        in-memory batch when the user is in "batched" mode. Without scheduled
        delivery, these notifications remain stuck in pending state forever.

        This loop delivers the digest at three scheduled times each day:
          - 09:00 (morning briefing)
          - 13:00 (midday update)
          - 18:00 (evening wrap-up)

        Each digest delivery:
          1. Calls notification_manager.get_digest() which returns all batched items
          2. Marks each notification as "delivered" in the database
          3. Marks any associated predictions as "surfaced"
          4. Clears the in-memory batch queue for the next cycle

        If a scheduled time is missed (e.g., system was off), the loop checks
        again on the next iteration. Notifications don't expire — they'll be
        delivered at the next available digest window.
        """
        # Scheduled delivery times (24-hour format, local time)
        digest_hours = [9, 13, 18]
        last_delivered_hour = None

        while not self.shutdown_event.is_set():
            try:
                now = datetime.now(timezone.utc).astimezone(ZoneInfo(self.user_tz))
                current_hour = now.hour

                # Check if we've reached a digest hour and haven't delivered yet this hour
                if current_hour in digest_hours and current_hour != last_delivered_hour:
                    # Deliver the batched digest
                    digest = await self.notification_manager.get_digest()
                    if digest:
                        logger.info("  DigestDelivery: delivered %d batched notifications", len(digest))

                    # Track that we've delivered for this hour to prevent duplicate deliveries
                    last_delivered_hour = current_hour

                # Reset tracking when we move to a different hour (digest or non-digest)
                # This allows the next digest hour to trigger delivery
                elif current_hour != last_delivered_hour and last_delivered_hour is not None:
                    last_delivered_hour = None

            except Exception as e:
                logger.error("Digest delivery error: %s", e)
                # Continue running even if delivery fails — we'll try again next cycle

            # Check every 5 minutes for the next digest window
            await asyncio.sleep(300)

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
                logger.info("       ✓ %s", connector.DISPLAY_NAME)
            except Exception as e:
                logger.error("       ✗ %s: %s", typedef.display_name, e)

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
        logger.info("Shutting down Life OS...")
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

        logger.info("Goodbye.")


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
