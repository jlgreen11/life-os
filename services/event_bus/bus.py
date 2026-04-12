"""
Life OS — Event Bus (NATS JetStream wrapper)

The nervous system of the entire application. Every service communicates
through events on the bus. This provides decoupling — add a new connector
or service without touching anything else.

Usage:
    bus = EventBus("nats://localhost:4222")
    await bus.connect()
    
    # Publish
    await bus.publish("email.received", payload)
    
    # Subscribe
    async def handler(event):
        print(f"Got: {event}")
    await bus.subscribe("email.*", handler)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

import nats
from nats.js.api import StreamConfig, ConsumerConfig, AckPolicy

logger = logging.getLogger(__name__)


class EventBus:
    """NATS JetStream event bus for Life OS.

    Acts as the central nervous system: every service (connectors, AI engine,
    storage) communicates exclusively through events on this bus. This enables
    full decoupling -- new services can be added by simply subscribing to the
    relevant event subjects without modifying existing code.

    All events live under the "lifeos." subject namespace and are persisted
    in the "LIFEOS" JetStream stream for durability and replay.
    """

    # Single stream that captures all Life OS events. Using one stream
    # simplifies management while still allowing fine-grained subscriptions
    # via subject filtering (e.g., "lifeos.email.*").
    STREAM_NAME = "LIFEOS"
    # Wildcard ">" captures all subjects under "lifeos." -- this is the
    # stream's subject filter, not a subscription pattern.
    SUBJECTS = "lifeos.>"

    def __init__(self, url: str = "nats://localhost:4222"):
        self.url = url
        # _nc: the raw NATS connection (None until connect() is called).
        self._nc: Optional[nats.NATS] = None
        # _js: the JetStream context derived from the NATS connection.
        # Provides publish/subscribe with at-least-once delivery guarantees.
        self._js = None
        # Track active subscriptions so they can be cleanly unsubscribed
        # during disconnect().
        self._subscriptions: list = []
        # Flag set when the client reconnects after a disconnect. Consumers
        # can poll this to detect connectivity blips (resets on read).
        self._reconnected_flag: bool = False

        # --- Throughput metrics ---
        # All counters live in a single asyncio event loop, so no locking
        # is needed — only one coroutine can mutate them at a time.
        self._publish_count: int = 0                  # Total publish() attempts (includes errors)
        self._publish_error_count: int = 0            # JetStream-level publish failures
        self._subscribe_count: int = 0                # Number of active subscriptions created
        self._last_publish_at: Optional[str] = None   # ISO-8601 UTC timestamp of most recent publish
        # Per-subject publish counts (full subject, e.g. "lifeos.email.received": 142).
        # Capped at 200 keys to prevent unbounded memory growth in long-running deployments.
        self._publish_by_subject: dict[str, int] = {}
        self._started_at: Optional[str] = None        # ISO-8601 UTC timestamp set when connect() succeeds

    async def _on_disconnect(self):
        """Called when the NATS client loses its connection.

        Logs a warning so operators can see connectivity blips in the logs.
        Without this callback, disconnections are completely silent.
        """
        logger.warning("NATS disconnected — event pipeline paused")

    async def _on_reconnect(self):
        """Called when the NATS client re-establishes a lost connection.

        Re-obtains the JetStream context (the old one may be stale) and sets
        the reconnected flag so callers can detect the blip.
        """
        logger.info("NATS reconnected — event pipeline resumed")
        # Re-obtain JetStream context after reconnection
        self._js = self._nc.jetstream()
        self._reconnected_flag = True

    async def _on_error(self, e):
        """Called for asynchronous NATS errors (e.g., slow consumer warnings).

        Logs the error so it surfaces in monitoring rather than being silently
        swallowed by the nats-py client.
        """
        logger.error("NATS async error: %s", e)

    @property
    def was_reconnected(self) -> bool:
        """True if the connection was lost and re-established since last check.

        Reading this property resets the flag, so it returns True only once
        per reconnection event. Useful for health-check endpoints that need
        to report recent connectivity issues.
        """
        val = self._reconnected_flag
        self._reconnected_flag = False
        return val

    async def connect(self):
        """Connect to NATS and ensure the JetStream stream exists."""
        # Establish the TCP connection to the NATS server with resilient
        # reconnection options. Without these, the client gives up after
        # 60 attempts (~2 minutes) and permanently disconnects.
        self._nc = await nats.connect(
            self.url,
            max_reconnect_attempts=-1,  # Never give up reconnecting
            reconnect_time_wait=5,  # 5 seconds between attempts
            disconnected_cb=self._on_disconnect,
            reconnected_cb=self._on_reconnect,
            error_cb=self._on_error,
        )
        # Obtain the JetStream context for persistent messaging (as opposed
        # to core NATS which is fire-and-forget).
        self._js = self._nc.jetstream()
        # Record when the bus came online so uptime can be reported in
        # get_metrics().  Set after a successful jetstream() call to avoid
        # counting time spent retrying a failed connect.
        self._started_at = datetime.now(timezone.utc).isoformat()

        # --- Idempotent stream creation ---
        # First, check if the stream already exists by looking up a subject.
        # If found, we skip creation (the stream is already configured).
        # If not found, we create it with our desired configuration.
        try:
            await self._js.find_stream_name_by_subject("lifeos.>")
        except Exception:
            # Stream does not exist yet -- create it with production defaults.
            await self._js.add_stream(
                StreamConfig(
                    name=self.STREAM_NAME,
                    # Capture ALL events under the lifeos namespace.
                    subjects=["lifeos.>"],
                    # "limits" retention: messages are kept until they hit the
                    # configured limits (count, size, or age), then oldest are
                    # discarded. This is appropriate for event log semantics.
                    retention="limits",
                    # Cap at 1M messages to prevent unbounded growth.
                    max_msgs=1_000_000,
                    # Cap at 1GB total storage.
                    max_bytes=1_073_741_824,  # 1GB
                    # Auto-expire messages older than 90 days.
                    max_age=60 * 60 * 24 * 90,  # 90 days in seconds
                    # "file" storage persists to disk (survives NATS restarts).
                    # Alternative is "memory" for ephemeral/test setups.
                    storage="file",
                    # Deduplication window: messages with the same Nats-Msg-Id
                    # header within 60 seconds are treated as duplicates.
                    duplicate_window=60,
                )
            )

    async def disconnect(self):
        """Gracefully disconnect.

        Unsubscribes all active subscriptions first to ensure no messages
        are delivered after the connection is closed, then closes the
        underlying NATS TCP connection.
        """
        for sub in self._subscriptions:
            await sub.unsubscribe()
        if self._nc:
            await self._nc.close()

    async def publish(self, event_type: str, payload: dict[str, Any],
                      source: str = "system", priority: str = "normal",
                      metadata: Optional[dict] = None) -> str:
        """
        Publish an event to the bus.
        
        Args:
            event_type: Dotted event type (e.g., "email.received")
            payload: Event data
            source: Which connector/service is publishing
            priority: Event priority level
            metadata: Optional metadata (contacts, domain, etc.)
            
        Returns:
            The event ID
        """
        # Generate a unique event ID (UUID4) for deduplication and tracing.
        event_id = str(uuid.uuid4())

        # --- Event envelope ---
        # Every event on the bus follows this standard envelope schema.
        # This ensures all subscribers can handle events uniformly regardless
        # of the payload structure.
        event = {
            "id": event_id,             # Unique identifier for this event instance
            "type": event_type,          # Dotted type (e.g., "email.received")
            "source": source,            # Originating service/connector name
            "timestamp": datetime.now(timezone.utc).isoformat(),  # ISO 8601 UTC
            "priority": priority,        # Routing hint for consumers
            "payload": payload,          # The actual event data (schema varies by type)
            "metadata": metadata or {},  # Optional: contact IDs, domain, tags, etc.
        }

        # Map the dotted event type to a NATS subject under the lifeos namespace.
        # e.g., "email.received" -> "lifeos.email.received"
        subject = f"lifeos.{event_type}"
        # Serialize the event envelope to JSON bytes for NATS transport.
        # json.dumps() raises TypeError for non-serializable values (caller
        # bug) — let that propagate before touching any counters.
        data = json.dumps(event).encode()

        # --- Throughput instrumentation ---
        # Increment counters *before* the actual NATS send so that attempted
        # publishes are counted even when the JetStream call fails.
        self._publish_count += 1
        self._publish_by_subject[subject] = self._publish_by_subject.get(subject, 0) + 1
        self._last_publish_at = datetime.now(timezone.utc).isoformat()

        # Keep the per-subject dict bounded at 200 keys.  When the cap is
        # exceeded, evict the subject with the lowest lifetime count.  In
        # practice this retains high-frequency subjects (email.received, etc.)
        # and drops rare one-offs (test runs, unknown event types).
        if len(self._publish_by_subject) > 200:
            least_used = min(self._publish_by_subject, key=self._publish_by_subject.get)
            del self._publish_by_subject[least_used]

        # JetStream publish provides at-least-once delivery: the message is
        # persisted to the stream before the publish ack is returned.
        try:
            await self._js.publish(subject, data)
        except Exception:
            # Track JetStream-level failures (connection lost, stream not
            # found, timeout, etc.) separately from total publish attempts so
            # callers can compute an error rate.
            self._publish_error_count += 1
            raise

        return event_id

    async def subscribe(
        self,
        pattern: str,
        handler: Callable[[dict], Coroutine],
        consumer_name: Optional[str] = None,
        durable: bool = True,
    ):
        """
        Subscribe to events matching a pattern.
        
        Args:
            pattern: Subject pattern (e.g., "email.*" or ">" for all)
            handler: Async callback that receives the event dict
            consumer_name: Name for durable consumer (auto-generated if None)
            durable: Whether the subscription survives restarts
        """
        # Prefix the pattern with "lifeos." to stay within our stream's subject space.
        subject = f"lifeos.{pattern}"
        # Auto-generate a deterministic consumer name from the pattern if not
        # provided. This ensures the same subscription pattern always maps to
        # the same durable consumer, surviving restarts without duplication.
        if consumer_name is None:
            consumer_name = f"consumer-{pattern.replace('.', '-').replace('*', 'all').replace('>', 'all')}"

        async def _wrapper(msg):
            """Internal message wrapper that handles deserialization, dispatch,
            acknowledgment, and error recovery for each delivered message."""
            try:
                # Deserialize the JSON event envelope from raw NATS bytes.
                event = json.loads(msg.data.decode())
                # Dispatch to the caller's async handler.
                await handler(event)
                # Explicit ACK tells JetStream this message was successfully
                # processed and should not be redelivered.
                await msg.ack()
            except Exception as e:
                logger.error("Event handler error for %s: %s", pattern, e, exc_info=True)
                # NAK (negative acknowledgment) with a 5-second delay triggers
                # automatic redelivery. This provides basic retry semantics
                # for transient failures (network blips, temporary DB locks).
                await msg.nak(delay=5)

        # --- Consumer configuration ---
        # durable_name: when set, JetStream remembers this consumer's position
        # across restarts. Set to None for ephemeral (test/dev) consumers.
        # ack_policy=EXPLICIT: messages must be explicitly ACK'd or NAK'd.
        # This prevents message loss -- unacknowledged messages are redelivered.
        config = ConsumerConfig(
            durable_name=consumer_name if durable else None,
            ack_policy=AckPolicy.EXPLICIT,
        )

        # Create the JetStream push subscription. The callback (_wrapper) is
        # invoked for each message matching the subject pattern.
        sub = await self._js.subscribe(
            subject,
            cb=_wrapper,
            config=config,
        )
        # Track the subscription handle for cleanup during disconnect().
        self._subscriptions.append(sub)
        # Count active subscriptions so get_metrics() can report them.
        self._subscribe_count += 1

    async def subscribe_all(self, handler: Callable[[dict], Coroutine],
                            consumer_name: str = "all-events"):
        """Subscribe to every event in the system.

        Convenience method that subscribes to the ">" wildcard, which matches
        all subjects under the lifeos namespace. Useful for event logging,
        metrics collection, or the event store persistence layer.
        """
        await self.subscribe(">", handler, consumer_name)

    async def request(self, event_type: str, payload: dict,
                      timeout: float = 5.0) -> Optional[dict]:
        """
        Publish an event and wait for a response (request-reply pattern).
        Useful for synchronous queries to services.

        This uses core NATS request-reply (not JetStream) for low-latency
        RPC-style communication. The responder must be online at call time.
        If no response arrives within the timeout, returns None rather than
        raising -- callers should handle the None case gracefully.
        """
        subject = f"lifeos.{event_type}"
        data = json.dumps(payload).encode()

        try:
            # Core NATS request(): publishes on the subject with an auto-
            # generated reply inbox, then waits for a single response.
            response = await self._nc.request(subject, data, timeout=timeout)
            # Deserialize the responder's JSON payload.
            return json.loads(response.data.decode())
        except Exception:
            # Timeout or connection error -- return None for graceful degradation.
            return None

    def get_metrics(self) -> dict:
        """Return a snapshot of event bus throughput and health metrics.

        All counters reflect the lifetime of this EventBus instance (since
        ``__init__`` was called).  Uptime is calculated from the moment
        ``connect()`` succeeded; it is 0 if the bus has never connected.

        Returns a dict with the following keys:

        - ``total_published``     — total publish() calls attempted (including errors)
        - ``total_errors``        — JetStream-level publish failures
        - ``active_subscriptions``— number of subscribe() calls that succeeded
        - ``last_publish_at``     — ISO-8601 UTC timestamp of the most recent publish, or None
        - ``started_at``          — ISO-8601 UTC timestamp when connect() first succeeded, or None
        - ``uptime_seconds``      — float seconds since connect() succeeded, 0 if not connected
        - ``top_subjects``        — dict of up to 20 highest-count subjects and their counts
        - ``error_rate``          — fraction of publish attempts that resulted in errors (0.0–1.0)
        - ``is_connected``        — current NATS connection state
        """
        now = datetime.now(timezone.utc)

        # Compute uptime from when the bus first connected.
        uptime_seconds = 0.0
        if self._started_at:
            started = datetime.fromisoformat(self._started_at)
            uptime_seconds = (now - started).total_seconds()

        # Sort by descending count and take the top 20 subjects.
        sorted_subjects = sorted(
            self._publish_by_subject.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_subjects = dict(sorted_subjects[:20])

        return {
            "total_published": self._publish_count,
            "total_errors": self._publish_error_count,
            "active_subscriptions": self._subscribe_count,
            "last_publish_at": self._last_publish_at,
            "started_at": self._started_at,
            "uptime_seconds": uptime_seconds,
            "top_subjects": top_subjects,
            # Avoid division by zero: denominator is at least 1.
            "error_rate": self._publish_error_count / max(self._publish_count, 1),
            "is_connected": self.is_connected,
        }

    @property
    def is_connected(self) -> bool:
        """Check if the NATS connection is alive.

        Returns True only when both the connection object exists AND the
        underlying TCP socket is still connected. Useful for health checks
        and reconnection logic in service supervisors.
        """
        return self._nc is not None and self._nc.is_connected
