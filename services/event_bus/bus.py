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
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

import nats
from nats.js.api import StreamConfig, ConsumerConfig, AckPolicy

logger = logging.getLogger(__name__)

# Rolling window (seconds) for computing per-subject throughput rates.
# Bus throughput is bursty (connector syncs run every N minutes), so a
# 60-second window gives a stable "events per minute" reading without
# being so long that it masks a sudden stall.
DEFAULT_METRICS_WINDOW_SECONDS = 60
# Per-subject deque size cap. At sustained ~16 msg/sec/subject this is
# >60 seconds of headroom; the bound prevents unbounded memory growth
# if a single subject ever floods.
METRICS_DEQUE_MAXLEN = 1000


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

    def __init__(self, url: str = "nats://localhost:4222",
                 metrics_window_seconds: int = DEFAULT_METRICS_WINDOW_SECONDS):
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

        # --- Observability counters ---
        # The data quality analyzer surfaces stale last_event timestamps but
        # cannot tell "connector not producing" apart from "bus dropping
        # messages". These per-subject counters give operators that signal.
        self.metrics_window_seconds: int = metrics_window_seconds
        # Lifetime totals — monotonically increasing. Keyed by full subject
        # (e.g. "lifeos.email.received").
        self.publish_total: dict[str, int] = defaultdict(int)
        self.publish_errors: dict[str, int] = defaultdict(int)
        self.consume_total: dict[str, int] = defaultdict(int)
        # Most recent activity timestamp per subject (UTC, ISO 8601 string).
        self.last_publish_at: dict[str, str] = {}
        self.last_consume_at: dict[str, str] = {}
        # Rolling per-subject timestamps (monotonic seconds) for rate
        # computation. Bounded to prevent memory growth under sustained load.
        self._publish_window: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=METRICS_DEQUE_MAXLEN)
        )
        self._consume_window: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=METRICS_DEQUE_MAXLEN)
        )

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
        logger.info(
            "EventBus metrics enabled (window=%ds, deque_maxlen=%d) — "
            "call get_metrics() for per-subject throughput",
            self.metrics_window_seconds,
            METRICS_DEQUE_MAXLEN,
        )

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
        # Serialization happens before any counters are bumped so that a
        # bad payload (non-JSON-serializable) raises before observers see
        # a phantom publish that never reached the stream.
        data = json.dumps(event).encode()
        try:
            # JetStream publish provides at-least-once delivery: the message
            # is persisted to the stream before the publish ack is returned.
            await self._js.publish(subject, data)
        except Exception:
            # Count the failure (so operators can see error rates) and
            # re-raise: callers decide whether to retry. We do NOT swallow
            # this — fail-open at the service level, not at the bus level.
            self.publish_errors[subject] += 1
            raise

        # Successful publish — update counters and rolling window.
        self.publish_total[subject] += 1
        self.last_publish_at[subject] = datetime.now(timezone.utc).isoformat()
        self._publish_window[subject].append(time.monotonic())

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
            # Bump consume counters before dispatch so metrics reflect "delivered
            # to subscriber" rather than "successfully handled" — a stuck or
            # crashing handler should still show up as bus traffic.
            msg_subject = getattr(msg, "subject", subject)
            self.consume_total[msg_subject] += 1
            self.last_consume_at[msg_subject] = datetime.now(timezone.utc).isoformat()
            self._consume_window[msg_subject].append(time.monotonic())
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

    def _rate_per_minute(self, window: deque, now: float) -> float:
        """Compute messages-per-minute over the configured rolling window.

        Counts entries in the deque whose timestamp falls within
        ``metrics_window_seconds`` of ``now`` and scales to a per-minute rate.
        Returns 0.0 when the window is empty.
        """
        if not window:
            return 0.0
        cutoff = now - self.metrics_window_seconds
        # Walk from the right (most recent) — once we drop below the cutoff,
        # everything older is also out of window. O(n) worst-case but bounded
        # by METRICS_DEQUE_MAXLEN.
        count = 0
        for ts in reversed(window):
            if ts < cutoff:
                break
            count += 1
        # Scale window-count to per-minute equivalent so dashboards can show
        # a consistent unit regardless of the configured window size.
        return count * (60.0 / self.metrics_window_seconds)

    async def get_metrics(self) -> dict[str, Any]:
        """Return per-subject throughput counters and connection state.

        Shape::

            {
                "connected": bool,
                "window_seconds": int,
                "subjects": {
                    "lifeos.email.received": {
                        "publish_total": int,
                        "publish_errors": int,
                        "consume_total": int,
                        "last_publish_at": "2026-05-17T10:00:00+00:00" | None,
                        "last_consume_at": str | None,
                        "publishes_per_minute": float,
                        "consumes_per_minute": float,
                    },
                    ...
                },
                "totals": {
                    "publish_total": int,
                    "publish_errors": int,
                    "consume_total": int,
                },
            }

        Designed for the ``/health`` endpoint: a connector showing
        ``last_publish_at`` from hours ago + ``publishes_per_minute == 0``
        is producing nothing; a non-zero error count is the bus rejecting
        messages.
        """
        now = time.monotonic()
        # Union of subjects we've ever seen on either side — a subject with
        # publishes but no consumers (or vice versa) should still appear.
        subjects = (
            set(self.publish_total)
            | set(self.consume_total)
            | set(self.publish_errors)
        )

        per_subject: dict[str, dict[str, Any]] = {}
        for subj in sorted(subjects):
            per_subject[subj] = {
                "publish_total": self.publish_total.get(subj, 0),
                "publish_errors": self.publish_errors.get(subj, 0),
                "consume_total": self.consume_total.get(subj, 0),
                "last_publish_at": self.last_publish_at.get(subj),
                "last_consume_at": self.last_consume_at.get(subj),
                "publishes_per_minute": self._rate_per_minute(
                    self._publish_window.get(subj, deque()), now
                ),
                "consumes_per_minute": self._rate_per_minute(
                    self._consume_window.get(subj, deque()), now
                ),
            }

        return {
            "connected": self.is_connected,
            "window_seconds": self.metrics_window_seconds,
            "subjects": per_subject,
            "totals": {
                "publish_total": sum(self.publish_total.values()),
                "publish_errors": sum(self.publish_errors.values()),
                "consume_total": sum(self.consume_total.values()),
            },
        }

    @property
    def is_connected(self) -> bool:
        """Check if the NATS connection is alive.

        Returns True only when both the connection object exists AND the
        underlying TCP socket is still connected. Useful for health checks
        and reconnection logic in service supervisors.
        """
        return self._nc is not None and self._nc.is_connected
