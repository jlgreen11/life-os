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
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

import nats
from nats.js.api import StreamConfig, ConsumerConfig, AckPolicy


class EventBus:
    """NATS JetStream event bus for Life OS."""

    STREAM_NAME = "LIFEOS"
    SUBJECTS = "lifeos.>"

    def __init__(self, url: str = "nats://localhost:4222"):
        self.url = url
        self._nc: Optional[nats.NATS] = None
        self._js = None
        self._subscriptions: list = []

    async def connect(self):
        """Connect to NATS and ensure the JetStream stream exists."""
        self._nc = await nats.connect(self.url)
        self._js = self._nc.jetstream()

        # Create or update the stream
        try:
            await self._js.find_stream_name_by_subject("lifeos.>")
        except Exception:
            await self._js.add_stream(
                StreamConfig(
                    name=self.STREAM_NAME,
                    subjects=["lifeos.>"],
                    retention="limits",
                    max_msgs=1_000_000,
                    max_bytes=1_073_741_824,  # 1GB
                    max_age=60 * 60 * 24 * 90,  # 90 days
                    storage="file",
                    duplicate_window=60,
                )
            )

    async def disconnect(self):
        """Gracefully disconnect."""
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
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "type": event_type,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": priority,
            "payload": payload,
            "metadata": metadata or {},
        }

        subject = f"lifeos.{event_type}"
        data = json.dumps(event).encode()
        await self._js.publish(subject, data)

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
        subject = f"lifeos.{pattern}"
        if consumer_name is None:
            consumer_name = f"consumer-{pattern.replace('.', '-').replace('*', 'all').replace('>', 'all')}"

        async def _wrapper(msg):
            try:
                event = json.loads(msg.data.decode())
                await handler(event)
                await msg.ack()
            except Exception as e:
                print(f"Event handler error for {pattern}: {e}")
                # NAK with delay for retry
                await msg.nak(delay=5)

        config = ConsumerConfig(
            durable_name=consumer_name if durable else None,
            ack_policy=AckPolicy.EXPLICIT,
        )

        sub = await self._js.subscribe(
            subject,
            cb=_wrapper,
            config=config,
        )
        self._subscriptions.append(sub)

    async def subscribe_all(self, handler: Callable[[dict], Coroutine],
                            consumer_name: str = "all-events"):
        """Subscribe to every event in the system."""
        await self.subscribe(">", handler, consumer_name)

    async def request(self, event_type: str, payload: dict,
                      timeout: float = 5.0) -> Optional[dict]:
        """
        Publish an event and wait for a response (request-reply pattern).
        Useful for synchronous queries to services.
        """
        subject = f"lifeos.{event_type}"
        data = json.dumps(payload).encode()

        try:
            response = await self._nc.request(subject, data, timeout=timeout)
            return json.loads(response.data.decode())
        except Exception:
            return None

    @property
    def is_connected(self) -> bool:
        return self._nc is not None and self._nc.is_connected
