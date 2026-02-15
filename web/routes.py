"""
Life OS — API Route Registration

All REST API routes, WebSocket handler, and the HTML UI.
Called by the app factory to register routes on the FastAPI instance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from web.schemas import (
    CommandRequest,
    ContextBatchRequest,
    ContextEventRequest,
    DraftRequest,
    FeedbackRequest,
    PreferenceUpdate,
    RuleCreateRequest,
    SearchRequest,
    TaskCreateRequest,
    TaskUpdateRequest,
)
from web.websocket import ws_manager


def register_routes(app: FastAPI, life_os) -> None:
    """Register all API routes on the FastAPI app.

    Routes are organized by domain:
        Health & Status    — /health, /api/status
        Command Bar        — /api/command  (NLP-like routing)
        Briefing           — /api/briefing
        Search             — /api/search
        Tasks              — /api/tasks (CRUD)
        Notifications      — /api/notifications (lifecycle)
        Draft Messages     — /api/draft
        Rules              — /api/rules (automation engine)
        User Model         — /api/user-model (facts, mood)
        Preferences        — /api/preferences
        Feedback           — /api/feedback
        Events             — /api/events (debug/inspection)
        Connectors         — /api/connectors, /api/browser/*
        WebSocket          — /ws (real-time push)
        Web UI             — / (HTML dashboard)

    All handlers access Life OS services through the ``life_os`` closure
    variable captured here, rather than through ``app.state``, keeping the
    route code concise.
    """

    # -------------------------------------------------------------------
    # Health & Status
    # -------------------------------------------------------------------

    @app.get("/health")
    async def health():
        """Aggregate health check — polls every connector and returns a
        unified status payload used by the dashboard status bar."""
        connectors = []
        for c in life_os.connectors:
            try:
                status = await c.health_check()
                connectors.append(status)
            except Exception as e:
                # Individual connector failures should not break the overall
                # health endpoint; report the error inline instead.
                connectors.append({"connector": c.CONNECTOR_ID, "status": "error", "details": str(e)})

        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_bus": life_os.event_bus.is_connected,
            "events_stored": life_os.event_store.get_event_count(),
            "vector_store": life_os.vector_store.get_stats(),
            "connectors": connectors,
        }

    @app.get("/api/status")
    async def status():
        return {
            "event_count": life_os.event_store.get_event_count(),
            "vector_store": life_os.vector_store.get_stats(),
            "user_model": life_os.signal_extractor.get_user_summary(),
            "notification_stats": life_os.notification_manager.get_stats(),
            "feedback_summary": life_os.feedback_collector.get_feedback_summary(),
        }

    # -------------------------------------------------------------------
    # Command Bar
    # -------------------------------------------------------------------

    @app.post("/api/command")
    async def command(req: CommandRequest):
        """Unified command bar — routes natural-language input to the right service.

        NLP-like routing:  The command text is matched against simple keyword
        prefixes to determine intent.  This avoids a full NLP pipeline for the
        most common actions while still falling back to the AI engine for
        anything unrecognized.

        Routing rules (evaluated top to bottom):
            "search ..." / "find ..."  ->  vector store semantic search
            "task ..." / "todo ..."    ->  task manager quick-capture
            "briefing"                 ->  AI-generated morning briefing
            "draft ..."               ->  AI-generated message draft
            (anything else)            ->  free-form AI search/answer
        """
        text = req.text.strip()
        if not text:
            raise HTTPException(400, "Empty command")

        lower = text.lower()

        # --- Intent: search / find ---
        if lower.startswith("search ") or lower.startswith("find "):
            query = text.split(" ", 1)[1]
            results = life_os.vector_store.search(query, limit=10)
            return {"type": "search_results", "results": results}

        # --- Intent: quick task creation ---
        elif lower.startswith("task ") or lower.startswith("todo "):
            title = text.split(" ", 1)[1]
            task_id = await life_os.task_manager.create_task(title=title)
            return {"type": "task_created", "task_id": task_id}

        # --- Intent: generate briefing ---
        elif lower == "briefing" or lower == "morning briefing":
            briefing = await life_os.ai_engine.generate_briefing()
            return {"type": "briefing", "content": briefing}

        # --- Intent: draft a message ---
        elif lower.startswith("draft "):
            context = text.split(" ", 1)[1]
            draft = await life_os.ai_engine.draft_reply(
                contact_id=None, channel="email",
                incoming_message=context,
            )
            return {"type": "draft", "content": draft}

        # --- Fallback: pass to AI engine for open-ended search/response ---
        else:
            response = await life_os.ai_engine.search_life(text)
            return {"type": "ai_response", "content": response}

    # -------------------------------------------------------------------
    # Briefing
    # -------------------------------------------------------------------

    @app.get("/api/briefing")
    async def get_briefing():
        briefing = await life_os.ai_engine.generate_briefing()
        return {"briefing": briefing, "generated_at": datetime.now(timezone.utc).isoformat()}

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------

    @app.post("/api/search")
    async def search(req: SearchRequest):
        results = life_os.vector_store.search(
            req.query, limit=req.limit, filter_metadata=req.filters
        )
        return {"query": req.query, "results": results, "count": len(results)}

    # -------------------------------------------------------------------
    # Tasks
    # -------------------------------------------------------------------

    @app.get("/api/tasks")
    async def list_tasks(status: str = "pending", limit: int = 50):
        tasks = life_os.task_manager.get_tasks(status=status, limit=limit)
        return {"tasks": tasks, "count": len(tasks)}

    @app.post("/api/tasks")
    async def create_task(req: TaskCreateRequest):
        task_id = await life_os.task_manager.create_task(
            title=req.title,
            description=req.description,
            domain=req.domain,
            priority=req.priority,
            due_date=req.due_date,
        )
        return {"task_id": task_id}

    @app.patch("/api/tasks/{task_id}")
    async def update_task(task_id: str, req: TaskUpdateRequest):
        life_os.task_manager.update_task(task_id, req.dict(exclude_none=True))
        return {"status": "updated"}

    @app.post("/api/tasks/{task_id}/complete")
    async def complete_task(task_id: str):
        life_os.task_manager.complete_task(task_id)
        return {"status": "completed"}

    # -------------------------------------------------------------------
    # Notifications
    # -------------------------------------------------------------------

    @app.get("/api/notifications")
    async def list_notifications(limit: int = 50):
        notifications = life_os.notification_manager.get_pending(limit=limit)
        return {"notifications": notifications}

    @app.post("/api/notifications/{notif_id}/read")
    async def mark_read(notif_id: str):
        await life_os.notification_manager.mark_read(notif_id)
        return {"status": "read"}

    @app.post("/api/notifications/{notif_id}/dismiss")
    async def dismiss_notification(notif_id: str):
        await life_os.notification_manager.dismiss(notif_id)
        return {"status": "dismissed"}

    @app.post("/api/notifications/{notif_id}/act")
    async def act_on_notification(notif_id: str):
        await life_os.notification_manager.mark_acted_on(notif_id)
        return {"status": "acted_on"}

    @app.get("/api/notifications/digest")
    async def get_digest():
        digest = await life_os.notification_manager.get_digest()
        return {"digest": digest}

    # -------------------------------------------------------------------
    # Draft Messages
    # -------------------------------------------------------------------

    @app.post("/api/draft")
    async def draft_message(req: DraftRequest):
        draft = await life_os.ai_engine.draft_reply(
            contact_id=req.contact_id,
            channel=req.channel,
            incoming_message=req.incoming_message,
        )
        return {"draft": draft}

    # -------------------------------------------------------------------
    # Rules
    # -------------------------------------------------------------------

    @app.get("/api/rules")
    async def list_rules():
        rules = life_os.rules_engine.get_all_rules()
        return {"rules": rules}

    @app.post("/api/rules")
    async def create_rule(req: RuleCreateRequest):
        rule_id = life_os.rules_engine.add_rule(
            name=req.name,
            trigger_event=req.trigger_event,
            conditions=req.conditions,
            actions=req.actions,
        )
        return {"rule_id": rule_id}

    @app.delete("/api/rules/{rule_id}")
    async def delete_rule(rule_id: str):
        life_os.rules_engine.remove_rule(rule_id)
        return {"status": "deactivated"}

    # -------------------------------------------------------------------
    # User Model
    # -------------------------------------------------------------------

    @app.get("/api/user-model")
    async def get_user_model():
        """Return the full user model summary from the signal extractor."""
        return life_os.signal_extractor.get_user_summary()

    @app.get("/api/user-model/facts")
    async def get_facts(min_confidence: float = 0.0):
        """Return semantic facts, optionally filtered by minimum confidence."""
        facts = life_os.user_model_store.get_semantic_facts(min_confidence=min_confidence)
        return {"facts": facts}

    @app.delete("/api/user-model/facts/{key}")
    async def delete_fact(key: str):
        """Delete a single semantic fact by key (user correction flow)."""
        with life_os.db.get_connection("user_model") as conn:
            conn.execute("DELETE FROM semantic_facts WHERE key = ?", (key,))
        return {"status": "deleted"}

    @app.get("/api/user-model/mood")
    async def get_mood():
        """Return the current inferred mood state.

        Handles both Pydantic models (with ``.dict()``) and plain dataclass-like
        objects by falling back to manual attribute access.
        """
        mood = life_os.signal_extractor.get_current_mood()
        return {
            "mood": mood.dict() if hasattr(mood, "dict") else {
                "energy_level": mood.energy_level,
                "stress_level": mood.stress_level,
                "social_battery": mood.social_battery,
                "cognitive_load": mood.cognitive_load,
                "emotional_valence": mood.emotional_valence,
                "confidence": mood.confidence,
                "trend": mood.trend,
            }
        }

    # -------------------------------------------------------------------
    # Preferences
    # -------------------------------------------------------------------

    @app.get("/api/preferences")
    async def get_preferences():
        with life_os.db.get_connection("preferences") as conn:
            rows = conn.execute("SELECT key, value, set_by, updated_at FROM user_preferences").fetchall()
            return {"preferences": [dict(r) for r in rows]}

    @app.put("/api/preferences")
    async def update_preference(req: PreferenceUpdate):
        with life_os.db.get_connection("preferences") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO user_preferences (key, value, set_by, updated_at)
                   VALUES (?, ?, 'user', ?)""",
                (req.key, json.dumps(req.value) if not isinstance(req.value, str) else req.value,
                 datetime.now(timezone.utc).isoformat()),
            )
        return {"status": "updated"}

    # -------------------------------------------------------------------
    # Feedback
    # -------------------------------------------------------------------

    @app.post("/api/feedback")
    async def submit_feedback(req: FeedbackRequest):
        await life_os.feedback_collector.process_explicit_feedback(req.message)
        return {"status": "received"}

    # -------------------------------------------------------------------
    # Events (debug/inspection)
    # -------------------------------------------------------------------

    @app.get("/api/events")
    async def list_events(event_type: Optional[str] = None,
                          source: Optional[str] = None,
                          since: Optional[str] = None,
                          limit: int = 50):
        events = life_os.event_store.get_events(
            event_type=event_type, source=source, since=since, limit=limit,
        )
        return {"events": events, "count": len(events)}

    # -------------------------------------------------------------------
    # Connectors
    # -------------------------------------------------------------------

    @app.get("/api/connectors")
    async def list_connectors():
        connectors = []
        for c in life_os.connectors:
            try:
                health = await c.health_check()
            except Exception:
                health = {"status": "error"}
            connectors.append({
                "id": c.CONNECTOR_ID,
                "name": c.DISPLAY_NAME,
                "health": health,
            })
        return {"connectors": connectors}

    @app.get("/api/browser/status")
    async def browser_status():
        return life_os.browser_orchestrator.get_status()

    @app.get("/api/browser/vault")
    async def browser_vault_sites():
        return {"sites": life_os.browser_orchestrator.get_vault_sites()}

    # -------------------------------------------------------------------
    # Context API (iOS / Mobile Device Context Ingestion)
    # -------------------------------------------------------------------

    @app.post("/api/context/event")
    async def submit_context_event(req: ContextEventRequest):
        """Ingest a single context event from a mobile device."""
        ts = req.timestamp or datetime.now(timezone.utc).isoformat()

        # Map context event type to internal event type
        event_type_map = {
            "context.location": "location.changed",
            "context.device_nearby": "home.device.state_changed",
            "context.time": "system.user.command",
            "context.background_refresh": "system.connector.sync_complete",
            "context.background_processing": "system.connector.sync_complete",
        }

        internal_type = event_type_map.get(req.type, "system.user.command")

        event = {
            "type": internal_type,
            "source": req.source,
            "timestamp": ts,
            "priority": "silent",
            "payload": req.payload.dict(exclude_none=True),
            "metadata": {
                "domain": "context",
                "mobile_event_type": req.type,
                **(req.metadata.dict(exclude_none=True) if req.metadata else {}),
            },
        }

        # Store the event
        event_id = life_os.event_store.store_event(event)

        # Publish to event bus for signal extraction
        try:
            await life_os.event_bus.publish(f"lifeos.context.{req.type}", event)
        except Exception:
            pass  # Event bus may not be connected

        # If location event, update places database
        if req.type == "context.location" and req.payload.latitude is not None:
            try:
                _update_place_from_context(life_os, req.payload)
            except Exception:
                pass

        # If device event, try to correlate with contacts
        if req.type == "context.device_nearby" and req.payload.device_name:
            try:
                _correlate_device_with_contact(life_os, req.payload)
            except Exception:
                pass

        return {"status": "received", "event_id": event_id}

    @app.post("/api/context/batch")
    async def submit_context_batch(req: ContextBatchRequest):
        """Ingest a batch of context events from a mobile device."""
        event_ids = []
        for event_req in req.events:
            ts = event_req.timestamp or datetime.now(timezone.utc).isoformat()
            event = {
                "type": "system.user.command",
                "source": event_req.source,
                "timestamp": ts,
                "priority": "silent",
                "payload": event_req.payload.dict(exclude_none=True),
                "metadata": {
                    "domain": "context",
                    "mobile_event_type": event_req.type,
                    **(event_req.metadata.dict(exclude_none=True) if event_req.metadata else {}),
                },
            }
            event_id = life_os.event_store.store_event(event)
            event_ids.append(event_id)

            # Publish to event bus
            try:
                await life_os.event_bus.publish(f"lifeos.context.{event_req.type}", event)
            except Exception:
                pass

        return {"status": "received", "count": len(event_ids), "event_ids": event_ids}

    @app.get("/api/context/summary")
    async def get_context_summary():
        """Get a summary of recently collected context data."""
        recent_events = life_os.event_store.get_events(
            source="ios_app", limit=100
        )

        # Aggregate context data
        locations = []
        devices = []
        for evt in recent_events:
            payload = evt.get("payload", {})
            meta = evt.get("metadata", {})
            mobile_type = meta.get("mobile_event_type", "")

            if mobile_type == "context.location" and payload.get("latitude"):
                locations.append({
                    "place": payload.get("place_name", "Unknown"),
                    "lat": payload.get("latitude"),
                    "lon": payload.get("longitude"),
                    "timestamp": evt.get("timestamp"),
                })
            elif mobile_type == "context.device_nearby":
                devices.append({
                    "name": payload.get("device_name", "Unknown"),
                    "type": payload.get("device_type"),
                    "signal": payload.get("signal_strength"),
                    "timestamp": evt.get("timestamp"),
                })

        # Unique locations and devices
        unique_places = list({l["place"] for l in locations if l["place"] != "Unknown"})
        unique_devices = list({d["name"] for d in devices if d["name"] != "Unknown"})

        return {
            "type": "context_summary",
            "content": f"Context: {len(locations)} location updates, {len(devices)} device sightings. "
                       f"Places: {', '.join(unique_places) or 'none tracked'}. "
                       f"Devices: {', '.join(unique_devices) or 'none detected'}.",
            "locations": locations[-10:],  # Last 10
            "devices": devices[-10:],
            "unique_places": unique_places,
            "unique_devices": unique_devices,
        }

    @app.get("/api/context/places")
    async def get_context_places():
        """Get learned places from context data."""
        with life_os.db.get_connection("entities") as conn:
            rows = conn.execute(
                "SELECT * FROM places ORDER BY visit_count DESC LIMIT 50"
            ).fetchall()
            return {"places": [dict(r) for r in rows]}

    def _update_place_from_context(life_os_instance, payload):
        """Update or create a place from mobile context location data."""
        if not payload.latitude or not payload.longitude:
            return

        place_name = payload.place_name or "Unknown"
        with life_os_instance.db.get_connection("entities") as conn:
            # Find nearby existing place (within ~100m)
            existing = conn.execute(
                """SELECT id, name, visit_count FROM places
                   WHERE ABS(latitude - ?) < 0.001 AND ABS(longitude - ?) < 0.001
                   LIMIT 1""",
                (payload.latitude, payload.longitude),
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE places SET visit_count = visit_count + 1, updated_at = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), existing["id"]),
                )
            else:
                import uuid
                conn.execute(
                    """INSERT INTO places (id, name, latitude, longitude, place_type, visit_count, created_at)
                       VALUES (?, ?, ?, ?, ?, 1, ?)""",
                    (str(uuid.uuid4()), place_name, payload.latitude, payload.longitude,
                     payload.place_type or "unknown", datetime.now(timezone.utc).isoformat()),
                )

    def _correlate_device_with_contact(life_os_instance, payload):
        """Try to match a nearby device to a known contact."""
        if not payload.device_name:
            return

        device_name = payload.device_name.lower()
        with life_os_instance.db.get_connection("entities") as conn:
            # Check if any contact has an alias matching the device name
            contacts = conn.execute(
                "SELECT id, name, aliases FROM contacts"
            ).fetchall()

            for contact in contacts:
                name_lower = contact["name"].lower()
                if name_lower in device_name or device_name in name_lower:
                    # Found a potential match - store as a context observation
                    life_os_instance.event_store.store_event({
                        "type": "system.ai.suggestion",
                        "source": "context_engine",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "priority": "silent",
                        "payload": {
                            "suggestion_type": "contact_nearby",
                            "contact_id": contact["id"],
                            "contact_name": contact["name"],
                            "device_name": payload.device_name,
                            "signal_strength": payload.signal_strength,
                        },
                    })
                    break

    # -------------------------------------------------------------------
    # WebSocket
    # -------------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time push updates.

        Connection lifecycle:
        1. Client connects -> ``ws_manager.connect`` accepts the handshake and
           adds the socket to the active connections list.
        2. The handler enters an infinite receive loop, keeping the connection
           alive.  Incoming messages can carry commands (type: "command"), but
           this is currently a placeholder for future client-to-server messages.
        3. When the client disconnects, ``WebSocketDisconnect`` is raised and
           the connection is removed from the manager's active list.

        Meanwhile, server-side services can call ``ws_manager.broadcast(...)``
        at any time to push notifications/events to all connected clients.
        """
        await ws_manager.connect(websocket)
        try:
            while True:
                # Block until the client sends a message (keeps the connection alive).
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "command":
                        pass  # Placeholder for future client-to-server commands.
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            # Client closed the connection — clean up the active connections list.
            ws_manager.disconnect(websocket)

    # -------------------------------------------------------------------
    # Web UI
    # -------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the single-page HTML dashboard.

        The template is imported lazily (inside the handler) to avoid circular
        imports at module load time and to keep the template module independent
        of the FastAPI app.
        """
        from web.template import HTML_TEMPLATE
        return HTML_TEMPLATE
