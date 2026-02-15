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
    ConnectorConfigRequest,
    ContextBatchRequest,
    ContextEventRequest,
    DraftRequest,
    FeedbackRequest,
    PreferenceUpdate,
    RuleCreateRequest,
    SearchRequest,
    SetupSubmitRequest,
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

        # --- Intent detection (also used for telemetry) ---
        if lower.startswith("search ") or lower.startswith("find "):
            command_type = "search"
        elif lower.startswith("task ") or lower.startswith("todo "):
            command_type = "task"
        elif lower == "briefing" or lower == "morning briefing":
            command_type = "briefing"
        elif lower.startswith("draft "):
            command_type = "draft"
        else:
            command_type = "ai_query"

        # Publish command telemetry
        if life_os.event_bus.is_connected:
            await life_os.event_bus.publish(
                "system.user.command",
                {
                    "command_type": command_type,
                    "text_length": len(text),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                source="web_api",
            )

        if command_type == "search":
            query = text.split(" ", 1)[1]
            results = life_os.vector_store.search(query, limit=10)
            return {"type": "search_results", "results": results}

        # --- Intent: quick task creation ---
        elif command_type == "task":
            title = text.split(" ", 1)[1]
            task_id = await life_os.task_manager.create_task(title=title)
            return {"type": "task_created", "task_id": task_id}

        # --- Intent: generate briefing ---
        elif command_type == "briefing":
            briefing = await life_os.ai_engine.generate_briefing()
            return {"type": "briefing", "content": briefing}

        # --- Intent: draft a message ---
        elif command_type == "draft":
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
        await life_os.task_manager.update_task(task_id, **req.dict(exclude_none=True))
        return {"status": "updated"}

    @app.post("/api/tasks/{task_id}/complete")
    async def complete_task(task_id: str):
        await life_os.task_manager.complete_task(task_id)
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
        rule_id = await life_os.rules_engine.add_rule(
            name=req.name,
            trigger_event=req.trigger_event,
            conditions=req.conditions,
            actions=req.actions,
        )
        return {"rule_id": rule_id}

    @app.delete("/api/rules/{rule_id}")
    async def delete_rule(rule_id: str):
        await life_os.rules_engine.remove_rule(rule_id)
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
        now = datetime.now(timezone.utc).isoformat()
        serialized_value = json.dumps(req.value) if not isinstance(req.value, str) else req.value

        with life_os.db.get_connection("preferences") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO user_preferences (key, value, set_by, updated_at)
                   VALUES (?, ?, 'user', ?)""",
                (req.key, serialized_value, now),
            )

        # Publish preference update telemetry
        if life_os.event_bus.is_connected:
            await life_os.event_bus.publish(
                "system.preference.updated",
                {
                    "key": req.key,
                    "updated_at": now,
                },
                source="web_api",
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
    # Admin — Connector Management
    # -------------------------------------------------------------------

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_page():
        from web.admin_template import ADMIN_HTML_TEMPLATE
        return ADMIN_HTML_TEMPLATE

    @app.get("/api/admin/connectors/registry")
    async def admin_connector_registry():
        """Return all connector type definitions with config schemas."""
        from connectors.registry import CONNECTOR_REGISTRY
        from dataclasses import asdict
        registry = {}
        for cid, typedef in CONNECTOR_REGISTRY.items():
            entry = asdict(typedef)
            # Remove internal fields not needed by frontend
            entry.pop("module_path", None)
            entry.pop("class_name", None)
            registry[cid] = entry
        return {"registry": registry}

    @app.get("/api/admin/connectors")
    async def admin_list_connectors():
        """Return all connectors with status, health, and masked config."""
        from connectors.registry import CONNECTOR_REGISTRY
        connectors = []
        for cid, typedef in CONNECTOR_REGISTRY.items():
            status = life_os.get_connector_status(cid)
            try:
                config = life_os.get_connector_config(cid)
            except Exception:
                config = {}
            connectors.append({
                "connector_id": cid,
                "display_name": typedef.display_name,
                "description": typedef.description,
                "category": typedef.category,
                "status": status,
                "config": config,
            })
        return {"connectors": connectors}

    @app.put("/api/admin/connectors/{connector_id}/config")
    async def admin_save_config(connector_id: str, req: ConnectorConfigRequest):
        """Save connector configuration (preserves unchanged password fields)."""
        try:
            life_os.save_connector_config(connector_id, req.config)
            return {"status": "saved"}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/admin/connectors/{connector_id}/test")
    async def admin_test_connector(connector_id: str, req: ConnectorConfigRequest):
        """Test connector credentials without saving."""
        try:
            result = await life_os.test_connector(connector_id, config=req.config)
            return result
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            return {"success": False, "detail": str(e)}

    @app.post("/api/admin/connectors/{connector_id}/enable")
    async def admin_enable_connector(connector_id: str):
        """Start a connector at runtime."""
        try:
            result = await life_os.enable_connector(connector_id)
            return result
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"Failed to start connector: {e}")

    @app.post("/api/admin/connectors/{connector_id}/disable")
    async def admin_disable_connector(connector_id: str):
        """Stop a running connector."""
        try:
            result = await life_os.disable_connector(connector_id)
            return result
        except Exception as e:
            raise HTTPException(500, f"Failed to stop connector: {e}")

    # -------------------------------------------------------------------
    # Admin — Google OAuth
    # -------------------------------------------------------------------

    @app.get("/api/admin/connectors/google/auth")
    async def admin_google_auth():
        """Start the Google OAuth flow — opens browser for user approval."""
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow

            SCOPES = [
                "https://www.googleapis.com/auth/gmail.modify",
                "https://www.googleapis.com/auth/calendar",
                "https://www.googleapis.com/auth/contacts.readonly",
            ]

            # Get config for file paths
            config = life_os.get_connector_config("google")
            credentials_file = config.get("credentials_file", "data/google_credentials.json")
            token_file = config.get("token_file", "data/google_token.json")

            import os
            if not os.path.exists(credentials_file):
                raise HTTPException(400,
                    f"Credentials file not found at {credentials_file}. "
                    "Download it from Google Cloud Console and place it there.")

            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0, open_browser=True)

            # Save token
            with open(token_file, "w") as f:
                f.write(creds.to_json())

            # Get email from profile
            from googleapiclient.discovery import build
            service = build("gmail", "v1", credentials=creds)
            profile = service.users().getProfile(userId="me").execute()
            email_addr = profile.get("emailAddress", "")

            return {"status": "authorized", "email": email_addr}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"OAuth flow failed: {e}")

    @app.get("/api/admin/connectors/google/status")
    async def admin_google_status():
        """Check if Google OAuth token exists and is valid."""
        import os

        config = life_os.get_connector_config("google")
        token_file = config.get("token_file", "data/google_token.json")

        if not os.path.exists(token_file):
            return {"authorized": False}

        try:
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_file(token_file)

            return {
                "authorized": creds.valid or bool(creds.refresh_token),
                "email": config.get("email_address", ""),
                "scopes": list(creds.scopes) if creds.scopes else [],
                "expired": creds.expired if hasattr(creds, "expired") else False,
            }
        except Exception as e:
            return {"authorized": False, "error": str(e)}

    # -------------------------------------------------------------------
    # Admin — Database Viewer
    # -------------------------------------------------------------------

    DB_NAMES = ["events", "entities", "state", "user_model", "preferences"]

    @app.get("/admin/db", response_class=HTMLResponse)
    async def admin_db_page():
        from web.db_template import DB_HTML_TEMPLATE
        return DB_HTML_TEMPLATE

    @app.get("/api/admin/db")
    async def admin_db_schema():
        """Return all databases, tables, columns, and row counts."""
        databases = {}
        for db_name in DB_NAMES:
            tables = {}
            with life_os.db.get_connection(db_name) as conn:
                tbl_rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
                for tbl_row in tbl_rows:
                    tbl = tbl_row["name"]
                    count = conn.execute(f'SELECT COUNT(*) as c FROM "{tbl}"').fetchone()["c"]
                    cols = [row["name"] for row in conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()]
                    tables[tbl] = {"columns": cols, "count": count}
            databases[db_name] = tables
        return {"databases": databases}

    @app.get("/api/admin/db/{db_name}/{table_name}")
    async def admin_db_query(db_name: str, table_name: str,
                             limit: int = 50, offset: int = 0,
                             search: Optional[str] = None,
                             sort: Optional[str] = None,
                             dir: str = "asc"):
        """Query rows from a specific table with optional search, sort, and pagination."""
        if db_name not in DB_NAMES:
            raise HTTPException(400, f"Unknown database: {db_name}")

        with life_os.db.get_connection(db_name) as conn:
            # Validate table exists
            tables = [r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()]
            if table_name not in tables:
                raise HTTPException(404, f"Table not found: {table_name}")

            # Get columns
            columns = [r["name"] for r in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()]

            # Build query
            where = ""
            params: list = []
            if search:
                # Search across all text columns
                clauses = [f'CAST("{col}" AS TEXT) LIKE ?' for col in columns]
                where = "WHERE " + " OR ".join(clauses)
                params = [f"%{search}%" for _ in columns]

            # Count total
            count_sql = f'SELECT COUNT(*) as c FROM "{table_name}" {where}'
            total = conn.execute(count_sql, params).fetchone()["c"]

            # Sort
            order = ""
            if sort and sort in columns:
                direction = "ASC" if dir == "asc" else "DESC"
                order = f'ORDER BY "{sort}" {direction}'
            else:
                # Default: try common columns
                for default_col in ("created_at", "timestamp", "updated_at", "id", "rowid"):
                    if default_col in columns:
                        order = f'ORDER BY "{default_col}" DESC'
                        break
                if not order:
                    order = "ORDER BY rowid DESC"

            query = f'SELECT * FROM "{table_name}" {where} {order} LIMIT ? OFFSET ?'
            rows = conn.execute(query, params + [limit, offset]).fetchall()

            return {
                "columns": columns,
                "rows": [dict(r) for r in rows],
                "total": total,
                "limit": limit,
                "offset": offset,
            }

    # -------------------------------------------------------------------
    # Setup / Onboarding
    # -------------------------------------------------------------------

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_page():
        from web.setup_template import SETUP_HTML_TEMPLATE
        return SETUP_HTML_TEMPLATE

    @app.get("/api/setup/status")
    async def setup_status():
        """Check if onboarding is complete, and return current answers."""
        with life_os.db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'onboarding_completed'"
            ).fetchone()
            completed = bool(row and row["value"] == "true")
        return {
            "completed": completed,
            "answers": life_os.onboarding.get_answers(),
        }

    @app.get("/api/setup/flow")
    async def setup_flow():
        """Return the onboarding flow phases."""
        from services.onboarding.manager import ONBOARDING_PHASES
        # Sanitize for JSON (convert any non-serializable values)
        phases = []
        for p in ONBOARDING_PHASES:
            phase = dict(p)
            if "options" in phase:
                phase["options"] = [
                    {"label": o["label"], "value": o["value"]}
                    for o in phase["options"]
                ]
            phases.append(phase)
        return {"phases": phases}

    @app.post("/api/setup/submit")
    async def setup_submit(req: SetupSubmitRequest):
        """Submit a single onboarding answer."""
        life_os.onboarding.submit_answer(req.step_id, req.value)
        return {"status": "ok"}

    @app.post("/api/setup/finalize")
    async def setup_finalize():
        """Finalize onboarding: save preferences, seed contacts, create vaults."""
        try:
            preferences = life_os.onboarding.finalize()

            # Seed priority contacts into the entities DB
            priority_contacts = preferences.get("priority_contacts", [])
            if priority_contacts:
                _seed_contacts(life_os, priority_contacts)

            # Create vault if requested
            vaults = preferences.get("vaults", [])
            for vault in vaults:
                _seed_vault(life_os, vault)

            return {"status": "ok", "preferences": preferences}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def _seed_contacts(life_os_ref, contacts: list[dict]):
        """Create contact records from onboarding priority people."""
        import uuid
        with life_os_ref.db.get_connection("entities") as conn:
            for contact in contacts:
                contact_id = str(uuid.uuid4())
                name = contact.get("name", "Unknown")
                relationship = contact.get("relationship")
                conn.execute(
                    """INSERT OR IGNORE INTO contacts
                       (id, name, relationship, is_priority, always_surface, domains)
                       VALUES (?, ?, ?, 1, 1, '["personal"]')""",
                    (contact_id, name, relationship),
                )

    def _seed_vault(life_os_ref, vault: dict):
        """Create a vault record."""
        with life_os_ref.db.get_connection("preferences") as conn:
            conn.execute(
                """INSERT OR IGNORE INTO vaults (name, auth_method)
                   VALUES (?, ?)""",
                (vault.get("name", "Vault"), vault.get("auth_method", "pin")),
            )

    # -------------------------------------------------------------------
    # Web UI
    # -------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        # Check if onboarding is complete — if not, redirect to setup
        with life_os.db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'onboarding_completed'"
            ).fetchone()
            if not row or row["value"] != "true":
                from fastapi.responses import RedirectResponse
                return RedirectResponse("/setup")
        from web.template import HTML_TEMPLATE
        return HTML_TEMPLATE
