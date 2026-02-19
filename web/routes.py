"""
Life OS — API Route Registration

All REST API routes, WebSocket handler, and the HTML UI.
Called by the app factory to register routes on the FastAPI instance.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

from web.schemas import (
    CommandRequest,
    ConnectorConfigRequest,
    ContextBatchRequest,
    ContextEventRequest,
    DraftRequest,
    FactCorrectionRequest,
    FeedbackRequest,
    PreferenceUpdate,
    RuleCreateRequest,
    SearchRequest,
    SetupSubmitRequest,
    SourceWeightCreate,
    SourceWeightUpdate,
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
        import asyncio

        async def _check(c):
            try:
                return await asyncio.wait_for(c.health_check(), timeout=5.0)
            except asyncio.TimeoutError:
                return {"connector": c.CONNECTOR_ID, "status": "error", "details": "timeout"}
            except Exception as e:
                return {"connector": c.CONNECTOR_ID, "status": "error", "details": str(e)}

        # Run all connector health checks concurrently instead of sequentially
        connectors = await asyncio.gather(*[_check(c) for c in life_os.connectors])

        # Offload synchronous SQLite calls to threads
        events_stored, vector_stats = await asyncio.gather(
            asyncio.to_thread(life_os.event_store.get_event_count),
            asyncio.to_thread(life_os.vector_store.get_stats),
        )

        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_bus": life_os.event_bus.is_connected,
            "events_stored": events_stored,
            "vector_store": vector_stats,
            "connectors": list(connectors),
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
    # Dashboard Feed (unified, priority-sorted view)
    # -------------------------------------------------------------------

    @app.get("/api/dashboard/feed")
    async def dashboard_feed(topic: Optional[str] = None, limit: int = 50):
        """Unified feed for the dashboard. Aggregates notifications, tasks,
        and recent events into a single priority-sorted list.

        Query params:
            topic: Filter by category — inbox (all), messages, email,
                   calendar, tasks, insights, system. Default: inbox.
            limit: Max items to return. Default: 50.
        """
        items = []

        # --- Notifications (all topics except 'system') ---
        if topic in (None, "inbox", "messages", "email"):
            try:
                notifications = life_os.notification_manager.get_pending(limit=limit)
                for n in notifications:
                    source_type = n.get("source", "")
                    if topic == "messages" and "message" not in source_type and "signal" not in source_type:
                        continue
                    if topic == "email" and "email" not in source_type:
                        continue
                    items.append({
                        "id": n.get("id"),
                        "kind": "notification",
                        "channel": "message" if "message" in source_type or "signal" in source_type else "email" if "email" in source_type else "system",
                        "title": n.get("title", ""),
                        "body": n.get("body", ""),
                        "priority": n.get("priority", "normal"),
                        "timestamp": n.get("created_at", n.get("timestamp", "")),
                        "source": source_type,
                        "domain": n.get("domain"),  # Include domain so UI can identify prediction notifications
                        "metadata": n.get("metadata", {}),
                    })
            except Exception:
                pass

        # --- Tasks ---
        if topic in (None, "inbox", "tasks"):
            try:
                tasks = life_os.task_manager.get_pending_tasks(limit=limit)
                for t in tasks:
                    items.append({
                        "id": t.get("id"),
                        "kind": "task",
                        "channel": "task",
                        "title": t.get("title", ""),
                        "body": t.get("description", ""),
                        "priority": t.get("priority", "normal"),
                        "timestamp": t.get("created_at", ""),
                        "source": t.get("domain", ""),
                        "metadata": {"due_date": t.get("due_date"), "domain": t.get("domain")},
                    })
            except Exception:
                pass

        # --- Recent events (email, messages, calendar) ---
        if topic in (None, "inbox", "messages", "email", "calendar"):
            try:
                event_types = []
                if topic in (None, "inbox", "email"):
                    event_types.append("email.received")
                if topic in (None, "inbox", "messages"):
                    event_types.append("message.received")
                if topic in (None, "inbox", "calendar"):
                    event_types.extend(["calendar.event.created", "calendar.event.updated", "calendar.event.reminder"])

                for et in event_types:
                    events = life_os.event_store.get_events(event_type=et, limit=20)
                    for ev in events:
                        payload = ev.get("payload", {}) if isinstance(ev.get("payload"), dict) else {}
                        channel = "email" if "email" in et else "message" if "message" in et else "calendar"
                        items.append({
                            "id": ev.get("id"),
                            "kind": "event",
                            "channel": channel,
                            "title": payload.get("subject", payload.get("title", et)),
                            "body": payload.get("snippet", payload.get("body", payload.get("description", "")))[:200],
                            "priority": "high" if payload.get("urgency", 0) > 0.7 else "normal",
                            "timestamp": ev.get("timestamp", ""),
                            "source": ev.get("source", ""),
                            "metadata": {
                                "sender": payload.get("sender", payload.get("from", "")),
                                "sentiment": payload.get("sentiment"),
                                "action_items": payload.get("action_items", []),
                                "attendees": payload.get("attendees", []),
                                "location": payload.get("location", ""),
                                "start_time": payload.get("start_time", payload.get("start", "")),
                                "end_time": payload.get("end_time", payload.get("end", "")),
                            },
                        })
            except Exception:
                pass

        # --- Sort by priority (critical > high > normal > low), then newest first ---
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        items.sort(key=lambda x: (
            priority_order.get(x["priority"], 2),
            "" if x.get("timestamp") else "z",  # items with timestamps before those without
        ))
        # Within same priority, sort by timestamp descending (newest first)
        from itertools import groupby
        sorted_items = []
        for _, group in groupby(items, key=lambda x: priority_order.get(x["priority"], 2)):
            group_list = list(group)
            group_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            sorted_items.extend(group_list)

        return {"items": sorted_items[:limit], "count": len(sorted_items[:limit]), "topic": topic or "inbox"}

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
        """List tasks filtered by status.

        Query parameters:
            status: Task status to filter on. One of ``pending`` (default),
                    ``completed``, ``in_progress``, ``archived``, ``cancelled``.
            limit:  Maximum number of results to return (default: 50).

        Previously this route accepted a ``status`` parameter but silently
        ignored it, always returning pending tasks regardless of what was
        requested.  It now delegates to ``get_tasks()`` so callers can retrieve
        completed, in-progress, or archived tasks through the same endpoint.
        """
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
        await life_os.task_manager.update_task(task_id, **req.model_dump(exclude_none=True))
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

    @app.patch("/api/user-model/facts/{key}")
    async def correct_fact(key: str, request: FactCorrectionRequest):
        """Correct a semantic fact by marking it as user-corrected and reducing confidence.

        When the user identifies an incorrect fact, this endpoint:
        1. Marks is_user_corrected = 1 to flag it in the learning loop
        2. Reduces confidence by 0.30 (significant penalty for being wrong)
        3. Optionally updates the value if a corrected value is provided
        4. Records the correction reason for audit trails

        This closes the feedback loop: the system can now learn from both
        positive signals (confirmations, times_confirmed++) and negative
        signals (corrections, confidence--).

        Args:
            key: The semantic fact key to correct
            request: Contains optional corrected_value and reason

        Returns:
            Updated fact with new confidence and correction status
        """
        import json
        from datetime import datetime, timezone

        with life_os.db.get_connection("user_model") as conn:
            # Retrieve the existing fact to compute new confidence
            existing = conn.execute(
                "SELECT * FROM semantic_facts WHERE key = ?", (key,)
            ).fetchone()

            if not existing:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail=f"Fact with key '{key}' not found")

            # Compute new confidence: reduce by 0.30 (significant penalty),
            # but never go below 0.0. This ensures corrected facts are
            # de-prioritized but remain visible for audit purposes.
            # Round to 2 decimal places to avoid floating point precision issues.
            new_confidence = round(max(0.0, existing["confidence"] - 0.30), 2)

            # Update the fact with correction metadata
            update_params = {
                "is_user_corrected": 1,
                "confidence": new_confidence,
                "last_confirmed": datetime.now(timezone.utc).isoformat(),
            }

            # If the user provided a corrected value, replace the existing one
            if request.corrected_value is not None:
                update_params["value"] = json.dumps(request.corrected_value)

            # Build the UPDATE query dynamically based on provided fields
            set_clause = ", ".join([f"{k} = ?" for k in update_params.keys()])
            values = list(update_params.values()) + [key]

            conn.execute(
                f"UPDATE semantic_facts SET {set_clause} WHERE key = ?",
                values,
            )

            # Retrieve the updated fact to return to the client
            updated = conn.execute(
                "SELECT * FROM semantic_facts WHERE key = ?", (key,)
            ).fetchone()

        # Log the correction to the feedback collector for analytics
        if life_os.feedback_collector:
            await life_os.feedback_collector._store_feedback({
                "action_id": f"fact_correction_{key}",
                "action_type": "semantic_fact",
                "feedback_type": "corrected",
                "response_latency_seconds": 0,
                "context": {
                    "fact_key": key,
                    "old_confidence": existing["confidence"],
                    "new_confidence": new_confidence,
                    "corrected_value_provided": request.corrected_value is not None,
                    "reason": request.reason,
                },
                "notes": request.reason,
            })

        # Publish a telemetry event for the correction
        if life_os.event_bus and life_os.event_bus.is_connected:
            await life_os.event_bus.publish(
                "usermodel.fact.corrected",
                {
                    "key": key,
                    "old_confidence": existing["confidence"],
                    "new_confidence": new_confidence,
                    "category": updated["category"],
                    "corrected_at": datetime.now(timezone.utc).isoformat(),
                },
                source="web_api",
            )

        return {
            "status": "corrected",
            "fact": dict(updated),
            "old_confidence": existing["confidence"],
            "new_confidence": new_confidence,
        }

    @app.get("/api/user-model/mood")
    async def get_mood():
        """Return the current inferred mood state.

        Uses Pydantic V2 ``model_dump()`` to serialize the ``MoodState``
        object.  Falls back to manual attribute access for any non-Pydantic
        object that may be returned during testing or early startup.

        Returns:
            dict: ``{"mood": {...}}`` with all MoodState fields.
        """
        mood = life_os.signal_extractor.get_current_mood()
        return {
            "mood": mood.model_dump() if hasattr(mood, "model_dump") else {
                "energy_level": mood.energy_level,
                "stress_level": mood.stress_level,
                "social_battery": mood.social_battery,
                "cognitive_load": mood.cognitive_load,
                "emotional_valence": mood.emotional_valence,
                "confidence": mood.confidence,
                "trend": mood.trend,
            }
        }

    @app.get("/api/user-model/workflows")
    async def get_workflows(min_success_rate: float = 0.0, min_observations: int = 0):
        """Return detected workflows from Layer 3 (procedural memory).

        Workflows are multi-step task-completion patterns learned from event
        sequences. Unlike routines (time/location-triggered), workflows are
        goal-driven processes that span multiple tools and interaction types.

        Examples:
        - "Responding to boss emails" (read → research → draft → send)
        - "Task completion workflow" (created → worked → completed)
        - "Calendar event workflow" (prep → attend → follow-up)

        Query params:
            min_success_rate: Filter workflows by minimum success rate (0.0-1.0)
            min_observations: Filter workflows by minimum times observed

        Returns:
            List of workflows with their steps, tools, success rates, and
            typical durations. Empty list if no workflows have been detected yet.
        """
        workflows = []
        try:
            with life_os.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT name, trigger_conditions, steps, typical_duration,
                              tools_used, success_rate, times_observed, updated_at
                       FROM workflows
                       WHERE success_rate >= ? AND times_observed >= ?
                       ORDER BY times_observed DESC, success_rate DESC""",
                    (min_success_rate, min_observations),
                ).fetchall()

                for row in rows:
                    workflow = dict(row)
                    # Parse JSON fields
                    for field in ("trigger_conditions", "steps", "tools_used"):
                        if isinstance(workflow.get(field), str):
                            try:
                                workflow[field] = json.loads(workflow[field])
                            except (json.JSONDecodeError, TypeError):
                                workflow[field] = []
                    workflows.append(workflow)
        except Exception as e:
            # Gracefully handle if workflows table doesn't exist yet or query fails
            logger.warning(f"Failed to fetch workflows: {e}")
            workflows = []

        return {
            "workflows": workflows,
            "count": len(workflows),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/user-model/routines")
    async def get_routines(trigger: Optional[str] = None, min_consistency: float = 0.0, min_observations: int = 0):
        """Return detected routines from Layer 3 (procedural memory).

        Routines are time- or location-triggered behavioral patterns learned
        from repeated event sequences. Unlike workflows (goal-driven processes),
        routines are habitual and occur automatically at predictable times or
        locations.

        Examples:
        - "morning_routine" (wake up → check email → calendar review → coffee)
        - "arrive_home" (unlock door → check messages → start dinner)
        - "weekly_review" (every Sunday evening: review tasks → plan ahead)

        Query params:
            trigger: Optional filter by trigger type (e.g., "morning", "arrive_home").
                     If omitted, all routines are returned.
            min_consistency: Filter by minimum consistency score (0.0–1.0).
                             Higher scores mean the routine fires more reliably.
            min_observations: Filter by minimum number of times observed.
                              Use to exclude one-off patterns.

        Returns:
            List of routines with their steps, consistency scores, and typical
            durations. Empty list if no routines have been detected yet.
        """
        routines = []
        try:
            # Delegate to UserModelStore which owns the routines schema and
            # handles JSON deserialization of the steps and variations fields.
            all_routines = life_os.user_model_store.get_routines(trigger=trigger)

            # Apply additional post-query filters that the store method doesn't
            # support natively (consistency threshold and min observations).
            for r in all_routines:
                if r["consistency_score"] >= min_consistency and r["times_observed"] >= min_observations:
                    routines.append(r)
        except Exception as e:
            # Gracefully handle if the routines table doesn't exist yet or the
            # query fails (e.g., during first-time startup before detection runs).
            logger.warning("Failed to fetch routines: %s", e)
            routines = []

        return {
            "routines": routines,
            "count": len(routines),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Known signal profile types maintained by the SignalPipeline.  This
    # constant serves as both documentation and the canonical list for the
    # signal-profiles endpoint so neither the route nor the tests need to
    # hard-code the profile names themselves.
    _SIGNAL_PROFILE_TYPES = [
        "linguistic",
        "linguistic_inbound",
        "cadence",
        "mood_signals",
        "relationships",
        "topics",
        "temporal",
        "spatial",
        "decision",
    ]

    @app.get("/api/user-model/signal-profiles")
    async def get_signal_profiles(profile_type: Optional[str] = None):
        """Return behavioral signal profiles from Layer 1 (episodic signals).

        Signal profiles are the raw aggregated behavioral data collected by the
        SignalPipeline from every event processed by Life OS.  Each profile type
        captures a different dimension of behavior:

        - **linguistic** — outbound writing style (formality, sentence length,
          punctuation patterns, vocabulary richness)
        - **linguistic_inbound** — aggregated writing style of contacts who email
          the user (formality mismatches trigger inbound_style insights)
        - **cadence** — response latency patterns per contact and per channel
          (who gets fast replies, who gets slow ones, and when)
        - **mood_signals** — valence, energy, and stress signals extracted from
          email/message content and timestamps
        - **relationships** — contact interaction graph: outbound count, inbound
          count, gap-days since last contact, and priority flags
        - **topics** — interest topic frequencies extracted from email subjects
          and message bodies (powers topic_interest insights)
        - **temporal** — chronotype (early bird vs. night owl), peak productive
          hours, and work-boundary adherence patterns
        - **spatial** — location visit frequencies and typical work locations
          extracted from calendar event metadata
        - **decision** — decision speed, delegation frequency, and recency of
          decision-making activity

        These profiles are aggregated by the SignalPipeline on every inbound event
        and are used internally by the InsightEngine, SemanticFactInferrer, and
        PredictionEngine.  This endpoint exposes the raw profiles directly so
        users and developers can inspect what the system has learned.

        Query params:
            profile_type: Optional.  If provided, return only the named profile
                          (e.g. ``?profile_type=linguistic``).  Raises 404 if
                          the profile type is unknown or has no data yet.
                          If omitted, all known profile types are returned.

        Returns:
            ``{"profiles": {<type>: {data, samples_count, updated_at}, ...},
               "types_with_data": [...], "generated_at": "..."}``

        Example (single profile)::

            GET /api/user-model/signal-profiles?profile_type=temporal
            → {
                "profiles": {
                    "temporal": {
                        "data": {"chronotype": "early_bird", ...},
                        "samples_count": 13814,
                        "updated_at": "2026-02-19T04:19:45.693Z"
                    }
                },
                "types_with_data": ["temporal"],
                "generated_at": "2026-02-19T04:20:00.000000+00:00"
            }
        """
        # Validate the requested profile_type before querying.
        if profile_type is not None and profile_type not in _SIGNAL_PROFILE_TYPES:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Unknown profile type {profile_type!r}. "
                    f"Valid types: {_SIGNAL_PROFILE_TYPES}"
                ),
            )

        # Determine which types to fetch: a specific one or all known types.
        types_to_fetch = [profile_type] if profile_type else _SIGNAL_PROFILE_TYPES

        profiles = {}
        for ptype in types_to_fetch:
            try:
                row = life_os.user_model_store.get_signal_profile(ptype)
                if row:
                    # Return the full profile including the deserialized data
                    # blob (not just the summary counts surfaced by pipeline
                    # snapshots) so callers can inspect raw signals.
                    profiles[ptype] = {
                        "data": row["data"],
                        "samples_count": row["samples_count"],
                        "updated_at": row["updated_at"],
                    }
            except Exception as e:
                # Never let a single broken profile abort the entire request —
                # return the other profiles and log the failure.
                logger.warning("Failed to fetch signal profile %r: %s", ptype, e)

        return {
            "profiles": profiles,
            # Convenience list so callers can quickly see which types have data.
            "types_with_data": list(profiles.keys()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # -------------------------------------------------------------------
    # Contacts (entities database — people the system has learned about)
    # -------------------------------------------------------------------

    @app.get("/api/contacts")
    async def get_contacts(
        is_priority: Optional[bool] = None,
        name: Optional[str] = None,
        has_metrics: Optional[bool] = None,
        limit: int = 100,
    ):
        """Return contacts from the entities database with relationship metrics.

        Exposes the ``contacts`` table from ``entities.db``, including the three
        denormalized relationship metrics written by
        ``RelationshipExtractor._sync_contact_metrics()``:

        - **typical_response_time** — median response latency in seconds
          (how quickly the user typically replies to this contact)
        - **last_contact** — ISO-8601 timestamp of the most recent two-way
          interaction (either an outbound email or a reply received)
        - **contact_frequency_days** — average gap in days between interactions
          (lower = more frequent contact)

        These metrics are derived from the ``relationships`` signal profile and
        are updated incrementally as new emails and messages are processed.

        Query params:
            is_priority: Optional bool.  ``true`` returns only contacts marked
                         as priority (``is_priority = 1``); ``false`` returns
                         non-priority contacts only.  Omit to return all.
            name: Optional string.  Case-insensitive substring match against
                  the contact's display name (``LIKE %name%``).
            has_metrics: Optional bool.  ``true`` returns only contacts that
                         have at least one denormalized metric populated
                         (i.e. ``contact_frequency_days IS NOT NULL``).
                         ``false`` returns contacts with no metrics yet.
                         Omit to return all.
            limit: Maximum number of contacts to return (default 100, max 500).
                   Contacts are ordered by priority first, then by most-recently
                   contacted.

        Returns:
            ``{"contacts": [...], "total": N, "generated_at": "..."}``

            Each contact object mirrors the DB schema.  JSON arrays
            (``aliases``, ``emails``, ``phones``, ``notes``) are deserialized
            from their stored JSON strings.  Numeric metrics are returned as
            floats (``null`` when not yet computed).

        Example::

            GET /api/contacts?is_priority=true
            → {
                "contacts": [
                    {
                        "id": "abc123",
                        "name": "Alice Smith",
                        "emails": ["alice@example.com"],
                        "relationship": "colleague",
                        "is_priority": true,
                        "typical_response_time": 7200.0,
                        "last_contact": "2026-02-18T14:30:00.000Z",
                        "contact_frequency_days": 3.5,
                        ...
                    }
                ],
                "total": 1,
                "generated_at": "2026-02-19T05:00:00.000000+00:00"
            }

            GET /api/contacts?has_metrics=true&limit=20
            → contacts that have interaction frequency data, up to 20 results
        """
        import json as _json

        # Hard cap to prevent accidental full-table dumps.
        effective_limit = min(limit, 500)

        # Build the WHERE clause dynamically based on provided filters.
        conditions: list[str] = []
        params: list = []

        if is_priority is not None:
            conditions.append("is_priority = ?")
            params.append(1 if is_priority else 0)

        if name is not None:
            conditions.append("LOWER(name) LIKE ?")
            params.append(f"%{name.lower()}%")

        if has_metrics is True:
            # At least one of the relationship metrics is populated.
            conditions.append("contact_frequency_days IS NOT NULL")
        elif has_metrics is False:
            conditions.append("contact_frequency_days IS NULL")

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        # ORDER BY: priority contacts first, then most-recently contacted,
        # then alphabetically for contacts with no last_contact date.
        order_clause = (
            "ORDER BY is_priority DESC, "
            "last_contact DESC NULLS LAST, "
            "name ASC"
        )

        query = f"""
            SELECT id, name, aliases, emails, phones, channels, relationship,
                   domains, is_priority, preferred_channel, always_surface,
                   typical_response_time, communication_style, last_contact,
                   contact_frequency_days, notes, created_at, updated_at
            FROM contacts
            {where_clause}
            {order_clause}
            LIMIT ?
        """
        params.append(effective_limit)

        # Separate count query (uses same conditions but no LIMIT).
        count_query = f"SELECT COUNT(*) FROM contacts {where_clause}"

        with life_os.db.get_connection("entities") as conn:
            total = conn.execute(count_query, params[:-1]).fetchone()[0]
            rows = conn.execute(query, params).fetchall()

        # Deserialize stored JSON strings into native Python lists/dicts so
        # the response payload is clean JSON rather than stringified arrays.
        contacts = []
        for row in rows:
            contact = dict(row)
            for json_field in ("aliases", "emails", "phones", "notes", "domains"):
                raw = contact.get(json_field)
                if isinstance(raw, str):
                    try:
                        contact[json_field] = _json.loads(raw)
                    except (_json.JSONDecodeError, ValueError):
                        contact[json_field] = []
            raw_channels = contact.get("channels")
            if isinstance(raw_channels, str):
                try:
                    contact["channels"] = _json.loads(raw_channels)
                except (_json.JSONDecodeError, ValueError):
                    contact["channels"] = {}
            # Coerce SQLite integers to booleans for cleaner JSON output.
            contact["is_priority"] = bool(contact.get("is_priority", 0))
            contact["always_surface"] = bool(contact.get("always_surface", 0))
            contacts.append(contact)

        return {
            "contacts": contacts,
            "total": total,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # -------------------------------------------------------------------
    # Insights (aggregated signal profiles → human-readable summaries)
    # -------------------------------------------------------------------

    @app.get("/api/insights/summary")
    async def insights_summary():
        """Return current insights generated by the InsightEngine.

        Triggers a fresh generation pass via InsightEngine.generate_insights(),
        then returns all non-expired, non-negative insights from the persistent
        store, sorted by confidence descending.

        The InsightEngine runs 14 correlators covering relationship gaps, place
        frequency, email volume, communication style, inbound style, actionable
        alerts, temporal patterns, mood trends, spending patterns, decision
        patterns, topic interests, cadence/response, routines, and spatial
        location — far more than the 3 categories the previous hand-rolled
        implementation covered.

        Each insight has:
            id          -- UUID for feedback routing
            type        -- Insight type (e.g. "relationship_intelligence")
            summary     -- Human-readable description
            confidence  -- 0–1 confidence score (weighted by source weights)
            category    -- Broad category (e.g. "contact_gap", "place")
            entity      -- Optional entity the insight is about (contact, place)
            evidence    -- List of supporting signal strings
            feedback    -- User feedback ("useful" / "dismissed" / None)
            created_at  -- ISO timestamp when insight was generated
        """
        # Trigger a fresh generation pass. generate_insights() is idempotent:
        # it deduplicates within the staleness TTL, so calling it on every
        # request does not cause insight floods — duplicate keys are silently
        # dropped until the TTL expires.
        try:
            await life_os.insight_engine.generate_insights()
        except Exception:
            # If InsightEngine fails, still return whatever is stored so the
            # caller gets the most-recent cached insights rather than an error.
            logger.exception(
                "InsightEngine.generate_insights() failed during /api/insights/summary"
            )

        # Read all non-expired, non-negative insights from the persistent store.
        # This mirrors the query used by services/ai_engine/context.py to build
        # the morning briefing context window.
        insights: list[dict] = []
        try:
            with life_os.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT id, type, summary, confidence, category, entity,
                              evidence, feedback, created_at
                       FROM insights
                       WHERE feedback IS NOT 'negative'
                         AND datetime(created_at) >
                             datetime('now', '-' || staleness_ttl_hours || ' hours')
                       ORDER BY confidence DESC"""
                ).fetchall()
            for row in rows:
                d = dict(row)
                # Parse JSON evidence list stored as a string
                if isinstance(d.get("evidence"), str):
                    try:
                        d["evidence"] = json.loads(d["evidence"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                insights.append(d)
        except Exception:
            logger.exception(
                "Failed to read insights from database in /api/insights/summary"
            )

        return {
            "insights": insights,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/insights")
    async def list_insights(limit: int = 20):
        """Return recent insights from the InsightEngine."""
        with life_os.db.get_connection("user_model") as conn:
            rows = conn.execute(
                """SELECT * FROM insights
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            # Parse JSON evidence field
            if isinstance(d.get("evidence"), str):
                try:
                    d["evidence"] = json.loads(d["evidence"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        return {"insights": results}

    @app.post("/api/insights/{insight_id}/feedback")
    async def insight_feedback(insight_id: str, feedback: str = "dismissed"):
        """Record user feedback on an insight (useful/dismissed).

        Also updates source weight engagement/dismissal counters so the
        AI drift can learn from the user's response patterns.
        """
        if feedback not in ("useful", "dismissed"):
            raise HTTPException(400, f"Invalid feedback value: {feedback}. Must be 'useful' or 'dismissed'.")

        # Look up the insight to find its source category for weight feedback
        source_key = None
        with life_os.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT category, entity FROM insights WHERE id = ?",
                (insight_id,),
            ).fetchone()
            conn.execute(
                "UPDATE insights SET feedback = ? WHERE id = ?",
                (feedback, insight_id),
            )

        # Map insight category back to source_key for weight learning.
        # Must match the category_to_source dict in InsightEngine._apply_source_weights()
        # exactly so that feedback adjusts the same source keys that were used
        # to weight the insight at generation time.
        if row:
            # This dict MUST mirror InsightEngine._apply_source_weights() exactly so
            # that feedback adjusts the same source weights used at generation time.
            # When a new correlator is added to engine.py, add its categories here too.
            category_to_source = {
                "place": "location.visits",
                "contact_gap": "messaging.direct",
                "email_volume": "email.work",
                "communication_style": "messaging.direct",
                # Inbound style mismatch — weighted against outbound messaging source
                "style_mismatch": "messaging.direct",
                # Temporal pattern insight categories (from _temporal_pattern_insights)
                "chronotype": "email.work",
                "peak_hour": "email.work",
                "busiest_day": "email.work",
                # Mood trend insight category (from _mood_trend_insights)
                "mood_trajectory": "messaging.direct",
                # Spending pattern insight categories (from _spending_pattern_insights)
                "top_spending_category": "finance.transactions",
                "spending_increase": "finance.transactions",
                "spending_decrease": "finance.transactions",
                "recurring_subscription": "finance.transactions",
                # Decision pattern insights (from _decision_pattern_insights)
                "decision_speed": "email.work",
                "delegation_tendency": "messaging.direct",
                "decision_fatigue": "messaging.direct",
                # Topic interest insights (from _topic_interest_insights)
                "top_interests": "email.work",
                "trending_topic": "email.work",
                # Cadence / response-time insights (from _cadence_response_insights)
                "response_time_baseline": "email.work",
                "fastest_contacts": "messaging.direct",
                "communication_peak_hours": "email.work",
                "channel_cadence": "email.work",
                # Routine pattern insights (from _routine_insights)
                "routine_pattern": "email.work",
                # Spatial insights (from _spatial_insights)
                "spatial_top_location": "location.visits",
                "spatial_work_location": "location.visits",
                "spatial_location_diversity": "location.visits",
                # Relationship intelligence insights (from _relationship_intelligence_insights, PR #278)
                # Reciprocity and response-time patterns come from email/messaging cadence data.
                "reciprocity_imbalance": "messaging.direct",
                "fast_responder": "messaging.direct",
                # Workflow pattern insights (from _workflow_pattern_insights, PR #279)
                # Email and task workflow patterns are primarily surfaced from email metadata;
                # calendar and interaction patterns map to messaging channels.
                "workflow_pattern_email": "email.work",
                "workflow_pattern_task": "email.work",
                "workflow_pattern_calendar": "email.work",
                "workflow_pattern_interaction": "messaging.direct",
                # actionable_alert categories (overdue_task, upcoming_calendar) are
                # intentionally absent: they bypass source-weight tuning entirely.
            }
            source_key = category_to_source.get(row["category"])

        # Update source weight engagement/dismissal counters
        if source_key and hasattr(life_os, "source_weight_manager"):
            try:
                if feedback == "useful":
                    life_os.source_weight_manager.record_engagement(source_key)
                else:
                    life_os.source_weight_manager.record_dismissal(source_key)
            except Exception:
                pass  # Source weight feedback is non-critical

        return {"status": "recorded"}

    # -------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------

    @app.get("/api/predictions")
    async def list_predictions(
        prediction_type: str = None,
        min_confidence: float = 0.0,
        include_resolved: bool = False,
        limit: int = 50,
    ):
        """Return active predictions from the prediction engine.

        Surfaces the system's forward-looking predictions — opportunity follow-ups,
        reminders, calendar conflicts, routine deviations, and more — so that clients
        can display them directly without parsing the morning briefing text.

        Each prediction includes:
          - ``id``: UUID for feedback and resolution calls
          - ``prediction_type``: one of opportunity, reminder, conflict,
            routine_deviation, preparation, spending_pattern
          - ``description``: human-readable prediction summary
          - ``confidence``: 0.0–1.0 gate-adjusted confidence score
          - ``confidence_gate``: OBSERVE / SUGGEST / DEFAULT / AUTONOMOUS
          - ``time_horizon``: optional ISO timestamp for when the event is expected
          - ``suggested_action``: optional next action string
          - ``supporting_signals``: dict of evidence used to generate the prediction
          - ``created_at``: ISO timestamp of when the prediction was generated
          - ``was_surfaced``: whether the prediction passed all gates and was shown

        Query parameters:
          - ``prediction_type``: filter to a single prediction type (optional)
          - ``min_confidence``: exclude predictions below this threshold (default 0.0)
          - ``include_resolved``: if True, also return resolved predictions (default False)
          - ``limit``: max number of predictions to return (default 50, max 200)

        Active predictions are defined as: ``was_surfaced=1``, ``resolved_at IS NULL``,
        ``filter_reason IS NULL``, generated within the last 7 days.

        Example::

            GET /api/predictions
            GET /api/predictions?prediction_type=opportunity&min_confidence=0.5
            GET /api/predictions?include_resolved=true&limit=100
        """
        # Clamp limit to prevent unbounded queries
        limit = min(limit, 200)

        conditions = [
            "filter_reason IS NULL",
            "confidence >= ?",
        ]
        params: list = [min_confidence]

        if prediction_type:
            conditions.append("prediction_type = ?")
            params.append(prediction_type)

        if not include_resolved:
            # Active-only: not yet resolved and generated within the last 7 days
            conditions.append("resolved_at IS NULL")
            conditions.append("datetime(created_at) > datetime('now', '-7 days')")
        else:
            # Include resolved, but still cap at 30 days to avoid huge result sets
            conditions.append("datetime(created_at) > datetime('now', '-30 days')")

        # Only surface predictions that passed the confidence gate (was_surfaced=1).
        # Filtered predictions (was_surfaced=0) are internal telemetry and are not
        # intended to be shown to the user.
        conditions.append("was_surfaced = 1")

        where_clause = " AND ".join(conditions)
        params.append(limit)

        with life_os.db.get_connection("user_model") as conn:
            rows = conn.execute(
                f"""SELECT id, prediction_type, description, confidence, confidence_gate,
                           time_horizon, suggested_action, supporting_signals,
                           was_surfaced, user_response, was_accurate,
                           filter_reason, resolution_reason,
                           created_at, resolved_at
                    FROM predictions
                    WHERE {where_clause}
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT ?""",
                params,
            ).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            # Deserialize supporting_signals from JSON string to dict.
            # The store serializes this as JSON because SQLite has no native JSON column.
            if isinstance(d.get("supporting_signals"), str):
                try:
                    d["supporting_signals"] = json.loads(d["supporting_signals"])
                except (json.JSONDecodeError, TypeError):
                    d["supporting_signals"] = {}
            results.append(d)

        return {
            "predictions": results,
            "count": len(results),
            "filters": {
                "prediction_type": prediction_type,
                "min_confidence": min_confidence,
                "include_resolved": include_resolved,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/api/predictions/{prediction_id}/feedback")
    async def prediction_feedback(
        prediction_id: str,
        was_accurate: bool,
        user_response: str = None,
    ):
        """Record user feedback on a prediction and mark it resolved.

        Feedback is the primary signal for the prediction accuracy learning loop.
        After resolving, the BehavioralAccuracyTracker and prediction engine use
        ``was_accurate`` to adjust per-type and per-contact confidence multipliers
        so that the system improves over time.

        Path parameter:
          - ``prediction_id``: UUID from the ``/api/predictions`` response

        Query parameters:
          - ``was_accurate``: True if the prediction was helpful/correct,
            False if it was wrong or annoying
          - ``user_response``: optional free-text label (e.g. "acted_on",
            "not_relevant", "already_done")

        Returns 404 when the prediction_id is not found.

        Example::

            POST /api/predictions/abc123/feedback?was_accurate=true
            POST /api/predictions/abc123/feedback?was_accurate=false&user_response=not_relevant
        """
        # Verify the prediction exists before resolving to give a clear 404
        with life_os.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT id FROM predictions WHERE id = ?",
                (prediction_id,),
            ).fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Prediction '{prediction_id}' not found.",
            )

        # Delegate to UserModelStore to update was_accurate, user_response,
        # resolved_at and emit telemetry in one atomic operation.
        life_os.user_model_store.resolve_prediction(
            prediction_id=prediction_id,
            was_accurate=was_accurate,
            user_response=user_response,
        )

        return {
            "status": "recorded",
            "prediction_id": prediction_id,
            "was_accurate": was_accurate,
        }

    # -------------------------------------------------------------------
    # Source Weights (tunable insight engine)
    # -------------------------------------------------------------------

    @app.get("/api/source-weights")
    async def list_source_weights():
        """Return all source weights with effective (user + AI drift) values.

        Each entry contains:
          - source_key: unique identifier (e.g. "email.marketing")
          - user_weight: the user's explicit setting (0.0-1.0)
          - ai_drift: the AI's learned adjustment (-0.3 to +0.3)
          - effective_weight: clamped(user_weight + decayed_ai_drift)
          - engagement/dismissal counts and rates
        """
        weights = life_os.source_weight_manager.get_all_weights()
        grouped = {}
        for w in weights:
            cat = w["category"]
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(w)
        return {
            "weights": weights,
            "by_category": grouped,
            "count": len(weights),
        }

    @app.get("/api/source-weights/{source_key:path}")
    async def get_source_weight(source_key: str):
        """Get detailed stats for a single source weight."""
        stats = life_os.source_weight_manager.get_source_stats(source_key)
        if not stats:
            raise HTTPException(404, f"Source weight not found: {source_key}")
        return stats

    @app.put("/api/source-weights/{source_key:path}")
    async def update_source_weight(source_key: str, req: SourceWeightUpdate):
        """Set the user-controlled weight for a source.

        The AI drift is preserved — the user is adjusting their base
        preference, and the AI will continue to learn on top of it.
        """
        try:
            updated = life_os.source_weight_manager.set_user_weight(
                source_key, req.weight,
            )
            return {"status": "updated", "weight": updated}
        except ValueError as e:
            raise HTTPException(404, str(e))

    @app.post("/api/source-weights/{source_key:path}/reset-drift")
    async def reset_source_drift(source_key: str):
        """Reset the AI drift for a source back to zero.

        Use this when the user changes their weight and wants the AI
        to start learning fresh from the new baseline.
        """
        try:
            updated = life_os.source_weight_manager.reset_ai_drift(source_key)
            return {"status": "reset", "weight": updated}
        except ValueError as e:
            raise HTTPException(404, str(e))

    @app.post("/api/source-weights")
    async def create_source_weight(req: SourceWeightCreate):
        """Create a custom source weight entry.

        Allows users to define fine-grained source categories beyond
        the defaults, e.g., "email.client_acme" for a specific sender.
        """
        result = life_os.source_weight_manager.add_source(
            source_key=req.source_key,
            category=req.category,
            label=req.label,
            description=req.description,
            user_weight=req.weight,
        )
        return {"status": "created", "weight": result}

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
            "payload": req.payload.model_dump(exclude_none=True),
            "metadata": {
                "domain": "context",
                "mobile_event_type": req.type,
                **(req.metadata.model_dump(exclude_none=True) if req.metadata else {}),
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
                "payload": event_req.payload.model_dump(exclude_none=True),
                "metadata": {
                    "domain": "context",
                    "mobile_event_type": event_req.type,
                    **(event_req.metadata.model_dump(exclude_none=True) if event_req.metadata else {}),
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
                        # Handle client-to-server commands from the UI
                        cmd = msg.get("command")

                        # Notification action commands: dismiss, act_on
                        if cmd == "dismiss_notification":
                            notif_id = msg.get("notification_id")
                            if notif_id:
                                await life_os.notification_manager.dismiss(notif_id)

                        elif cmd == "act_on_notification":
                            notif_id = msg.get("notification_id")
                            if notif_id:
                                await life_os.notification_manager.mark_acted_on(notif_id)

                        # Prediction feedback commands: for direct prediction
                        # resolution with custom user response
                        elif cmd == "resolve_prediction":
                            prediction_id = msg.get("prediction_id")
                            was_accurate = msg.get("was_accurate", False)
                            user_response = msg.get("user_response")
                            if prediction_id is not None:
                                life_os.user_model_store.resolve_prediction(
                                    prediction_id=prediction_id,
                                    was_accurate=was_accurate,
                                    user_response=user_response
                                )

                except json.JSONDecodeError:
                    # Silently ignore malformed JSON messages — fail-open keeps
                    # the connection alive even if the client sends bad data
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
    # Semantic Fact Inference
    # -------------------------------------------------------------------

    @app.post("/api/admin/semantic-facts/infer")
    async def trigger_semantic_fact_inference():
        """
        Trigger semantic fact inference across all signal profiles.

        Analyzes accumulated signal profiles (linguistic, relationship, topic,
        cadence, mood) and derives high-level semantic facts about the user's
        preferences, expertise, values, and patterns.

        This endpoint is useful for:
          - Manual testing during development
          - On-demand inference after bulk data ingestion
          - Admin troubleshooting

        In production, inference runs automatically every 6 hours via background task.
        """
        life_os.semantic_fact_inferrer.run_all_inference()

        # Return count of facts after inference
        facts = life_os.user_model_store.get_semantic_facts()
        facts_by_category = {}
        for fact in facts:
            category = fact.get("category", "unknown")
            facts_by_category[category] = facts_by_category.get(category, 0) + 1

        return {
            "status": "success",
            "message": "Semantic fact inference completed",
            "total_facts": len(facts),
            "facts_by_category": facts_by_category,
        }

    # -------------------------------------------------------------------
    # Prediction Diagnostics
    # -------------------------------------------------------------------

    @app.get("/api/admin/predictions/diagnostics")
    async def prediction_diagnostics():
        """
        Get comprehensive prediction engine diagnostics.

        Returns detailed analysis of why each prediction type is or isn't working,
        including data availability, configuration gaps, and actionable recommendations.

        This is the single source of truth for understanding prediction engine health
        and debugging issues.

        Example response:
        {
            "prediction_types": {
                "reminder": {
                    "status": "active",
                    "generated_last_7d": 2116,
                    "data_available": {
                        "unreplied_emails_24h": 97674,
                        ...
                    },
                    "blockers": [],
                    "recommendations": []
                },
                ...
            },
            "overall": {
                "total_predictions_7d": 2116,
                "active_types": 1,
                "blocked_types": 5,
                "health": "degraded"
            }
        }
        """
        diagnostics = await life_os.prediction_engine.get_diagnostics()
        return diagnostics

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
