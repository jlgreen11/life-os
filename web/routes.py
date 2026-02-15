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
        tasks = life_os.task_manager.get_pending_tasks(limit=limit)
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
    # Insights (aggregated signal profiles → human-readable summaries)
    # -------------------------------------------------------------------

    @app.get("/api/insights/summary")
    async def insights_summary():
        """Aggregate signal profiles into human-readable insights.

        Pulls data from relationship, cadence, linguistic, and topic signal
        profiles plus the events and places tables.  Each insight is a dict
        with ``type``, ``summary``, ``confidence``, ``category``, and an
        optional ``entity``.  Missing or empty profiles are silently skipped.
        """
        insights: list[dict] = []

        # -- 1. Relationship profile: contacts overdue vs their usual interval --
        try:
            rel_profile = life_os.user_model_store.get_signal_profile("relationships")
            if rel_profile and rel_profile.get("data"):
                contacts = rel_profile["data"].get("contacts", {})
                for address, profile in contacts.items():
                    timestamps = profile.get("interaction_timestamps", [])
                    if len(timestamps) >= 2:
                        # Compute the average interval between interactions
                        from datetime import datetime as _dt
                        parsed = []
                        for ts in timestamps:
                            try:
                                parsed.append(_dt.fromisoformat(ts.replace("Z", "+00:00")))
                            except (ValueError, AttributeError):
                                continue
                        if len(parsed) >= 2:
                            parsed.sort()
                            deltas = [
                                (parsed[i + 1] - parsed[i]).total_seconds()
                                for i in range(len(parsed) - 1)
                            ]
                            avg_interval = sum(deltas) / len(deltas)
                            last = parsed[-1]
                            now = datetime.now(timezone.utc)
                            seconds_since_last = (now - last).total_seconds()
                            if avg_interval > 0 and seconds_since_last > avg_interval * 1.5:
                                overdue_days = round((seconds_since_last - avg_interval) / 86400, 1)
                                usual_days = round(avg_interval / 86400, 1)
                                insights.append({
                                    "type": "relationship_overdue",
                                    "summary": (
                                        f"You usually interact with {address} every "
                                        f"{usual_days} days, but it has been "
                                        f"{round(seconds_since_last / 86400, 1)} days "
                                        f"since your last contact ({overdue_days} days overdue)."
                                    ),
                                    "confidence": min(0.9, 0.5 + len(timestamps) * 0.02),
                                    "category": "relationships",
                                    "entity": address,
                                })

                    # Top contacts by interaction count
                    count = profile.get("interaction_count", 0)
                    if count >= 5:
                        inbound = profile.get("inbound_count", 0)
                        outbound = profile.get("outbound_count", 0)
                        if count > 0:
                            ratio = outbound / count
                            if ratio < 0.3:
                                direction_note = "They reach out far more than you reply."
                            elif ratio > 0.7:
                                direction_note = "You initiate most conversations."
                            else:
                                direction_note = "Communication is roughly balanced."
                            insights.append({
                                "type": "relationship_dynamics",
                                "summary": (
                                    f"{address}: {count} total interactions "
                                    f"({inbound} inbound, {outbound} outbound). "
                                    f"{direction_note}"
                                ),
                                "confidence": min(0.85, 0.4 + count * 0.01),
                                "category": "relationships",
                                "entity": address,
                            })
        except Exception:
            pass  # Gracefully skip if profile is missing or malformed

        # -- 2. Cadence profile: per-contact response times vs overall average --
        try:
            cadence_profile = life_os.user_model_store.get_signal_profile("cadence")
            if cadence_profile and cadence_profile.get("data"):
                cdata = cadence_profile["data"]
                global_rts = cdata.get("response_times", [])
                if global_rts:
                    avg_rt = sum(global_rts) / len(global_rts)
                    avg_rt_hours = round(avg_rt / 3600, 1)
                    insights.append({
                        "type": "cadence_overall",
                        "summary": (
                            f"Your average response time across all contacts is "
                            f"{avg_rt_hours} hours."
                        ),
                        "confidence": min(0.9, 0.5 + len(global_rts) * 0.01),
                        "category": "cadence",
                    })

                per_contact = cdata.get("per_contact_response_times", {})
                if global_rts and per_contact:
                    avg_global = sum(global_rts) / len(global_rts)
                    for contact_id, rts in per_contact.items():
                        if len(rts) >= 3:
                            avg_contact = sum(rts) / len(rts)
                            if avg_global > 0:
                                ratio = avg_contact / avg_global
                                if ratio < 0.5:
                                    insights.append({
                                        "type": "cadence_fast_responder",
                                        "summary": (
                                            f"You reply to {contact_id} significantly faster "
                                            f"than average ({round(avg_contact / 3600, 1)}h vs "
                                            f"{round(avg_global / 3600, 1)}h overall)."
                                        ),
                                        "confidence": min(0.85, 0.5 + len(rts) * 0.02),
                                        "category": "cadence",
                                        "entity": contact_id,
                                    })
                                elif ratio > 2.0:
                                    insights.append({
                                        "type": "cadence_slow_responder",
                                        "summary": (
                                            f"You take much longer to reply to {contact_id} "
                                            f"({round(avg_contact / 3600, 1)}h vs "
                                            f"{round(avg_global / 3600, 1)}h overall)."
                                        ),
                                        "confidence": min(0.85, 0.5 + len(rts) * 0.02),
                                        "category": "cadence",
                                        "entity": contact_id,
                                    })

                # Peak activity hours
                hourly = cdata.get("hourly_activity", {})
                if hourly:
                    sorted_hours = sorted(hourly.items(), key=lambda x: x[1], reverse=True)
                    top_hours = sorted_hours[:3]
                    if top_hours:
                        hour_labels = [f"{int(h)}:00" for h, _ in top_hours]
                        insights.append({
                            "type": "cadence_peak_hours",
                            "summary": (
                                f"Your most active communication hours are "
                                f"{', '.join(hour_labels)}."
                            ),
                            "confidence": min(0.9, 0.5 + sum(hourly.values()) * 0.005),
                            "category": "cadence",
                        })
        except Exception:
            pass  # Gracefully skip if profile is missing or malformed

        # -- 3. Linguistic profile: formality score --
        try:
            ling_profile = life_os.user_model_store.get_signal_profile("linguistic")
            if ling_profile and ling_profile.get("data"):
                ldata = ling_profile["data"]
                averages = ldata.get("averages", {})
                formality = averages.get("formality")
                if formality is not None:
                    if formality > 0.7:
                        style = "formal"
                    elif formality < 0.3:
                        style = "casual"
                    else:
                        style = "balanced"
                    insights.append({
                        "type": "linguistic_formality",
                        "summary": (
                            f"Your overall writing style is {style} "
                            f"(formality score: {round(formality, 2)})."
                        ),
                        "confidence": min(0.9, 0.5 + len(ldata.get("samples", [])) * 0.01),
                        "category": "communication_style",
                    })

                # Greetings and closings
                greetings = ldata.get("common_greetings", [])
                closings = ldata.get("common_closings", [])
                if greetings:
                    insights.append({
                        "type": "linguistic_greetings",
                        "summary": (
                            f"Your most common greetings: {', '.join(greetings)}."
                        ),
                        "confidence": 0.7,
                        "category": "communication_style",
                    })
                if closings:
                    insights.append({
                        "type": "linguistic_closings",
                        "summary": (
                            f"Your most common sign-offs: {', '.join(closings)}."
                        ),
                        "confidence": 0.7,
                        "category": "communication_style",
                    })
        except Exception:
            pass  # Gracefully skip if profile is missing or malformed

        # -- 4. Events table: email volume by day of week (last 30 days) --
        try:
            with life_os.db.get_connection("events") as conn:
                rows = conn.execute(
                    """SELECT timestamp FROM events
                       WHERE type IN ('email.received', 'email.sent')
                       AND timestamp >= datetime('now', '-30 days')"""
                ).fetchall()
                if rows:
                    from collections import Counter as _Counter
                    day_counts = _Counter()
                    for row in rows:
                        try:
                            dt = datetime.fromisoformat(
                                row["timestamp"].replace("Z", "+00:00")
                            )
                            day_counts[dt.strftime("%A")] += 1
                        except (ValueError, AttributeError):
                            continue
                    if day_counts:
                        busiest = day_counts.most_common(1)[0]
                        quietest = day_counts.most_common()[-1]
                        insights.append({
                            "type": "email_volume_pattern",
                            "summary": (
                                f"In the last 30 days, your busiest email day is "
                                f"{busiest[0]} ({busiest[1]} emails) and quietest is "
                                f"{quietest[0]} ({quietest[1]} emails)."
                            ),
                            "confidence": min(0.85, 0.4 + len(rows) * 0.005),
                            "category": "activity_patterns",
                        })
        except Exception:
            pass  # Gracefully skip if query fails

        # -- 5. Places table: most visited places --
        try:
            with life_os.db.get_connection("entities") as conn:
                rows = conn.execute(
                    """SELECT name, place_type, visit_count
                       FROM places
                       WHERE visit_count > 0
                       ORDER BY visit_count DESC
                       LIMIT 5"""
                ).fetchall()
                if rows:
                    place_summaries = []
                    for row in rows:
                        name = row["name"]
                        visits = row["visit_count"]
                        place_type = row["place_type"] or "unknown"
                        place_summaries.append(f"{name} ({visits} visits, {place_type})")
                    insights.append({
                        "type": "frequent_places",
                        "summary": (
                            f"Your most visited places: {'; '.join(place_summaries)}."
                        ),
                        "confidence": min(0.9, 0.5 + len(rows) * 0.08),
                        "category": "location_patterns",
                    })
        except Exception:
            pass  # Gracefully skip if query fails

        # Sort all insights by confidence descending
        insights.sort(key=lambda x: x.get("confidence", 0), reverse=True)

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

        # Map insight category back to source_key for weight learning
        if row:
            category_to_source = {
                "place": "location.visits",
                "contact_gap": "messaging.direct",
                "email_volume": "email.work",
                "communication_style": "messaging.direct",
                "relationships": "messaging.direct",
                "cadence": "messaging.direct",
                "activity_patterns": "email.work",
                "location_patterns": "location.visits",
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
                       (id, name, relationship, is_priority, always_surface,
                        contact_type, domains)
                       VALUES (?, ?, ?, 1, 1, 'person', '["personal"]')""",
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
