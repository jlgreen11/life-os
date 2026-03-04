"""
Life OS — API Route Registration

All REST API routes, WebSocket handler, and the HTML UI.
Called by the app factory to register routes on the FastAPI instance.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

from web.schemas import (
    BackupRestoreRequest,
    CommandRequest,
    ConnectorConfigRequest,
    ContextBatchRequest,
    ContextEventRequest,
    DraftRequest,
    FactConfirmationRequest,
    FactCorrectionRequest,
    FeedbackRequest,
    PreferenceUpdate,
    RuleCreateRequest,
    SearchRequest,
    SendMessageRequest,
    SetupSubmitRequest,
    SourceWeightCreate,
    SourceWeightUpdate,
    TaskCreateRequest,
    TaskUpdateRequest,
    TemplateUpdateRequest,
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
        events_stored, vector_stats, db_health = await asyncio.gather(
            asyncio.to_thread(life_os.event_store.get_event_count),
            asyncio.to_thread(life_os.vector_store.get_stats),
            asyncio.to_thread(life_os.db.get_database_health),
        )

        # Event flow stats — wrapped so a failure doesn't break /health
        try:
            flow_stats = await asyncio.to_thread(life_os.event_store.get_event_flow_stats)
        except Exception:
            flow_stats = None

        # Derive overall DB status: "ok" if all databases are healthy,
        # "degraded" if one or more are corrupted.  Callers can inspect
        # db_health for per-database detail.
        corrupted_dbs = [name for name, info in db_health.items() if info["status"] != "ok"]
        db_status = "degraded" if corrupted_dbs else "ok"

        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_bus": life_os.event_bus.is_connected,
            "events_stored": events_stored,
            "data_flow": flow_stats,
            "vector_store": vector_stats,
            "connectors": list(connectors),
            "db_health": db_health,
            "db_status": db_status,
        }

    @app.get("/api/status")
    async def status():
        """System status — offload synchronous DB/service calls to threads."""
        import asyncio

        event_count, vector_stats, user_model, notif_stats, feedback = await asyncio.gather(
            asyncio.to_thread(life_os.event_store.get_event_count),
            asyncio.to_thread(life_os.vector_store.get_stats),
            asyncio.to_thread(life_os.signal_extractor.get_user_summary),
            asyncio.to_thread(life_os.notification_manager.get_stats),
            asyncio.to_thread(life_os.feedback_collector.get_feedback_summary),
        )
        return {
            "event_count": event_count,
            "vector_store": vector_stats,
            "user_model": user_model,
            "notification_stats": notif_stats,
            "feedback_summary": feedback,
        }

    @app.get("/api/diagnostics/pipeline")
    async def pipeline_diagnostics():
        """Return comprehensive data-flow health diagnostics.

        Checks each stage of the processing pipeline — signal profiles,
        user model tables, predictions, notifications, and events — so
        that broken or degraded stages can be identified without external
        scripts or direct database access.

        Each section is wrapped in its own try/except so a failure in one
        area (e.g. a corrupt user_model.db) doesn't prevent the other
        sections from reporting.  The endpoint always returns 200 with
        diagnostic data; errors are reported inline.
        """
        import asyncio

        EXPECTED_PROFILES = [
            "relationships", "temporal", "topics", "linguistic",
            "cadence", "mood_signals", "spatial", "decision",
        ]

        result: dict = {}
        has_errors = False

        # --- signal_profiles ---
        def _check_signal_profiles():
            """Query each expected signal profile type."""
            profiles = {}
            for ptype in EXPECTED_PROFILES:
                try:
                    profile = life_os.user_model_store.get_signal_profile(ptype)
                    if profile:
                        profiles[ptype] = {
                            "exists": True,
                            "samples_count": profile.get("samples_count", 0),
                            "updated_at": profile.get("updated_at"),
                        }
                    else:
                        profiles[ptype] = {
                            "exists": False,
                            "samples_count": 0,
                            "updated_at": None,
                        }
                except Exception as exc:
                    profiles[ptype] = {"exists": False, "error": str(exc)}
            return profiles

        try:
            result["signal_profiles"] = await asyncio.to_thread(_check_signal_profiles)
        except Exception as exc:
            result["signal_profiles"] = {"error": str(exc)}
            has_errors = True

        # --- user_model ---
        def _check_user_model():
            """Count rows in core user-model tables."""
            counts: dict = {}
            tables = {
                "episodes_count": "SELECT COUNT(*) as c FROM episodes",
                "semantic_facts_count": "SELECT COUNT(*) as c FROM semantic_facts",
                "routines_count": "SELECT COUNT(*) as c FROM routines",
                "mood_readings_count": "SELECT COUNT(*) as c FROM mood_history",
            }
            with life_os.db.get_connection("user_model") as conn:
                for key, sql in tables.items():
                    try:
                        row = conn.execute(sql).fetchone()
                        counts[key] = row["c"] if row else 0
                    except Exception as exc:
                        counts[key] = f"error: {exc}"
            return counts

        try:
            result["user_model"] = await asyncio.to_thread(_check_user_model)
        except Exception as exc:
            result["user_model"] = {"error": str(exc)}
            has_errors = True

        # --- predictions (lives in user_model.db) ---
        def _check_predictions():
            """Query prediction pipeline health."""
            with life_os.db.get_connection("user_model") as conn:
                total_row = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()
                last_24h_row = conn.execute(
                    "SELECT COUNT(*) as c FROM predictions WHERE created_at > datetime('now', '-1 day')"
                ).fetchone()
                last_row = conn.execute("SELECT MAX(created_at) as ts FROM predictions").fetchone()
                return {
                    "total_generated": total_row["c"] if total_row else 0,
                    "last_24h": last_24h_row["c"] if last_24h_row else 0,
                    "last_prediction_at": last_row["ts"] if last_row else None,
                }

        try:
            result["predictions"] = await asyncio.to_thread(_check_predictions)
        except Exception as exc:
            result["predictions"] = {"error": str(exc)}
            has_errors = True

        # --- notifications (lives in state.db) ---
        def _check_notifications():
            """Query notification counts.

            'actionable' counts notifications the user hasn't interacted with yet
            (pending or delivered). The old query used status='unread' which doesn't
            exist in the schema — valid statuses are: pending, delivered, read,
            acted_on, dismissed, suppressed, expired.
            """
            with life_os.db.get_connection("state") as conn:
                total_row = conn.execute("SELECT COUNT(*) as c FROM notifications").fetchone()
                actionable_row = conn.execute(
                    "SELECT COUNT(*) as c FROM notifications WHERE status IN ('pending', 'delivered')"
                ).fetchone()
                last_24h_row = conn.execute(
                    "SELECT COUNT(*) as c FROM notifications WHERE created_at > datetime('now', '-1 day')"
                ).fetchone()
                return {
                    "total": total_row["c"] if total_row else 0,
                    "actionable": actionable_row["c"] if actionable_row else 0,
                    "last_24h": last_24h_row["c"] if last_24h_row else 0,
                }

        try:
            result["notifications"] = await asyncio.to_thread(_check_notifications)
        except Exception as exc:
            result["notifications"] = {"error": str(exc)}
            has_errors = True

        # --- events_pipeline ---
        def _check_events():
            """Query event pipeline health."""
            total = life_os.event_store.get_event_count()
            with life_os.db.get_connection("events") as conn:
                last_row = conn.execute("SELECT MAX(timestamp) as ts FROM events").fetchone()
                last_24h_row = conn.execute(
                    "SELECT COUNT(*) as c FROM events WHERE timestamp > datetime('now', '-1 day')"
                ).fetchone()
                return {
                    "total_events": total,
                    "last_event_at": last_row["ts"] if last_row else None,
                    "last_24h": last_24h_row["c"] if last_24h_row else 0,
                }

        try:
            result["events_pipeline"] = await asyncio.to_thread(_check_events)
        except Exception as exc:
            result["events_pipeline"] = {"error": str(exc)}
            has_errors = True

        # --- overall_status ---
        if has_errors:
            result["overall_status"] = "error"
        else:
            profiles = result.get("signal_profiles", {})
            existing_count = sum(
                1 for v in profiles.values()
                if isinstance(v, dict) and v.get("exists")
            )
            preds_24h = result.get("predictions", {}).get("last_24h", 0)

            if existing_count == 0:
                result["overall_status"] = "broken"
            elif existing_count < len(EXPECTED_PROFILES) or preds_24h == 0:
                result["overall_status"] = "degraded"
            else:
                result["overall_status"] = "healthy"

        # --- recommendations ---
        try:
            recommendations = []

            # Check signal profiles
            profiles = result.get("signal_profiles", {})
            if isinstance(profiles, dict) and not profiles.get("error"):
                missing_profiles = [
                    k for k, v in profiles.items()
                    if isinstance(v, dict) and (not v.get("exists") or v.get("error"))
                ]
                if missing_profiles:
                    recommendations.append({
                        "severity": "high",
                        "area": "signal_profiles",
                        "message": f"{len(missing_profiles)} signal profile(s) missing: {', '.join(missing_profiles)}",
                        "action": "POST /api/admin/backfills/trigger to repopulate signal profiles from event history",
                    })

            # Check user model tables
            um = result.get("user_model", {})
            if isinstance(um, dict) and not um.get("error"):
                um_error_found = False
                for table_key in ["episodes_count", "semantic_facts_count", "routines_count"]:
                    value = um.get(table_key)
                    if isinstance(value, str) and "error" in value:
                        recommendations.append({
                            "severity": "critical",
                            "area": "user_model",
                            "message": f"user_model.db query error on {table_key}: {value}",
                            "action": "POST /api/admin/rebuild-user-model to rebuild the corrupted database",
                        })
                        um_error_found = True
                        break  # One recommendation for the whole DB
                if not um_error_found:
                    for table_key in ["episodes_count", "semantic_facts_count", "routines_count"]:
                        value = um.get(table_key)
                        if value == 0:
                            friendly = table_key.replace("_count", "")
                            recommendations.append({
                                "severity": "medium",
                                "area": "user_model",
                                "message": f"{friendly} table is empty",
                                "action": "POST /api/admin/backfills/trigger to populate from event history",
                            })

            # Check predictions
            preds = result.get("predictions", {})
            if isinstance(preds, dict) and not preds.get("error"):
                if preds.get("last_24h", 0) == 0:
                    recommendations.append({
                        "severity": "high",
                        "area": "predictions",
                        "message": "No predictions generated in the last 24 hours",
                        "action": (
                            "Check signal profiles and connector status — "
                            "predictions require populated signal profiles and recent event data"
                        ),
                    })

            # Check notifications
            notifs = result.get("notifications", {})
            if isinstance(notifs, dict) and not notifs.get("error"):
                if notifs.get("total", 0) == 0 and notifs.get("last_24h", 0) == 0:
                    recommendations.append({
                        "severity": "low",
                        "area": "notifications",
                        "message": "No notifications have been generated",
                        "action": (
                            "Notifications are generated from predictions and rule actions — "
                            "ensure the prediction pipeline is healthy"
                        ),
                    })

            # Check events pipeline
            events = result.get("events_pipeline", {})
            if isinstance(events, dict) and not events.get("error"):
                if events.get("last_24h", 0) == 0:
                    recommendations.append({
                        "severity": "high",
                        "area": "events_pipeline",
                        "message": "No events received in the last 24 hours",
                        "action": (
                            "Check connector status at /admin — "
                            "at least one connector must be syncing to feed the pipeline"
                        ),
                    })

            result["recommendations"] = recommendations
            result["recommendations_count"] = len(recommendations)
        except Exception as exc:
            logger.warning("Failed to build diagnostics recommendations: %s", exc)
            result["recommendations"] = {"error": str(exc)}
            result["recommendations_count"] = 0

        return result

    @app.get("/api/system/sources")
    async def get_system_sources():
        """Return per-source event statistics for the System health dashboard.

        Queries the event log and groups by ``source`` to surface how recently
        each data source has contributed events.  A source that hasn't sent
        events for more than ``STALE_HOURS`` is flagged as stale so the user
        can diagnose connector problems (e.g. an expired OAuth token or
        network outage) before they impact their daily briefing.

        The staleness threshold is 6 hours for external connectors (which
        should sync at least hourly) and 24 hours for internal services
        (user_model_store, rules_engine) that are driven by incoming data.

        Returns a JSON object with::

            {
                "sources": [
                    {
                        "source": "google",
                        "last_event": "2026-02-22T08:54:57Z",
                        "total_events": 16545,
                        "events_24h": 0,
                        "events_7d": 0,
                        "stale": true,
                        "hours_since": 150.3
                    },
                    ...
                ],
                "generated_at": "2026-02-28T19:00:00Z",
                "stale_count": 1,
                "active_count": 4
            }
        """
        now = datetime.now(timezone.utc)
        cutoff_24h = (now - timedelta(hours=24)).isoformat()
        cutoff_7d = (now - timedelta(days=7)).isoformat()

        import asyncio

        # Offload the synchronous SQLite query to a thread so we don't block
        # the async event loop for what could be a scan of a large events table.
        raw = await asyncio.to_thread(
            life_os.event_store.get_source_stats, cutoff_24h, cutoff_7d
        )

        # Internal pipeline sources (driven by incoming events) — only flag
        # stale if silent for 24+ hours since they don't self-initiate.
        # External connectors — flag stale after 6 hours (they should sync
        # at minimum every hour via their SYNC_INTERVAL_SECONDS setting).
        INTERNAL_SOURCES = {"user_model_store", "rules_engine", "task_manager",
                            "routine_detector", "workflow_detector", "feedback_collector"}
        STALE_EXTERNAL_HOURS = 6
        STALE_INTERNAL_HOURS = 24

        sources = []
        for row in raw:
            source_name = row["source"]
            last_event_str = row["last_event"]
            hours_since: float | None = None
            stale = False

            if last_event_str:
                try:
                    last_dt = datetime.fromisoformat(last_event_str.replace("Z", "+00:00"))
                    hours_since = round((now - last_dt).total_seconds() / 3600, 1)
                    threshold = (STALE_INTERNAL_HOURS
                                 if source_name in INTERNAL_SOURCES
                                 else STALE_EXTERNAL_HOURS)
                    stale = hours_since > threshold
                except ValueError:
                    pass  # Malformed timestamp — treat as unknown

            sources.append({
                "source": source_name,
                "last_event": last_event_str,
                "total_events": row["total_events"],
                "events_24h": row["events_24h"],
                "events_7d": row["events_7d"],
                "stale": stale,
                "hours_since": hours_since,
            })

        stale_count = sum(1 for s in sources if s["stale"])
        active_count = sum(1 for s in sources if not s["stale"])

        return {
            "sources": sources,
            "generated_at": now.isoformat(),
            "stale_count": stale_count,
            "active_count": active_count,
        }

    @app.get("/api/system/intelligence")
    async def get_system_intelligence():
        """Return prediction engine diagnostics for the System health dashboard.

        Exposes the prediction engine's internal state: which prediction types
        are active or blocked, what data each generator needs, and how many
        predictions were generated in the last 7 days.  Also reports the
        current user-model depth (signal profiles, routines, workflows,
        semantic facts) so the System tab gives a complete picture of
        intelligence layer health without requiring database access.

        Returns::

            {
                "prediction_types": {
                    "reminder": {
                        "status": "active" | "limited" | "blocked",
                        "generated_last_7d": int,
                        "blockers": ["list"],
                        "recommendations": ["list"]
                    },
                    ...
                },
                "overall": {
                    "total_predictions_7d": int,
                    "active_types": int,
                    "blocked_types": int,
                    "health": "healthy" | "degraded" | "broken"
                },
                "user_model_depth": {
                    "signal_profiles": int,
                    "routines": int,
                    "workflows": int,
                    "semantic_facts": int,
                    "episodes": int
                },
                "generated_at": "ISO-8601 timestamp"
            }

        Example::

            GET /api/system/intelligence
            → {
                "overall": {"total_predictions_7d": 4, "health": "degraded", ...},
                "prediction_types": {
                    "opportunity": {"status": "active", "generated_last_7d": 4, ...},
                    "reminder": {"status": "blocked", "generated_last_7d": 0, ...}
                },
                "user_model_depth": {
                    "signal_profiles": 8, "routines": 0, "workflows": 0,
                    "semantic_facts": 23, "episodes": 56876
                }
              }
        """
        try:
            diagnostics = await life_os.prediction_engine.get_diagnostics()
        except Exception:
            diagnostics = {"prediction_types": {}, "overall": {"health": "unknown"}}

        # Augment with user-model depth metrics pulled from the user model store.
        # This surfaces how populated each intelligence layer is so the System
        # tab can explain *why* predictions might be limited (e.g. "0 routines
        # detected" directly explains why routine-deviation predictions are blocked).
        user_model_depth = {"signal_profiles": 0, "routines": 0, "workflows": 0,
                            "semantic_facts": 0, "episodes": 0}
        # Query each table independently so one missing/broken table
        # doesn't zero out the counts for all subsequent tables.
        try:
            with life_os.db.get_connection("user_model") as um_conn:
                for table_name in ("signal_profiles", "routines", "workflows",
                                   "semantic_facts", "episodes"):
                    try:
                        user_model_depth[table_name] = um_conn.execute(
                            f"SELECT COUNT(*) FROM {table_name}"  # noqa: S608
                        ).fetchone()[0]
                    except Exception as e:
                        logger.warning("Failed to count %s: %s", table_name, e)
        except Exception as e:
            logger.warning("Failed to open user_model connection: %s", e)

        diagnostics["user_model_depth"] = user_model_depth
        diagnostics["generated_at"] = datetime.now(timezone.utc).isoformat()
        return diagnostics

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
            try:
                briefing = await life_os.ai_engine.generate_briefing()
                return {"type": "briefing", "content": briefing}
            except Exception as e:
                logger.warning("Command briefing failed: %s", e)
                return {"type": "error", "content": "Briefing generation temporarily unavailable. The AI engine may be offline."}

        # --- Intent: draft a message ---
        elif command_type == "draft":
            try:
                context = text.split(" ", 1)[1]
                draft = await life_os.ai_engine.draft_reply(
                    contact_id=None, channel="email",
                    incoming_message=context,
                )
                return {"type": "draft", "content": draft}
            except Exception as e:
                logger.warning("Command draft failed: %s", e)
                return {"type": "error", "content": "Draft generation temporarily unavailable. The AI engine may be offline."}

        # --- Fallback: pass to AI engine for open-ended search/response ---
        else:
            try:
                response = await life_os.ai_engine.search_life(text)
                return {"type": "ai_response", "content": response}
            except Exception as e:
                logger.warning("Command AI response failed: %s", e)
                return {"type": "error", "content": "AI processing temporarily unavailable. The AI engine may be offline."}

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
        sections_loaded = []
        sections_failed = []

        # --- Notifications (all topics except 'system') ---
        if topic in (None, "inbox", "messages", "email"):
            try:
                notifications = life_os.notification_manager.get_pending(limit=limit)
                for n in notifications:
                    source_type = n.get("domain", "")
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
                        "metadata": {"source_event_id": n.get("source_event_id")},
                    })
                sections_loaded.append("notifications")
            except Exception as e:
                logger.warning("dashboard_feed: failed to load %s section: %s", "notifications", e)
                sections_failed.append({"section": "notifications", "error": str(e)})

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
                sections_loaded.append("tasks")
            except Exception as e:
                logger.warning("dashboard_feed: failed to load %s section: %s", "tasks", e)
                sections_failed.append({"section": "tasks", "error": str(e)})

        # --- Calendar (upcoming events for badge count) ---
        if topic in (None, "inbox", "calendar"):
            try:
                import asyncio

                now = datetime.now(timezone.utc)
                # Show events in the next 7 days for the feed/badge
                end_window = now + timedelta(days=7)
                # Generous lookback: event row timestamps don't always match
                # the calendar start_time, so fetch recent rows and filter in
                # Python.  LIMIT 500 caps the worst case.
                lookback = (now - timedelta(days=30)).isoformat()

                def _cal_query():
                    """Run the calendar query off the async event loop."""
                    with life_os.db.get_connection("events") as conn:
                        return conn.execute(
                            """SELECT id, payload, timestamp FROM events
                               WHERE type = 'calendar.event.created'
                                 AND timestamp > ?
                               ORDER BY timestamp DESC
                               LIMIT 500""",
                            (lookback,),
                        ).fetchall()

                rows = await asyncio.to_thread(_cal_query)

                # Deduplicate by event_id
                seen_eids: set[str] = set()
                for row in rows:
                    try:
                        payload = json.loads(row["payload"])
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    eid = payload.get("event_id", row["id"])
                    if eid in seen_eids:
                        continue
                    seen_eids.add(eid)

                    evt_start = payload.get("start_time", "")
                    if not evt_start:
                        continue

                    # Parse start_time to a proper datetime for comparison.
                    # Date-only strings (e.g. '2026-03-05') represent all-day
                    # events and are treated as midnight UTC on that day.
                    try:
                        if len(evt_start) <= 10:
                            evt_start_dt = datetime.fromisoformat(evt_start).replace(tzinfo=timezone.utc)
                        else:
                            evt_start_dt = datetime.fromisoformat(evt_start.replace("Z", "+00:00"))
                            # Ensure timezone-aware for comparison
                            if evt_start_dt.tzinfo is None:
                                evt_start_dt = evt_start_dt.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        continue

                    if evt_start_dt < now or evt_start_dt > end_window:
                        continue

                    start_time = payload.get("start_time", "")
                    end_time = payload.get("end_time", "")
                    location = payload.get("location", "")
                    items.append({
                        "id": row["id"],
                        "kind": "event",
                        "channel": "calendar",
                        "title": payload.get("title", "Calendar Event"),
                        "body": payload.get("description", ""),
                        "priority": "normal",
                        "timestamp": start_time,
                        "source": "calendar",
                        "metadata": {
                            "start_time": start_time,
                            "end_time": end_time,
                            "location": location,
                            "attendees": payload.get("attendees", []),
                            "is_all_day": payload.get("is_all_day", False),
                        },
                    })
                sections_loaded.append("calendar")
            except Exception as e:
                logger.warning("dashboard_feed: failed to load %s section: %s", "calendar", e)
                sections_failed.append({"section": "calendar", "error": str(e)})

        # --- Actual email events (email.received) ---
        # The notifications section above only surfaces emails that triggered a
        # rules-engine notification (very few).  This section fetches real inbound
        # email events directly from the events store so the "Email" topic always
        # has meaningful content, regardless of whether any notifications fired.
        if topic in (None, "inbox", "email"):
            try:
                from services.signal_extractor.marketing_filter import is_marketing_or_noreply

                # Collect notification source_event_ids to skip events that are
                # already represented by a notification in the feed above.
                notif_event_ids = {
                    item["metadata"].get("source_event_id")
                    for item in items
                    if item.get("kind") == "notification" and item.get("metadata")
                }

                with life_os.db.get_connection("events") as conn:
                    email_rows = conn.execute(
                        """SELECT id, payload, timestamp FROM events
                           WHERE type = 'email.received'
                           ORDER BY timestamp DESC
                           LIMIT ?""",
                        (limit * 2,),  # Fetch extra to absorb marketing/dup filtering
                    ).fetchall()

                for row in email_rows:
                    if len(items) >= limit:
                        break
                    try:
                        ep = json.loads(row["payload"])
                        if isinstance(ep, str):
                            ep = json.loads(ep)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    # Skip if this event already appears as a notification.
                    if row["id"] in notif_event_ids:
                        continue

                    # Skip marketing / automated senders — they add noise.
                    from_addr = ep.get("from_address", "")
                    if is_marketing_or_noreply(from_addr, ep):
                        continue

                    subject = ep.get("subject") or "(no subject)"
                    snippet = ep.get("snippet") or ep.get("body", "")
                    # Truncate snippet to keep feed items lightweight.
                    if snippet and len(snippet) > 300:
                        snippet = snippet[:297] + "…"

                    items.append({
                        "id": row["id"],
                        "kind": "email",
                        "channel": "email",
                        "title": subject,
                        "body": snippet,
                        "priority": "normal",
                        "timestamp": ep.get("timestamp", row["timestamp"]),
                        "source": from_addr,
                        "contact_id": from_addr,
                        "metadata": {
                            "from_address": from_addr,
                            "from_name": ep.get("from_name", ""),
                            "to_addresses": ep.get("to_addresses", []),
                            "thread_id": ep.get("thread_id"),
                            "message_id": ep.get("message_id"),
                            "has_attachments": ep.get("has_attachments", False),
                        },
                    })
                sections_loaded.append("email")
            except Exception as e:
                logger.warning("dashboard_feed: failed to load %s section: %s", "email", e)
                sections_failed.append({"section": "email", "error": str(e)})

        # --- Actual message events (message.received) ---
        # Same rationale as email: show real inbound messages even when no
        # notification was generated for them.
        if topic in (None, "inbox", "messages"):
            try:
                notif_event_ids_msg = {
                    item["metadata"].get("source_event_id")
                    for item in items
                    if item.get("kind") == "notification" and item.get("metadata")
                }

                with life_os.db.get_connection("events") as conn:
                    msg_rows = conn.execute(
                        """SELECT id, payload, timestamp FROM events
                           WHERE type = 'message.received'
                           ORDER BY timestamp DESC
                           LIMIT ?""",
                        (limit,),
                    ).fetchall()

                for row in msg_rows:
                    if len(items) >= limit:
                        break
                    try:
                        ep = json.loads(row["payload"])
                        if isinstance(ep, str):
                            ep = json.loads(ep)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    if row["id"] in notif_event_ids_msg:
                        continue

                    from_addr = ep.get("from_address", "")
                    channel = ep.get("channel", "message")
                    body = ep.get("body", "")
                    if body and len(body) > 300:
                        body = body[:297] + "…"

                    items.append({
                        "id": row["id"],
                        "kind": "message",
                        "channel": channel,
                        "title": ep.get("contact_name") or from_addr or "Message",
                        "body": body,
                        "priority": "normal",
                        "timestamp": ep.get("timestamp", row["timestamp"]),
                        "source": from_addr,
                        "contact_id": from_addr,
                        "metadata": {
                            "from_address": from_addr,
                            "channel": channel,
                            "is_group": ep.get("is_group", False),
                            "group_name": ep.get("group_name"),
                            "message_id": ep.get("message_id"),
                        },
                    })
                sections_loaded.append("messages")
            except Exception as e:
                logger.warning("dashboard_feed: failed to load %s section: %s", "messages", e)
                sections_failed.append({"section": "messages", "error": str(e)})

        # --- Enrich email items with AI-extracted action items ---
        # The task manager extracts actionable tasks from every email it processes
        # and stores them in the tasks table with a ``source_event_id`` pointing
        # back to the originating email event.  We batch-fetch those tasks here
        # and attach them as ``action_items`` lists so the dashboard card renderer
        # can display them as chips — surfacing AI intelligence directly on the
        # email card without the user having to open a drill-down view.
        #
        # Email items from email.received events use the event row id as their
        # feed item id.  Notification-based email items store the original event
        # id in metadata.source_event_id.  Both are collected here so all email
        # cards in the feed can benefit from the enrichment.
        email_event_ids: list[str] = []
        for item in items:
            if item.get("kind") == "email":
                # Direct email events: item id == events.id
                if item.get("id"):
                    email_event_ids.append(item["id"])
            elif item.get("kind") == "notification" and item.get("channel") == "email":
                # Notification-backed email items: source_event_id is in metadata
                src_id = (item.get("metadata") or {}).get("source_event_id")
                if src_id:
                    email_event_ids.append(src_id)

        if email_event_ids:
            try:
                placeholders = ",".join("?" * len(email_event_ids))
                with life_os.db.get_connection("state") as state_conn:
                    task_rows = state_conn.execute(
                        f"""SELECT source_event_id, title
                               FROM tasks
                              WHERE source_event_id IN ({placeholders})
                                AND status NOT IN ('dismissed', 'cancelled')
                           ORDER BY created_at ASC""",
                        email_event_ids,
                    ).fetchall()

                # Build event_id → [task titles] mapping in one pass
                tasks_by_event: dict[str, list[str]] = {}
                for tr in task_rows:
                    eid = tr["source_event_id"]
                    tasks_by_event.setdefault(eid, []).append(tr["title"])

                # Attach action_items to each matching feed item.
                # For direct email items, the item id is the event id.
                # For notification items, the event id lives in metadata.
                for item in items:
                    if item.get("kind") == "email":
                        action_items = tasks_by_event.get(item.get("id"))
                    elif item.get("kind") == "notification" and item.get("channel") == "email":
                        src_id = (item.get("metadata") or {}).get("source_event_id")
                        action_items = tasks_by_event.get(src_id) if src_id else None
                    else:
                        continue
                    if action_items:
                        if item.get("metadata") is None:
                            item["metadata"] = {}
                        item["metadata"]["action_items"] = action_items
            except Exception as e:
                # Action-item enrichment is non-critical.  If the tasks DB is
                # unavailable or the query fails, the feed still returns all items
                # — they just won't have action_items chips on email cards.
                logger.warning("dashboard_feed: action-item enrichment failed: %s", e)
                sections_failed.append({"section": "action_items", "error": str(e)})

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

        return {
            "items": sorted_items[:limit],
            "count": len(sorted_items[:limit]),
            "topic": topic or "inbox",
            "sections_loaded": sections_loaded,
            "sections_failed": sections_failed,
        }

    @app.get("/api/dashboard/badges")
    async def dashboard_badges():
        """Return per-topic badge counts without loading full feed items.

        Replaces the previous pattern of firing 5 separate /api/dashboard/feed
        requests with limit=100 just to get counts.  This endpoint directly
        queries the same underlying data stores but only counts items, keeping
        the payload tiny (~100 bytes vs ~50 KB for five full-feed responses).

        Counts match the ``dashboard_feed`` definition:
          - inbox:    all pending notifications + pending tasks + upcoming calendar events
          - messages: notifications whose source contains "message" or "signal"
          - email:    notifications whose source contains "email"
          - calendar: calendar.event.created events starting within the next 7 days
          - tasks:    pending tasks

        Returns:
            JSON ``{"badges": {topic: count, ...}}`` for topics inbox, messages,
            email, calendar, and tasks.

        Example response::

            {
                "badges": {
                    "inbox": 12,
                    "messages": 3,
                    "email": 7,
                    "calendar": 4,
                    "tasks": 2
                }
            }
        """
        # --- Notifications ---
        # Fetch once and count per-channel; avoids 3 separate queries.
        try:
            all_notifications = life_os.notification_manager.get_pending(limit=500)
        except Exception:
            all_notifications = []

        email_count = sum(
            1 for n in all_notifications if "email" in n.get("domain", "")
        )
        msg_count = sum(
            1 for n in all_notifications
            if "message" in n.get("domain", "") or "signal" in n.get("domain", "")
        )

        # --- Tasks ---
        try:
            pending_tasks = life_os.task_manager.get_pending_tasks(limit=500)
            task_count = len(pending_tasks)
        except Exception:
            task_count = 0

        # --- Calendar (upcoming events in the next 7 days) ---
        try:
            now = datetime.now(timezone.utc)
            end_window = (now + timedelta(days=7)).isoformat()
            with life_os.db.get_connection("events") as conn:
                # Count distinct calendar events (deduplicate by event_id payload field)
                row = conn.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT json_extract(payload, '$.event_id') AS eid
                        FROM events
                        WHERE type = 'calendar.event.created'
                          AND json_extract(payload, '$.start_time') > ?
                          AND json_extract(payload, '$.start_time') <= ?
                        GROUP BY eid
                    )
                    """,
                    (now.isoformat(), end_window),
                ).fetchone()
                cal_count = row[0] if row else 0
        except Exception:
            cal_count = 0

        # --- Insights (active, non-dismissed, non-expired) ---
        # Counts insights the user hasn't yet dismissed so the Insights tab
        # badge reflects how many fresh behavioral observations are waiting.
        # Uses the same staleness window as /api/insights/summary (TTL-driven).
        try:
            with life_os.db.get_connection("user_model") as conn:
                row = conn.execute(
                    """SELECT COUNT(*) FROM insights
                       WHERE (feedback IS NULL OR feedback NOT IN ('negative', 'dismissed', 'not_relevant'))
                         AND datetime(created_at) >
                             datetime('now', '-' || staleness_ttl_hours || ' hours')"""
                ).fetchone()
                insights_count = row[0] if row else 0
        except Exception:
            insights_count = 0

        # Inbox aggregates all three categories
        inbox_count = len(all_notifications) + task_count + cal_count

        return {
            "badges": {
                "inbox": inbox_count,
                "messages": msg_count,
                "email": email_count,
                "calendar": cal_count,
                "tasks": task_count,
                "insights": insights_count,
            }
        }

    # -------------------------------------------------------------------
    # Calendar Events
    # -------------------------------------------------------------------

    @app.get("/api/calendar/events")
    async def calendar_events(start: str, end: str):
        """Return deduplicated calendar events overlapping [start, end].

        Query params:
            start: Start of range as YYYY-MM-DD (inclusive).
            end:   End of range as YYYY-MM-DD (exclusive).

        Deduplicates by payload.event_id, keeping the most recent sync.
        Returns events whose start_time/end_time overlap the requested range.
        """
        import asyncio

        def _query():
            with life_os.db.get_connection("events") as conn:
                # Use the caller-provided start/end to narrow the SQL query.
                # json_extract filters on the payload's start_time so SQLite
                # only scans relevant rows.  LIMIT 1000 is a safety cap.
                rows = conn.execute(
                    """SELECT id, payload, timestamp
                       FROM events
                       WHERE type = 'calendar.event.created'
                         AND json_extract(payload, '$.start_time') < ?
                         AND json_extract(payload, '$.start_time') >= ?
                       ORDER BY timestamp DESC
                       LIMIT 1000""",
                    (end, start),
                ).fetchall()
            return [dict(r) for r in rows]

        rows = await asyncio.to_thread(_query)

        # Deduplicate by event_id — keep most recent sync per calendar event
        seen: dict[str, dict] = {}
        for row in rows:
            try:
                payload = json.loads(row["payload"])
                if isinstance(payload, str):
                    payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                continue

            eid = payload.get("event_id", row["id"])
            if eid in seen:
                continue  # rows already ordered by timestamp DESC
            seen[eid] = payload

        # Filter to events overlapping [start, end)
        results = []
        for payload in seen.values():
            evt_start = payload.get("start_time", "")
            evt_end = payload.get("end_time", "")
            if not evt_start:
                continue
            # An event overlaps if it starts before range-end AND ends after range-start
            if evt_start < end and (evt_end > start if evt_end else evt_start >= start):
                results.append({
                    "id": payload.get("event_id", ""),
                    "title": payload.get("title", ""),
                    "start_time": evt_start,
                    "end_time": evt_end,
                    "is_all_day": payload.get("is_all_day", False),
                    "location": payload.get("location", ""),
                    "attendees": payload.get("attendees", []),
                    "description": payload.get("description", ""),
                    "calendar_id": payload.get("calendar_id", ""),
                })

        # Sort by start_time
        results.sort(key=lambda e: e["start_time"])
        return {"events": results, "count": len(results)}

    # -------------------------------------------------------------------
    # Briefing
    # -------------------------------------------------------------------

    @app.get("/api/briefing")
    async def get_briefing():
        """Generate the user's daily briefing via the AI engine."""
        try:
            briefing = await life_os.ai_engine.generate_briefing()
            return {"briefing": briefing, "generated_at": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            logger.warning("Briefing generation failed: %s", e)
            return {
                "briefing": None,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "error": "Briefing generation temporarily unavailable",
            }

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------

    @app.post("/api/search")
    async def search(req: SearchRequest):
        """Perform semantic vector search across ingested events."""
        try:
            results = life_os.vector_store.search(
                req.query, limit=req.limit, filter_metadata=req.filters
            )
            return {"query": req.query, "results": results, "count": len(results)}
        except Exception as e:
            logger.warning("Search failed: %s", e)
            return {"query": req.query, "results": [], "count": 0, "error": "Search temporarily unavailable"}

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

    @app.get("/api/tasks/stats")
    async def get_task_stats():
        """Get task dashboard statistics.

        Returns summary counts for the task management widget:
        pending tasks, tasks completed today, overdue tasks,
        and a breakdown of pending tasks by life domain.
        """
        try:
            return life_os.task_manager.get_task_stats()
        except Exception as e:
            logger.warning("Failed to get task stats: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.patch("/api/tasks/{task_id}")
    async def update_task(task_id: str, req: TaskUpdateRequest):
        try:
            await life_os.task_manager.update_task(task_id, **req.model_dump(exclude_none=True))
            return {"status": "updated"}
        except Exception as e:
            logger.warning("Failed to update task '%s': %s", task_id, e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/api/tasks/{task_id}/complete")
    async def complete_task(task_id: str):
        try:
            await life_os.task_manager.complete_task(task_id)
            return {"status": "completed"}
        except Exception as e:
            logger.warning("Failed to complete task '%s': %s", task_id, e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    # -------------------------------------------------------------------
    # Notifications
    # -------------------------------------------------------------------

    async def _classify_notification_source(notif_id: str) -> Optional[str]:
        """Look up a notification's originating source and classify it for weight learning.

        Traces from notification -> source_event_id -> event -> source_key.
        For prediction-domain notifications without a direct source event,
        falls back to domain-based classification using the same mapping as
        the insight_feedback handler.

        Returns the source_key string (e.g. "email.work") or None if
        classification is not possible.
        """
        import asyncio

        def _sync_classify(nid: str) -> Optional[str]:
            """Synchronous DB lookups, run via asyncio.to_thread to avoid blocking."""
            # Step 1: Look up the notification to get source_event_id and domain
            with life_os.db.get_connection("state") as conn:
                notif = conn.execute(
                    "SELECT source_event_id, domain FROM notifications WHERE id = ?",
                    (nid,),
                ).fetchone()

            if not notif:
                return None

            source_event_id = notif["source_event_id"]
            domain = notif["domain"]

            # Step 2: If there's a source event, look it up and classify it
            if source_event_id:
                with life_os.db.get_connection("events") as conn:
                    event_row = conn.execute(
                        "SELECT type, payload, metadata FROM events WHERE id = ?",
                        (source_event_id,),
                    ).fetchone()

                if event_row:
                    event = {
                        "type": event_row["type"],
                        "payload": json.loads(event_row["payload"] or "{}"),
                        "metadata": json.loads(event_row["metadata"] or "{}"),
                    }
                    source_key = life_os.source_weight_manager.classify_event(event)
                    if source_key:
                        return source_key

            # Step 3: Fallback for domain-based classification.
            # Domains that map naturally to a single source weight key get classified;
            # cross-domain origins like "prediction" return None to avoid misattributing
            # weight updates to an unrelated source (e.g. email.work).
            if domain:
                domain_to_source = {
                    "email": "email.work",
                    "messaging": "messaging.direct",
                    "calendar": "calendar.meetings",
                    "finance": "finance.transactions",
                    "health": "health.activity",
                    "location": "location.visits",
                    "home": "home.devices",
                }
                return domain_to_source.get(domain)

            return None

        return await asyncio.to_thread(_sync_classify, notif_id)

    @app.get("/api/notifications")
    async def list_notifications(limit: int = 50):
        try:
            notifications = life_os.notification_manager.get_pending(limit=limit)
            return {"notifications": notifications}
        except Exception as e:
            logger.warning("Failed to list notifications: %s", e)
            return JSONResponse(status_code=500, content={"notifications": [], "error": str(e)})

    @app.post("/api/notifications/{notif_id}/read")
    async def mark_read(notif_id: str):
        try:
            await life_os.notification_manager.mark_read(notif_id)
            return {"status": "read"}
        except Exception as e:
            logger.warning("Failed to mark notification '%s' as read: %s", notif_id, e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/api/notifications/{notif_id}/dismiss")
    async def dismiss_notification(notif_id: str):
        await life_os.notification_manager.dismiss(notif_id)
        # Update source weights — dismissal is a negative signal
        try:
            if hasattr(life_os, "source_weight_manager"):
                source_key = await _classify_notification_source(notif_id)
                if source_key:
                    life_os.source_weight_manager.record_dismissal(source_key)
        except Exception as e:
            logger.debug("Source weight feedback failed: %s", e)  # Source weight feedback is non-critical
        return {"status": "dismissed"}

    @app.post("/api/notifications/{notif_id}/act")
    async def act_on_notification(notif_id: str):
        await life_os.notification_manager.mark_acted_on(notif_id)
        # Update source weights — acting on a notification is a positive signal
        try:
            if hasattr(life_os, "source_weight_manager"):
                source_key = await _classify_notification_source(notif_id)
                if source_key:
                    life_os.source_weight_manager.record_engagement(source_key)
        except Exception as e:
            logger.debug("Source weight feedback failed: %s", e)  # Source weight feedback is non-critical
        return {"status": "acted_on"}

    @app.get("/api/notifications/digest")
    async def get_digest():
        try:
            digest = await life_os.notification_manager.get_digest()
            return {"digest": digest}
        except Exception as e:
            logger.warning("Failed to get notification digest: %s", e)
            return JSONResponse(status_code=500, content={"digest": None, "error": str(e)})

    # -------------------------------------------------------------------
    # Draft Messages
    # -------------------------------------------------------------------

    @app.post("/api/draft")
    async def draft_message(req: DraftRequest):
        """Generate an AI-drafted reply for a given message."""
        try:
            draft = await life_os.ai_engine.draft_reply(
                contact_id=req.contact_id,
                channel=req.channel,
                incoming_message=req.incoming_message,
            )
            return {"draft": draft}
        except Exception as e:
            logger.warning("Draft generation failed: %s", e)
            return {"draft": None, "error": "Draft generation temporarily unavailable"}

    @app.post("/api/messages/send")
    async def send_message(req: SendMessageRequest):
        """Send a direct message via the appropriate messaging connector.

        Routes the outbound message to whichever connector matches the
        requested channel:
          - ``channel="imessage"`` → iMessage connector (AppleScript-based)
          - ``channel="signal"``   → Signal connector (signal-cli-based)
          - ``channel="message"``  → tries iMessage then Signal (first active wins)

        The connector's ``execute("send_message", {...})`` method handles the
        actual delivery.  If no matching connector is active (e.g. not
        configured), the endpoint returns ``{"status": "no_connector"}`` so the
        UI can show a helpful message rather than a 500 error.

        Args:
            req: SendMessageRequest with ``recipient``, ``message``, and
                 optional ``channel`` hint.

        Returns:
            ``{"status": "sent", "details": {...}}`` on success.
            ``{"status": "no_connector", "details": "..."}`` when no connector.
            ``{"status": "error", "details": "..."}`` on connector error.

        Example::

            POST /api/messages/send
            {"recipient": "+15555550100", "message": "See you at 3!", "channel": "imessage"}
            -> {"status": "sent", "details": {"title": "See you at 3!"}}
        """
        # --- Build the ordered list of candidate connector IDs ---
        # Explicit channel names map directly to their connector IDs.
        # The generic "message" channel tries iMessage then Signal so the user
        # gets a best-effort send without knowing which connector is active.
        channel = req.channel.lower()
        if channel == "imessage":
            candidate_ids = ["imessage"]
        elif channel == "signal":
            candidate_ids = ["signal"]
        else:
            # Generic "message" or unknown channel: try both in preference order
            candidate_ids = ["imessage", "signal"]

        # --- Find the first active connector from the candidates ---
        connector = None
        matched_id = None
        for cid in candidate_ids:
            c = life_os.connector_map.get(cid)
            if c is not None:
                connector = c
                matched_id = cid
                break

        if connector is None:
            return {
                "status": "no_connector",
                "details": (
                    f"No active messaging connector found for channel '{req.channel}'. "
                    "Configure iMessage or Signal in the Admin panel to enable sending."
                ),
            }

        # --- Delegate to the connector's execute() method ---
        try:
            result = await connector.execute(
                "send_message",
                {"recipient": req.recipient, "message": req.message},
            )
            # Connectors return {"status": "sent", ...} or {"status": "error", ...}
            if isinstance(result, dict) and result.get("status") == "error":
                return {"status": "error", "details": result.get("details", "Unknown error")}
            return {"status": "sent", "connector": matched_id, "details": result}
        except Exception as e:
            logger.error("send_message via %s failed: %s", matched_id, e)
            return {"status": "error", "details": str(e)}

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
        try:
            return life_os.signal_extractor.get_user_summary()
        except Exception as e:
            logger.warning("Failed to get user model summary: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/api/user-model/facts")
    async def get_facts(min_confidence: float = 0.0):
        """Return semantic facts, optionally filtered by minimum confidence."""
        try:
            facts = life_os.user_model_store.get_semantic_facts(min_confidence=min_confidence)
            return {"facts": facts}
        except Exception as e:
            logger.warning("Failed to get semantic facts: %s", e)
            return JSONResponse(status_code=500, content={"facts": [], "error": str(e)})

    @app.delete("/api/user-model/facts/{key}")
    async def delete_fact(key: str):
        """Delete a single semantic fact by key (user correction flow)."""
        try:
            with life_os.db.get_connection("user_model") as conn:
                conn.execute("DELETE FROM semantic_facts WHERE key = ?", (key,))
            return {"status": "deleted"}
        except Exception as e:
            logger.warning("Failed to delete fact '%s': %s", key, e)
            return JSONResponse(status_code=500, content={"error": str(e)})

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

        # Fast-path: skip query if user_model.db is known to be corrupted
        if getattr(life_os.db, "user_model_degraded", False) is True:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "User model database is temporarily unavailable",
                    "detail": "Database corruption detected. The system is attempting automatic recovery.",
                },
            )

        try:
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
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            logger.warning("correct_fact: user_model.db query failed: %s", e)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "User model database is temporarily unavailable",
                    "detail": "Database corruption detected. The system is attempting automatic recovery.",
                },
            )

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

    @app.post("/api/user-model/facts/{key}/confirm")
    async def confirm_fact(key: str, request: FactConfirmationRequest = FactConfirmationRequest()):
        """Confirm a semantic fact is correct, bumping confidence by +0.05.

        This is the positive counterpart to the correction endpoint: when the
        user explicitly confirms that an inferred fact is accurate, this
        endpoint increments ``times_confirmed`` and increases confidence by
        the architectural standard of +0.05 (capped at 1.0).  This closes
        the positive feedback loop that was previously missing — confidence
        could only grow through automated re-inference, never through
        explicit user validation.

        Args:
            key: The semantic fact key to confirm
            request: Optional reason for the confirmation

        Returns:
            Confirmed fact with old and new confidence values
        """
        import json
        from datetime import datetime, timezone

        # Fast-path: skip query if user_model.db is known to be corrupted
        if getattr(life_os.db, "user_model_degraded", False) is True:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "User model database is temporarily unavailable",
                    "detail": "Database corruption detected. The system is attempting automatic recovery.",
                },
            )

        try:
            with life_os.db.get_connection("user_model") as conn:
                existing = conn.execute(
                    "SELECT * FROM semantic_facts WHERE key = ?", (key,)
                ).fetchone()

                if not existing:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=404, detail=f"Fact with key '{key}' not found")

                old_confidence = existing["confidence"]
                # Confidence grows by +0.05 per confirmation, capped at 1.0.
                # Round to 2 decimal places to avoid floating point drift.
                new_confidence = round(min(1.0, old_confidence + 0.05), 2)

                conn.execute(
                    """UPDATE semantic_facts
                       SET confidence = ?,
                           times_confirmed = times_confirmed + 1,
                           last_confirmed = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                       WHERE key = ?""",
                    (new_confidence, key),
                )

                updated = conn.execute(
                    "SELECT * FROM semantic_facts WHERE key = ?", (key,)
                ).fetchone()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            logger.warning("confirm_fact: user_model.db query failed: %s", e)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "User model database is temporarily unavailable",
                    "detail": "Database corruption detected. The system is attempting automatic recovery.",
                },
            )

        # Log confirmation to the feedback collector for analytics
        if life_os.feedback_collector:
            await life_os.feedback_collector._store_feedback({
                "action_id": f"fact_confirmation_{key}",
                "action_type": "semantic_fact",
                "feedback_type": "confirmed",
                "response_latency_seconds": 0,
                "context": {
                    "fact_key": key,
                    "old_confidence": old_confidence,
                    "new_confidence": new_confidence,
                    "reason": request.reason,
                },
                "notes": request.reason,
            })

        # Publish telemetry event for the confirmation
        if life_os.event_bus and life_os.event_bus.is_connected:
            await life_os.event_bus.publish(
                "usermodel.fact.confirmed",
                {
                    "key": key,
                    "old_confidence": old_confidence,
                    "new_confidence": new_confidence,
                    "category": updated["category"],
                    "times_confirmed": updated["times_confirmed"],
                    "confirmed_at": datetime.now(timezone.utc).isoformat(),
                },
                source="web_api",
            )

        return {
            "status": "confirmed",
            "fact": dict(updated),
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "times_confirmed": updated["times_confirmed"],
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

    @app.get("/api/user-model/templates")
    async def get_communication_templates(
        contact_id: Optional[str] = None,
        channel: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 50,
    ):
        """Return learned communication style templates from Layer 3 (procedural memory).

        Communication templates are per-contact, per-channel writing-style summaries
        derived from analyzing the user's outbound messages and the contact's inbound
        messages.  Each template captures:

        - **context**: Direction — ``user_to_contact`` (user's outbound style) or
          ``contact_to_user`` (contact's inbound style).
        - **contact_id**: Email or phone of the other party.
        - **channel**: Communication channel — ``email``, ``message``, etc.
        - **greeting** / **closing**: Detected opening and sign-off phrases.
        - **formality**: 0.0 (casual) to 1.0 (formal).
        - **typical_length**: Average word count for messages to/from this contact.
        - **uses_emoji**: Whether emoji appear in the conversation.
        - **samples_analyzed**: Number of messages used to derive this template.

        Templates are populated by the relationship extractor on every inbound/outbound
        communication event, and are used by the AI engine to style-match when
        generating draft replies.  This endpoint surfaces them so the user can verify
        what writing patterns the system has learned.

        Query params:
            contact_id: Filter to a specific contact (email/phone).
            channel: Filter to a specific channel (e.g. ``email``).
            context: Filter to ``user_to_contact`` or ``contact_to_user``.
            limit: Max templates to return (default 50, capped at 200).

        Returns::

            {
                "templates": [
                    {
                        "id": "...",
                        "context": "user_to_contact",
                        "contact_id": "alice@example.com",
                        "channel": "email",
                        "greeting": "hey",
                        "closing": "thanks",
                        "formality": 0.25,
                        "typical_length": 42.0,
                        "uses_emoji": false,
                        "common_phrases": ["sounds good", "let me know"],
                        "avoids_phrases": [],
                        "tone_notes": ["casual"],
                        "samples_analyzed": 17,
                        "updated_at": "2026-02-20T10:15:30.000Z"
                    }
                ],
                "total": 1,
                "generated_at": "..."
            }
        """
        import json as _json

        # Cap limit to prevent excessively large responses.
        limit = min(limit, 200)

        conditions = []
        params: list = []

        if contact_id is not None:
            conditions.append("contact_id = ?")
            params.append(contact_id)
        if channel is not None:
            conditions.append("channel = ?")
            params.append(channel)
        if context is not None:
            conditions.append("context = ?")
            params.append(context)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        try:
            with life_os.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    f"""SELECT id, context, contact_id, channel, greeting, closing,
                               formality, typical_length, uses_emoji,
                               common_phrases, avoids_phrases, tone_notes,
                               samples_analyzed, updated_at
                          FROM communication_templates
                         {where_clause}
                         ORDER BY samples_analyzed DESC, updated_at DESC
                         LIMIT ?""",
                    params + [limit],
                ).fetchall()

            templates = []
            for row in rows:
                t = dict(row)
                # Deserialize JSON list fields so callers receive native arrays.
                t["common_phrases"] = _json.loads(t["common_phrases"] or "[]")
                t["avoids_phrases"] = _json.loads(t["avoids_phrases"] or "[]")
                t["tone_notes"] = _json.loads(t["tone_notes"] or "[]")
                t["uses_emoji"] = bool(t["uses_emoji"])
                templates.append(t)

            return {
                "templates": templates,
                "total": len(templates),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error("Failed to fetch communication templates: %s", e)
            return {"templates": [], "total": 0, "generated_at": datetime.now(timezone.utc).isoformat()}

    @app.delete("/api/user-model/templates/{template_id}")
    async def delete_communication_template(template_id: str):
        """Delete a communication template by ID.

        Removes a template that the system inferred incorrectly, allowing the
        relationship extractor to re-learn the style from future messages.

        Returns 404 if no template with the given ID exists.
        """
        try:
            deleted = life_os.user_model_store.delete_communication_template(template_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Template not found")
            return {"deleted": True, "template_id": template_id}
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to delete template %s: %s", template_id, e)
            raise HTTPException(status_code=500, detail="Internal error")

    @app.patch("/api/user-model/templates/{template_id}")
    async def update_communication_template(template_id: str, request: TemplateUpdateRequest):
        """Partially update a communication template.

        Accepts any subset of mutable style fields (greeting, closing,
        formality, typical_length, uses_emoji, common_phrases, avoids_phrases,
        tone_notes). Structural fields (id, context, contact_id, channel) are
        immutable and cannot be changed via this endpoint.

        Returns the full updated template on success, 404 if not found, or
        400 if no updatable fields were provided.
        """
        try:
            updates = request.model_dump(exclude_unset=True)
            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update")
            result = life_os.user_model_store.update_communication_template(template_id, updates)
            if result is None:
                raise HTTPException(status_code=404, detail="Template not found")
            return {"template": result}
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to update template %s: %s", template_id, e)
            raise HTTPException(status_code=500, detail="Internal error")

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

    @app.get("/api/contacts/{contact_email}/interactions")
    async def get_contact_interactions(
        contact_email: str,
        limit: int = 5,
    ):
        """Return recent interactions with a specific contact.

        Queries the events database for the last N email/message exchanges
        with the given contact, plus the contact's profile from entities.db
        if one exists.

        Args:
            contact_email: URL-decoded email address or identifier to match.
            limit: Number of interactions to return (default 5, max 20).

        Returns:
            ``{"contact_email": "...", "contact": {...}|null,
              "interactions": [...], "total_interactions": N}``
        """
        import json as _json
        from urllib.parse import unquote

        decoded_email = unquote(contact_email)
        effective_limit = min(max(limit, 1), 20)

        # --- Contact lookup from entities.db ---
        contact = None
        try:
            with life_os.db.get_connection("entities") as conn:
                row = conn.execute(
                    """
                    SELECT id, name, aliases, emails, phones, channels,
                           relationship, is_priority, preferred_channel,
                           always_surface, typical_response_time,
                           last_contact, contact_frequency_days,
                           communication_style, notes, created_at
                    FROM contacts
                    WHERE id IN (
                        SELECT contacts.id FROM contacts, json_each(contacts.emails)
                        WHERE json_each.value = ?
                    )
                    LIMIT 1
                    """,
                    (decoded_email,),
                ).fetchone()
                if row:
                    contact = dict(row)
                    for json_field in ("aliases", "emails", "phones", "notes"):
                        raw = contact.get(json_field)
                        if isinstance(raw, str):
                            try:
                                contact[json_field] = _json.loads(raw)
                            except (_json.JSONDecodeError, ValueError):
                                contact[json_field] = []
                    contact["is_priority"] = bool(contact.get("is_priority", 0))
                    contact["always_surface"] = bool(contact.get("always_surface", 0))
        except Exception as e:
            logger.warning("Contact lookup failed for %s: %s", decoded_email, e)

        # --- Interaction history from events.db ---
        interaction_types = (
            "email.received", "email.sent",
            "message.received", "message.sent",
            "imessage.received", "imessage.sent",
        )
        type_placeholders = ",".join("?" for _ in interaction_types)

        # Match contact on from_address (inbound), to_addresses (outbound),
        # or sender (fallback).  to_addresses is a JSON array so we use LIKE.
        interaction_query = f"""
            SELECT id, type, timestamp, payload
            FROM events
            WHERE type IN ({type_placeholders})
              AND (
                  json_extract(payload, '$.from_address') = ?
                  OR payload LIKE ?
                  OR json_extract(payload, '$.sender') = ?
              )
            ORDER BY timestamp DESC
            LIMIT ?
        """

        # Count query uses same WHERE clause but no LIMIT.
        count_query = f"""
            SELECT COUNT(*)
            FROM events
            WHERE type IN ({type_placeholders})
              AND (
                  json_extract(payload, '$.from_address') = ?
                  OR payload LIKE ?
                  OR json_extract(payload, '$.sender') = ?
              )
        """

        # The LIKE pattern wraps the email in quotes to match within a JSON
        # array value like ["alice@example.com", "bob@example.com"].
        like_pattern = f'%"{decoded_email}"%'
        base_params = list(interaction_types) + [decoded_email, like_pattern, decoded_email]

        interactions = []
        total_interactions = 0
        try:
            with life_os.db.get_connection("events") as conn:
                total_interactions = conn.execute(
                    count_query, base_params
                ).fetchone()[0]

                rows = conn.execute(
                    interaction_query, base_params + [effective_limit]
                ).fetchall()

            for row in rows:
                row_dict = dict(row)
                payload = {}
                if isinstance(row_dict.get("payload"), str):
                    try:
                        payload = _json.loads(row_dict["payload"])
                    except (_json.JSONDecodeError, ValueError):
                        pass

                # Determine channel from event type
                evt_type = row_dict.get("type", "")
                if "email" in evt_type:
                    channel = "email"
                elif "imessage" in evt_type:
                    channel = "imessage"
                else:
                    channel = "message"

                # Build snippet from body/text content
                body = payload.get("body") or payload.get("text") or payload.get("content") or ""
                snippet = body[:100] + ("..." if len(body) > 100 else "")

                interactions.append({
                    "id": row_dict["id"],
                    "type": evt_type,
                    "timestamp": row_dict.get("timestamp", ""),
                    "subject": payload.get("subject", ""),
                    "snippet": snippet,
                    "channel": channel,
                })
        except Exception as e:
            logger.error("Failed to query interactions for %s: %s", decoded_email, e)

        return {
            "contact_email": decoded_email,
            "contact": contact,
            "interactions": interactions,
            "total_interactions": total_interactions,
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
        # Capture the return value so we can use it as a fallback if the
        # DB read below fails (e.g. user_model.db corruption).
        generated: list = []
        try:
            generated = await life_os.insight_engine.generate_insights()
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
                       WHERE (feedback IS NULL OR feedback NOT IN ('negative', 'dismissed', 'not_relevant'))
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

        # Fallback: if DB read returned nothing but generate_insights produced
        # results, serialize the in-memory Insight objects directly so the API
        # still returns useful data in degraded mode (e.g. user_model.db corruption).
        if not insights and generated:
            for ins in generated:
                insights.append(ins.model_dump())

        return {
            "insights": insights,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/insights")
    async def list_insights(limit: int = 20):
        """Return recent insights from the InsightEngine."""
        try:
            with life_os.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT * FROM insights
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (limit,),
                ).fetchall()
        except Exception as e:
            logger.warning("Failed to read insights from database: %s", e)
            # Fallback: generate insights in-memory and return those instead
            # of an empty list, so the API is useful even when user_model.db
            # is corrupted.
            fallback: list[dict] = []
            try:
                generated = await life_os.insight_engine.generate_insights()
                for ins in (generated or [])[:limit]:
                    fallback.append(ins.model_dump())
            except Exception:
                logger.exception("Fallback generate_insights() also failed in /api/insights")
            return {"insights": fallback, "error": str(e)}
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
        """Record user feedback on an insight (useful/dismissed/not_relevant).

        Also updates source weight engagement/dismissal counters so the
        AI drift can learn from the user's response patterns.
        """
        if feedback not in ("useful", "dismissed", "not_relevant", "negative"):
            raise HTTPException(400, f"Invalid feedback value: {feedback}. Must be 'useful', 'dismissed', or 'not_relevant'.")

        # Fast-path: skip query if user_model.db is known to be corrupted
        if getattr(life_os.db, "user_model_degraded", False) is True:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "User model database is temporarily unavailable",
                    "detail": "Database corruption detected. The system is attempting automatic recovery.",
                },
            )

        # Look up the insight to find its source category for weight feedback
        source_key = None
        try:
            with life_os.db.get_connection("user_model") as conn:
                row = conn.execute(
                    "SELECT category, entity FROM insights WHERE id = ?",
                    (insight_id,),
                ).fetchone()
                conn.execute(
                    "UPDATE insights SET feedback = ? WHERE id = ?",
                    (feedback, insight_id),
                )
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            logger.warning("insight_feedback: user_model.db query failed: %s", e)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "User model database is temporarily unavailable",
                    "detail": "Database corruption detected. The system is attempting automatic recovery.",
                },
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
                # Email peak-hour insights (from _email_peak_hour_insights)
                "email_timing": "email.work",
                # Meeting density insights (from _meeting_density_insights)
                "meeting_density": "calendar.meetings",
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
            except Exception as e:
                logger.debug("Source weight feedback failed: %s", e)  # Source weight feedback is non-critical

        # When user marks an insight as "not about me", create a suppression fact
        # so the system learns not to regenerate similar insights.
        if feedback == "not_relevant" and row:
            try:
                suppression_key = f"relevance_suppression:{row['category']}:{row['entity'] or 'general'}"
                suppression_value = json.dumps({
                    "insight_id": insight_id,
                    "category": row["category"],
                    "entity": row["entity"],
                    "reason": "User indicated this insight is not about them",
                })
                with life_os.db.get_connection("user_model") as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO semantic_facts
                           (key, category, value, confidence, is_user_corrected, source_episodes)
                           VALUES (?, 'relevance_suppression', ?, 1.0, 1, '[]')""",
                        (suppression_key, suppression_value),
                    )
            except Exception:
                logger.exception("Failed to create relevance suppression fact")

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
        # Fast-path: skip query if user_model.db is known to be corrupted
        if getattr(life_os.db, "user_model_degraded", False) is True:
            return JSONResponse(
                status_code=503,
                content={
                    "predictions": [],
                    "count": 0,
                    "error": "User model database is temporarily unavailable",
                },
            )

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

        try:
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
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            logger.warning("list_predictions: user_model.db query failed: %s", e)
            return JSONResponse(
                status_code=503,
                content={
                    "predictions": [],
                    "count": 0,
                    "error": "User model database is temporarily unavailable",
                },
            )

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
        # Fast-path: skip query if user_model.db is known to be corrupted
        if getattr(life_os.db, "user_model_degraded", False) is True:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "User model database is temporarily unavailable",
                    "detail": "Database corruption detected. The system is attempting automatic recovery.",
                },
            )

        # Verify the prediction exists before resolving to give a clear 404
        try:
            with life_os.db.get_connection("user_model") as conn:
                row = conn.execute(
                    "SELECT id FROM predictions WHERE id = ?",
                    (prediction_id,),
                ).fetchone()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            logger.warning("prediction_feedback: user_model.db query failed: %s", e)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "User model database is temporarily unavailable",
                    "detail": "Database corruption detected. The system is attempting automatic recovery.",
                },
            )

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
        try:
            with life_os.db.get_connection("preferences") as conn:
                rows = conn.execute("SELECT key, value, set_by, updated_at FROM user_preferences").fetchall()
                return {"preferences": [dict(r) for r in rows]}
        except Exception as e:
            logger.warning("Failed to get preferences: %s", e)
            return JSONResponse(status_code=500, content={"preferences": [], "error": str(e)})

    @app.put("/api/preferences")
    async def update_preference(req: PreferenceUpdate):
        try:
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
        except Exception as e:
            logger.warning("Failed to update preference '%s': %s", req.key, e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    # -------------------------------------------------------------------
    # Feedback
    # -------------------------------------------------------------------

    @app.post("/api/feedback")
    async def submit_feedback(req: FeedbackRequest):
        try:
            await life_os.feedback_collector.process_explicit_feedback(req.message)
            return {"status": "received"}
        except Exception as e:
            logger.warning("Failed to process feedback: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

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
        """Return the current status of the browser orchestrator."""
        try:
            if life_os.browser_orchestrator is None:
                return {"status": "unavailable", "error": "Browser orchestrator not configured"}
            return life_os.browser_orchestrator.get_status()
        except Exception as e:
            logger.warning("Browser status check failed: %s", e)
            return {"status": "unavailable", "error": "Browser orchestrator not configured"}

    @app.get("/api/browser/vault")
    async def browser_vault_sites():
        """Return the list of credential vault sites for browser connectors."""
        try:
            if life_os.browser_orchestrator is None:
                return {"sites": [], "error": "Browser orchestrator not configured"}
            return {"sites": life_os.browser_orchestrator.get_vault_sites()}
        except Exception as e:
            logger.warning("Browser vault lookup failed: %s", e)
            return {"sites": [], "error": "Browser orchestrator not configured"}

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
        except Exception as e:
            logger.warning("Context event bus publish failed for %s: %s", req.type, e)

        # If location event, update places database
        if req.type == "context.location" and req.payload.latitude is not None:
            try:
                _update_place_from_context(life_os, req.payload)
            except Exception as e:
                logger.warning("Place update from context event failed: %s", e)

        # If device event, try to correlate with contacts
        if req.type == "context.device_nearby" and req.payload.device_name:
            try:
                _correlate_device_with_contact(life_os, req.payload)
            except Exception as e:
                logger.warning("Device-contact correlation failed for %s: %s", req.payload.device_name, e)

        return {"status": "received", "event_id": event_id}

    @app.post("/api/context/batch")
    async def submit_context_batch(req: ContextBatchRequest):
        """Ingest a batch of context events from a mobile device."""
        # Same type mapping as the single-event endpoint so batch events
        # are stored with the correct internal type for pipeline routing.
        event_type_map = {
            "context.location": "location.changed",
            "context.device_nearby": "home.device.state_changed",
            "context.time": "system.user.command",
            "context.background_refresh": "system.connector.sync_complete",
            "context.background_processing": "system.connector.sync_complete",
        }

        event_ids = []
        publish_fail_count = 0
        for event_req in req.events:
            ts = event_req.timestamp or datetime.now(timezone.utc).isoformat()
            internal_type = event_type_map.get(event_req.type, "system.user.command")
            event = {
                "type": internal_type,
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
            except Exception as e:
                publish_fail_count += 1
                logger.warning("Context batch event bus publish failed for %s: %s", event_req.type, e)

        return {"status": "received", "count": len(event_ids), "event_ids": event_ids, "publish_failures": publish_fail_count}

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
                                # Update source weights — dismissal is a negative signal
                                try:
                                    if hasattr(life_os, "source_weight_manager"):
                                        source_key = await _classify_notification_source(notif_id)
                                        if source_key:
                                            life_os.source_weight_manager.record_dismissal(source_key)
                                except Exception as e:
                                    logger.debug("Source weight feedback failed: %s", e)  # Source weight feedback is non-critical

                        elif cmd == "act_on_notification":
                            notif_id = msg.get("notification_id")
                            if notif_id:
                                await life_os.notification_manager.mark_acted_on(notif_id)
                                # Update source weights — engagement is a positive signal
                                try:
                                    if hasattr(life_os, "source_weight_manager"):
                                        source_key = await _classify_notification_source(notif_id)
                                        if source_key:
                                            life_os.source_weight_manager.record_engagement(source_key)
                                except Exception as e:
                                    logger.debug("Source weight feedback failed: %s", e)  # Source weight feedback is non-critical

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
    # Admin — Pipeline Health Diagnostics
    # -------------------------------------------------------------------

    @app.get("/admin/pipeline-health")
    async def pipeline_health():
        """Return a comprehensive diagnostic snapshot of all pipeline components.

        Each database probe and pipeline metric is wrapped in its own
        try/except so that a corrupted or unavailable database never
        blocks the healthy ones from reporting.
        """
        health: dict = {"databases": {}, "pipeline": {}, "source_weights": {}}

        # --- Database probes ---
        # Each probe runs an isolated COUNT query against a known table.
        for db_name, probe_sql in [
            ("events", "SELECT COUNT(*) FROM events"),
            ("user_model", "SELECT COUNT(*) FROM signal_profiles"),
            ("state", "SELECT COUNT(*) FROM tasks"),
            ("preferences", "SELECT COUNT(*) FROM rules"),
            ("entities", "SELECT COUNT(*) FROM contacts"),
        ]:
            try:
                with life_os.db.get_connection(db_name) as conn:
                    count = conn.execute(probe_sql).fetchone()[0]
                health["databases"][db_name] = {"status": "ok", "probe_count": count}
            except Exception as e:
                health["databases"][db_name] = {"status": "error", "error": str(e)}

        # --- Pipeline component metrics ---
        # Signal profiles
        try:
            with life_os.db.get_connection("user_model") as conn:
                count = conn.execute("SELECT COUNT(*) FROM signal_profiles").fetchone()[0]
            health["pipeline"]["signal_profiles"] = count
        except Exception as e:
            health["pipeline"]["signal_profiles"] = {"error": str(e)}

        # Episodes
        try:
            with life_os.db.get_connection("user_model") as conn:
                count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            health["pipeline"]["episodes"] = count
        except Exception as e:
            health["pipeline"]["episodes"] = {"error": str(e)}

        # Predictions
        try:
            with life_os.db.get_connection("user_model") as conn:
                count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            health["pipeline"]["predictions"] = count
        except Exception as e:
            health["pipeline"]["predictions"] = {"error": str(e)}

        # Pending notifications
        try:
            with life_os.db.get_connection("state") as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM notifications WHERE status IN ('pending', 'delivered')"
                ).fetchone()[0]
            health["pipeline"]["pending_notifications"] = count
        except Exception as e:
            health["pipeline"]["pending_notifications"] = {"error": str(e)}

        # Pending tasks
        try:
            with life_os.db.get_connection("state") as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE completed_at IS NULL"
                ).fetchone()[0]
            health["pipeline"]["pending_tasks"] = count
        except Exception as e:
            health["pipeline"]["pending_tasks"] = {"error": str(e)}

        # Events in the last 24 hours
        try:
            with life_os.db.get_connection("events") as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE timestamp > datetime('now', '-1 day')"
                ).fetchone()[0]
            health["pipeline"]["events_last_24h"] = count
        except Exception as e:
            health["pipeline"]["events_last_24h"] = {"error": str(e)}

        # --- Source weight activity ---
        try:
            with life_os.db.get_connection("preferences") as conn:
                row = conn.execute(
                    "SELECT COUNT(*) as total, "
                    "SUM(CASE WHEN user_set_at IS NOT NULL THEN 1 ELSE 0 END) as user_set, "
                    "SUM(CASE WHEN ai_drift != 0 THEN 1 ELSE 0 END) as drifted "
                    "FROM source_weights"
                ).fetchone()
                health["source_weights"] = {
                    "total": row[0],
                    "with_user_set": row[1],
                    "with_drift": row[2],
                }
        except Exception as e:
            health["source_weights"] = {"error": str(e)}

        return health

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
        """Start the Google OAuth flow — opens browser for user approval.

        The OAuth server and subsequent API calls are synchronous and
        long-running, so the entire blocking section is offloaded to a
        thread to avoid stalling the FastAPI event loop.
        """
        import asyncio
        import os

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

        if not os.path.exists(credentials_file):
            raise HTTPException(400,
                f"Credentials file not found at {credentials_file}. "
                "Download it from Google Cloud Console and place it there.")

        def _run_oauth_flow():
            """Run the blocking OAuth flow, token save, and profile fetch in a thread."""
            from googleapiclient.discovery import build

            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0, open_browser=True)

            with open(token_file, "w") as f:
                f.write(creds.to_json())

            service = build("gmail", "v1", credentials=creds)
            profile = service.users().getProfile(userId="me").execute()
            return profile.get("emailAddress", "")

        try:
            email_addr = await asyncio.to_thread(_run_oauth_flow)
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
        import asyncio

        await asyncio.to_thread(life_os.semantic_fact_inferrer.run_all_inference)

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
    # Signal Profile Backfills (Admin)
    # -------------------------------------------------------------------

    @app.get("/api/admin/backfills/status")
    async def get_backfill_status():
        """
        Return the current population state of all signal profiles.

        Signal profiles are the foundation of the intelligence layer.  When they
        are empty — typically after a database migration that wipes
        ``signal_profiles``, or when the live connector is stale — the following
        features degrade silently:

          - Relationship maintenance predictions (People Radar contacts)
          - Semantic fact inference (expertise / interest / communication-style facts)
          - Behavioral insights (cadence, temporal, and topic patterns)

        Use this endpoint to determine whether a backfill is needed and to track
        progress after triggering one via ``POST /api/admin/backfills/trigger``.

        Returns:
            {
                "status": "ok" | "needs_backfill",
                "profiles": {
                    "<profile_name>": {
                        "populated": bool,
                        "samples_count": int,
                        "last_updated": str | null
                    },
                    ...
                },
                "generated_at": "<ISO-8601>"
            }

        Example::

            GET /api/admin/backfills/status
            → {
                "status": "needs_backfill",
                "profiles": {
                    "relationships": {"populated": false, "samples_count": 0, "last_updated": null},
                    "temporal":      {"populated": false, "samples_count": 0, "last_updated": null},
                    ...
                },
                "generated_at": "2026-03-01T00:00:00.000000+00:00"
              }
        """
        # The eight signal profiles that drive the intelligence layer.
        # "relationships" powers People Radar and maintenance predictions.
        # "temporal" drives preparation-needs and routine-deviation predictions.
        # "topics" feeds expertise/interest semantic fact inference.
        # "linguistic" drives communication-style fact inference and tone matching.
        # "cadence" tracks reply latency and peak communication hours.
        # "mood_signals" drives stress detection in the reaction prediction gatekeeper.
        # "spatial" tracks place-based behavior (visit frequency, duration, dominant domain).
        # "decision" tracks decision-making patterns (speed, delegation, risk tolerance).
        profile_names = [
            "relationships", "temporal", "topics", "linguistic",
            "cadence", "mood_signals", "spatial", "decision",
        ]

        profiles: dict[str, dict] = {}
        for name in profile_names:
            profile = life_os.user_model_store.get_signal_profile(name)
            if profile:
                samples = profile.get("samples_count", 0)
                profiles[name] = {
                    "populated": samples >= 1,
                    "samples_count": samples,
                    "last_updated": profile.get("updated_at"),
                }
            else:
                profiles[name] = {
                    "populated": False,
                    "samples_count": 0,
                    "last_updated": None,
                }

        all_populated = all(p["populated"] for p in profiles.values())
        return {
            "status": "ok" if all_populated else "needs_backfill",
            "profiles": profiles,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Guard against concurrent backfill invocations.  Since asyncio is
    # single-threaded this simple variable check is race-free — no lock needed.
    _backfill_task: Optional["asyncio.Task"] = None  # noqa: F821 — asyncio imported inside handler

    @app.post("/api/admin/backfills/trigger")
    async def trigger_signal_profile_backfills():
        """
        Trigger all signal profile backfills without requiring a system restart.

        After a database migration or when the live connector is stale, signal
        profiles may be completely empty.  Empty profiles block the entire
        intelligence layer: relationship maintenance predictions, People Radar
        contacts, semantic fact inference, and behavioral insights.

        This endpoint replays historical events from ``events.db`` through each
        signal extractor to rebuild the profiles.  The work runs in a background
        ``asyncio`` task so this endpoint returns immediately.  Use
        ``GET /api/admin/backfills/status`` to poll progress.

        Backfills are idempotent: each one checks whether the profile already
        has meaningful data (``samples_count >= 10``) and skips silently if so.
        To force a re-run on an already-populated profile, first clear it via the
        database browser, then call this endpoint again.

        If a backfill is already running, returns HTTP 409 Conflict instead of
        starting a duplicate run (which would cause double-counted signal samples).

        Backfills triggered (in order):
          1. relationship  — per-contact interaction graph (People Radar, maintenance predictions)
          2. temporal      — activity-hour patterns (preparation needs, routine detection)
          3. topic         — email topic frequencies (expertise/interest semantic facts)
          4. linguistic    — writing-style metrics (communication-style semantic facts)
          5. cadence       — response times, activity heatmaps (priority contact detection)
          6. mood_signals  — mood signal ring buffer (dashboard mood widget, episode energy)
          7. spatial       — place-based behavior patterns (visit frequency, duration, domain)
          8. decision      — decision-making patterns (speed, delegation, risk tolerance)

        Returns:
            {"status": "started", "backfills": [...], "message": "..."}
            or HTTP 409 with {"status": "already_running", "message": "..."}

        Example::

            POST /api/admin/backfills/trigger
            → {
                "status": "started",
                "message": "Signal profile backfills triggered in background. ...",
                "backfills": ["relationship", "temporal", "topic", "linguistic", "cadence", "mood_signals", "spatial", "decision"]
              }
        """
        import asyncio as _asyncio

        nonlocal _backfill_task

        # Reject concurrent invocations to prevent double-counted signal samples
        if _backfill_task is not None and not _backfill_task.done():
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=409,
                content={
                    "status": "already_running",
                    "message": "A backfill is already in progress. Poll /api/admin/backfills/status for progress.",
                },
            )

        async def _run_all_backfills():
            """Run all signal profile backfills sequentially in the background.

            Order matters: relationship first because it also populates contact stubs
            that downstream profiles (cadence, temporal) can enrich.  Marketing cleanup
            runs right after relationship so the other profiles see clean data.
            """
            await life_os._backfill_relationship_profile_if_needed()
            await life_os._clean_relationship_profile_if_needed()
            await life_os._backfill_temporal_profile_if_needed()
            await life_os._backfill_topic_profile_if_needed()
            await life_os._backfill_linguistic_profile_if_needed()
            await life_os._backfill_cadence_profile_if_needed()
            await life_os._backfill_mood_signals_profile_if_needed()
            await life_os._backfill_spatial_profile_if_needed()
            await life_os._backfill_decision_profile_if_needed()

        # Fire-and-forget: background task so the HTTP response returns immediately.
        # The caller should poll /api/admin/backfills/status to track completion.
        _backfill_task = _asyncio.create_task(_run_all_backfills())

        return {
            "status": "started",
            "message": (
                "Signal profile backfills triggered in background. "
                "Poll /api/admin/backfills/status to track progress."
            ),
            "backfills": [
                "relationship", "temporal", "topic", "linguistic",
                "cadence", "mood_signals", "spatial", "decision",
            ],
        }

    # -------------------------------------------------------------------
    # Database Health & Recovery
    # -------------------------------------------------------------------

    @app.get("/api/admin/db-integrity")
    async def check_db_integrity():
        """Run thorough integrity checks on all databases and report status.

        Uses ``DatabaseManager.get_database_health()`` which runs
        ``PRAGMA integrity_check`` plus blob overflow page probes for
        ``user_model.db`` (catching corruption that ``PRAGMA quick_check``
        misses in large JSON TEXT columns), and ``PRAGMA quick_check`` for
        the other four databases.

        Returns a map of database name → {status, detail} for each of the five
        SQLite databases.  ``status`` is one of ``"ok"``, ``"corrupted"``, or
        ``"error"`` (when the file cannot be opened at all).

        Example response::

            {
                "databases": {
                    "events": {"status": "ok", "detail": "ok"},
                    "user_model": {"status": "corrupted", "detail": "..."}
                },
                "checked_at": "2026-03-02T14:00:00+00:00"
            }
        """
        import asyncio

        health = await asyncio.to_thread(life_os.db.get_database_health)

        # Adapt get_database_health() output to the endpoint's response format
        results = {}
        for db_name, info in health.items():
            detail = "; ".join(info["errors"]) if info["errors"] else "ok"
            results[db_name] = {
                "status": info["status"],
                "detail": detail,
            }
        return {"databases": results, "checked_at": datetime.now(timezone.utc).isoformat()}

    @app.post("/api/admin/rebuild-user-model")
    async def rebuild_user_model():
        """Rebuild user_model.db from scratch if corrupted.

        Uses ``DatabaseManager.get_database_health()`` (integrity_check +
        blob overflow probes) to detect corruption — this catches overflow
        page damage that ``PRAGMA quick_check`` misses.

        When corruption is detected the flow is:

        1. Back up the corrupted file (``user_model.db.corrupted-<ts>``).
        2. Reinitialise the schema via ``DatabaseManager._init_user_model_db()``.
        3. Kick off signal-profile backfills in the background.

        The endpoint returns immediately; poll ``GET /api/admin/backfills/status``
        to track the background backfill progress.
        """
        import asyncio as _asyncio
        import os

        # Step 1: Check if actually corrupted using thorough health check
        # (integrity_check + blob probes — catches overflow page corruption)
        try:
            health = await _asyncio.to_thread(life_os.db.get_database_health)
            um_status = health.get("user_model", {}).get("status", "corrupted")
            if um_status == "ok":
                return {"status": "skipped", "reason": "Database is healthy, no rebuild needed"}
        except Exception:
            pass  # Health check itself failed — proceed with rebuild

        # Step 2: Use DatabaseManager's built-in recovery (backs up corrupt file)
        try:
            recovered = life_os.db._check_and_recover_db("user_model")
            if not recovered:
                # _check_and_recover_db returned False but our quick_check above
                # failed — force a manual backup as a fallback.
                db_path = os.path.join(str(life_os.db.data_dir), "user_model.db")
                backup_suffix = f".corrupted-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                for ext in ("", "-wal", "-shm"):
                    src = db_path + ext
                    if os.path.exists(src):
                        os.rename(src, db_path + ext + backup_suffix)
                logger.warning("Manually backed up corrupted user_model.db")

            # Step 3: Reinitialise schema (creates fresh tables)
            life_os.db._init_user_model_db()
            logger.info("Rebuilt user_model.db schema after runtime recovery")

        except Exception as e:
            logger.error("Runtime user_model.db rebuild failed: %s", e)
            return {"status": "error", "detail": str(e)}

        # Step 4: Trigger signal-profile backfill in background
        async def _post_rebuild_backfill():
            """Re-run signal profile backfills after a runtime DB rebuild."""
            try:
                # Use the same backfill sequence as /api/admin/backfills/trigger
                for method_name in [
                    "_backfill_relationship_profile_if_needed",
                    "_clean_relationship_profile_if_needed",
                    "_backfill_temporal_profile_if_needed",
                    "_backfill_topic_profile_if_needed",
                    "_backfill_linguistic_profile_if_needed",
                    "_backfill_cadence_profile_if_needed",
                    "_backfill_mood_signals_profile_if_needed",
                    "_backfill_spatial_profile_if_needed",
                    "_backfill_decision_profile_if_needed",
                ]:
                    if hasattr(life_os, method_name):
                        try:
                            await getattr(life_os, method_name)()
                        except Exception as e:
                            logger.error("Post-rebuild backfill %s failed: %s", method_name, e)
            except Exception as e:
                logger.error("Post-rebuild backfill failed: %s", e)

            # Backfill episodes from events.db (foundation of cognitive pipeline)
            try:
                if hasattr(life_os, "_backfill_episodes_from_events_if_needed"):
                    await life_os._backfill_episodes_from_events_if_needed()
            except Exception as e:
                logger.error("Post-rebuild episode backfill failed: %s", e)

            # Backfill communication templates (Layer 3 procedural memory)
            try:
                if hasattr(life_os, "_backfill_communication_templates_if_needed"):
                    await life_os._backfill_communication_templates_if_needed()
            except Exception as e:
                logger.error("Post-rebuild communication template backfill failed: %s", e)

        _asyncio.create_task(_post_rebuild_backfill())

        return {
            "status": "rebuilt",
            "message": (
                "Database rebuilt with fresh schema. "
                "Signal profile, episode, and communication template backfills started in background. "
                "Poll /api/admin/backfills/status to track progress."
            ),
        }

    # -------------------------------------------------------------------
    # Backup Listing & Restore
    # -------------------------------------------------------------------

    @app.get("/api/admin/backups")
    async def list_backups(db_name: str = "user_model"):
        """List available database backups for recovery.

        Scans ``data/backups/`` for backup files matching the given database
        name and returns metadata (path, size, age) sorted newest-first.

        Args:
            db_name: Logical database name to list backups for.  Must be one
                of ``events``, ``user_model``, ``state``, ``preferences``,
                or ``entities``.
        """
        valid_dbs = {"events", "user_model", "state", "preferences", "entities"}
        if db_name not in valid_dbs:
            raise HTTPException(status_code=400, detail=f"Invalid db_name. Must be one of: {sorted(valid_dbs)}")
        try:
            import asyncio

            backups = await asyncio.to_thread(life_os.db.list_backups, db_name)
            return {"db_name": db_name, "backups": backups, "count": len(backups)}
        except Exception as exc:
            logger.error("list_backups endpoint failed for %s: %s", db_name, exc)
            return {"db_name": db_name, "backups": [], "count": 0, "error": str(exc)}

    @app.post("/api/admin/backups/restore")
    async def restore_backup(body: BackupRestoreRequest):
        """Restore a database from a specific backup file.

        Validates the requested backup path is within the ``data/backups/``
        directory (preventing path-traversal attacks), checks that the file
        exists, and delegates to ``DatabaseManager.restore_from_backup()``
        which archives the current database before replacing it.

        Request body (JSON):
            - ``backup_path`` (str, required): Absolute path to the backup file.
            - ``db_name`` (str, optional): Database to restore. Defaults to
              ``"user_model"``.
        """
        import asyncio

        backup_path = body.backup_path
        db_name = body.db_name

        valid_dbs = {"events", "user_model", "state", "preferences", "entities"}
        if db_name not in valid_dbs:
            raise HTTPException(status_code=400, detail=f"Invalid db_name. Must be one of: {sorted(valid_dbs)}")

        # Path traversal protection: backup must be inside data/backups/
        from pathlib import Path

        backup_dir = (life_os.db.data_dir / "backups").resolve()
        resolved = Path(backup_path).resolve()
        if not str(resolved).startswith(str(backup_dir) + "/") and resolved.parent != backup_dir:
            raise HTTPException(status_code=400, detail="backup_path must be within the backups directory")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="Backup file not found")

        success = await asyncio.to_thread(life_os.db.restore_from_backup, str(resolved), db_name)
        if success:
            return {"status": "restored", "db_name": db_name, "backup_path": str(resolved)}
        else:
            raise HTTPException(status_code=500, detail="Restore failed — check server logs")

    # -------------------------------------------------------------------
    # Data Quality Diagnostics
    # -------------------------------------------------------------------

    @app.get("/api/admin/data-quality")
    async def data_quality_diagnostics():
        """Return real-time data quality and pipeline health diagnostics.

        Queries all five databases for key metrics — event volume, signal
        profile freshness, prediction pipeline stats, source weight
        staleness, connector health, and task/notification summaries.

        Each section is independently resilient: if one database is
        corrupted or unavailable, that section returns ``{"error": "..."}``
        while the remaining sections still return data.
        """
        import asyncio

        generated_at = datetime.now(timezone.utc).isoformat()

        # -- a. Event stats (events.db) ------------------------------------
        def _query_event_stats():
            """Gather event volume and source breakdown from events.db."""
            try:
                with life_os.db.get_connection("events") as conn:
                    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                    last_24h = conn.execute(
                        "SELECT COUNT(*) FROM events WHERE timestamp > datetime('now', '-1 day')"
                    ).fetchone()[0]
                    top_types = [
                        dict(row)
                        for row in conn.execute(
                            "SELECT type, COUNT(*) as count FROM events GROUP BY type ORDER BY count DESC LIMIT 10"
                        ).fetchall()
                    ]
                    sources = [
                        dict(row)
                        for row in conn.execute(
                            "SELECT source, COUNT(*) as count, MAX(timestamp) as last_event "
                            "FROM events GROUP BY source ORDER BY count DESC"
                        ).fetchall()
                    ]
                return {
                    "total": total,
                    "last_24h": last_24h,
                    "top_types": top_types,
                    "sources": sources,
                }
            except Exception as e:
                return {"error": str(e)}

        # -- b. Signal profiles (user_model.db) ----------------------------
        def _query_signal_profiles():
            """Fetch signal profile freshness from user_model.db."""
            try:
                with life_os.db.get_connection("user_model") as conn:
                    rows = conn.execute(
                        "SELECT profile_type, samples_count, updated_at FROM signal_profiles"
                    ).fetchall()
                return [dict(row) for row in rows]
            except Exception as e:
                return {"error": str(e)}

        # -- c. Prediction pipeline (user_model.db) ------------------------
        def _query_prediction_pipeline():
            """Compute prediction pipeline stats from user_model.db."""
            try:
                with life_os.db.get_connection("user_model") as conn:
                    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                    surfaced = conn.execute(
                        "SELECT COUNT(*) FROM predictions WHERE was_surfaced = 1"
                    ).fetchone()[0]
                    accuracy_row = conn.execute(
                        "SELECT AVG(CASE WHEN was_accurate = 1 THEN 1.0 ELSE 0.0 END) "
                        "FROM predictions WHERE was_accurate IS NOT NULL"
                    ).fetchone()
                    accuracy = accuracy_row[0] if accuracy_row else None
                return {"total": total, "surfaced": surfaced, "accuracy": accuracy}
            except Exception as e:
                return {"error": str(e)}

        # -- d. Source weight staleness (preferences.db) -------------------
        def _query_source_weights():
            """Fetch source weights and flag stale entries from preferences.db."""
            try:
                with life_os.db.get_connection("preferences") as conn:
                    rows = conn.execute(
                        "SELECT source_key, user_weight, ai_drift, ai_updated_at, "
                        "interactions, engagements, dismissals FROM source_weights"
                    ).fetchall()
                result = []
                for row in rows:
                    entry = dict(row)
                    entry["never_updated"] = entry.get("ai_updated_at") is None
                    result.append(entry)
                return result
            except Exception as e:
                return {"error": str(e)}

        # -- e. Connector health -------------------------------------------
        async def _query_connector_health():
            """Check health of all registered connectors."""
            try:
                results = {}
                for c in life_os.connectors:
                    try:
                        health = await asyncio.wait_for(c.health_check(), timeout=5.0)
                        results[getattr(c, "CONNECTOR_ID", str(c))] = health
                    except asyncio.TimeoutError:
                        results[getattr(c, "CONNECTOR_ID", str(c))] = {
                            "status": "error",
                            "details": "timeout",
                        }
                    except Exception as e:
                        results[getattr(c, "CONNECTOR_ID", str(c))] = {
                            "status": "error",
                            "details": str(e),
                        }
                return results
            except Exception as e:
                return {"error": str(e)}

        # -- f. Task summary (state.db) ------------------------------------
        def _query_task_summary():
            """Aggregate task counts by status from state.db."""
            try:
                with life_os.db.get_connection("state") as conn:
                    rows = conn.execute(
                        "SELECT status, COUNT(*) as count FROM tasks GROUP BY status"
                    ).fetchall()
                return {row["status"]: row["count"] for row in rows}
            except Exception as e:
                return {"error": str(e)}

        # -- g. Notification summary (state.db) ----------------------------
        def _query_notification_summary():
            """Aggregate notification counts by status from state.db."""
            try:
                with life_os.db.get_connection("state") as conn:
                    rows = conn.execute(
                        "SELECT status, COUNT(*) as count FROM notifications GROUP BY status"
                    ).fetchall()
                return {row["status"]: row["count"] for row in rows}
            except Exception as e:
                return {"error": str(e)}

        # Run all synchronous DB queries concurrently via to_thread,
        # plus the async connector health check.
        (
            event_stats,
            signal_profiles,
            prediction_pipeline,
            source_weights,
            connector_health,
            task_summary,
            notification_summary,
        ) = await asyncio.gather(
            asyncio.to_thread(_query_event_stats),
            asyncio.to_thread(_query_signal_profiles),
            asyncio.to_thread(_query_prediction_pipeline),
            asyncio.to_thread(_query_source_weights),
            _query_connector_health(),
            asyncio.to_thread(_query_task_summary),
            asyncio.to_thread(_query_notification_summary),
        )

        return {
            "generated_at": generated_at,
            "event_stats": event_stats,
            "signal_profiles": signal_profiles,
            "prediction_pipeline": prediction_pipeline,
            "source_weight_staleness": source_weights,
            "connector_health": connector_health,
            "task_summary": task_summary,
            "notification_summary": notification_summary,
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
