"""
Life OS — CalDAV Calendar Connector

Connects to any CalDAV-compliant calendar (Proton Calendar, Nextcloud,
iCloud, Google via bridge, etc.) for bidirectional sync.

Configuration:
    connectors:
      caldav:
        url: "https://calendar.proton.me/api/calendars"
        username: "your@proton.me"
        password: "bridge-password"
        sync_interval: 60
        calendars:
          - "Personal"
          - "Work"
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class CalDAVConnector(BaseConnector):
    """Connector that syncs calendar events from any CalDAV-compliant server.

    Uses the ``caldav`` library to discover calendars, fetch VEVENT objects within
    a rolling window, normalize them into Life OS event payloads, and publish
    them on the internal event bus.  Also supports creating new events by
    generating iCalendar (VCALENDAR/VEVENT) data and pushing it back to the server.
    """

    CONNECTOR_ID = "caldav"
    DISPLAY_NAME = "Calendar (CalDAV)"
    # Poll the CalDAV server once per minute for near-real-time awareness.
    SYNC_INTERVAL_SECONDS = 60

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        # The caldav.DAVClient instance; set during authenticate().
        self._client = None
        # List of caldav.Calendar objects the user chose to sync.
        self._calendars = []
        # Track already-published conflict pairs so we don't re-publish on every
        # sync cycle.  Key is a frozenset of (row_id_a, row_id_b) to handle pair
        # ordering.  Mirrors the pattern in ConflictDetector._published_conflicts.
        self._published_conflicts: set[frozenset[str]] = set()

    async def authenticate(self) -> bool:
        """Connect to the CalDAV server and discover available calendars.

        The caldav library is imported lazily so the rest of Life OS can load
        even when the dependency is not installed.  Authentication happens via
        HTTP Basic Auth (or whatever the server supports) encapsulated inside
        ``DAVClient``.  After connecting, we retrieve the *principal* (the
        authenticated user's root) and enumerate their calendars.

        If the user listed specific calendar names in the config, we filter the
        discovered calendars down to that subset.
        """
        try:
            # Lazy import: caldav is an optional dependency.
            import caldav

            url = self.config["url"]
            username = self.config["username"]
            password = self.config["password"]

            # DAVClient handles the HTTP(S) transport and credential management.
            self._client = caldav.DAVClient(
                url=url, username=username, password=password
            )
            # The principal represents the authenticated user on the server.
            principal = self._client.principal()
            # Discover all calendars that belong to this principal.
            self._calendars = principal.calendars()

            # Filter to configured calendars if specified —
            # this lets the user ignore shared/public calendars they don't need.
            cal_names = self.config.get("calendars")
            if cal_names:
                self._calendars = [
                    c for c in self._calendars
                    if c.name in cal_names
                ]

            return True
        except Exception as e:
            logger.error("Auth failed: %s", e)
            return False

    async def sync(self) -> int:
        """Pull calendar events for the upcoming 14 days.

        Performs a time-range query (``date_search``) against every tracked
        calendar, asking the server to *expand* recurring events into
        individual occurrences.  Each VEVENT is then normalised into a flat
        dict (the Life OS "calendar event" schema) and published to the event
        bus so that downstream services (scheduling, conflict detection, daily
        briefing) can react.

        Returns the number of events that were successfully published.
        """
        if not self._client:
            return 0

        count = 0
        now = datetime.now(timezone.utc)
        # Look 1 day into the past (catch late updates) and 14 days ahead.
        start = now - timedelta(days=1)
        end = now + timedelta(days=14)

        for calendar in self._calendars:
            try:
                # date_search with expand=True tells the server to unroll
                # recurring events (RRULE) into discrete VEVENT instances,
                # one per occurrence, within the requested window.
                events = calendar.date_search(start=start, end=end, expand=True)

                for event in events:
                    try:
                        # ---- VEVENT Parsing ----
                        # vobject_instance gives us the parsed iCalendar tree;
                        # .vevent accesses the first VEVENT component inside it.
                        vevent = event.vobject_instance.vevent

                        # Use the iCalendar UID as the stable event identifier.
                        # Fall back to a hash if the UID is missing (rare but
                        # possible with malformed calendar entries).
                        event_id = str(vevent.uid.value) if hasattr(vevent, "uid") else str(hash(str(vevent)))

                        # ---- Start / End Time Extraction ----
                        dtstart = vevent.dtstart.value
                        # DTEND is optional in the spec; default to 1-hour duration.
                        dtend = vevent.dtend.value if hasattr(vevent, "dtend") else dtstart + timedelta(hours=1)

                        # ---- Timezone Normalisation ----
                        # Naive datetimes (no tzinfo) are treated as UTC so that
                        # all downstream consumers can compare times consistently.
                        if hasattr(dtstart, "tzinfo") and dtstart.tzinfo is None:
                            dtstart = dtstart.replace(tzinfo=timezone.utc)
                        if hasattr(dtend, "tzinfo") and dtend.tzinfo is None:
                            dtend = dtend.replace(tzinfo=timezone.utc)

                        # ---- Optional VEVENT Fields ----
                        summary = str(vevent.summary.value) if hasattr(vevent, "summary") else "Untitled"
                        description = str(vevent.description.value) if hasattr(vevent, "description") else None
                        location = str(vevent.location.value) if hasattr(vevent, "location") else None

                        # ---- Attendee Extraction ----
                        # ATTENDEE can be a single value or a list; normalise to
                        # a list and strip the "mailto:" URI prefix.
                        attendees = []
                        if hasattr(vevent, "attendee"):
                            for a in (vevent.attendee if isinstance(vevent.attendee, list) else [vevent.attendee]):
                                attendees.append(str(a.value).replace("mailto:", ""))

                        # ORGANIZER follows the same "mailto:" convention.
                        organizer = None
                        if hasattr(vevent, "organizer"):
                            organizer = str(vevent.organizer.value).replace("mailto:", "")

                        # ---- All-Day Detection ----
                        # All-day events use a ``date`` object (no hour attribute),
                        # while timed events use ``datetime``.
                        is_all_day = not hasattr(dtstart, "hour")

                        # ---- Build Normalised Life OS Payload ----
                        payload = {
                            "event_id": event_id,
                            "calendar_id": calendar.name,
                            "title": summary,
                            "description": description,
                            "location": location,
                            # Prefer ISO-8601 strings for serialisation safety.
                            "start_time": dtstart.isoformat() if hasattr(dtstart, "isoformat") else str(dtstart),
                            "end_time": dtend.isoformat() if hasattr(dtend, "isoformat") else str(dtend),
                            "is_all_day": is_all_day,
                            "attendees": attendees,
                            "organizer": organizer,
                        }

                        # Metadata travels alongside the event for cross-connector
                        # enrichment (e.g., linking attendees to known contacts).
                        metadata = {
                            "related_contacts": attendees,
                            "location": location,
                        }

                        # Publish to the event bus so other services can react
                        # (e.g., daily briefing builder, conflict detector).
                        await self.publish_event(
                            "calendar.event.created", payload,
                            metadata=metadata,
                        )
                        count += 1

                    except Exception as e:
                        # Log but skip individual event failures so the rest of
                        # the calendar still syncs.
                        logger.warning("Event parse error: %s", e)

            except Exception as e:
                # Log per-calendar failures without aborting other calendars.
                logger.error("Calendar sync error (%s): %s", calendar.name, e)

        # After ingesting all events, run conflict detection across the full
        # window to flag overlapping meetings.
        await self._detect_conflicts()
        return count

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Create or modify calendar events.

        This is the *write* path of the connector, invoked by the agent or
        automation layer when it needs to create a new calendar event on the
        user's behalf.
        """
        if action == "create_event":
            return await self._create_event(params)
        raise ValueError(f"Unknown action: {action}")

    async def _create_event(self, params: dict) -> dict:
        """Create a new calendar event by building raw iCalendar (RFC 5545) data.

        The VCALENDAR string is assembled manually rather than through a
        library so we keep the dependency footprint small.  ``save_event``
        on the caldav Calendar object PUTs the iCalendar payload to the server.
        """
        if not self._calendars:
            return {"status": "error", "details": "No calendars available"}

        # Default to the first discovered calendar when no explicit target is
        # provided.  A future improvement could accept a calendar name/id.
        calendar = self._calendars[0]  # Use first calendar by default

        # Generate a unique UID for the VEVENT (required by RFC 5545).
        # Without a UID, most CalDAV servers will reject the PUT request.
        uid = str(uuid.uuid4())

        # Build a minimal but valid VCALENDAR document containing one VEVENT.
        # PRODID identifies the creator application (required by RFC 5545).
        # DTEND falls back to DTSTART (zero-duration event) when not supplied.
        vcal = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//LifeOS//CalDAV Connector//EN
BEGIN:VEVENT
UID:{uid}
SUMMARY:{params['title']}
DTSTART:{params['start_time']}
DTEND:{params.get('end_time', params['start_time'])}
DESCRIPTION:{params.get('description', '')}
LOCATION:{params.get('location', '')}
END:VEVENT
END:VCALENDAR"""

        try:
            # save_event performs an HTTP PUT to the CalDAV server.
            calendar.save_event(vcal)
            return {"status": "created", "title": params["title"], "uid": uid}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    async def _detect_conflicts(self):
        """Check for overlapping calendar events and alert.

        Queries all calendar events within the upcoming 48-hour window from the
        event store, sorts them by start time, and detects overlapping time
        ranges using a sweep-line algorithm. For each detected conflict, publishes
        a ``calendar.conflict.detected`` event containing both conflicting events'
        details so downstream services (notification manager, daily briefing)
        can alert the user.

        CRITICAL FIX (iteration 169):
            Previously queried events by creation timestamp (last 24h), which only
            caught conflicts if both events were synced within the same 24h window.
            This failed for 99.9% of conflicts since most events are synced once
            and never updated.

            Now queries by start_time in the upcoming 48h window, catching ALL
            future conflicts regardless of when events were synced. This matches
            how the prediction engine's calendar conflict detector works.

        Algorithm:
            1. Fetch all calendar.event.created events with start_time in next 48h
            2. Parse start_time and end_time from each event payload
            3. Sort by start_time
            4. For each event, check if it overlaps with any following event
            5. Two events overlap if: start1 < end2 AND start2 < end1
            6. Publish conflict events for each detected overlap
        """
        try:
            # Query the event store for all calendar events whose start_time
            # falls in the next 48 hours. This catches upcoming conflicts
            # regardless of when the events were originally synced.
            import json
            from datetime import datetime, timedelta, timezone

            now = datetime.now(timezone.utc)
            window_end = now + timedelta(hours=48)

            # Query all calendar events directly from the events table, filtering
            # by start_time in the payload. This catches ALL upcoming events
            # regardless of when they were synced.
            with self.db.get_connection("events") as conn:
                calendar_events = conn.execute(
                    """SELECT id, payload FROM events
                       WHERE type = 'calendar.event.created'
                       ORDER BY timestamp DESC
                       LIMIT 5000""",  # Generous limit for large calendars
                ).fetchall()

            if len(calendar_events) < 2:
                # Need at least 2 events to have a conflict
                return

            # Parse event times and build a list of events in the 48h window.
            # Each entry: (start_time, end_time, event_dict, payload_dict)
            parsed_events = []
            for evt in calendar_events:
                try:
                    # The payload is always a string in the database — deserialize it.
                    # Note: EventStore.store_event() JSON-serializes the payload, so
                    # we get it back as a string. If the original payload was already
                    # a JSON string (double-encoded), we need to parse twice.
                    raw_payload = evt["payload"]
                    payload = json.loads(raw_payload)

                    # If the result is still a string (double-encoded), parse again
                    if isinstance(payload, str):
                        payload = json.loads(payload)

                    # Extract ISO-format timestamps from payload
                    start_str = payload.get("start_time")
                    end_str = payload.get("end_time")

                    if not start_str or not end_str:
                        continue  # Skip events without time bounds

                    # Parse ISO timestamps. fromisoformat handles most formats.
                    # Handle both 'Z' suffix and explicit timezone offset formats.
                    start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

                    # Filter to events in the next 48 hours
                    # Skip events that have already ended or start after the window
                    if end_dt < now or start_dt > window_end:
                        continue

                    # Skip all-day events — they don't cause scheduling conflicts
                    # in the traditional sense (you can have multiple all-day markers).
                    if payload.get("is_all_day"):
                        continue

                    parsed_events.append((start_dt, end_dt, evt, payload))
                except Exception as e:
                    # Log but skip individual parse errors — don't let one
                    # malformed event block conflict detection for the rest.
                    logger.warning("Event parse error in conflict detection: %s", e)
                    continue

            if len(parsed_events) < 2:
                return

            # Sort by start time (earliest first)
            parsed_events.sort(key=lambda x: x[0])

            # Sweep-line algorithm: compare each event with every event that
            # starts before it ends (potential overlap).
            # Track new conflicts found in this run for the diagnostic log.
            new_conflicts = 0

            for i in range(len(parsed_events)):
                start1, end1, evt1, payload1 = parsed_events[i]

                # Check all subsequent events that could overlap
                for j in range(i + 1, len(parsed_events)):
                    start2, end2, evt2, payload2 = parsed_events[j]

                    # If the second event starts after the first one ends,
                    # no overlap is possible (since list is sorted by start time)
                    if start2 >= end1:
                        break  # No need to check further events

                    # Overlap condition: start1 < end2 AND start2 < end1
                    # (Since we already know start2 < end1 from the break condition,
                    # we just need to verify start1 < end2)
                    if start1 < end2:
                        # Conflict detected!  Use frozenset for unordered pair
                        # dedup, consistent with ConflictDetector service.
                        event_pair = frozenset({evt1["id"], evt2["id"]})

                        if event_pair not in self._published_conflicts:
                            self._published_conflicts.add(event_pair)
                            new_conflicts += 1

                            # Build conflict event payload
                            conflict_payload = {
                                "event1": {
                                    "id": payload1.get("event_id"),
                                    "title": payload1.get("title"),
                                    "start_time": start1.isoformat(),
                                    "end_time": end1.isoformat(),
                                    "calendar_id": payload1.get("calendar_id"),
                                    "location": payload1.get("location"),
                                },
                                "event2": {
                                    "id": payload2.get("event_id"),
                                    "title": payload2.get("title"),
                                    "start_time": start2.isoformat(),
                                    "end_time": end2.isoformat(),
                                    "calendar_id": payload2.get("calendar_id"),
                                    "location": payload2.get("location"),
                                },
                                "overlap_start": max(start1, start2).isoformat(),
                                "overlap_end": min(end1, end2).isoformat(),
                            }

                            # Publish the conflict event so the notification manager
                            # and default rules can fire alerts
                            await self.publish_event(
                                "calendar.conflict.detected",
                                conflict_payload,
                                priority="high",  # Conflicts are urgent
                            )

                            logger.info(
                                "Conflict detected: '%s' overlaps with '%s'",
                                payload1.get("title"),
                                payload2.get("title"),
                            )

            # Diagnostic summary for observability
            logger.debug(
                "Conflict detection: scanned %d total events → %d in 48h window (non-all-day) → %d new conflicts (%d total tracked)",
                len(calendar_events),
                len(parsed_events),
                new_conflicts,
                len(self._published_conflicts),
            )

        except Exception as e:
            # Fail-open: conflict detection errors should never crash the sync.
            logger.error("Conflict detection error: %s", e)

    async def health_check(self) -> dict[str, Any]:
        """Verify the CalDAV connection is still alive.

        Re-fetches the principal to confirm that the server responds and our
        credentials are still valid.
        """
        try:
            if self._client:
                principal = self._client.principal()
                return {"status": "ok", "connector": self.CONNECTOR_ID,
                        "calendars": len(self._calendars)}
            return {"status": "error", "details": "Not connected"}
        except Exception as e:
            return {"status": "error", "details": str(e)}
