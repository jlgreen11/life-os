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

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager


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
            print(f"[caldav] Auth failed: {e}")
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
                        print(f"[caldav] Event parse error: {e}")

            except Exception as e:
                # Log per-calendar failures without aborting other calendars.
                print(f"[caldav] Calendar sync error ({calendar.name}): {e}")

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

        from datetime import datetime
        # Build a minimal but valid VCALENDAR document containing one VEVENT.
        # DTEND falls back to DTSTART (zero-duration event) when not supplied.
        vcal = f"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
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
            return {"status": "created", "title": params["title"]}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    async def _detect_conflicts(self):
        """Check for overlapping calendar events and alert.

        Stub: in production this would query the event store for all events
        in the sync window, sort them by start time, and flag any pairs whose
        time ranges overlap.  Detected conflicts would be published as
        ``calendar.conflict.detected`` events.
        """
        pass

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
