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

    CONNECTOR_ID = "caldav"
    DISPLAY_NAME = "Calendar (CalDAV)"
    SYNC_INTERVAL_SECONDS = 60

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        self._client = None
        self._calendars = []

    async def authenticate(self) -> bool:
        """Connect to the CalDAV server."""
        try:
            import caldav

            url = self.config["url"]
            username = self.config["username"]
            password = self.config["password"]

            self._client = caldav.DAVClient(
                url=url, username=username, password=password
            )
            principal = self._client.principal()
            self._calendars = principal.calendars()

            # Filter to configured calendars if specified
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
        """Pull calendar events for the upcoming 14 days."""
        if not self._client:
            return 0

        count = 0
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=1)
        end = now + timedelta(days=14)

        for calendar in self._calendars:
            try:
                events = calendar.date_search(start=start, end=end, expand=True)

                for event in events:
                    try:
                        vevent = event.vobject_instance.vevent
                        event_id = str(vevent.uid.value) if hasattr(vevent, "uid") else str(hash(str(vevent)))

                        dtstart = vevent.dtstart.value
                        dtend = vevent.dtend.value if hasattr(vevent, "dtend") else dtstart + timedelta(hours=1)

                        # Ensure timezone-aware
                        if hasattr(dtstart, "tzinfo") and dtstart.tzinfo is None:
                            dtstart = dtstart.replace(tzinfo=timezone.utc)
                        if hasattr(dtend, "tzinfo") and dtend.tzinfo is None:
                            dtend = dtend.replace(tzinfo=timezone.utc)

                        summary = str(vevent.summary.value) if hasattr(vevent, "summary") else "Untitled"
                        description = str(vevent.description.value) if hasattr(vevent, "description") else None
                        location = str(vevent.location.value) if hasattr(vevent, "location") else None

                        # Extract attendees
                        attendees = []
                        if hasattr(vevent, "attendee"):
                            for a in (vevent.attendee if isinstance(vevent.attendee, list) else [vevent.attendee]):
                                attendees.append(str(a.value).replace("mailto:", ""))

                        organizer = None
                        if hasattr(vevent, "organizer"):
                            organizer = str(vevent.organizer.value).replace("mailto:", "")

                        is_all_day = not hasattr(dtstart, "hour")

                        payload = {
                            "event_id": event_id,
                            "calendar_id": calendar.name,
                            "title": summary,
                            "description": description,
                            "location": location,
                            "start_time": dtstart.isoformat() if hasattr(dtstart, "isoformat") else str(dtstart),
                            "end_time": dtend.isoformat() if hasattr(dtend, "isoformat") else str(dtend),
                            "is_all_day": is_all_day,
                            "attendees": attendees,
                            "organizer": organizer,
                        }

                        metadata = {
                            "related_contacts": attendees,
                            "location": location,
                        }

                        await self.publish_event(
                            "calendar.event.created", payload,
                            metadata=metadata,
                        )
                        count += 1

                    except Exception as e:
                        print(f"[caldav] Event parse error: {e}")

            except Exception as e:
                print(f"[caldav] Calendar sync error ({calendar.name}): {e}")

        # Check for conflicts
        await self._detect_conflicts()
        return count

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Create or modify calendar events."""
        if action == "create_event":
            return await self._create_event(params)
        raise ValueError(f"Unknown action: {action}")

    async def _create_event(self, params: dict) -> dict:
        """Create a new calendar event."""
        if not self._calendars:
            return {"status": "error", "details": "No calendars available"}

        calendar = self._calendars[0]  # Use first calendar by default

        from datetime import datetime
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
            calendar.save_event(vcal)
            return {"status": "created", "title": params["title"]}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    async def _detect_conflicts(self):
        """Check for overlapping calendar events and alert."""
        # Simplified: in production, query events from the event store
        # and check for time overlaps
        pass

    async def health_check(self) -> dict[str, Any]:
        try:
            if self._client:
                principal = self._client.principal()
                return {"status": "ok", "connector": self.CONNECTOR_ID,
                        "calendars": len(self._calendars)}
            return {"status": "error", "details": "Not connected"}
        except Exception as e:
            return {"status": "error", "details": str(e)}
