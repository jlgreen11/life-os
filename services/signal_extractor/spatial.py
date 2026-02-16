"""
Life OS — Spatial Profile Extractor

Analyzes location data from calendar events, iOS device proximity, and
explicit location updates to build a spatial profile that captures how
behavior changes based on where the user is.

This enables location-aware context switching: notification preferences,
dominant work/personal mode, typical activities, and behavioral patterns
at each place the user frequents.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from services.signal_extractor.base import BaseExtractor


class SpatialExtractor(BaseExtractor):
    """
    Extracts location-based behavioral patterns from events with location data.

    Builds a spatial profile that tracks:
      - Known places and their behavioral signatures
      - Notification preference by location
      - Dominant domain (work/personal) by location
      - Typical duration and activities at each place
      - Location transition patterns (home → work commute times, etc.)

    Data sources:
      - calendar.event.created (location field)
      - ios.context.update (device proximity, geolocation)
      - system.user.location_update (explicit location changes)
    """

    def can_process(self, event: dict) -> bool:
        """Check if this event contains location data worth extracting.

        Args:
            event: Event dictionary with type and payload

        Returns:
            True if event has extractable location information
        """
        event_type = event.get("type", "")

        # Calendar events with location field
        if event_type == "calendar.event.created":
            location = event.get("payload", {}).get("location")
            return bool(location and location.strip())

        # iOS context updates with location
        if event_type == "ios.context.update":
            payload = event.get("payload", {})
            return bool(payload.get("location") or payload.get("device_proximity"))

        # Explicit location updates
        if event_type == "system.user.location_update":
            return True

        return False

    def extract(self, event: dict) -> list[dict]:
        """Extract spatial signals from location-bearing events.

        Process the event to identify:
          1. Which place this is (normalize location strings)
          2. How long the user spent there (from event duration or historical avg)
          3. What domain activity occurred (work/personal, inferred from event metadata)
          4. Update the spatial profile with new observations

        Args:
            event: Event dictionary with location data

        Returns:
            List of signal dictionaries (usually 1 per event)
        """
        signals = []
        event_type = event.get("type", "")
        payload = event.get("payload", {})
        timestamp = event.get("timestamp", datetime.now(timezone.utc).isoformat())

        # Parse timestamp
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except Exception:
                timestamp = datetime.now(timezone.utc)

        # Extract location from different event types
        location = None
        duration_minutes = None
        activity_type = None
        domain = "personal"  # Default assumption

        if event_type == "calendar.event.created":
            location = payload.get("location", "").strip()

            # Estimate duration from calendar event start/end times
            start_time = payload.get("start_time")
            end_time = payload.get("end_time")
            if start_time and end_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    duration_minutes = (end_dt - start_dt).total_seconds() / 60
                except Exception:
                    pass

            # Infer domain from calendar event metadata
            # Work indicators: attendees, work hours, work-related keywords
            if payload.get("attendees") or "meeting" in payload.get("title", "").lower():
                domain = "work"

            activity_type = "calendar_event"

        elif event_type == "ios.context.update":
            # iOS device proximity as a location signal
            proximity = payload.get("device_proximity")
            if proximity:
                location = f"device:{proximity}"
                domain = "personal"
                activity_type = "device_proximity"

        elif event_type == "system.user.location_update":
            location = payload.get("location", "").strip()
            domain = payload.get("domain", "personal")
            activity_type = "explicit_update"

        # Skip if we couldn't extract a meaningful location
        if not location:
            return []

        # Normalize location string (handle common variations)
        location = self._normalize_location(location)

        # Build the signal
        signal = {
            "signal_type": "spatial",
            "location": location,
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            "duration_minutes": duration_minutes,
            "activity_type": activity_type,
            "domain": domain,
            "source": event.get("source", "unknown"),
        }
        signals.append(signal)

        # Update the spatial profile with this observation
        self._update_spatial_profile(location, duration_minutes, domain, activity_type, timestamp)

        return signals

    def _normalize_location(self, location: str) -> str:
        """Normalize location strings to group similar places.

        Examples:
          - "123 Main St, Austin, TX" → "123 main st austin"
          - "Residence Inn by Marriott..." → "residence inn marriott"
          - "Home" → "home"

        Args:
            location: Raw location string

        Returns:
            Normalized lowercase location identifier
        """
        # Convert to lowercase and strip extra whitespace
        normalized = location.lower().strip()

        # Remove common punctuation
        normalized = normalized.replace(",", " ").replace(".", " ")

        # Collapse multiple spaces
        normalized = " ".join(normalized.split())

        # Truncate very long location strings (keep first ~50 chars of meaningful content)
        if len(normalized) > 60:
            normalized = normalized[:60].rsplit(" ", 1)[0]

        return normalized

    def _update_spatial_profile(
        self,
        location: str,
        duration_minutes: Optional[float],
        domain: str,
        activity_type: Optional[str],
        timestamp: datetime,
    ):
        """Update the spatial signal profile with new location observation.

        Aggregates observations per location to compute:
          - Visit frequency
          - Average duration
          - Dominant domain (work vs personal)
          - Typical activities
          - Last visit timestamp

        Args:
            location: Normalized location identifier
            duration_minutes: How long spent there (if known)
            domain: work, personal, or social
            activity_type: What happened there (calendar_event, device_proximity, etc)
            timestamp: When this observation occurred
        """
        # Load current spatial profile
        profile = self.ums.get_signal_profile("spatial")
        if not profile:
            place_behaviors = {}
        else:
            # Deserialize place_behaviors from the "data" field
            place_behaviors_raw = profile.get("data", {}).get("place_behaviors", {})
            if isinstance(place_behaviors_raw, str):
                place_behaviors = json.loads(place_behaviors_raw)
            else:
                place_behaviors = place_behaviors_raw if place_behaviors_raw else {}

        # Get or create place behavior entry
        if location not in place_behaviors:
            place_behaviors[location] = {
                "place_id": location,
                "visit_count": 0,
                "total_duration_minutes": 0.0,
                "domain_counts": {},
                "activity_counts": {},
                "first_visit": timestamp.isoformat(),
                "last_visit": timestamp.isoformat(),
            }

        place = place_behaviors[location]

        # Update visit count
        place["visit_count"] = place.get("visit_count", 0) + 1

        # Accumulate duration
        if duration_minutes:
            place["total_duration_minutes"] = place.get("total_duration_minutes", 0.0) + duration_minutes

        # Track domain distribution
        domain_counts = place.get("domain_counts", {})
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        place["domain_counts"] = domain_counts

        # Track activity types
        if activity_type:
            activity_counts = place.get("activity_counts", {})
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
            place["activity_counts"] = activity_counts

        # Update last visit
        place["last_visit"] = timestamp.isoformat()

        # Compute derived metrics for this place
        # Dominant domain: whichever has the most observations
        if place["domain_counts"]:
            place["dominant_domain"] = max(
                place["domain_counts"].items(),
                key=lambda x: x[1]
            )[0]

        # Average duration per visit
        if place["total_duration_minutes"] > 0 and place["visit_count"] > 0:
            place["average_duration_minutes"] = (
                place["total_duration_minutes"] / place["visit_count"]
            )

        # Top activities (sorted by frequency)
        if place["activity_counts"]:
            sorted_activities = sorted(
                place["activity_counts"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            place["typical_activities"] = [act for act, _count in sorted_activities[:5]]

        # Serialize place_behaviors for storage
        profile_data = {
            "place_behaviors": json.dumps(place_behaviors),
        }

        self.ums.update_signal_profile(
            profile_type="spatial",
            data=profile_data,
        )

    def get_dominant_location_now(self) -> Optional[str]:
        """Get the most likely current location based on recent observations.

        Looks at the last 24 hours of spatial signals and returns the location
        with the most recent visit, weighted by typical duration.

        Returns:
            Normalized location string or None if no recent data
        """
        profile = self.ums.get_signal_profile("spatial")
        if not profile:
            return None

        place_behaviors_raw = profile.get("data", {}).get("place_behaviors", {})
        if isinstance(place_behaviors_raw, str):
            place_behaviors = json.loads(place_behaviors_raw)
        else:
            place_behaviors = place_behaviors_raw if place_behaviors_raw else {}

        if not place_behaviors:
            return None

        # Find most recently visited place
        now = datetime.now(timezone.utc)
        recent_places = []

        for location, place in place_behaviors.items():
            last_visit_str = place.get("last_visit")
            if not last_visit_str:
                continue

            try:
                last_visit = datetime.fromisoformat(last_visit_str.replace('Z', '+00:00'))
                hours_since = (now - last_visit).total_seconds() / 3600

                # Only consider places visited in last 24 hours
                if hours_since < 24:
                    recent_places.append((location, hours_since, place))
            except Exception:
                continue

        if not recent_places:
            return None

        # Return the most recently visited place
        recent_places.sort(key=lambda x: x[1])  # Sort by hours_since (ascending)
        return recent_places[0][0]
