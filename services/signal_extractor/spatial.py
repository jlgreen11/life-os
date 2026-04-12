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
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from services.signal_extractor.base import BaseExtractor

logger = logging.getLogger(__name__)


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
      - calendar.event.created / calendar.event.updated (location field)
      - ios.context.update (device proximity, geolocation)
      - system.user.location_update (explicit location changes)
      - email.received (timezone/location hints — low confidence)
    """

    def can_process(self, event: dict) -> bool:
        """Check if this event contains location data worth extracting.

        Args:
            event: Event dictionary with type and payload

        Returns:
            True if event has extractable location information
        """
        event_type = event.get("type", "")

        # Calendar events with location field (created or updated)
        if event_type in ("calendar.event.created", "calendar.event.updated"):
            location = event.get("payload", {}).get("location")
            return bool(location and location.strip())

        # iOS context updates with location
        if event_type == "ios.context.update":
            payload = event.get("payload", {})
            return bool(payload.get("location") or payload.get("device_proximity"))

        # Explicit location updates
        if event_type == "system.user.location_update":
            return True

        # Email events with timezone or location metadata (weak spatial signals)
        if event_type == "email.received":
            payload = event.get("payload", {})
            return bool(
                payload.get("timezone")
                or payload.get("sender_timezone")
                or payload.get("location")
            )

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

        if event_type in ("calendar.event.created", "calendar.event.updated"):
            location = payload.get("location", "").strip()

            # Estimate duration from calendar event start/end times
            start_time = payload.get("start_time")
            end_time = payload.get("end_time")
            if start_time and end_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    duration_minutes = (end_dt - start_dt).total_seconds() / 60
                except Exception as e:
                    logger.debug('spatial_extractor: skipping duration calc — malformed time: %s', e)

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

        elif event_type == "email.received":
            # Extract low-confidence location hints from email metadata
            return self._extract_email_location_hint(event)

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

    # Common timezone prefixes → approximate region labels.
    # Kept intentionally coarse: the goal is directional signal, not GPS accuracy.
    TIMEZONE_REGION_MAP: dict[str, str] = {
        "America/New_York": "East Coast US",
        "America/Chicago": "Central US",
        "America/Denver": "Mountain US",
        "America/Los_Angeles": "West Coast US",
        "America/Phoenix": "Arizona US",
        "America/Anchorage": "Alaska US",
        "Pacific/Honolulu": "Hawaii US",
        "America/Toronto": "Eastern Canada",
        "America/Vancouver": "Western Canada",
        "Europe/London": "London UK",
        "Europe/Paris": "Western Europe",
        "Europe/Berlin": "Central Europe",
        "Europe/Amsterdam": "Western Europe",
        "Europe/Rome": "Southern Europe",
        "Europe/Madrid": "Southern Europe",
        "Europe/Moscow": "Russia",
        "Asia/Tokyo": "Japan",
        "Asia/Shanghai": "China",
        "Asia/Kolkata": "India",
        "Asia/Dubai": "Middle East",
        "Asia/Singapore": "Southeast Asia",
        "Australia/Sydney": "Eastern Australia",
        "Australia/Melbourne": "Eastern Australia",
        "Australia/Perth": "Western Australia",
    }

    def _extract_email_location_hint(self, event: dict) -> list[dict]:
        """Extract a low-confidence location hint from email timezone/location metadata.

        Email headers and connector metadata often carry timezone information
        that reveals the approximate region where the sender (or recipient) is
        active. These signals are intentionally low-confidence (0.3) so they
        contribute to time-of-day-by-location patterns without overriding
        high-confidence GPS or calendar location data.

        Args:
            event: An email.received event dictionary.

        Returns:
            List containing zero or one location_hint signal dicts.
        """
        payload = event.get("payload", {})
        timestamp = event.get("timestamp", datetime.now(timezone.utc).isoformat())

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                timestamp = datetime.now(timezone.utc)

        # Try explicit location first (highest value), then timezone fields
        location_str = payload.get("location")
        source_field = "email_location"

        if not location_str:
            tz_value = payload.get("timezone") or payload.get("sender_timezone") or ""
            location_str = self.TIMEZONE_REGION_MAP.get(tz_value)
            source_field = "email_timezone"

        if not location_str:
            return []

        location_str = self._normalize_location(location_str)

        signal = {
            "signal_type": "spatial",
            "type": "location_hint",
            "location": location_str,
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            "duration_minutes": None,
            "activity_type": "email_hint",
            "domain": "personal",
            "source": event.get("source", "unknown"),
            "confidence": 0.3,
        }

        # Update inferred locations in the spatial profile
        self._update_inferred_location(location_str, source_field, timestamp)

        return [signal]

    def _update_inferred_location(
        self, location: str, source_field: str, timestamp: datetime
    ):
        """Store a low-confidence inferred location separately from known places.

        Inferred locations (from email timezone headers, etc.) are kept in a
        dedicated 'inferred_locations' dict within the spatial profile so they
        do not pollute the high-confidence 'place_behaviors' data used for
        routine detection and notification routing.

        Args:
            location: Normalized location/region string.
            source_field: How the location was derived (e.g. 'email_timezone').
            timestamp: When this observation occurred.
        """
        profile = self.ums.get_signal_profile("spatial")
        if not profile:
            inferred = {}
        else:
            data = profile.get("data", {})
            # get_signal_profile already deserializes the outer JSON blob; the
            # value here is a plain dict (not a JSON string).
            inferred = data.get("inferred_locations") or {}

        if location not in inferred:
            inferred[location] = {
                "observation_count": 0,
                "sources": {},
                "first_seen": timestamp.isoformat(),
                "last_seen": timestamp.isoformat(),
            }

        entry = inferred[location]
        entry["observation_count"] = entry.get("observation_count", 0) + 1
        entry["last_seen"] = timestamp.isoformat()

        # Track which source types contribute to this inferred location
        sources = entry.get("sources", {})
        sources[source_field] = sources.get(source_field, 0) + 1
        entry["sources"] = sources

        # Build updated profile data — preserve existing place_behaviors
        profile_data = {}
        if profile:
            existing_data = profile.get("data", {})
            if "place_behaviors" in existing_data:
                profile_data["place_behaviors"] = existing_data["place_behaviors"]

        # Pass the raw dict — update_signal_profile() handles JSON serialization.
        profile_data["inferred_locations"] = inferred

        # --- Defensive serialization check ---
        # update_signal_profile() catches all exceptions with a silent warning, so a
        # TypeError from json.dumps() (e.g. a set, datetime, or Enum sneaking into
        # inferred_locations) would cause a zero-trace write failure.  We catch this
        # here first so we can log the exact field causing the problem.
        try:
            json.dumps(profile_data)
        except (TypeError, ValueError) as exc:
            bad_fields: list[str] = []
            inferred_locs = profile_data.get("inferred_locations", {})
            if isinstance(inferred_locs, dict):
                for loc_key, loc_val in inferred_locs.items():
                    if not isinstance(loc_val, dict):
                        bad_fields.append(
                            f"inferred_locations[{loc_key!r}]={type(loc_val).__name__}"
                        )
                        continue
                    for field, fval in loc_val.items():
                        if field == "sources" and isinstance(fval, dict):
                            # Inspect the nested sources sub-dict
                            for src_key, src_val in fval.items():
                                try:
                                    json.dumps(src_val)
                                except (TypeError, ValueError):
                                    bad_fields.append(
                                        f"inferred_locations[{loc_key!r}]"
                                        f"['sources'][{src_key!r}]"
                                        f"={type(src_val).__name__}"
                                    )
                        else:
                            try:
                                json.dumps(fval)
                            except (TypeError, ValueError):
                                bad_fields.append(
                                    f"inferred_locations[{loc_key!r}][{field!r}]"
                                    f"={type(fval).__name__}"
                                )
            for key, val in profile_data.items():
                if key != "inferred_locations":
                    try:
                        json.dumps(val)
                    except (TypeError, ValueError):
                        bad_fields.append(f"{key!r}={type(val).__name__}")

            logger.error(
                "SpatialExtractor._update_inferred_location: profile_data contains "
                "non-JSON-serializable types — write SKIPPED to avoid silent data loss. "
                "Non-serializable fields: %s. Error: %s",
                bad_fields or ["unknown"],
                exc,
            )
            return

        self.ums.update_signal_profile(
            profile_type="spatial",
            data=profile_data,
        )

        # Post-write verification: confirm the profile is readable after the write.
        # update_signal_profile() silently swallows DB errors (fail-open), so without
        # this check a corrupt user_model.db would cause 0 profile data while producing
        # no visible error in the pipeline logs.
        verify = self.ums.get_signal_profile("spatial")
        if not verify:
            logger.error(
                "SpatialExtractor._update_inferred_location: spatial profile FAILED "
                "to persist after write (inferred_locations=%d) — "
                "user_model.db may be corrupt",
                len(inferred),
            )

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
            # get_signal_profile already deserializes the outer JSON blob; the
            # value here is a plain dict (not a JSON string).
            place_behaviors = profile.get("data", {}).get("place_behaviors") or {}

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

        # Pass the raw dict — update_signal_profile() handles JSON serialization.
        profile_data = {
            "place_behaviors": place_behaviors,
        }

        # --- Defensive serialization check ---
        # update_signal_profile() catches all exceptions with a silent warning, so a
        # TypeError from json.dumps() (e.g. a set sneaking into activity_counts or
        # typical_activities) would cause a zero-trace write failure.  We catch this
        # here first so we can log the exact field causing the problem.
        try:
            json.dumps(profile_data)
        except (TypeError, ValueError) as exc:
            bad_fields: list[str] = []
            behaviors = profile_data.get("place_behaviors", {})
            if isinstance(behaviors, dict):
                for place_key, place_val in behaviors.items():
                    if not isinstance(place_val, dict):
                        bad_fields.append(
                            f"place_behaviors[{place_key!r}]={type(place_val).__name__}"
                        )
                        continue
                    for field, fval in place_val.items():
                        if field == "activity_counts" and isinstance(fval, dict):
                            # Inspect the nested activity_counts sub-dict
                            for act_key, act_val in fval.items():
                                try:
                                    json.dumps(act_val)
                                except (TypeError, ValueError):
                                    bad_fields.append(
                                        f"place_behaviors[{place_key!r}]"
                                        f"['activity_counts'][{act_key!r}]"
                                        f"={type(act_val).__name__}"
                                    )
                        elif field == "typical_activities" and isinstance(fval, list):
                            # Inspect each item in the typical_activities list
                            for idx, item in enumerate(fval):
                                try:
                                    json.dumps(item)
                                except (TypeError, ValueError):
                                    bad_fields.append(
                                        f"place_behaviors[{place_key!r}]"
                                        f"['typical_activities'][{idx}]"
                                        f"={type(item).__name__}"
                                    )
                        else:
                            try:
                                json.dumps(fval)
                            except (TypeError, ValueError):
                                bad_fields.append(
                                    f"place_behaviors[{place_key!r}][{field!r}]"
                                    f"={type(fval).__name__}"
                                )

            logger.error(
                "SpatialExtractor._update_spatial_profile: profile_data contains "
                "non-JSON-serializable types — write SKIPPED to avoid silent data loss. "
                "Non-serializable fields: %s. Error: %s",
                bad_fields or ["unknown"],
                exc,
            )
            return

        self.ums.update_signal_profile(
            profile_type="spatial",
            data=profile_data,
        )

        # Post-write verification: confirm the profile is readable after the write.
        # update_signal_profile() silently swallows DB errors (fail-open), so without
        # this check a corrupt user_model.db would cause 0 profile data while producing
        # no visible error in the pipeline logs.
        verify = self.ums.get_signal_profile("spatial")
        if not verify:
            logger.error(
                "SpatialExtractor._update_spatial_profile: spatial profile FAILED "
                "to persist after write (place_behaviors=%d) — "
                "user_model.db may be corrupt",
                len(place_behaviors),
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

        # get_signal_profile already deserializes the outer JSON blob; the
        # value here is a plain dict (not a JSON string).
        place_behaviors = profile.get("data", {}).get("place_behaviors") or {}

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
