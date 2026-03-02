"""
Life OS — Cadence Signal Extractor

Tracks when and how quickly the user communicates.
Reveals priorities, avoidance patterns, and natural rhythms.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from models.core import EventType
from services.signal_extractor.base import BaseExtractor

logger = logging.getLogger(__name__)


class CadenceExtractor(BaseExtractor):
    """
    Tracks when and how quickly the user communicates.
    Reveals priorities, avoidance patterns, and natural rhythms.
    """

    def can_process(self, event: dict) -> bool:
        # Cadence analysis applies to all communication events (both directions)
        # because we need inbound events to anchor response-time calculations
        # and outbound events to measure the user's actual reply latency.
        return event.get("type") in [
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_SENT.value,
            EventType.EMAIL_RECEIVED.value,
            EventType.MESSAGE_RECEIVED.value,
        ]

    def extract(self, event: dict) -> list[dict]:
        payload = event.get("payload", {})
        timestamp = event.get("timestamp", "")
        event_type = event.get("type", "")
        source = event.get("source", "")

        signals = []

        # ----- Response-time tracking -----
        # Only outbound (user-authored) replies produce a response-time signal.
        # We look up the original inbound message by ID to compute the delta.
        if event_type in [EventType.EMAIL_SENT.value, EventType.MESSAGE_SENT.value]:
            if payload.get("is_reply") and payload.get("in_reply_to"):
                response_time = self._calculate_response_time(
                    payload["in_reply_to"], timestamp
                )
                if response_time is not None:
                    # Grab the first recipient as the contact identifier for
                    # per-contact response-time breakdowns.
                    contact = (
                        payload.get("to_addresses", [None])[0]
                        if payload.get("to_addresses")
                        else None
                    )
                    signals.append({
                        "type": "cadence_response_time",
                        "timestamp": timestamp,
                        "contact_id": contact,
                        "channel": source,
                        "response_time_seconds": response_time,
                    })

        # ----- Activity-window detection -----
        # Record the hour-of-day and day-of-week for every communication event
        # (both inbound and outbound).  Over time this builds a heatmap of the
        # user's natural activity windows — e.g., "most active 9-11am on weekdays".
        try:
            # Normalise the trailing "Z" to a proper UTC offset so fromisoformat
            # can parse it consistently across Python versions.
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            signals.append({
                "type": "cadence_activity",
                "timestamp": timestamp,
                "hour": dt.hour,
                "day_of_week": dt.strftime("%A").lower(),
                "direction": "outbound" if "sent" in event_type.lower() else "inbound",
                "channel": source,
            })
        except (ValueError, AttributeError) as e:
            # If the timestamp is missing or malformed we skip the activity
            # signal rather than failing the whole extraction.
            logger.debug('cadence_extractor: skipping event %s — malformed timestamp: %s',
                         event.get('id', 'unknown'), e)

        # Persist the signals into the running cadence profile.
        self._update_profile(signals)
        return signals

    def _calculate_response_time(self, original_message_id: str,
                                  response_timestamp: str) -> Optional[float]:
        """Look up the original inbound message and calculate response time.

        Queries the event store for the event whose payload.message_id
        matches ``original_message_id``, then returns the delta in seconds
        between the original timestamp and the user's reply.  This reveals
        how quickly the user replies to specific contacts or channels — a
        strong signal of priority and engagement.

        Uses an expression index on ``json_extract(payload, '$.message_id')``
        for O(log n) lookups (see storage/manager.py schema migration).

        Returns None if the original message isn't found or if timestamps
        can't be parsed (fail-open: the cadence profile simply doesn't
        record a response-time sample for this event).
        """
        with self.db.get_connection("events") as conn:
            row = conn.execute(
                """SELECT timestamp FROM events
                   WHERE json_extract(payload, '$.message_id') = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (original_message_id,),
            ).fetchone()

        if not row:
            return None

        try:
            original_dt = datetime.fromisoformat(
                row["timestamp"].replace("Z", "+00:00")
            )
            response_dt = datetime.fromisoformat(
                response_timestamp.replace("Z", "+00:00")
            )
            delta = (response_dt - original_dt).total_seconds()
            # Only return positive deltas — a negative value would mean the
            # "reply" was timestamped before the original, which indicates a
            # clock-skew or message-ID collision.  Silently discard.
            return delta if delta > 0 else None
        except (ValueError, AttributeError):
            return None

    def _update_profile(self, signals: list[dict]):
        """Incrementally merge new signals into the persisted cadence profile.

        The profile stores four running aggregates:
          - response_times:             global list (capped at 1000 entries)
          - per_contact_response_times: response times bucketed by contact
          - per_channel_response_times: response times bucketed by channel
          - hourly_activity / daily_activity: histogram counters for the
            activity-window heatmap
        """
        # Load the existing profile or bootstrap with empty structures.
        existing = self.ums.get_signal_profile("cadence")
        data = existing["data"] if existing else {
            "response_times": [],
            "hourly_activity": defaultdict(int),
            "daily_activity": defaultdict(int),
            "per_contact_response_times": defaultdict(list),
            "per_channel_response_times": defaultdict(list),
        }

        for signal in signals:
            if signal["type"] == "cadence_response_time":
                rt = signal["response_time_seconds"]
                # Append to the global response-time list for overall statistics.
                data["response_times"].append(rt)

                # Also bucket by contact so we can compare how fast the user
                # replies to different people (priority signal).
                contact = signal.get("contact_id")
                if contact:
                    if contact not in data["per_contact_response_times"]:
                        data["per_contact_response_times"][contact] = []
                    data["per_contact_response_times"][contact].append(rt)

                # Bucket by channel (email vs. Slack vs. SMS) to detect
                # channel-specific habits — e.g., user replies to Slack in
                # minutes but to email in hours.
                channel = signal.get("channel")
                if channel:
                    if channel not in data["per_channel_response_times"]:
                        data["per_channel_response_times"][channel] = []
                    data["per_channel_response_times"][channel].append(rt)

            elif signal["type"] == "cadence_activity":
                # Increment histogram counters. These are stored as string keys
                # ("0"-"23" for hours, "monday"-"sunday" for days) to stay
                # JSON-serialisable.
                hour = str(signal["hour"])
                day = signal["day_of_week"]
                if hour not in data["hourly_activity"]:
                    data["hourly_activity"][hour] = 0
                data["hourly_activity"][hour] += 1
                if day not in data["daily_activity"]:
                    data["daily_activity"][day] = 0
                data["daily_activity"][day] += 1

        # Cap the global response-time list to prevent unbounded growth.
        # Keeping the most recent 1000 entries provides enough data for
        # statistical baselines while bounding storage.
        if len(data.get("response_times", [])) > 1000:
            data["response_times"] = data["response_times"][-1000:]

        # Recompute derived metrics from the updated raw histograms so that
        # CadenceProfile fields (peak_hours, quiet_hours_observed,
        # avg_response_time_by_domain) always reflect the latest data.
        self._compute_derived_metrics(data)

        self.ums.update_signal_profile("cadence", data)

    # ------------------------------------------------------------------
    # Derived-metric computation — run after every profile update
    # ------------------------------------------------------------------

    def _compute_derived_metrics(self, data: dict) -> None:
        """Recompute all derived CadenceProfile fields from raw histogram data.

        Called at the end of every ``_update_profile`` invocation so that the
        three aggregate fields defined in ``CadenceProfile`` but not produced
        by the incremental signal collection stay in sync with the raw data:

          - ``peak_hours``               — hours of highest communication activity
          - ``quiet_hours_observed``     — naturally quiet (typically sleep) windows
          - ``avg_response_time_by_domain`` — reply latency grouped by email domain

        This is a pure recomputation (no I/O): it mutates *data* in place and
        the caller is responsible for persisting to the store.

        Requires a minimum of 50 total activity samples before producing any
        derived output, so early-lifecycle profiles don't generate noisy results.
        """
        self._compute_peak_hours(data)
        self._compute_quiet_hours(data)
        self._compute_domain_response_times(data)

    def _compute_peak_hours(self, data: dict) -> None:
        """Derive peak activity hours from the hourly_activity histogram.

        Peak hours are those where the activity count exceeds the mean by at
        least half a standard deviation.  Using 0.5σ (rather than 1σ) keeps
        broad active windows (e.g., a user active 9 AM–5 PM) fully labelled
        rather than only capturing the single busiest hour.

        Requires at least 50 total samples across all hour buckets to produce a
        statistically meaningful result; silently skips if insufficient data.

        Stores results in ``data["peak_hours"]`` as a sorted list of ints
        in the range 0–23 (UTC hours).

        Example:
            If a user sends most messages between 9 and 17:
            >>> data["peak_hours"]
            [9, 10, 11, 12, 13, 14, 15, 16, 17]
        """
        hourly = data.get("hourly_activity", {})
        if not hourly:
            return

        # Convert string keys ("0"–"23") to int counts, filling missing hours
        # with 0 so that the mean and standard deviation are computed across
        # all 24 hour buckets — not just the ones that have observed activity.
        # This prevents inflated thresholds when only a few hours have data.
        counts = {int(h): v for h, v in hourly.items()}
        total_samples = sum(counts.values())

        # Need at least 50 samples for a reliable estimate.
        if total_samples < 50:
            return

        # Full 24-element array including hours with zero activity.
        values = [counts.get(h, 0) for h in range(24)]
        mean = sum(values) / len(values)
        # Population standard deviation across all known hour buckets.
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5

        # Threshold: mean + 0.5 * σ so broad active windows are fully captured.
        threshold = mean + 0.5 * std_dev
        data["peak_hours"] = sorted(h for h, v in counts.items() if v > threshold)

    def _compute_quiet_hours(self, data: dict) -> None:
        """Derive quiet (typically sleep) windows from the hourly_activity histogram.

        A quiet window is a contiguous span of hours whose activity count falls
        at or below 10% of the peak hour's count.  The algorithm handles
        midnight-wrapping windows correctly (e.g., 22:00–06:00) by using an
        active (non-quiet) hour as the scan anchor rather than always starting
        at 00:00.

        Requires at least 50 total samples and a minimum span of 3 contiguous
        hours to qualify as a quiet window.

        Stores results in ``data["quiet_hours_observed"]`` as a list of
        ``(start_hour, end_hour)`` tuples where both values are in 0–23.
        When a span crosses midnight, ``end_hour < start_hour``
        (e.g., ``(22, 6)`` means 22:00 to 06:00).

        Example:
            >>> data["quiet_hours_observed"]
            [(22, 6)]   # sleep window from 10 PM to 6 AM
        """
        hourly = data.get("hourly_activity", {})
        if not hourly:
            return

        counts = {int(h): v for h, v in hourly.items()}
        total_samples = sum(counts.values())

        if total_samples < 50:
            return

        # Build a full 24-element list filling missing hours with 0.
        full = [counts.get(h, 0) for h in range(24)]
        peak = max(full)
        if peak == 0:
            return

        # Threshold: ≤10% of peak count qualifies as "quiet".
        threshold = peak * 0.10
        quiet_set = {h for h in range(24) if full[h] <= threshold}

        if not quiet_set:
            return

        # Use the first non-quiet hour as the scan anchor so that spans which
        # cross midnight (e.g., 22–06) are detected as a single contiguous run.
        non_quiet_anchor = next(
            (h for h in range(24) if full[h] > threshold), None
        )
        if non_quiet_anchor is None:
            # Pathological case: every hour is quiet — store as single 24-h span.
            data["quiet_hours_observed"] = [(0, 0)]
            return

        spans: list[tuple[int, int]] = []
        i = 0
        while i < 24:
            hour = (non_quiet_anchor + i) % 24
            if hour in quiet_set:
                # Entering a new quiet span — extend until the span ends.
                span_start_offset = i
                while i < 24 and (non_quiet_anchor + i) % 24 in quiet_set:
                    i += 1
                length = i - span_start_offset
                if length >= 3:
                    start_h = (non_quiet_anchor + span_start_offset) % 24
                    end_h = (non_quiet_anchor + i) % 24
                    spans.append((start_h, end_h))
            else:
                i += 1

        if spans:
            data["quiet_hours_observed"] = spans

    def _compute_domain_response_times(self, data: dict) -> None:
        """Derive average response times grouped by email domain.

        Iterates over the raw ``per_contact_response_times`` dict (contact →
        list[float seconds]) and groups entries by the domain portion of each
        email address (the part after ``@``).  Phone numbers and other
        non-email identifiers are skipped.

        Domains with fewer than 3 data points are excluded to avoid
        single-sample noise (e.g., one email from an unusual domain).

        Stores results in ``data["avg_response_time_by_domain"]`` as a dict
        mapping domain string → average response time in seconds.

        Example:
            If the user replies to gmail.com addresses in ~30 min and to
            corp.example.com addresses in ~4 hours:
            >>> data["avg_response_time_by_domain"]
            {"gmail.com": 1820.0, "corp.example.com": 14400.0}
        """
        per_contact = data.get("per_contact_response_times", {})
        if not per_contact:
            return

        # Accumulate per-domain response time lists.
        domain_times: dict[str, list[float]] = {}
        for contact, times in per_contact.items():
            if "@" not in contact:
                # Skip phone numbers and other non-email identifiers.
                continue
            domain = contact.split("@")[-1].lower()
            if domain not in domain_times:
                domain_times[domain] = []
            domain_times[domain].extend(times)

        # Compute average per domain; require ≥3 samples for reliability.
        avg_by_domain = {
            domain: sum(times) / len(times)
            for domain, times in domain_times.items()
            if len(times) >= 3
        }

        if avg_by_domain:
            data["avg_response_time_by_domain"] = avg_by_domain
