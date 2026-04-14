"""
Life OS -- Cohort Profiler

Builds an anonymous, privacy-preserving cohort profile from the user's
signal profiles.  The cohort profile is a set of coarse-grained dimension
buckets (e.g. chronotype=early_bird, communication_volume=high) that
characterize the user without revealing any PII.

The combined profile vector is hashed into a cohort_key -- an 8-character
hex string that can match the user to aggregate baselines from users with
similar behavioral patterns, without ever transmitting individual data.

Dimensions:
    chronotype          — early_bird / night_owl / mixed
    communication_volume — low / medium / high
    stress_profile      — low / moderate / high
    meeting_load        — low / medium / heavy
    response_speed      — fast / medium / slow
    social_breadth      — narrow / medium / broad

Architecture:
    The profiler reads from signal_profiles and mood_history, computes
    each dimension, stores results in cohort_profiles, and returns the
    cohort_key.  The population_baselines table can then be filtered
    by cohort to give "people like you" baselines instead of generic
    population baselines.

Privacy model:
    - Dimensions are coarse (3-4 buckets each) so they cannot be
      reversed into individual data points.
    - The cohort_key is a SHA-256 hash of sorted dimension:bucket pairs.
    - No raw signal data is ever included in the profile.
    - Contribution to a peer network is optional and always differential-
      privacy noised before leaving the device.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore

logger = logging.getLogger(__name__)

# Current profile schema version.  Increment when dimension definitions change.
PROFILE_VERSION = 1


class CohortProfiler:
    """Computes and stores the user's anonymous cohort classification."""

    def __init__(self, db: DatabaseManager, ums: UserModelStore):
        self.db = db
        self.ums = ums

    def compute_profile(self) -> dict[str, str]:
        """Compute all cohort dimensions from current signal profiles.

        Returns:
            Dict mapping dimension name to bucket string,
            e.g. {"chronotype": "early_bird", "communication_volume": "high"}.
        """
        profile: dict[str, str] = {}

        profile["chronotype"] = self._classify_chronotype()
        profile["communication_volume"] = self._classify_communication_volume()
        profile["stress_profile"] = self._classify_stress_profile()
        profile["meeting_load"] = self._classify_meeting_load()
        profile["response_speed"] = self._classify_response_speed()
        profile["social_breadth"] = self._classify_social_breadth()

        return profile

    def compute_and_store(self) -> str | None:
        """Compute profile, store dimensions, return cohort_key.

        Returns:
            8-character hex cohort key, or None if insufficient data.
        """
        profile = self.compute_profile()

        # Require at least 3 non-"unknown" dimensions for a meaningful cohort
        known = {k: v for k, v in profile.items() if v != "unknown"}
        if len(known) < 3:
            logger.debug(
                "CohortProfiler: only %d known dimensions, need 3+", len(known)
            )
            return None

        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection("user_model") as conn:
            for dimension, bucket in profile.items():
                conn.execute(
                    """INSERT OR REPLACE INTO cohort_profiles
                       (profile_version, dimension, bucket, computed_at,
                        evidence_count, confidence)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (PROFILE_VERSION, dimension, bucket, now,
                     0, 0.5 if bucket == "unknown" else 0.7),
                )

        cohort_key = self._compute_cohort_key(known)
        logger.info(
            "CohortProfiler: computed profile %s -> cohort_key=%s",
            known, cohort_key,
        )
        return cohort_key

    def get_stored_profile(self) -> dict[str, str]:
        """Load the most recently stored cohort profile."""
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT dimension, bucket FROM cohort_profiles
                       WHERE profile_version = ?""",
                    (PROFILE_VERSION,),
                ).fetchall()
                return {row["dimension"]: row["bucket"] for row in rows}
        except Exception:
            return {}

    def get_cohort_key(self) -> str | None:
        """Return the cohort key from the stored profile, or None."""
        profile = self.get_stored_profile()
        known = {k: v for k, v in profile.items() if v != "unknown"}
        if len(known) < 3:
            return None
        return self._compute_cohort_key(known)

    # ------------------------------------------------------------------
    # Dimension classifiers
    # ------------------------------------------------------------------

    def _classify_chronotype(self) -> str:
        """Classify as early_bird, night_owl, or mixed from temporal profile."""
        profile = self.ums.get_signal_profile("temporal")
        if not profile or profile.get("samples_count", 0) < 30:
            return "unknown"

        data = profile.get("data", {})
        activity_by_hour = data.get("activity_by_hour", {})
        if not activity_by_hour:
            return "unknown"

        morning = sum(activity_by_hour.get(str(h), 0) for h in range(5, 11))
        evening = sum(activity_by_hour.get(str(h), 0) for h in range(18, 24))
        total = morning + evening
        if total < 10:
            return "unknown"

        morning_ratio = morning / total
        if morning_ratio >= 0.65:
            return "early_bird"
        elif morning_ratio <= 0.35:
            return "night_owl"
        return "mixed"

    def _classify_communication_volume(self) -> str:
        """Classify email volume as low, medium, or high."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        try:
            with self.db.get_connection("events") as conn:
                row = conn.execute(
                    """SELECT COUNT(*) as cnt
                       FROM events
                       WHERE type IN ('email.received', 'email.sent')
                         AND timestamp > ?""",
                    (cutoff,),
                ).fetchone()
                daily_avg = (row["cnt"] or 0) / 30.0
        except Exception:
            return "unknown"

        if daily_avg < 1:
            return "unknown"
        if daily_avg < 25:
            return "low"
        elif daily_avg < 60:
            return "medium"
        return "high"

    def _classify_stress_profile(self) -> str:
        """Classify average stress as low, moderate, or high."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        try:
            with self.db.get_connection("user_model") as conn:
                row = conn.execute(
                    """SELECT AVG(stress_level) as avg_stress, COUNT(*) as cnt
                       FROM mood_history
                       WHERE timestamp > ?
                         AND stress_level IS NOT NULL""",
                    (cutoff,),
                ).fetchone()
                if (row["cnt"] or 0) < 5:
                    return "unknown"
                avg = row["avg_stress"]
        except Exception:
            return "unknown"

        if avg is None:
            return "unknown"
        if avg < 0.35:
            return "low"
        elif avg < 0.60:
            return "moderate"
        return "high"

    def _classify_meeting_load(self) -> str:
        """Classify meeting load as low, medium, or heavy."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        try:
            with self.db.get_connection("events") as conn:
                row = conn.execute(
                    """SELECT COUNT(*) as cnt
                       FROM events
                       WHERE type IN ('calendar.event.created', 'calendar.event.updated')
                         AND timestamp > ?""",
                    (cutoff,),
                ).fetchone()
                weekly_avg = (row["cnt"] or 0) / 4.3  # 30 days ~ 4.3 weeks
        except Exception:
            return "unknown"

        if weekly_avg < 1:
            return "unknown"
        if weekly_avg < 5:
            return "low"
        elif weekly_avg < 15:
            return "medium"
        return "heavy"

    def _classify_response_speed(self) -> str:
        """Classify response speed as fast, medium, or slow from cadence profile."""
        profile = self.ums.get_signal_profile("cadence")
        if not profile or profile.get("samples_count", 0) < 10:
            return "unknown"

        data = profile.get("data", {})
        response_times = data.get("avg_response_time_by_contact", {})
        if not response_times:
            return "unknown"

        # Median response time across contacts (values are in seconds)
        times = sorted(response_times.values())
        if not times:
            return "unknown"
        n = len(times)
        if n % 2 == 1:
            median_seconds = times[n // 2]
        else:
            median_seconds = (times[n // 2 - 1] + times[n // 2]) / 2
        median_minutes = median_seconds / 60.0

        if median_minutes < 15:
            return "fast"
        elif median_minutes < 120:
            return "medium"
        return "slow"

    def _classify_social_breadth(self) -> str:
        """Classify social breadth as narrow, medium, or broad."""
        profile = self.ums.get_signal_profile("relationships")
        if not profile or profile.get("samples_count", 0) < 10:
            return "unknown"

        data = profile.get("data", {})
        # Count contacts with at least 2 interactions
        contacts = data.get("interaction_timestamps", {})
        active_count = sum(
            1 for ts_list in contacts.values()
            if isinstance(ts_list, list) and len(ts_list) >= 2
        )

        if active_count < 3:
            return "narrow"
        elif active_count < 15:
            return "medium"
        return "broad"

    # ------------------------------------------------------------------
    # Cohort key computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cohort_key(profile: dict[str, str]) -> str:
        """Hash the profile into an 8-character hex cohort key."""
        canonical = json.dumps(sorted(profile.items()), separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:8]
