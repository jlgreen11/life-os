"""
Life OS — Mood Inference Engine

Infers the user's current emotional/cognitive state from a composite
of all available signals. Never asks "how are you feeling?"

Key principle: The AI never tells the user their mood. It silently
adjusts its behavior (tone, timing, proactivity) based on the inference.
"""

from __future__ import annotations

import logging

from models.core import EventType
from models.user_model import MoodSignal, MoodState
from services.signal_extractor.base import BaseExtractor

logger = logging.getLogger(__name__)


class MoodInferenceEngine(BaseExtractor):
    """
    Infers the user's current emotional/cognitive state from a composite
    of all available signals.
    """

    # Default baseline thresholds used before enough data has been collected to
    # compute personalised baselines.  Each metric's delta-from-baseline drives
    # the mood signal's magnitude.  Over time these are replaced by per-user
    # values stored in the "baselines" signal profile.
    DEFAULT_BASELINES = {
        "typing_speed_wpm": 40.0,
        "response_latency_seconds": 300.0,    # 5 minutes
        "message_length_words": 25.0,
        "exclamation_rate": 0.1,
        "emoji_rate": 0.05,
    }

    # Negative-valence words used to score both outbound (the user's own
    # language) and inbound (stress-inducing content arriving at the user).
    NEGATIVE_WORDS = {
        "frustrated", "annoyed", "tired", "exhausted", "stressed",
        "worried", "overwhelmed", "confused", "angry", "upset",
        "sorry", "unfortunately", "problem", "issue", "difficult",
        "urgent", "critical", "emergency", "immediately", "asap",
        "disappointing", "concerned", "unacceptable", "overdue",
        "escalate", "failed", "failure", "broken", "blocked",
    }

    def can_process(self, event: dict) -> bool:
        # The mood engine casts a wide net: it ingests communication events
        # in BOTH directions (outbound for self-expression analysis, inbound
        # for incoming-stress detection), health data (sleep quality/duration),
        # calendar density (busyness/stress), financial transactions (spending
        # spikes), and location changes.
        return event.get("type") in [
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_SENT.value,
            EventType.EMAIL_RECEIVED.value,
            EventType.MESSAGE_RECEIVED.value,
            EventType.HEALTH_METRIC_UPDATED.value,
            EventType.SLEEP_RECORDED.value,
            EventType.CALENDAR_EVENT_CREATED.value,
            EventType.TRANSACTION_NEW.value,
            EventType.LOCATION_CHANGED.value,
            "system.user.command",
        ]

    def extract(self, event: dict) -> list[dict]:
        """Collect one or more mood-relevant signals from this event.

        Each signal carries:
          - signal_type:          what was measured (e.g., "message_length")
          - value:                the raw measurement
          - delta_from_baseline:  how far the value deviates from the user's
                                  personal baseline (positive = above, negative = below)
          - weight:               how strongly this signal should influence the
                                  overall mood estimate (0-1)
          - source:               originating data channel

        Signals are accumulated in the "mood_signals" profile and consumed
        periodically by ``compute_current_mood()``.
        """
        mood_signals = []
        event_type = event.get("type", "")
        payload = event.get("payload", {})

        # --- Outbound communication signals ---
        # The user's own writing: message length relative to baseline reveals
        # terse/stressed vs. engaged communication.  Negative-valence words in
        # outbound text are a strong mood indicator (weight=0.6) because the
        # user chose those words deliberately.
        is_outbound = event_type in [
            EventType.EMAIL_SENT.value, EventType.MESSAGE_SENT.value,
        ]
        is_inbound = event_type in [
            EventType.EMAIL_RECEIVED.value, EventType.MESSAGE_RECEIVED.value,
        ]

        if is_outbound:
            text = payload.get("body", "") or payload.get("body_plain", "")
            if text:
                word_count = len(text.split())
                baseline_length = self._get_baseline("message_length_words")

                mood_signals.append({
                    "signal_type": "message_length",
                    "value": word_count,
                    # Fractional deviation: +0.5 means 50% longer than baseline.
                    "delta_from_baseline": (word_count - baseline_length) / max(baseline_length, 1),
                    "weight": 0.3,
                    "source": event.get("source", "unknown"),
                })

                negative_count = sum(
                    1 for w in text.lower().split() if w in self.NEGATIVE_WORDS
                )
                if negative_count > 0:
                    mood_signals.append({
                        "signal_type": "negative_language",
                        "value": negative_count / max(word_count, 1),
                        "delta_from_baseline": negative_count / max(word_count, 1),
                        "weight": 0.6,
                        "source": event.get("source", "unknown"),
                    })

        # --- Inbound communication signals ---
        # Content arriving at the user directly affects mood.  An angry email
        # from a boss or a tense message from a partner is stress-inducing
        # regardless of whether the user has replied yet.  Inbound signals
        # carry lower weight than outbound (0.4 vs 0.6) because someone else's
        # words are a weaker indicator than the user's own expression, but they
        # are still significant — and they arrive *immediately*, not after the
        # user composes a reply.
        if is_inbound:
            text = payload.get("body", "") or payload.get("body_plain", "")
            if text:
                word_count = len(text.split())
                negative_count = sum(
                    1 for w in text.lower().split() if w in self.NEGATIVE_WORDS
                )
                if negative_count > 0:
                    mood_signals.append({
                        "signal_type": "incoming_negative_language",
                        "value": negative_count / max(word_count, 1),
                        "delta_from_baseline": negative_count / max(word_count, 1),
                        "weight": 0.4,
                        "source": event.get("source", "unknown"),
                    })

                # Incoming urgency pressure: messages with high exclamation
                # density or ALL-CAPS words create pressure on the user even
                # before they open the message.
                caps_words = sum(
                    1 for w in text.split()
                    if w.isupper() and len(w) > 1
                )
                exclamation_density = text.count("!") / max(word_count, 1)
                if caps_words >= 2 or exclamation_density > 0.1:
                    mood_signals.append({
                        "signal_type": "incoming_pressure",
                        "value": caps_words + exclamation_density,
                        "delta_from_baseline": 0.3,
                        "weight": 0.3,
                        "source": event.get("source", "unknown"),
                    })

        # --- Sleep signals ---
        # Sleep quality and duration carry high weight because poor sleep
        # reliably predicts lower energy and elevated stress the next day.
        if event_type == EventType.SLEEP_RECORDED.value:
            hours = payload.get("duration_hours", 7)
            quality = payload.get("quality_score", 0.5)
            mood_signals.append({
                "signal_type": "sleep_quality",
                "value": quality,
                # 0.7 is treated as the baseline for "good" sleep quality.
                "delta_from_baseline": quality - 0.7,
                "weight": 0.8,
                "source": "health",
            })
            mood_signals.append({
                "signal_type": "sleep_duration",
                "value": hours,
                # 7.5 hours is the assumed baseline for adequate sleep.
                "delta_from_baseline": (hours - 7.5) / 7.5,
                "weight": 0.5,
                "source": "health",
            })

        # --- Proxy energy signals (when no health data available) ---
        # In the absence of sleep/activity trackers, infer energy from behavioral
        # proxies. These signals are essential for populating episode.energy_level
        # which drives mood-aware features throughout the system.
        #
        # CRITICAL FIX (iteration 146):
        # Episodes had 0% energy_level population because compute_current_mood()
        # only considers ["sleep_quality", "sleep_duration", "activity_level"]
        # signals, but ZERO of these existed in the database (all 200 mood signals
        # were incoming_pressure/negative_language/message_length). Without health
        # connectors, we need proxy energy signals to populate this critical field.
        #
        # Time-of-day energy proxy:
        # People naturally have higher energy mid-morning to mid-afternoon (9am-3pm)
        # and lower energy early morning (5-8am) and late evening (9pm-midnight).
        # This circadian pattern provides a baseline energy estimate even without
        # any direct physiological data.
        #
        # Only extract when there's actual message content (text), to avoid polluting
        # tests that expect no signals from empty messages.
        if (is_outbound or is_inbound):
            text = payload.get("body", "") or payload.get("body_plain", "")
            if text and len(text.strip()) > 0:  # Only if there's actual content
                from datetime import datetime, timezone
                try:
                    timestamp_str = event.get("timestamp", "")
                    if timestamp_str:
                        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        hour = dt.hour  # 0-23

                        # Energy curve based on circadian rhythm:
                        # 0-5: very low (0.2)
                        # 5-8: ramping up (0.4-0.6)
                        # 8-12: peak morning (0.8)
                        # 12-14: post-lunch dip (0.6)
                        # 14-17: afternoon peak (0.7)
                        # 17-21: declining (0.5)
                        # 21-24: very low (0.3)
                        if 0 <= hour < 5:
                            energy_estimate = 0.2
                        elif 5 <= hour < 8:
                            energy_estimate = 0.4 + (hour - 5) * 0.07  # ramp 0.4→0.6
                        elif 8 <= hour < 12:
                            energy_estimate = 0.8
                        elif 12 <= hour < 14:
                            energy_estimate = 0.6
                        elif 14 <= hour < 17:
                            energy_estimate = 0.7
                        elif 17 <= hour < 21:
                            energy_estimate = 0.5
                        else:  # 21-24
                            energy_estimate = 0.3

                        mood_signals.append({
                            "signal_type": "circadian_energy",
                            "value": energy_estimate,
                            # Delta from baseline (0.5 = neutral energy)
                            "delta_from_baseline": energy_estimate - 0.5,
                            # Lower weight than sleep (0.3 vs 0.8) since it's indirect
                            "weight": 0.3,
                            "source": event.get("source", "unknown"),
                        })
                except Exception:
                    pass  # Malformed timestamp, skip circadian signal

        # Activity-level proxy from communication cadence:
        # Outbound communication (emails sent, messages sent) indicates active
        # engagement. The longer the message, the more energy investment it
        # represents. Use message length as a proxy for current activity level.
        if is_outbound:
            text = payload.get("body", "") or payload.get("body_plain", "")
            if text:
                word_count = len(text.split())
                baseline_length = self._get_baseline("message_length_words")

                # Longer-than-baseline messages suggest active engagement (high energy)
                # Shorter suggests low-effort communication (potentially low energy)
                if word_count > baseline_length * 1.2:  # 20%+ above baseline
                    mood_signals.append({
                        "signal_type": "communication_energy",
                        "value": min(1.0, word_count / baseline_length),
                        "delta_from_baseline": 0.2,
                        "weight": 0.2,
                        "source": event.get("source", "unknown"),
                    })
                elif word_count < baseline_length * 0.5:  # 50%+ below baseline
                    mood_signals.append({
                        "signal_type": "communication_energy",
                        "value": word_count / baseline_length,
                        "delta_from_baseline": -0.3,
                        "weight": 0.2,
                        "source": event.get("source", "unknown"),
                    })

        # --- Calendar density (stress indicator) ---
        # Each new calendar event increments the density counter.  A burst of
        # calendar events in a short window signals a packed schedule and
        # potential stress.
        if event_type == EventType.CALENDAR_EVENT_CREATED.value:
            mood_signals.append({
                "signal_type": "calendar_density",
                "value": 1.0,
                "delta_from_baseline": 0.0,
                "weight": 0.2,
                "source": "calendar",
            })

            # --- Social battery (meeting load indicator) ---
            # Social interactions — particularly multi-person meetings — drain
            # the social battery for most people.  A calendar event with ≥1
            # attendees (or a title containing "meeting", "standup", "sync",
            # "review", "interview") is treated as a social draw-down; a solo
            # block is treated as a recovery opportunity.
            #
            # Why derive social_battery from calendar events rather than actual
            # attendance?  Because Life OS does not yet track real-time
            # attendance; it only sees CalDAV/Google Calendar sync events.
            # Using scheduled meetings as a proxy is the best available signal
            # without adding a new connector.
            #
            # Social-drain heuristic (value = residual battery level, 0-1):
            #   - Solo block (no attendees, no social keywords): 0.7  (mild recovery)
            #   - Small meeting (1-2 attendees):                 0.5  (moderate drain)
            #   - Medium meeting (3-5 attendees):                0.3  (significant drain)
            #   - Large meeting (6+ attendees):                  0.1  (heavy drain)
            #
            # The weight is intentionally low (0.4) because a single meeting
            # cannot override the overall daily mood — many such signals must
            # accumulate to move the average materially.
            attendees = payload.get("attendees", [])
            title = payload.get("title", "").lower()
            social_keywords = {"meeting", "standup", "stand-up", "sync", "review",
                               "interview", "workshop", "presentation", "all-hands",
                               "team", "1:1", "one-on-one"}
            is_social = bool(attendees) or any(kw in title for kw in social_keywords)

            if is_social:
                n = len(attendees)
                if n >= 6:
                    battery_after = 0.1   # Heavy drain: large meeting
                elif n >= 3:
                    battery_after = 0.3   # Significant drain: medium meeting
                elif n >= 1:
                    battery_after = 0.5   # Moderate drain: small meeting
                else:
                    # Social keyword in title but no explicit attendees — treat
                    # as a small meeting (conservative estimate).
                    battery_after = 0.5
            else:
                # Solo block: treat as partial recovery from previous social drain.
                battery_after = 0.7

            mood_signals.append({
                "signal_type": "social_battery",
                "value": battery_after,
                # Delta relative to a neutral 0.5 baseline (positive = above, negative = below).
                "delta_from_baseline": battery_after - 0.5,
                "weight": 0.4,
                "source": "calendar",
            })

        # --- Spending anomaly (stress signal) ---
        # Unusually large transactions can correlate with stress-spending or
        # an out-of-routine event.  Only flag amounts > $100.
        if event_type == EventType.TRANSACTION_NEW.value:
            amount = abs(payload.get("amount", 0))
            if amount > 100:
                mood_signals.append({
                    "signal_type": "spending_spike",
                    "value": amount,
                    "delta_from_baseline": 0.5,  # Simplified fixed delta for now.
                    "weight": 0.3,
                    "source": "finance",
                })

        # Persist accumulated signals so compute_current_mood can consume them.
        if mood_signals:
            self._update_mood_state(mood_signals)

        # Wrap all signals in a single envelope dict for the pipeline's return list.
        return [{"type": "mood_signal", "signals": mood_signals}] if mood_signals else []

    def compute_current_mood(self) -> MoodState:
        """
        Compute the current mood state from recent signals.
        Called periodically (every 15 minutes) and on-demand.

        The mood is represented as a multi-dimensional state:
          - energy_level:       derived from sleep and activity signals (0-1)
          - stress_level:       derived from calendar density, negative language,
                                spending spikes, and response latency (0-1)
          - social_battery:     weighted average of "social_battery" signals derived
                                from calendar events (attendee count + title keywords)
          - cognitive_load:     approximated by the sheer count of stress signals
          - emotional_valence:  inverted weighted average of negativity signals
                                (1.0 = positive, 0.0 = negative)
          - confidence:         scales linearly with signal count, capped at 1.0
          - trend:              "improving", "declining", or "stable" from history

        Returns a neutral MoodState when no signals are available.
        """
        existing = self.ums.get_signal_profile("mood_signals")
        if not existing:
            return MoodState()

        recent_signals = existing["data"].get("recent_signals", [])
        if not recent_signals:
            return MoodState()

        # Partition signals into the mood dimensions they inform.
        # A single signal type may contribute to multiple dimensions (e.g.,
        # "negative_language" feeds both stress and valence).  Inbound signal
        # types (incoming_negative_language, incoming_pressure) feed stress
        # and valence so that stress-inducing content arriving at the user
        # is reflected immediately rather than only after the user replies.
        #
        # CRITICAL FIX (iteration 146):
        # Added proxy energy signals (circadian_energy, communication_energy) to
        # enable energy_level population when health trackers are unavailable.
        # Previously only ["sleep_quality", "sleep_duration", "activity_level"]
        # were considered, but ZERO of these signals existed in production,
        # causing 100% of episodes to have NULL energy_level despite 27K+ mood
        # signals being available.
        energy_signals = [s for s in recent_signals if s["signal_type"] in [
            "sleep_quality", "sleep_duration", "activity_level",
            "circadian_energy", "communication_energy",
        ]]
        stress_signals = [s for s in recent_signals if s["signal_type"] in [
            "calendar_density", "negative_language", "spending_spike",
            "response_latency", "incoming_negative_language", "incoming_pressure",
        ]]
        valence_signals = [s for s in recent_signals if s["signal_type"] in [
            "negative_language", "emoji_usage", "message_length",
            "incoming_negative_language",
        ]]

        # Social battery: derived from accumulated social_battery signals emitted
        # by the calendar event handler above.  Falls back to 0.5 (neutral) when
        # no meeting data has been seen yet — e.g., for a user who has not
        # connected a calendar connector.
        social_signals = [s for s in recent_signals if s["signal_type"] == "social_battery"]

        mood = MoodState(
            energy_level=self._weighted_average(energy_signals, default=0.5),
            stress_level=self._weighted_average(stress_signals, default=0.3),
            social_battery=self._weighted_average(social_signals, default=0.5),
            # Cognitive load is a rough proxy: each stress signal adds 0.15,
            # capped at 1.0 to stay within the 0-1 range.
            cognitive_load=min(1.0, len(stress_signals) * 0.15),
            # Valence is inverted: high negative-signal averages yield low valence.
            emotional_valence=1.0 - self._weighted_average(valence_signals, default=0.3),
            # Confidence ramps with signal volume; 10+ signals reach full confidence.
            confidence=min(1.0, len(recent_signals) * 0.1),
            # Attach the most recent 10 raw signals for explainability/debugging.
            contributing_signals=[
                MoodSignal(**s) for s in recent_signals[-10:]
            ],
        )

        # Overlay a directional trend by comparing the current snapshot against
        # the historical mood baseline.
        mood.trend = self._compute_trend()

        return mood

    def _weighted_average(self, signals: list[dict], default: float = 0.5) -> float:
        """Compute a weight-normalised average of signal values.

        Each signal's value (preferred) or delta_from_baseline (fallback) is
        multiplied by its declared weight, summed, and divided by the total
        weight. The result is clamped to [0.0, 1.0]. When no signals are
        available, ``default`` is returned so the mood dimensions degrade
        gracefully to neutral values.

        CRITICAL FIX (iteration 146):
        Previously used only delta_from_baseline, which caused incorrect mood
        calculations. For example, circadian_energy with value=0.8 (morning peak)
        and delta=0.3 would return 0.3 instead of 0.8. Energy/stress/valence
        should reflect absolute states (0-1 scale), not deviations from baseline.

        For backward compatibility with existing signals that only have
        delta_from_baseline (e.g., from tests), we use delta as a fallback.
        """
        if not signals:
            return default
        total_weight = sum(abs(s.get("weight", 1.0)) for s in signals)
        if total_weight == 0:
            return default
        weighted_sum = sum(
            # Prefer 'value' (absolute 0-1 scale), fallback to 'delta_from_baseline'
            s.get("value", s.get("delta_from_baseline", 0)) * s.get("weight", 1.0)
            for s in signals
        )
        return min(1.0, max(0.0, weighted_sum / total_weight))

    def _compute_trend(self) -> str:
        """Compare recent mood to historical baseline to determine trajectory.

        Queries the ``mood_history`` time-series table (written by
        ``SignalExtractorPipeline.get_current_mood()``) and computes a
        composite mood score for two windows:

        - **Recent window** (last 4 snapshots, ~1 hour at 15-min cadence):
          the user's current trajectory.
        - **Baseline window** (snapshots 5–12, the hour before that):
          the recent historical baseline to compare against.

        A single *composite score* is computed for each window:

            composite = energy_level + emotional_valence − stress_level

        This combines the three most meaningful mood dimensions into a single
        scalar in the range [−1, 2] (high energy and positive affect decrease
        stress → high composite; exhausted and stressed → low composite).

        Thresholds (empirically chosen to avoid hair-trigger changes):
        - composite delta > +0.10  → "improving"
        - composite delta < −0.10  → "declining"
        - otherwise                → "stable"

        Returns "stable" when there are fewer than 5 history rows (not enough
        data to establish a baseline), or when a database error occurs.  This
        matches the safe default so the calling code never crashes.

        Algorithm notes:
        - We use ``ORDER BY timestamp DESC LIMIT 12`` to bound the query cost.
        - Confidence-weighted averaging would be more accurate but adds
          complexity without proportionate benefit given the ~15-min cadence.
        - The threshold is intentionally conservative (0.10) to avoid surfacing
          noise from a single outlier signal as a "declining" mood alert.
        """
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT energy_level, stress_level, emotional_valence
                       FROM mood_history
                       ORDER BY timestamp DESC
                       LIMIT 12""",
                ).fetchall()
        except Exception:
            # Fail-open: return "stable" if history cannot be read.
            return "stable"

        # Need at least 5 rows to split into a recent window and a baseline.
        if len(rows) < 5:
            return "stable"

        def _composite(subset: list) -> float:
            """Compute average composite mood score for a list of DB rows.

            composite = energy_level + emotional_valence - stress_level

            Each dimension defaults to 0.5 on NULL (the neutral/default value
            used at insertion time) so missing data does not skew the score.
            """
            total = 0.0
            for row in subset:
                energy = row["energy_level"] if row["energy_level"] is not None else 0.5
                stress = row["stress_level"] if row["stress_level"] is not None else 0.5
                valence = row["emotional_valence"] if row["emotional_valence"] is not None else 0.5
                total += energy + valence - stress
            return total / len(subset)

        # Rows come back newest-first.  Slice into the two windows.
        recent_window = rows[:4]      # Most recent ~1 hour
        baseline_window = rows[4:]    # Previous ~2 hours

        recent_score = _composite(recent_window)
        baseline_score = _composite(baseline_window)
        delta = recent_score - baseline_score

        if delta > 0.10:
            return "improving"
        if delta < -0.10:
            return "declining"
        return "stable"

    def _get_baseline(self, metric: str) -> float:
        """Get the user's personal baseline for a metric.

        First checks the "baselines" signal profile (populated once enough
        data has been collected).  Falls back to the class-level
        DEFAULT_BASELINES dict, and ultimately to 0.5 as a last resort.
        """
        profile = self.ums.get_signal_profile("baselines")
        if profile and metric in profile["data"]:
            return profile["data"][metric]
        return self.DEFAULT_BASELINES.get(metric, 0.5)

    def _update_mood_state(self, signals: list[dict]):
        """Accumulate mood signals into the "mood_signals" profile.

        New signals are appended to a ring buffer capped at 200 entries
        (roughly 2-3 days of typical activity).  This buffer is the input
        that ``compute_current_mood()`` reads to produce the MoodState.
        """
        existing = self.ums.get_signal_profile("mood_signals")
        data = existing["data"] if existing else {"recent_signals": []}

        data["recent_signals"].extend(signals)
        # Cap at 200 to bound memory and keep the mood estimate focused on
        # recent behaviour rather than stale historical signals.
        if len(data["recent_signals"]) > 200:
            data["recent_signals"] = data["recent_signals"][-200:]

        self.ums.update_signal_profile("mood_signals", data)
        # Post-write verification: immediately read back to confirm the profile
        # was persisted.  A missing read-back indicates a silent write failure
        # (e.g. WAL corruption, DB locked, JSON serialization error silently
        # caught by update_signal_profile's try/except).  This diagnostic log
        # surfaces the failure so operators can investigate rather than
        # discovering the problem indirectly from missing mood widget data.
        verify = self.ums.get_signal_profile("mood_signals")
        if not verify:
            logger.error(
                "MoodExtractor: mood_signals profile FAILED to persist after write "
                "(data keys=%s, signals=%d)",
                list(data.keys()), len(data.get("recent_signals", [])),
            )
