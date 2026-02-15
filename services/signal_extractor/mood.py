"""
Life OS — Mood Inference Engine

Infers the user's current emotional/cognitive state from a composite
of all available signals. Never asks "how are you feeling?"

Key principle: The AI never tells the user their mood. It silently
adjusts its behavior (tone, timing, proactivity) based on the inference.
"""

from __future__ import annotations

from models.core import EventType
from models.user_model import MoodSignal, MoodState
from services.signal_extractor.base import BaseExtractor


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

    def can_process(self, event: dict) -> bool:
        # The mood engine casts a wide net: it ingests communication events
        # (language sentiment), health data (sleep quality/duration), calendar
        # density (busyness/stress), financial transactions (spending spikes),
        # and location changes.  Each event type contributes a different facet
        # to the multi-dimensional mood estimate.
        return event.get("type") in [
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_SENT.value,
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

        # --- Communication signals ---
        # Message length relative to baseline: unusually short messages may
        # indicate terse/stressed communication; long ones may indicate
        # engagement or venting.
        if event_type in [EventType.EMAIL_SENT.value, EventType.MESSAGE_SENT.value]:
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

                # Scan for negative-valence words.  The density of negative
                # language is a strong (weight=0.6) mood indicator because
                # it is more deliberate than structural features.
                negative_words = sum(1 for w in text.lower().split() if w in [
                    "frustrated", "annoyed", "tired", "exhausted", "stressed",
                    "worried", "overwhelmed", "confused", "angry", "upset",
                    "sorry", "unfortunately", "problem", "issue", "difficult",
                ])
                if negative_words > 0:
                    mood_signals.append({
                        "signal_type": "negative_language",
                        "value": negative_words / max(word_count, 1),
                        "delta_from_baseline": negative_words / max(word_count, 1),
                        "weight": 0.6,
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
          - social_battery:     placeholder — would need meeting/interaction data
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
        # "negative_language" feeds both stress and valence).
        energy_signals = [s for s in recent_signals if s["signal_type"] in [
            "sleep_quality", "sleep_duration", "activity_level"
        ]]
        stress_signals = [s for s in recent_signals if s["signal_type"] in [
            "calendar_density", "negative_language", "spending_spike", "response_latency"
        ]]
        valence_signals = [s for s in recent_signals if s["signal_type"] in [
            "negative_language", "emoji_usage", "message_length"
        ]]

        mood = MoodState(
            energy_level=self._weighted_average(energy_signals, default=0.5),
            stress_level=self._weighted_average(stress_signals, default=0.3),
            social_battery=0.5,  # Requires meeting/social-interaction signals to estimate.
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
        """Compute a weight-normalised average of delta-from-baseline values.

        Each signal's absolute delta is multiplied by its declared weight,
        summed, and divided by the total weight.  The result is clamped to
        [0.0, 1.0].  When no signals are available, ``default`` is returned
        so the mood dimensions degrade gracefully to neutral values.
        """
        if not signals:
            return default
        total_weight = sum(abs(s.get("weight", 1.0)) for s in signals)
        if total_weight == 0:
            return default
        weighted_sum = sum(
            abs(s.get("delta_from_baseline", 0)) * s.get("weight", 1.0)
            for s in signals
        )
        return min(1.0, max(0.0, weighted_sum / total_weight))

    def _compute_trend(self) -> str:
        """Compare recent mood to historical baseline.

        In a full implementation this would query the mood_history table,
        compare the latest N snapshots against the running average, and
        return "improving", "declining", or "stable".  Currently simplified
        to always return "stable".
        """
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
