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

    # Baseline thresholds (calibrated per-user over time)
    DEFAULT_BASELINES = {
        "typing_speed_wpm": 40.0,
        "response_latency_seconds": 300.0,
        "message_length_words": 25.0,
        "exclamation_rate": 0.1,
        "emoji_rate": 0.05,
    }

    def can_process(self, event: dict) -> bool:
        # The mood engine processes a wide range of events
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
        """Collect a mood signal from this event."""
        mood_signals = []
        event_type = event.get("type", "")
        payload = event.get("payload", {})

        # --- Communication signals ---
        if event_type in [EventType.EMAIL_SENT.value, EventType.MESSAGE_SENT.value]:
            text = payload.get("body", "") or payload.get("body_plain", "")
            if text:
                word_count = len(text.split())
                baseline_length = self._get_baseline("message_length_words")

                mood_signals.append({
                    "signal_type": "message_length",
                    "value": word_count,
                    "delta_from_baseline": (word_count - baseline_length) / max(baseline_length, 1),
                    "weight": 0.3,
                    "source": event.get("source", "unknown"),
                })

                # Negative word detection
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
        if event_type == EventType.SLEEP_RECORDED.value:
            hours = payload.get("duration_hours", 7)
            quality = payload.get("quality_score", 0.5)
            mood_signals.append({
                "signal_type": "sleep_quality",
                "value": quality,
                "delta_from_baseline": quality - 0.7,  # Assume 0.7 is baseline good sleep
                "weight": 0.8,
                "source": "health",
            })
            mood_signals.append({
                "signal_type": "sleep_duration",
                "value": hours,
                "delta_from_baseline": (hours - 7.5) / 7.5,
                "weight": 0.5,
                "source": "health",
            })

        # --- Calendar density (stress indicator) ---
        if event_type == EventType.CALENDAR_EVENT_CREATED.value:
            mood_signals.append({
                "signal_type": "calendar_density",
                "value": 1.0,
                "delta_from_baseline": 0.0,
                "weight": 0.2,
                "source": "calendar",
            })

        # --- Spending anomaly (stress signal) ---
        if event_type == EventType.TRANSACTION_NEW.value:
            amount = abs(payload.get("amount", 0))
            if amount > 100:  # Flag larger-than-usual transactions
                mood_signals.append({
                    "signal_type": "spending_spike",
                    "value": amount,
                    "delta_from_baseline": 0.5,  # Simplified
                    "weight": 0.3,
                    "source": "finance",
                })

        if mood_signals:
            self._update_mood_state(mood_signals)

        return [{"type": "mood_signal", "signals": mood_signals}] if mood_signals else []

    def compute_current_mood(self) -> MoodState:
        """
        Compute the current mood state from recent signals.
        Called periodically (every 15 minutes) and on-demand.
        """
        existing = self.ums.get_signal_profile("mood_signals")
        if not existing:
            return MoodState()

        recent_signals = existing["data"].get("recent_signals", [])
        if not recent_signals:
            return MoodState()

        # Weighted composite across dimensions
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
            social_battery=0.5,  # Requires more signals to estimate
            cognitive_load=min(1.0, len(stress_signals) * 0.15),
            emotional_valence=1.0 - self._weighted_average(valence_signals, default=0.3),
            confidence=min(1.0, len(recent_signals) * 0.1),
            contributing_signals=[
                MoodSignal(**s) for s in recent_signals[-10:]
            ],
        )

        # Determine trend from mood history
        mood.trend = self._compute_trend()

        return mood

    def _weighted_average(self, signals: list[dict], default: float = 0.5) -> float:
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
        """Compare recent mood to historical baseline."""
        # Simplified; in production this would look at mood_history table
        return "stable"

    def _get_baseline(self, metric: str) -> float:
        """Get the user's personal baseline for a metric."""
        profile = self.ums.get_signal_profile("baselines")
        if profile and metric in profile["data"]:
            return profile["data"][metric]
        return self.DEFAULT_BASELINES.get(metric, 0.5)

    def _update_mood_state(self, signals: list[dict]):
        """Accumulate mood signals for periodic mood computation."""
        existing = self.ums.get_signal_profile("mood_signals")
        data = existing["data"] if existing else {"recent_signals": []}

        data["recent_signals"].extend(signals)
        # Keep last 200 signals (roughly 2-3 days of activity)
        if len(data["recent_signals"]) > 200:
            data["recent_signals"] = data["recent_signals"][-200:]

        self.ums.update_signal_profile("mood_signals", data)
