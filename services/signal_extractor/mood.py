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
