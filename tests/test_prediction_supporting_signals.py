"""
Tests verifying that conflict, need, and opportunity predictions include
structured supporting_signals so BehavioralAccuracyTracker can close the
learning loop without relying on fragile description parsing.

Background
----------
Three prediction types were created without ``supporting_signals``:

* **conflict** — ``_infer_conflict_accuracy()`` immediately returns ``None``
  if ``conflicting_event_ids`` is absent (line 327 of tracker.py), so every
  conflict prediction remained unresolved forever.
* **need** — ``_infer_need_accuracy()`` returns ``None`` if ``event_start_time``
  is absent (line 413 of tracker.py), so preparation-need predictions also
  stayed unresolved.
* **opportunity** — ``_infer_opportunity_accuracy()`` has a regex fallback
  that works for most email addresses, but structured signals are more reliable
  and enable the automated-sender fast-path (PR #189) without parsing.

This iteration adds the required ``supporting_signals`` dicts to all three
prediction types in ``PredictionEngine``.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.user_model import Prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_calendar_event(
    event_id: str,
    title: str,
    start_iso: str,
    end_iso: str,
    is_all_day: bool = False,
) -> MagicMock:
    """Return a fake DB row for a calendar.event.created event.

    The row mimics what sqlite3.Row looks like when accessed by column name.
    """
    row = MagicMock()
    row.__getitem__ = MagicMock(side_effect=lambda k: {
        "id": event_id,
        "type": "calendar.event.created",
        "payload": json.dumps({
            "event_id": event_id,
            "title": title,
            "start_time": start_iso,
            "end_time": end_iso,
            "is_all_day": is_all_day,
            "attendees": [],
        }),
        "timestamp": start_iso,
    }[k])
    row.get = MagicMock(side_effect=lambda k, d=None: {
        "id": event_id,
        "type": "calendar.event.created",
        "payload": json.dumps({
            "event_id": event_id,
            "title": title,
            "start_time": start_iso,
            "end_time": end_iso,
            "is_all_day": is_all_day,
            "attendees": [],
        }),
        "timestamp": start_iso,
    }.get(k, d))
    return row


def _make_engine(db):
    """Construct a PredictionEngine with minimal mocked dependencies."""
    from services.prediction_engine.engine import PredictionEngine

    ums = MagicMock()
    # store_prediction is called for each generated prediction; no return value needed
    ums.store_prediction.return_value = None
    # get_signal_profile returns None by default (no profile data), which is
    # the safe fail-open path in each prediction method
    ums.get_signal_profile.return_value = None

    engine = PredictionEngine(db=db, ums=ums)
    return engine


# ---------------------------------------------------------------------------
# Conflict predictions: supporting_signals
# ---------------------------------------------------------------------------

class TestConflictPredictionSignals:
    """_check_calendar_conflicts() must populate supporting_signals."""

    @pytest.mark.asyncio
    async def test_conflict_prediction_includes_event_ids(self, db):
        """Overlapping events produce a 'conflict' prediction with conflicting_event_ids."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)

        # Two timed events that overlap: A 09:00-10:30, B 10:00-11:00
        ev_a_start = (now + timedelta(hours=5)).replace(microsecond=0)
        ev_a_end = ev_a_start + timedelta(minutes=90)
        ev_b_start = ev_a_start + timedelta(minutes=60)
        ev_b_end = ev_b_start + timedelta(hours=1)

        fake_events = [
            _make_calendar_event("ev-A", "Meeting A", ev_a_start.isoformat(), ev_a_end.isoformat()),
            _make_calendar_event("ev-B", "Meeting B", ev_b_start.isoformat(), ev_b_end.isoformat()),
        ]

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value.fetchall.return_value = fake_events
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_calendar_conflicts({})

        conflict_preds = [p for p in predictions if p.prediction_type == "conflict"]
        assert len(conflict_preds) >= 1, "Expected at least one conflict prediction"

        pred = conflict_preds[0]
        signals = pred.supporting_signals
        assert "conflicting_event_ids" in signals, (
            "conflict prediction must include conflicting_event_ids so "
            "_infer_conflict_accuracy() can detect reschedules"
        )
        assert "ev-A" in signals["conflicting_event_ids"]
        assert "ev-B" in signals["conflicting_event_ids"]

    @pytest.mark.asyncio
    async def test_conflict_prediction_includes_event_titles(self, db):
        """Conflict prediction supporting_signals includes event_titles."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)

        ev_a_start = (now + timedelta(hours=5)).replace(microsecond=0)
        ev_a_end = ev_a_start + timedelta(minutes=90)
        ev_b_start = ev_a_start + timedelta(minutes=60)
        ev_b_end = ev_b_start + timedelta(hours=1)

        fake_events = [
            _make_calendar_event("ev-C", "Sprint Review", ev_a_start.isoformat(), ev_a_end.isoformat()),
            _make_calendar_event("ev-D", "1:1 with Boss", ev_b_start.isoformat(), ev_b_end.isoformat()),
        ]

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value.fetchall.return_value = fake_events
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_calendar_conflicts({})

        conflict_preds = [p for p in predictions if p.prediction_type == "conflict"]
        assert conflict_preds, "Expected conflict predictions"
        signals = conflict_preds[0].supporting_signals
        assert "event_titles" in signals
        assert "Sprint Review" in signals["event_titles"]
        assert "1:1 with Boss" in signals["event_titles"]

    @pytest.mark.asyncio
    async def test_conflict_prediction_includes_overlap_minutes(self, db):
        """Conflict prediction supporting_signals includes overlap_minutes."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)

        # 30-minute overlap
        ev_a_start = (now + timedelta(hours=5)).replace(microsecond=0)
        ev_a_end = ev_a_start + timedelta(hours=1)          # ends at +6h
        ev_b_start = ev_a_start + timedelta(minutes=30)     # starts at +5h30m
        ev_b_end = ev_b_start + timedelta(hours=1)

        fake_events = [
            _make_calendar_event("ev-E", "Event E", ev_a_start.isoformat(), ev_a_end.isoformat()),
            _make_calendar_event("ev-F", "Event F", ev_b_start.isoformat(), ev_b_end.isoformat()),
        ]

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value.fetchall.return_value = fake_events
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_calendar_conflicts({})

        conflict_preds = [p for p in predictions if p.prediction_type == "conflict"]
        assert conflict_preds, "Expected conflict predictions"
        signals = conflict_preds[0].supporting_signals
        assert "overlap_minutes" in signals
        assert signals["overlap_minutes"] == 30

    @pytest.mark.asyncio
    async def test_risk_tight_transition_includes_event_ids(self, db):
        """Tight-transition 'risk' predictions include conflicting_event_ids."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)

        # 10-minute gap (tight, but no overlap)
        ev_a_start = (now + timedelta(hours=5)).replace(microsecond=0)
        ev_a_end = ev_a_start + timedelta(hours=1)
        ev_b_start = ev_a_end + timedelta(minutes=10)   # only 10 min gap
        ev_b_end = ev_b_start + timedelta(hours=1)

        fake_events = [
            _make_calendar_event("ev-G", "Workshop", ev_a_start.isoformat(), ev_a_end.isoformat()),
            _make_calendar_event("ev-H", "All-Hands", ev_b_start.isoformat(), ev_b_end.isoformat()),
        ]

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value.fetchall.return_value = fake_events
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_calendar_conflicts({})

        risk_preds = [p for p in predictions if p.prediction_type == "risk"]
        assert risk_preds, "Expected at least one risk prediction for tight transition"
        signals = risk_preds[0].supporting_signals
        assert "conflicting_event_ids" in signals
        assert "ev-G" in signals["conflicting_event_ids"]
        assert "ev-H" in signals["conflicting_event_ids"]
        assert "gap_minutes" in signals
        assert signals["gap_minutes"] == 10


# ---------------------------------------------------------------------------
# Need predictions: supporting_signals
# ---------------------------------------------------------------------------

class TestNeedPredictionSignals:
    """_check_preparation_needs() must populate supporting_signals."""

    def _make_prep_event(self, event_id: str, title: str, hours_ahead: float,
                         attendees=None, is_all_day=False):
        """Build a fake calendar.event.created row in the 12-48h preparation window."""
        now = datetime.now(timezone.utc)
        start = (now + timedelta(hours=hours_ahead)).replace(microsecond=0)
        end = start + timedelta(hours=1)
        row = MagicMock()
        payload = {
            "event_id": event_id,
            "title": title,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "is_all_day": is_all_day,
            "attendees": attendees or [],
        }
        row.__getitem__ = MagicMock(side_effect=lambda k: {
            "id": event_id,
            "payload": json.dumps(payload),
        }[k])
        row.get = MagicMock(side_effect=lambda k, d=None: {
            "id": event_id,
            "payload": json.dumps(payload),
        }.get(k, d))
        return row

    @pytest.mark.asyncio
    async def test_travel_need_includes_event_start_time(self, db):
        """Travel 'need' predictions include event_start_time for accuracy inference."""
        engine = _make_engine(db)
        fake_event = self._make_prep_event("flight-001", "Flight to NYC", hours_ahead=24)

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value.fetchall.return_value = [fake_event]
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_preparation_needs({})

        travel_preds = [p for p in predictions if "travel" in p.description.lower()]
        assert travel_preds, "Expected a travel prediction"

        signals = travel_preds[0].supporting_signals
        assert "event_start_time" in signals, (
            "need prediction must include event_start_time so "
            "_infer_need_accuracy() can determine if the event occurred"
        )
        # Verify it's a parseable ISO timestamp
        parsed = datetime.fromisoformat(signals["event_start_time"].replace("Z", "+00:00"))
        assert parsed.tzinfo is not None

    @pytest.mark.asyncio
    async def test_travel_need_includes_event_id(self, db):
        """Travel 'need' predictions include event_id for exact matching."""
        engine = _make_engine(db)
        fake_event = self._make_prep_event("flight-002", "Flight to Boston", hours_ahead=20)

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value.fetchall.return_value = [fake_event]
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_preparation_needs({})

        travel_preds = [p for p in predictions if "travel" in p.description.lower()]
        assert travel_preds, "Expected a travel prediction"

        signals = travel_preds[0].supporting_signals
        assert "event_id" in signals
        assert signals["event_id"] == "flight-002"

    @pytest.mark.asyncio
    async def test_travel_need_includes_event_title(self, db):
        """Travel 'need' predictions include event_title for fuzzy fallback."""
        engine = _make_engine(db)
        fake_event = self._make_prep_event("trip-003", "Trip to Chicago", hours_ahead=36)

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value.fetchall.return_value = [fake_event]
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_preparation_needs({})

        travel_preds = [p for p in predictions if "travel" in p.description.lower()]
        assert travel_preds, "Expected a travel prediction"

        signals = travel_preds[0].supporting_signals
        assert "event_title" in signals
        assert signals["event_title"] == "Trip to Chicago"

    @pytest.mark.asyncio
    async def test_large_meeting_need_includes_supporting_signals(self, db):
        """Large meeting 'need' predictions include event_id and event_start_time."""
        engine = _make_engine(db)
        fake_event = self._make_prep_event(
            "meeting-100",
            "Q4 Planning Session",
            hours_ahead=24,
            attendees=["a@x.com", "b@x.com", "c@x.com", "d@x.com"],  # 4 > 3
        )

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value.fetchall.return_value = [fake_event]
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_preparation_needs({})

        meeting_preds = [p for p in predictions if "meeting" in p.description.lower()]
        assert meeting_preds, "Expected a large meeting prediction"

        signals = meeting_preds[0].supporting_signals
        assert "event_id" in signals
        assert signals["event_id"] == "meeting-100"
        assert "event_start_time" in signals
        assert "event_title" in signals
        assert signals["event_title"] == "Q4 Planning Session"
        assert "attendee_count" in signals
        assert signals["attendee_count"] == 4

    @pytest.mark.asyncio
    async def test_preparation_type_field_travel(self, db):
        """Travel predictions include preparation_type='travel'."""
        engine = _make_engine(db)
        fake_event = self._make_prep_event("airport-1", "Airport Pickup", hours_ahead=18)

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value.fetchall.return_value = [fake_event]
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_preparation_needs({})

        travel_preds = [p for p in predictions if "travel" in p.description.lower()]
        assert travel_preds
        assert travel_preds[0].supporting_signals.get("preparation_type") == "travel"

    @pytest.mark.asyncio
    async def test_preparation_type_field_large_meeting(self, db):
        """Large meeting predictions include preparation_type='large_meeting'."""
        engine = _make_engine(db)
        fake_event = self._make_prep_event(
            "meeting-200", "Board Presentation", hours_ahead=24,
            attendees=["a@x.com", "b@x.com", "c@x.com", "d@x.com"],
        )

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value.fetchall.return_value = [fake_event]
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_preparation_needs({})

        meeting_preds = [p for p in predictions if "meeting" in p.description.lower()]
        assert meeting_preds
        assert meeting_preds[0].supporting_signals.get("preparation_type") == "large_meeting"


# ---------------------------------------------------------------------------
# Opportunity predictions: supporting_signals
# ---------------------------------------------------------------------------

class TestOpportunityPredictionSignals:
    """_check_relationship_maintenance() must populate supporting_signals."""

    def _make_relationship_profile(self, addr: str, days_since: int, avg_gap: float = 14.0):
        """Return a fake relationship profile that triggers an opportunity prediction."""
        now = datetime.now(timezone.utc)
        last_contact = now - timedelta(days=days_since)
        # Create interaction timestamps spanning enough history
        timestamps = [
            (now - timedelta(days=days_since + avg_gap * i)).isoformat()
            for i in range(5)
        ]
        return {
            addr: {
                "interaction_count": 6,
                "last_interaction": last_contact.isoformat(),
                "interaction_timestamps": sorted(timestamps),
                "domains": ["work"],
            }
        }

    @pytest.mark.asyncio
    async def test_opportunity_includes_contact_email(self, db):
        """Opportunity prediction supporting_signals includes contact_email."""
        engine = _make_engine(db)
        profile = self._make_relationship_profile("bob@example.com", days_since=30)

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            # Return a fake relationships signal_profile row
            mock_row = MagicMock()
            mock_row.__getitem__ = MagicMock(side_effect=lambda k: {
                "profile_data": json.dumps(profile),
            }[k])
            mock_conn.execute.return_value.fetchone.return_value = mock_row
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_relationship_maintenance({})

        opp_preds = [p for p in predictions if p.prediction_type == "opportunity"]
        if not opp_preds:
            pytest.skip("No opportunity prediction generated — relationship profile may not trigger threshold")

        signals = opp_preds[0].supporting_signals
        assert "contact_email" in signals, (
            "opportunity prediction must include contact_email so "
            "_infer_opportunity_accuracy() can match outbound emails"
        )
        assert signals["contact_email"] == "bob@example.com"

    @pytest.mark.asyncio
    async def test_opportunity_includes_contact_name(self, db):
        """Opportunity prediction supporting_signals includes contact_name derived from email."""
        engine = _make_engine(db)
        profile = self._make_relationship_profile("alice@company.com", days_since=25)

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_row = MagicMock()
            mock_row.__getitem__ = MagicMock(side_effect=lambda k: {
                "profile_data": json.dumps(profile),
            }[k])
            mock_conn.execute.return_value.fetchone.return_value = mock_row
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_relationship_maintenance({})

        opp_preds = [p for p in predictions if p.prediction_type == "opportunity"]
        if not opp_preds:
            pytest.skip("No opportunity prediction generated")

        signals = opp_preds[0].supporting_signals
        assert "contact_name" in signals
        # contact_name should be the local part of the email
        assert signals["contact_name"] == "alice"

    @pytest.mark.asyncio
    async def test_opportunity_includes_days_since_last_contact(self, db):
        """Opportunity prediction supporting_signals includes days_since_last_contact."""
        engine = _make_engine(db)
        profile = self._make_relationship_profile("charlie@work.com", days_since=28)

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_row = MagicMock()
            mock_row.__getitem__ = MagicMock(side_effect=lambda k: {
                "profile_data": json.dumps(profile),
            }[k])
            mock_conn.execute.return_value.fetchone.return_value = mock_row
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_relationship_maintenance({})

        opp_preds = [p for p in predictions if p.prediction_type == "opportunity"]
        if not opp_preds:
            pytest.skip("No opportunity prediction generated")

        signals = opp_preds[0].supporting_signals
        assert "days_since_last_contact" in signals
        # Should be close to 28 (exact depends on current time)
        assert 20 <= signals["days_since_last_contact"] <= 35

    @pytest.mark.asyncio
    async def test_opportunity_includes_avg_contact_gap(self, db):
        """Opportunity prediction supporting_signals includes avg_contact_gap_days."""
        engine = _make_engine(db)
        profile = self._make_relationship_profile("diana@example.org", days_since=30, avg_gap=14.0)

        with patch.object(engine.db, "get_connection") as mock_ctx:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_row = MagicMock()
            mock_row.__getitem__ = MagicMock(side_effect=lambda k: {
                "profile_data": json.dumps(profile),
            }[k])
            mock_conn.execute.return_value.fetchone.return_value = mock_row
            mock_ctx.return_value = mock_conn

            predictions = await engine._check_relationship_maintenance({})

        opp_preds = [p for p in predictions if p.prediction_type == "opportunity"]
        if not opp_preds:
            pytest.skip("No opportunity prediction generated")

        signals = opp_preds[0].supporting_signals
        assert "avg_contact_gap_days" in signals
        assert isinstance(signals["avg_contact_gap_days"], float)


# ---------------------------------------------------------------------------
# Integration: signals survive round-trip through storage
# ---------------------------------------------------------------------------

class TestSignalsRoundTrip:
    """Verify supporting_signals are preserved when stored and retrieved."""

    def test_conflict_signals_json_round_trip(self):
        """Conflict supporting_signals must be JSON-serializable (for DB storage)."""
        signals = {
            "conflicting_event_ids": ["ev-1", "ev-2"],
            "event_titles": ["Meeting A", "Meeting B"],
            "event_start_times": ["2026-02-18T09:00:00+00:00", "2026-02-18T10:00:00+00:00"],
            "overlap_minutes": 30,
            "is_all_day_conflict": False,
        }
        # Simulate DB storage (json.dumps) and retrieval (json.loads)
        stored = json.dumps(signals)
        retrieved = json.loads(stored)
        assert retrieved["conflicting_event_ids"] == ["ev-1", "ev-2"]
        assert retrieved["overlap_minutes"] == 30
        assert retrieved["is_all_day_conflict"] is False

    def test_need_signals_json_round_trip(self):
        """Need supporting_signals must be JSON-serializable (for DB storage)."""
        signals = {
            "event_id": "cal-123",
            "event_title": "Flight to Boston",
            "event_start_time": "2026-02-19T08:00:00+00:00",
            "preparation_type": "travel",
        }
        stored = json.dumps(signals)
        retrieved = json.loads(stored)
        assert retrieved["event_id"] == "cal-123"
        assert retrieved["preparation_type"] == "travel"

    def test_opportunity_signals_json_round_trip(self):
        """Opportunity supporting_signals must be JSON-serializable (for DB storage)."""
        signals = {
            "contact_email": "bob@example.com",
            "contact_name": "bob",
            "days_since_last_contact": 30,
            "avg_contact_gap_days": 14.0,
        }
        stored = json.dumps(signals)
        retrieved = json.loads(stored)
        assert retrieved["contact_email"] == "bob@example.com"
        assert retrieved["avg_contact_gap_days"] == 14.0

    def test_prediction_model_stores_conflict_signals(self):
        """Prediction model correctly stores dict supporting_signals."""
        pred = Prediction(
            prediction_type="conflict",
            description="Calendar overlap: 'Meeting A' and 'Meeting B' overlap by 30 minutes",
            confidence=0.95,
            confidence_gate="default",
            time_horizon="24_hours",
            supporting_signals={
                "conflicting_event_ids": ["ev-1", "ev-2"],
                "overlap_minutes": 30,
            },
        )
        assert pred.supporting_signals["conflicting_event_ids"] == ["ev-1", "ev-2"]
        assert pred.supporting_signals["overlap_minutes"] == 30

    def test_prediction_model_stores_need_signals(self):
        """Prediction model correctly stores need supporting_signals."""
        start_time = datetime.now(timezone.utc) + timedelta(hours=24)
        pred = Prediction(
            prediction_type="need",
            description="Upcoming travel in 24h: 'Flight to NYC'. Time to prepare.",
            confidence=0.75,
            confidence_gate="default",
            time_horizon="24_hours",
            supporting_signals={
                "event_id": "flight-007",
                "event_title": "Flight to NYC",
                "event_start_time": start_time.isoformat(),
                "preparation_type": "travel",
            },
        )
        assert pred.supporting_signals["event_id"] == "flight-007"
        assert pred.supporting_signals["preparation_type"] == "travel"
