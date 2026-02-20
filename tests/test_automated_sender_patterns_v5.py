"""
Tests for expanded automated-sender patterns in BehavioralAccuracyTracker and
PredictionEngine (iteration 179).

Problem (pre-fix):
    After the marketing filter expansion in PRs #183–#194, 67 opportunity predictions
    for automated senders remained unresolved because _is_automated_sender() in the
    tracker and _is_marketing_or_noreply() in the prediction engine did not cover
    several common automated local-part prefixes observed in production data:

        - top@raymore.com         (15 predictions — Raymore & Flanigan promo)
        - stay@hotelvandivort.com  (5 predictions — hotel booking CRM)
        - irrigation@ryanlawn.com  (4 predictions — landscaping service)
        - mail@cardsagainsthumanity.com / mail@directpay.irs.gov  (2 predictions)
        - alumni@mst.edu / kummercollege@mst.edu  (3 predictions — educational bulk)
        - msftpc@microsoft.com    (3 predictions — Microsoft PC-fleet)
        - claims@treasurer.mo.gov (2 predictions — government automated)

    The tracker's _is_automated_sender() also lacked the marketing service provider
    (ESP) domain suffix checks present in the prediction engine, meaning predictions
    for addresses like user@updates.sendgrid.net could avoid the fast-path.

Fix (iteration 179):
    Added 7 new bulk local-part prefixes to both methods:
        mail, alumni, top, stay, msftpc, irrigation, claims

    Added full ESP domain suffix checks to BehavioralAccuracyTracker._is_automated_sender()
    to match the prediction engine's marketing_service_patterns list.

    Result: 34 additional opportunity predictions immediately resolved as inaccurate
    (automated senders) instead of waiting 7 days; 17 contacts filtered from future
    relationship_maintenance predictions.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker
from services.prediction_engine.engine import PredictionEngine


# ============================================================================
# Fixtures
# ============================================================================


def _make_tracker() -> BehavioralAccuracyTracker:
    """Return a BehavioralAccuracyTracker with a mock DB (no I/O needed for unit tests)."""
    tracker = BehavioralAccuracyTracker.__new__(BehavioralAccuracyTracker)
    tracker.db = MagicMock()
    return tracker


# ============================================================================
# New bulk local-part patterns — _is_automated_sender (tracker)
# ============================================================================


def test_tracker_mail_prefix_is_automated():
    """mail@ prefix is always automated — generic mail-sending address."""
    tracker = _make_tracker()
    automated = [
        "mail@directpay.irs.gov",
        "mail@cardsagainsthumanity.com",
        "mail@ifttt.com",
        "mail@cdkeys.com",
        "mail@eg.vrbo.com",
        "mail@invite.beforethepremiere.com",
        "mail@mail2.creditkarma.com",
    ]
    for addr in automated:
        assert tracker._is_automated_sender(addr), (
            f"Expected mail@ address to be detected as automated: {addr!r}"
        )


def test_tracker_alumni_prefix_is_automated():
    """alumni@ prefix indicates mass university/college mailing list, never a person."""
    tracker = _make_tracker()
    assert tracker._is_automated_sender("alumni@mst.edu"), (
        "Expected alumni@mst.edu to be automated"
    )
    assert tracker._is_automated_sender("alumni@harvard.edu")
    assert tracker._is_automated_sender("alumni@stanford.edu")


def test_tracker_top_prefix_is_automated():
    """top@ is a promotional/campaign prefix (e.g. top@raymore.com furniture promo)."""
    tracker = _make_tracker()
    assert tracker._is_automated_sender("top@raymore.com"), (
        "Expected top@raymore.com (Raymore & Flanigan promo) to be automated"
    )
    assert tracker._is_automated_sender("top@deals.example.com")


def test_tracker_stay_prefix_is_automated():
    """stay@ is a hotel/hospitality booking CRM automated sender."""
    tracker = _make_tracker()
    assert tracker._is_automated_sender("stay@hotelvandivort.com"), (
        "Expected stay@hotelvandivort.com (hotel CRM) to be automated"
    )


def test_tracker_msftpc_prefix_is_automated():
    """msftpc@ is Microsoft's PC-fleet management automated sender."""
    tracker = _make_tracker()
    assert tracker._is_automated_sender("msftpc@microsoft.com"), (
        "Expected msftpc@microsoft.com to be automated"
    )


def test_tracker_irrigation_prefix_is_automated():
    """irrigation@ is a landscaping/lawn-service automated sender."""
    tracker = _make_tracker()
    assert tracker._is_automated_sender("irrigation@ryanlawn.com"), (
        "Expected irrigation@ryanlawn.com (Ryan Lawn & Tree) to be automated"
    )


def test_tracker_claims_prefix_is_automated():
    """claims@ is a government/insurance automated sender, never a personal address."""
    tracker = _make_tracker()
    assert tracker._is_automated_sender("claims@treasurer.mo.gov"), (
        "Expected claims@treasurer.mo.gov (MO Treasurer) to be automated"
    )
    assert tracker._is_automated_sender("claims@insurance.com")


# ============================================================================
# New bulk local-part patterns — _is_marketing_or_noreply (prediction engine)
# ============================================================================


def test_prediction_engine_mail_prefix_is_marketing():
    """PredictionEngine._is_marketing_or_noreply catches mail@ prefix."""
    automated = [
        "mail@directpay.irs.gov",
        "mail@cardsagainsthumanity.com",
        "mail@ifttt.com",
    ]
    for addr in automated:
        assert PredictionEngine._is_marketing_or_noreply(addr, {}), (
            f"Expected {addr!r} to be filtered by prediction engine"
        )


def test_prediction_engine_new_prefixes_are_marketing():
    """All 7 new local-part prefixes are caught by the prediction engine."""
    automated = [
        "top@raymore.com",
        "stay@hotelvandivort.com",
        "alumni@mst.edu",
        "msftpc@microsoft.com",
        "irrigation@ryanlawn.com",
        "claims@treasurer.mo.gov",
    ]
    for addr in automated:
        assert PredictionEngine._is_marketing_or_noreply(addr, {}), (
            f"Expected {addr!r} to be filtered by prediction engine"
        )


# ============================================================================
# ESP domain suffix checks — _is_automated_sender (tracker)
# ============================================================================


def test_tracker_esp_sendgrid_domain_is_automated():
    """Addresses on sendgrid.net subdomains are always ESP-delivered, hence automated."""
    tracker = _make_tracker()
    assert tracker._is_automated_sender("bounce@em1234.sendgrid.net")
    assert tracker._is_automated_sender("support@updates.sendgrid.net")


def test_tracker_esp_klaviyo_domain_is_automated():
    """Addresses on klaviyo.com subdomains (e-commerce marketing) are automated."""
    tracker = _make_tracker()
    assert tracker._is_automated_sender("promo@company.klaviyo.com")


def test_tracker_esp_mailchimp_domain_is_automated():
    """Addresses on mailchimp.com subdomains are automated."""
    tracker = _make_tracker()
    assert tracker._is_automated_sender("hello@brand.mailchimp.com")


def test_tracker_esp_amazonses_domain_is_automated():
    """Addresses on amazonses.com are AWS SES — transactional/bulk by definition."""
    tracker = _make_tracker()
    assert tracker._is_automated_sender("no_reply@amazonses.com")


def test_tracker_esp_patterns_dont_block_personal_email():
    """ESP domain checks must not falsely flag personal email providers."""
    tracker = _make_tracker()
    personal = [
        "alice@gmail.com",
        "bob@hotmail.com",
        "carol@protonmail.com",
        "dave@outlook.com",
        "eve@yahoo.com",
    ]
    for addr in personal:
        assert not tracker._is_automated_sender(addr), (
            f"Expected personal email {addr!r} NOT to be flagged as automated"
        )


# ============================================================================
# No false positives for legitimate human contacts
# ============================================================================


def test_new_patterns_no_false_positives_for_human_contacts():
    """New patterns must not flag legitimate human email addresses as automated.

    Tests contacts from production data that were correctly identified as human
    and should continue to reach the 7-day wait window.
    """
    tracker = _make_tracker()
    human_contacts = [
        "tulsi724@gmail.com",
        "shelbyhiter@gmail.com",
        "193arm24@gmail.com",
        "bdc.umr@gmail.com",
        "sockscondo681@gmail.com",
        "natasha_mayekar2131@elcamino.edu",
        "gerardo_sanchez2096@elcamino.edu",
        "victoria.galindo@harvestright.com",
        "ernest.lim@harvestright.com",
        "Pattie.HERROD@capstonelogistics.com",
        "ashley.schuch@hilton.com",
        "jason.stull@hyatt.com",
        "dino@ventiscafe.com",
        "dorrian@vestaboard.com",
        # Named Microsoft employees (not the PC-fleet system)
        "john.smith@microsoft.com",
        "alice.jones@hilton.com",
    ]
    for addr in human_contacts:
        assert not tracker._is_automated_sender(addr), (
            f"New patterns incorrectly flagged human contact as automated: {addr!r}"
        )


def test_prediction_engine_no_false_positives_for_human_contacts():
    """PredictionEngine._is_marketing_or_noreply must not block legitimate human contacts."""
    human_contacts = [
        "tulsi724@gmail.com",
        "shelbyhiter@gmail.com",
        "victoria.galolia@harvestright.com",
        "alice.jones@hilton.com",
        "john.smith@microsoft.com",
    ]
    for addr in human_contacts:
        assert not PredictionEngine._is_marketing_or_noreply(addr, {}), (
            f"Prediction engine incorrectly filtered human contact: {addr!r}"
        )


# ============================================================================
# Both filters are consistent for new patterns
# ============================================================================


def test_tracker_and_prediction_engine_agree_on_new_patterns():
    """Tracker _is_automated_sender and engine _is_marketing_or_noreply must agree.

    Both methods must return the same result for the 7 new patterns added in
    iteration 179. Divergence would cause predictions to be generated (engine
    doesn't filter) but never resolved (tracker doesn't fast-path).
    """
    tracker = _make_tracker()

    test_cases = [
        "mail@directpay.irs.gov",
        "alumni@mst.edu",
        "top@raymore.com",
        "stay@hotelvandivort.com",
        "msftpc@microsoft.com",
        "irrigation@ryanlawn.com",
        "claims@treasurer.mo.gov",
        "mail@cardsagainsthumanity.com",
        # Human contacts — both must return False
        "tulsi724@gmail.com",
        "victoria.galindo@harvestright.com",
        "alice@company.com",
    ]
    for addr in test_cases:
        tracker_result = tracker._is_automated_sender(addr)
        engine_result = PredictionEngine._is_marketing_or_noreply(addr, {})
        assert tracker_result == engine_result, (
            f"Filter disagreement for {addr!r}: "
            f"tracker={tracker_result}, engine={engine_result}"
        )
