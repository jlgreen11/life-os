"""
Tests for actionable prediction cards in the dashboard template.

Validates that:
1. loadPredictions() renders Draft Reply button for reminder predictions with contact_email
2. loadPredictions() renders View Calendar button for conflict predictions
3. Done button is always rendered for surfaced predictions
4. suggested_action hint text is rendered when present
5. predictionActedOn() calls the correct feedback endpoint
6. draftPredictionReply() calls /api/draft with contact_id and incoming_message
7. Inline draft placeholder is injected for reminder predictions
"""

import pytest


def _get_template_js() -> str:
    """Extract the full JavaScript block from web/template.py for assertion checks."""
    from web.template import HTML_TEMPLATE

    return HTML_TEMPLATE


def test_load_predictions_shows_draft_reply_for_reminder():
    """loadPredictions should render a Draft Reply button for reminder predictions."""
    js = _get_template_js()
    # The draft reply button condition: reminder type AND contactEmail present
    assert "prediction_type === 'reminder' && contactEmail" in js
    assert "Draft Reply" in js
    assert "draftPredictionReply" in js


def test_load_predictions_shows_view_calendar_for_conflict():
    """loadPredictions should render a View Calendar button for conflict predictions."""
    js = _get_template_js()
    assert "prediction_type === 'conflict'" in js
    assert "View Calendar" in js
    # switchTopic call may have escaped quotes in the Python string literal
    assert "switchTopic" in js and "calendar" in js


def test_load_predictions_shows_done_button():
    """loadPredictions should render a Done button for every prediction."""
    js = _get_template_js()
    assert "predictionActedOn" in js
    assert "Done" in js
    assert "btn-success" in js


def test_load_predictions_shows_suggested_action_hint():
    """loadPredictions should render the suggested_action as a hint line."""
    js = _get_template_js()
    # Arrow prefix on the hint line
    assert "suggested_action" in js
    assert "\\u2192" in js  # → right-arrow prefix for hint


def test_prediction_acted_on_function_exists():
    """predictionActedOn() function must be defined in the template."""
    js = _get_template_js()
    assert "function predictionActedOn(id)" in js


def test_prediction_acted_on_calls_correct_endpoint():
    """predictionActedOn() must POST to /api/predictions/{id}/feedback with acted_on."""
    js = _get_template_js()
    assert "was_accurate=true&user_response=acted_on" in js
    assert "/api/predictions/" in js


def test_draft_prediction_reply_function_exists():
    """draftPredictionReply() function must be defined in the template."""
    js = _get_template_js()
    assert "function draftPredictionReply(predId, contactEmail, context)" in js


def test_draft_prediction_reply_calls_draft_endpoint():
    """draftPredictionReply() must POST to /api/draft with contact_id."""
    js = _get_template_js()
    assert "'/api/draft'" in js
    assert "contact_id: contactEmail" in js


def test_draft_prediction_reply_passes_incoming_message():
    """draftPredictionReply() must send incoming_message to give AI context."""
    js = _get_template_js()
    assert "incoming_message: context" in js


def test_inline_draft_placeholder_injected():
    """For reminder predictions with a contact, a draft placeholder div must be injected."""
    js = _get_template_js()
    # The placeholder ID pattern used for inline drafts
    assert "pred-draft-" in js
    assert "getElementById('pred-draft-' + predId)" in js


def test_signals_object_extracted_from_prediction():
    """supporting_signals must be extracted to get contact_email for each prediction."""
    js = _get_template_js()
    assert "supporting_signals" in js
    assert "contact_email" in js


def test_btn_success_css_defined():
    """The btn-success CSS class must be defined for the Done button styling."""
    js = _get_template_js()
    assert "btn-success" in js
    # Green accent color
    assert "accent-green" in js


def test_prediction_feedback_inaccurate_still_present():
    """Inaccurate feedback button must still appear (not removed, just renamed)."""
    js = _get_template_js()
    assert "Inaccurate" in js
    assert "btn-danger" in js


def test_not_me_feedback_abbreviated():
    """Not About Me is abbreviated to Not Me to save space in the prediction card."""
    js = _get_template_js()
    assert "Not Me" in js
    assert "not_relevant" in js


def test_prediction_card_fade_animation():
    """predictionActedOn() should apply a CSS transition before hiding the card."""
    js = _get_template_js()
    # Fade animation: opacity transition + delay before display:none
    assert "style.transition" in js
    assert "style.opacity" in js
    assert "setTimeout" in js


@pytest.mark.asyncio
async def test_prediction_feedback_endpoint_accepts_acted_on(db):
    """The /api/predictions/{id}/feedback endpoint must accept user_response=acted_on."""
    # Verify the resolve_prediction method in UserModelStore handles acted_on
    from storage.database import UserModelStore

    ums = UserModelStore(db)

    # Store a minimal prediction record
    import json
    import uuid
    from datetime import datetime, timezone

    pred_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Unreplied email from test@example.com",
                0.7,
                "DEFAULT",
                1,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    # Resolve with acted_on — this is the code path triggered by the Done button
    ums.resolve_prediction(
        prediction_id=pred_id,
        was_accurate=True,
        user_response="acted_on",
    )

    # Verify the prediction was updated correctly
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None, "Prediction must exist after resolve"
    assert row["was_accurate"] == 1, "was_accurate should be 1 (True)"
    assert row["user_response"] == "acted_on"
    assert row["resolved_at"] is not None, "resolved_at must be set after resolving"
