"""
Smoke tests for the PredictionEngine.

Verifies that the engine can be instantiated with real database
fixtures and that its basic attributes are set correctly.
"""

from services.prediction_engine.engine import PredictionEngine


def test_prediction_engine_initializes(db, user_model_store):
    """PredictionEngine can be created with a real DB and UserModelStore."""
    engine = PredictionEngine(db=db, ums=user_model_store)

    assert engine.db is db
    assert engine.ums is user_model_store
