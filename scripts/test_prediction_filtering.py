#!/usr/bin/env python3
"""
Test why non-reminder predictions are being filtered out.

Traces through the full prediction pipeline including reaction prediction
and confidence gates to identify where predictions are being dropped.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore
from services.prediction_engine.engine import PredictionEngine


async def test_filtering():
    """Test the full prediction generation pipeline."""
    print("=== Prediction Filtering Test ===\n")

    db = DatabaseManager("data")
    ums = UserModelStore(db)
    engine = PredictionEngine(db, ums)

    # Generate predictions using the full pipeline
    print("Running full prediction pipeline...")
    predictions = await engine.generate_predictions({})

    print(f"\n✓ Pipeline returned {len(predictions)} surfaced predictions")

    if predictions:
        for pred in predictions:
            print(f"  - {pred.prediction_type}: {pred.description[:60]}... (conf={pred.confidence:.2f})")
    else:
        print("  No predictions were surfaced")

    # Now test each method individually and trace filtering
    print("\n\n=== Testing Individual Methods ===\n")

    methods = [
        ("follow_up_needs", engine._check_follow_up_needs),
        ("calendar_conflicts", engine._check_calendar_conflicts),
        ("routine_deviations", engine._check_routine_deviations),
    ]

    for name, method in methods:
        print(f"\nTesting {name}...")
        raw_predictions = await method({})
        print(f"  Generated: {len(raw_predictions)} predictions")

        if raw_predictions:
            for pred in raw_predictions:
                print(f"\n  Prediction: {pred.description[:60]}...")
                print(f"    Type: {pred.prediction_type}")
                print(f"    Confidence: {pred.confidence:.2f} ({pred.confidence_gate})")

                # Test accuracy multiplier
                mult = engine._get_accuracy_multiplier(pred.prediction_type)
                adjusted_conf = pred.confidence * mult
                print(f"    Accuracy multiplier: {mult:.2f}x -> {adjusted_conf:.2f}")

                # Test reaction prediction
                reaction = await engine.predict_reaction(pred, {})
                print(f"    Reaction: {reaction.predicted_reaction} (score={reaction.reasoning})")

                # Determine if it would pass filters
                if adjusted_conf < 0.3:
                    print(f"    ✗ FILTERED: Confidence too low ({adjusted_conf:.2f} < 0.3)")
                elif reaction.predicted_reaction not in ("helpful", "neutral"):
                    print(f"    ✗ FILTERED: Reaction {reaction.predicted_reaction}")
                else:
                    print(f"    ✓ WOULD SURFACE")


if __name__ == "__main__":
    asyncio.run(test_filtering())
