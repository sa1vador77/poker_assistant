"""Public API of the hand evaluation package."""

from __future__ import annotations

from poker_assistant.domain.hand.evaluator import (
    MAX_HAND_SIZE,
    MIN_HAND_SIZE,
    STRAIGHT_LENGTH,
    EvaluatedHand,
    HandCategory,
    HandRank,
    evaluate_best_hand_detailed,
    evaluate_five_card_hand,
)

__all__ = [
    "MAX_HAND_SIZE",
    "MIN_HAND_SIZE",
    "STRAIGHT_LENGTH",
    "EvaluatedHand",
    "HandCategory",
    "HandRank",
    "evaluate_best_hand_detailed",
    "evaluate_five_card_hand",
]
