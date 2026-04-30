"""Public API of the board analysis package."""

from __future__ import annotations

from poker_assistant.domain.board.texture import (
    MAX_BOARD_SIZE,
    MIN_BOARD_SIZE,
    BoardFacts,
    BoardTexture,
    BoardTextureKind,
    analyze_board_texture,
)

__all__ = [
    "MAX_BOARD_SIZE",
    "MIN_BOARD_SIZE",
    "BoardFacts",
    "BoardTexture",
    "BoardTextureKind",
    "analyze_board_texture",
]
