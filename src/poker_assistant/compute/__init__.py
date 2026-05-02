"""Public API of the compute layer.

This layer hosts computational backends (currently the equity engines)
and the encodings they share with the native extension. Domain code
consumes ``compute`` only through protocols and the encoding helpers
exposed here.
"""

from __future__ import annotations

from poker_assistant.compute.card_encoding import (
    DECK_SIZE,
    RANKS_PER_SUIT,
    decode_card,
    decode_cards,
    encode_card,
    encode_cards,
)

__all__ = [
    "DECK_SIZE",
    "RANKS_PER_SUIT",
    "decode_card",
    "decode_cards",
    "encode_card",
    "encode_cards",
]
