"""Bidirectional encoding between :class:`Card` and integer card ids.

The native equity backend operates on cards encoded as a single integer
in the range ``[0, 51]``. This module is the single source of truth
for that encoding on the Python side; both the Python equity backend
(when implemented as a reference) and any code that hands cards to
the native extension must go through these functions.

Encoding scheme
---------------
Given a card with rank ``r`` (2..14) and suit ``s`` (a :class:`Suit`):

    card_id = suit_order(s) * 13 + (r - 2)

This matches the decode logic in ``_native_equity.cpp``::

    suit = card_id // 13
    rank = (card_id %% 13) + 2

The mapping is fully bijective: every valid :class:`Card` produces
exactly one id in ``[0, 51]``, and every id maps back to exactly
one :class:`Card`.

Why this lives in ``compute`` and not in ``domain``
---------------------------------------------------
The integer encoding is a contract between the Python and native
backends. It is an implementation detail of the compute layer, not a
fact about poker. Domain code should never see card ids; conversions
happen at the boundary, here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from poker_assistant.domain.cards import (
    Card,
    Rank,
    SuitOrder,
    suit_from_order,
    suit_order,
)

# Number of ranks per suit. Texas Hold'em uses a 52-card deck, four
# suits of 13 cards each. The constant is named explicitly so that
# the encoding/decoding arithmetic reads as poker rather than as
# magic numbers.
RANKS_PER_SUIT: Final[int] = 13

# Total number of distinct cards. The encoded id occupies [0, DECK_SIZE).
DECK_SIZE: Final[int] = 52

# Lowest rank value in the :class:`Rank` enum. Used as an offset so
# that the first rank (TWO) maps to 0 and the highest (ACE) maps to 12.
_RANK_OFFSET: Final[int] = int(Rank.TWO)


def encode_card(card: Card) -> int:
    """Encode a :class:`Card` into an integer id in ``[0, 51]``.

    The function is total: every well-formed :class:`Card` has a valid
    encoding. There is no failure mode; type-checking enforces that
    the input is a real :class:`Card`.
    """
    return suit_order(card.suit) * RANKS_PER_SUIT + (int(card.rank) - _RANK_OFFSET)


def decode_card(card_id: int) -> Card:
    """Decode an integer id in ``[0, 51]`` back into a :class:`Card`.

    Args:
        card_id: An integer in ``[0, 51]``.

    Raises:
        ValueError: If ``card_id`` is outside the valid range. The
            error message states the range explicitly so that callers
            can diagnose off-by-one bugs at the encoding boundary.
    """
    if not 0 <= card_id < DECK_SIZE:
        raise ValueError(
            f"card_id must be in range [0, {DECK_SIZE}), got {card_id}",
        )
    suit = suit_from_order(SuitOrder(card_id // RANKS_PER_SUIT))
    rank = Rank(card_id % RANKS_PER_SUIT + _RANK_OFFSET)
    return Card(rank=rank, suit=suit)


def encode_cards(cards: Iterable[Card]) -> tuple[int, ...]:
    """Encode an iterable of cards into a tuple of integer ids.

    The result is a :class:`tuple` (not a list) so that callers can
    safely use it as a dict key or pass it across the native boundary
    without worrying about accidental mutation.
    """
    return tuple(encode_card(card) for card in cards)


def decode_cards(card_ids: Iterable[int]) -> tuple[Card, ...]:
    """Decode an iterable of integer ids back into a tuple of cards.

    Raises:
        ValueError: If any id is outside ``[0, 51]``; the failing id
            is reported with the same message as :func:`decode_card`.
    """
    return tuple(decode_card(card_id) for card_id in card_ids)
