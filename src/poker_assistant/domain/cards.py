"""Card domain primitives: suit, rank, card, and parsing utilities.

This module is a leaf of the domain layer: it defines the smallest
building blocks of poker logic and depends on nothing inside the
project. Every other domain module — ranges, equity, hand evaluation,
state — builds on top of these types.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final


class Suit(StrEnum):
    """Card suit.

    The string value is the Unicode glyph used throughout the project,
    which lets ``Suit("♠")`` round-trip with any rendered text without
    extra conversion.

    For a numeric ordering of suits — needed by combo canonicalisation
    and by integer card encoding — see :class:`SuitOrder`.
    """

    SPADES = "♠"
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"


class SuitOrder(IntEnum):
    """Canonical numeric ordering of suits used across the project.

    The order is arbitrary but fixed. Two unrelated systems rely on it:

    * :class:`poker_assistant.domain.ranges.HoleCombo` canonicalises
      pairs of equal-rank cards by this order, so that a given pair
      of cards always produces the same combo regardless of the
      argument order.
    * The native equity backend encodes a card as
      ``card_id = suit_order * 13 + (rank - 2)``. Both backends and
      every test fixture must agree on this mapping.

    Centralising the ordering here gives us a single source of truth.
    Reordering this enum would silently change the meaning of every
    integer card id seen by the native code; it must not be edited
    without coordinated changes elsewhere.
    """

    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3


# Lookup table from a suit to its numeric order. Building it once at
# module import is significantly faster than calling SuitOrder[name] in
# hot paths (combo enumeration, card encoding for the native backend).
_SUIT_ORDER_BY_SUIT: Final[dict[Suit, SuitOrder]] = {
    Suit.SPADES: SuitOrder.SPADES,
    Suit.HEARTS: SuitOrder.HEARTS,
    Suit.DIAMONDS: SuitOrder.DIAMONDS,
    Suit.CLUBS: SuitOrder.CLUBS,
}

_SUIT_BY_SUIT_ORDER: Final[dict[SuitOrder, Suit]] = {
    order: suit for suit, order in _SUIT_ORDER_BY_SUIT.items()
}


def suit_order(suit: Suit) -> SuitOrder:
    """Return the canonical numeric order of ``suit``."""
    return _SUIT_ORDER_BY_SUIT[suit]


def suit_from_order(order: SuitOrder) -> Suit:
    """Return the :class:`Suit` whose canonical order is ``order``."""
    return _SUIT_BY_SUIT_ORDER[order]


class Rank(IntEnum):
    """Card rank with poker-standard ordering (2 lowest, ace highest).

    The integer value matches the rank's strength, so ranks compare
    naturally and can be used as keys in arithmetic (e.g. equity tables,
    hand evaluators).
    """

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass(frozen=True, slots=True)
class Card:
    """A playing card identified by its rank and suit.

    Cards are immutable, hashable, and memory-efficient (``slots=True``),
    which makes them suitable as keys in sets and dicts and as elements
    of large collections (e.g. enumerated ranges, dead-card sets).
    """

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        """Render the card as ``<rank_label><suit_glyph>`` (e.g. ``"A♠"``, ``"T♦"``)."""
        return f"{rank_to_label(self.rank)}{self.suit.value}"


# Read-side label table: accepts both the two-character "10" form (emitted
# by the vision pipeline) and the canonical single-character "T".
_RANK_BY_LABEL: Final[dict[str, Rank]] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "10": Rank.TEN,
    "T": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
}

# Write-side label table: every rank maps to exactly one short label.
# Output is always single-character ("T" for ten), never the two-digit form.
_LABEL_BY_RANK: Final[dict[Rank, str]] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}


def rank_to_label(rank: Rank) -> str:
    """Return the short poker label for a rank (``"A"``, ``"K"``, …, ``"2"``)."""
    return _LABEL_BY_RANK[rank]


def _consume_rank(compact: str, position: int) -> tuple[Rank, int]:
    """Consume one rank token starting at ``position``.

    Accepts either a single-character label (``"2"``…``"9"``, ``"T"``,
    ``"J"``, ``"Q"``, ``"K"``, ``"A"``) or the two-character ``"10"``.

    Returns:
        A tuple of the parsed :class:`Rank` and the index of the first
        character that follows the rank token.

    Raises:
        ValueError: If the substring at ``position`` is not a valid rank.
    """
    if position + 1 < len(compact) and compact[position : position + 2] == "10":
        return Rank.TEN, position + 2

    label = compact[position].upper()
    rank = _RANK_BY_LABEL.get(label)
    if rank is None:
        raise ValueError(
            f"Unknown rank {label!r} at position {position} in {compact!r}",
        )
    return rank, position + 1


def _consume_suit(compact: str, position: int) -> tuple[Suit, int]:
    """Consume one suit glyph starting at ``position``.

    Returns:
        A tuple of the parsed :class:`Suit` and the index of the first
        character that follows the suit glyph.

    Raises:
        ValueError: If ``position`` is past the end of the string or the
            character at ``position`` is not one of the four suit glyphs.
    """
    if position >= len(compact):
        raise ValueError(f"Truncated card token in {compact!r}: missing suit")
    glyph = compact[position]
    try:
        suit = Suit(glyph)
    except ValueError:
        raise ValueError(
            f"Unknown suit {glyph!r} at position {position} in {compact!r}",
        ) from None
    return suit, position + 1


def parse_cards_compact(compact: str) -> list[Card]:
    """Parse a sequence of cards in compact poker notation.

    Each card is a rank followed by a suit glyph; cards are concatenated
    without separators. The ten can be written as either ``"10"`` (the
    form emitted by the vision pipeline) or ``"T"`` (canonical short
    form). Rank letters are case-insensitive.

    Args:
        compact: A non-empty string such as ``"7♠3♠J♣"`` or ``"5♥10♣6♦"``.

    Returns:
        Cards in the same left-to-right order they appear in ``compact``.
        Duplicates are not detected here — see :func:`cards_are_unique`.

    Raises:
        ValueError: If the string is malformed (unknown rank, unknown
            suit, or a trailing rank without a suit).
    """
    cards: list[Card] = []
    position = 0
    while position < len(compact):
        rank, position = _consume_rank(compact, position)
        suit, position = _consume_suit(compact, position)
        cards.append(Card(rank=rank, suit=suit))
    return cards


def parse_card_token(token: str) -> Card:
    """Parse exactly one card from a compact token (e.g. ``"A♠"``, ``"10♦"``).

    Args:
        token: A string that is expected to encode one and only one card.

    Returns:
        The parsed :class:`Card`.

    Raises:
        ValueError: If ``token`` is empty, malformed, or encodes more
            than one card.
    """
    cards = parse_cards_compact(token)
    if len(cards) != 1:
        raise ValueError(f"Expected exactly one card token, got {len(cards)} in {token!r}")
    return cards[0]


def cards_are_unique(cards: Collection[Card]) -> bool:
    """Return ``True`` iff no card appears more than once in ``cards``.

    Args:
        cards: Any collection of cards. ``Collection`` is required (not
            ``Iterable``) because the function reads the size twice and
            must not exhaust a generator.
    """
    return len(set(cards)) == len(cards)
