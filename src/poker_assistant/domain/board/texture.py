"""Board texture analysis: objective facts and heuristic classification.

This module sits in the domain layer next to :mod:`poker_assistant.domain.hand`.
It depends only on :mod:`poker_assistant.domain.cards` and produces two
objects from a 3- to 5-card board:

* :class:`BoardFacts` — strictly objective properties of the cards
  (paired? monotone? max suit count? broadway count? …). These are
  what they are, regardless of strategy.
* :class:`BoardTexture` — a heuristic classification on top of the
  facts: a single :class:`BoardTextureKind` label, plus a few summary
  flags (``is_dry``, ``is_dynamic``) used by the decision layer.

The split matters: facts are stable, heuristics are tunable. Other
domain modules are encouraged to consume :class:`BoardFacts` directly
and only fall back to :class:`BoardTexture` when they want the rolled-up
classification.

Heuristics here (connectedness thresholds, ``is_dry`` formula,
``BoardTextureKind`` priority order) are inherited from earlier
iterations of the project. They are not solver-derived; they will be
revisited when the decision engine accumulates evidence about which
labels actually drive recommendations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Final

from poker_assistant.domain.cards import Card, Rank

# Texas Hold'em board sizes accepted by :func:`analyze_board_texture`.
# Three is the flop, four is the turn, five is the river.
MIN_BOARD_SIZE: Final[int] = 3
MAX_BOARD_SIZE: Final[int] = 5

# A flush requires five cards of the same suit; four of the same suit
# means "four to a flush" — a draw on the board itself.
FLUSH_SUITED_COUNT: Final[int] = 5
FOUR_FLUSH_SUITED_COUNT: Final[int] = 4

# A straight is five consecutive ranks; "four to a straight" means a
# four-card run on the board.
STRAIGHT_RUN_LENGTH: Final[int] = 5
FOUR_TO_STRAIGHT_RUN_LENGTH: Final[int] = 4

# Threshold ranks for grouping cards by strength. The exact boundaries
# match the conventional broadway/high-card splits used in poker
# literature: T-J-Q-K-A are broadway, J-Q-K-A are "high cards".
BROADWAY_MIN_RANK: Final[Rank] = Rank.TEN
HIGH_CARD_MIN_RANK: Final[Rank] = Rank.JACK

# A board is considered "low" when its highest card is no greater than
# the ten. This is a heuristic from the legacy project, kept verbatim.
LOW_BOARD_MAX_RANK: Final[Rank] = Rank.TEN

# Inherited connectedness thresholds. They classify the spread of
# ranks on the board into "connected" and "very connected" buckets.
# These numbers are heuristics, not derived from poker theory.
_THREE_CARD_CONNECTED_SPAN: Final[int] = 4
_FOUR_PLUS_CARD_CONNECTED_SPAN: Final[int] = 5
_VERY_CONNECTED_RUN_LENGTH: Final[int] = 3

# A board is considered "dry" when it offers little draw or
# coordination potential. Inherited heuristic from legacy project.
_DRY_BOARD_MAX_RUN_LENGTH: Final[int] = 2


class BoardTextureKind(StrEnum):
    """Single-label classification of a board's texture.

    The enum is closed: every board falls into exactly one kind. The
    order of evaluation in :func:`_classify_texture` reflects priority
    — earlier checks dominate later ones. For example, a paired
    monotone board is classified as :attr:`PAIRED_MONOTONE`, never
    :attr:`MONOTONE` or :attr:`PAIRED`.
    """

    # Five same-suit + paired beats five same-suit alone, etc.
    PAIRED_FIVE_FLUSH = "paired_five_flush_board"
    FIVE_FLUSH = "five_flush_board"
    PAIRED_FOUR_FLUSH = "paired_four_flush_board"
    FOUR_FLUSH = "four_flush_board"
    PAIRED_MONOTONE = "paired_monotone_board"
    MONOTONE = "monotone_board"
    TRIPS = "trips_board"
    DOUBLE_PAIRED = "double_paired_board"
    PAIRED_CONNECTED = "paired_connected_board"
    PAIRED = "paired_board"
    TWO_TONE_CONNECTED = "two_tone_connected_board"
    TWO_TONE = "two_tone_board"
    VERY_CONNECTED = "very_connected_board"
    CONNECTED = "connected_board"
    DRY_HIGH = "dry_high_board"
    DRY = "dry_board"
    DYNAMIC = "dynamic_board"
    NEUTRAL = "neutral_board"


@dataclass(frozen=True, slots=True)
class BoardFacts:
    """Objective, strategy-independent properties of a board."""

    board_size: int

    # Pairing structure.
    is_paired: bool
    is_double_paired: bool
    is_trips: bool

    # Suit structure.
    is_monotone: bool  # exactly one suit on the board
    is_two_tone: bool  # exactly two suits on the board
    is_rainbow: bool  # every card a different suit
    max_same_suit: int

    # Flush availability on the board itself.
    has_four_flush: bool
    has_five_flush: bool

    # Rank structure.
    highest_rank: Rank
    lowest_rank: Rank
    broadway_count: int
    high_card_count: int

    # Connectedness.
    max_consecutive_run: int
    has_four_to_straight: bool
    has_straight: bool


@dataclass(frozen=True, slots=True)
class BoardTexture:
    """Heuristic classification of a board, built on top of :class:`BoardFacts`."""

    facts: BoardFacts
    kind: BoardTextureKind
    is_connected: bool
    is_very_connected: bool
    is_low_board: bool
    is_dry: bool
    is_dynamic: bool


def analyze_board_texture(board_cards: Sequence[Card]) -> BoardTexture:
    """Analyse a 3- to 5-card board.

    Args:
        board_cards: Three (flop), four (turn), or five (river) cards.

    Returns:
        A :class:`BoardTexture` with both objective facts and the
        derived classification.

    Raises:
        ValueError: If the number of cards is outside the supported
            range or contains duplicates.
    """
    if not MIN_BOARD_SIZE <= len(board_cards) <= MAX_BOARD_SIZE:
        raise ValueError(
            f"analyze_board_texture expects {MIN_BOARD_SIZE} to "
            f"{MAX_BOARD_SIZE} board cards, got {len(board_cards)}",
        )
    if len({*board_cards}) != len(board_cards):
        raise ValueError("analyze_board_texture received duplicate cards")

    facts = _build_facts(board_cards)
    is_connected = _is_connected(board_cards)
    is_very_connected = facts.max_consecutive_run >= _VERY_CONNECTED_RUN_LENGTH
    is_low_board = facts.highest_rank <= LOW_BOARD_MAX_RANK
    is_dry = (
        facts.is_rainbow
        and not facts.is_paired
        and facts.max_consecutive_run <= _DRY_BOARD_MAX_RUN_LENGTH
    )
    is_dynamic = (
        is_very_connected or facts.max_same_suit >= 3 or (is_connected and facts.max_same_suit >= 2)
    )
    kind = _classify_texture(
        facts=facts,
        is_connected=is_connected,
        is_very_connected=is_very_connected,
        is_dry=is_dry,
        is_dynamic=is_dynamic,
    )
    return BoardTexture(
        facts=facts,
        kind=kind,
        is_connected=is_connected,
        is_very_connected=is_very_connected,
        is_low_board=is_low_board,
        is_dry=is_dry,
        is_dynamic=is_dynamic,
    )


def _build_facts(board_cards: Sequence[Card]) -> BoardFacts:
    """Compute objective :class:`BoardFacts` from a validated board."""
    ranks: list[Rank] = [card.rank for card in board_cards]
    rank_counter: Counter[Rank] = Counter(ranks)
    suit_counter: Counter[str] = Counter(card.suit.value for card in board_cards)

    distinct_suits = len(suit_counter)
    max_same_suit = max(suit_counter.values())

    paired_groups = sum(1 for count in rank_counter.values() if count >= 2)
    is_paired = paired_groups >= 1
    is_double_paired = paired_groups >= 2
    is_trips = any(count >= 3 for count in rank_counter.values())

    is_monotone = distinct_suits == 1
    is_rainbow = distinct_suits == len(board_cards)
    # Exactly two suits — distinct from "not monotone" and "not rainbow".
    is_two_tone = distinct_suits == 2

    has_four_flush = max_same_suit >= FOUR_FLUSH_SUITED_COUNT
    has_five_flush = max_same_suit >= FLUSH_SUITED_COUNT

    broadway_count = sum(1 for rank in ranks if rank >= BROADWAY_MIN_RANK)
    high_card_count = sum(1 for rank in ranks if rank >= HIGH_CARD_MIN_RANK)

    max_consecutive_run = _max_consecutive_run_length(ranks)
    has_four_to_straight = max_consecutive_run >= FOUR_TO_STRAIGHT_RUN_LENGTH
    has_straight = max_consecutive_run >= STRAIGHT_RUN_LENGTH

    return BoardFacts(
        board_size=len(board_cards),
        is_paired=is_paired,
        is_double_paired=is_double_paired,
        is_trips=is_trips,
        is_monotone=is_monotone,
        is_two_tone=is_two_tone,
        is_rainbow=is_rainbow,
        max_same_suit=max_same_suit,
        has_four_flush=has_four_flush,
        has_five_flush=has_five_flush,
        highest_rank=max(ranks),
        lowest_rank=min(ranks),
        broadway_count=broadway_count,
        high_card_count=high_card_count,
        max_consecutive_run=max_consecutive_run,
        has_four_to_straight=has_four_to_straight,
        has_straight=has_straight,
    )


def _max_consecutive_run_length(ranks: Sequence[Rank]) -> int:
    """Return the length of the longest run of consecutive ranks.

    The ace is treated as both high (14) and low (1) so that the wheel
    pattern A-2-3-4-5 produces a run of length five.
    """
    rank_ints = {int(rank) for rank in ranks}
    if int(Rank.ACE) in rank_ints:
        rank_ints.add(1)

    sorted_ranks = sorted(rank_ints)
    current_run = 1
    max_run = 1
    for previous, current in pairwise(sorted_ranks):
        if current == previous + 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def _is_connected(board_cards: Sequence[Card]) -> bool:
    """Return ``True`` iff the board is "connected" by inherited heuristic.

    A 3-card board is connected when its rank span is at most
    :data:`_THREE_CARD_CONNECTED_SPAN`; a 4- or 5-card board when its
    span is at most :data:`_FOUR_PLUS_CARD_CONNECTED_SPAN`. These
    thresholds are heuristic and predate this rewrite; they are
    preserved verbatim so that the decision engine sees the same
    classification as before.
    """
    unique_ranks = sorted({int(card.rank) for card in board_cards})
    if len(unique_ranks) < 2:
        return False

    span = unique_ranks[-1] - unique_ranks[0]
    if len(unique_ranks) == 3:
        return span <= _THREE_CARD_CONNECTED_SPAN
    return span <= _FOUR_PLUS_CARD_CONNECTED_SPAN


def _classify_texture(
    *,
    facts: BoardFacts,
    is_connected: bool,
    is_very_connected: bool,
    is_dry: bool,
    is_dynamic: bool,
) -> BoardTextureKind:
    """Map facts and heuristics to a single :class:`BoardTextureKind`.

    Evaluation is ordered: the first matching rule wins. Order matters
    because boards can satisfy multiple criteria (e.g. paired + monotone),
    and we want the most specific label.
    """
    if facts.has_five_flush:
        return (
            BoardTextureKind.PAIRED_FIVE_FLUSH if facts.is_paired else BoardTextureKind.FIVE_FLUSH
        )
    if facts.has_four_flush:
        return (
            BoardTextureKind.PAIRED_FOUR_FLUSH if facts.is_paired else BoardTextureKind.FOUR_FLUSH
        )
    if facts.is_monotone:
        return BoardTextureKind.PAIRED_MONOTONE if facts.is_paired else BoardTextureKind.MONOTONE
    if facts.is_trips:
        return BoardTextureKind.TRIPS
    if facts.is_double_paired:
        return BoardTextureKind.DOUBLE_PAIRED
    if facts.is_paired and is_very_connected:
        return BoardTextureKind.PAIRED_CONNECTED
    if facts.is_paired:
        return BoardTextureKind.PAIRED
    if facts.is_two_tone and is_very_connected:
        return BoardTextureKind.TWO_TONE_CONNECTED
    if facts.is_two_tone:
        return BoardTextureKind.TWO_TONE
    if is_very_connected:
        return BoardTextureKind.VERY_CONNECTED
    if is_connected:
        return BoardTextureKind.CONNECTED
    if is_dry and facts.broadway_count >= 1:
        return BoardTextureKind.DRY_HIGH
    if is_dry:
        return BoardTextureKind.DRY
    if is_dynamic:
        return BoardTextureKind.DYNAMIC
    return BoardTextureKind.NEUTRAL
