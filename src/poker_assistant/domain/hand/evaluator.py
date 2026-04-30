"""Poker hand evaluation: best 5-card combination out of 5 to 7 cards.

This module is a pure-algorithmic primitive of the domain layer. It
maps any 5 to 7 cards to a comparable :class:`HandRank` and to the
specific 5 cards that realised that rank. It does not know about the
hero, the board, dead cards, or the rules of betting.

The two entry points are:

* :func:`evaluate_five_card_hand` — fast path for an exact 5-card hand;
* :func:`evaluate_best_hand_detailed` — chooses the best 5-card combo
  out of a 5-7 card pool and returns both the rank and the cards.

Two :class:`HandRank` values can be compared directly with ``<``,
``==``, ``>``: a strictly higher rank means a strictly stronger hand,
ties compare equal.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import Final

from poker_assistant.domain.cards import Card, Rank

# Texas Hold'em hand sizes accepted by :func:`evaluate_best_hand_detailed`.
# Five is the size of a complete poker hand; seven is the maximum a
# Hold'em player ever has to evaluate (two hole cards plus a full board).
MIN_HAND_SIZE: Final[int] = 5
MAX_HAND_SIZE: Final[int] = 7

# A straight is exactly five consecutive ranks. The wheel (A-2-3-4-5)
# is the special case where the ace plays low.
STRAIGHT_LENGTH: Final[int] = 5


class HandCategory(IntEnum):
    """Category of a 5-card poker hand, ordered by strength.

    A higher integer value means a strictly stronger category. The
    royal flush is not a separate category: it is the special case of
    a :attr:`STRAIGHT_FLUSH` whose top card is an ace.
    """

    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9


@dataclass(frozen=True, slots=True, order=True)
class HandRank:
    """A comparable representation of a hand's strength.

    Two ranks are compared lexicographically by ``(category,
    tiebreakers)``. The dataclass-generated ordering matches poker
    semantics exactly: a stronger category beats a weaker one, and
    within the same category the longer-prefix tiebreaker tuple wins.

    The tiebreaker tuple holds rank values (``int(Rank.X)``) ordered
    from most to least significant. For example, two pair of aces and
    kings with a queen kicker is stored as ``(14, 13, 12)``; a flush
    is stored with all five card ranks descending.
    """

    category: HandCategory
    tiebreakers: tuple[int, ...]

    def __str__(self) -> str:
        """Render the rank as ``CATEGORY (t1, t2, ...)`` for diagnostics."""
        return f"{self.category.name} {self.tiebreakers}"


@dataclass(frozen=True, slots=True)
class EvaluatedHand:
    """The detailed result of evaluating a hand.

    Holds both the comparable :class:`HandRank` and the specific five
    cards that realised it. Consumers that only need the rank can read
    :attr:`rank`; consumers that want to display or audit the hand
    (e.g. UIs, logs) read :attr:`best_five_cards`.
    """

    rank: HandRank
    best_five_cards: tuple[Card, ...]


def evaluate_five_card_hand(cards: Sequence[Card]) -> HandRank:
    """Evaluate exactly five cards into a :class:`HandRank`.

    Args:
        cards: A sequence of exactly five cards.

    Returns:
        The hand's rank.

    Raises:
        ValueError: If ``cards`` does not contain exactly five cards.
    """
    if len(cards) != STRAIGHT_LENGTH:
        raise ValueError(
            f"evaluate_five_card_hand expects exactly {STRAIGHT_LENGTH} cards, "
            f"got {len(cards)}",
        )

    ranks_desc: list[int] = sorted((int(card.rank) for card in cards), reverse=True)
    rank_counter = Counter(ranks_desc)

    flush = _all_same_suit(cards)
    straight_high = _straight_high_card(ranks_desc)

    if flush and straight_high is not None:
        return HandRank(category=HandCategory.STRAIGHT_FLUSH, tiebreakers=(straight_high,))

    # ``grouped`` orders rank/count pairs so that the most significant
    # group comes first: more copies first, ties broken by higher rank.
    # Examples:
    #   AAA KK -> [(14, 3), (13, 2)]
    #   QQ 99 A -> [(12, 2), (9, 2), (14, 1)]
    grouped = sorted(
        rank_counter.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    counts = sorted(rank_counter.values(), reverse=True)

    if counts == [4, 1]:
        quads_rank, kicker_rank = grouped[0][0], grouped[1][0]
        return HandRank(
            category=HandCategory.FOUR_OF_A_KIND,
            tiebreakers=(quads_rank, kicker_rank),
        )

    if counts == [3, 2]:
        trips_rank, pair_rank = grouped[0][0], grouped[1][0]
        return HandRank(
            category=HandCategory.FULL_HOUSE,
            tiebreakers=(trips_rank, pair_rank),
        )

    if flush:
        return HandRank(category=HandCategory.FLUSH, tiebreakers=tuple(ranks_desc))

    if straight_high is not None:
        return HandRank(category=HandCategory.STRAIGHT, tiebreakers=(straight_high,))

    if counts == [3, 1, 1]:
        trips_rank = grouped[0][0]
        kickers = sorted(
            (rank for rank, count in grouped if count == 1),
            reverse=True,
        )
        return HandRank(
            category=HandCategory.THREE_OF_A_KIND,
            tiebreakers=(trips_rank, *kickers),
        )

    if counts == [2, 2, 1]:
        pair_ranks = sorted(
            (rank for rank, count in grouped if count == 2),
            reverse=True,
        )
        kicker = next(rank for rank, count in grouped if count == 1)
        return HandRank(
            category=HandCategory.TWO_PAIR,
            tiebreakers=(pair_ranks[0], pair_ranks[1], kicker),
        )

    if counts == [2, 1, 1, 1]:
        pair_rank = next(rank for rank, count in grouped if count == 2)
        kickers = sorted(
            (rank for rank, count in grouped if count == 1),
            reverse=True,
        )
        return HandRank(
            category=HandCategory.ONE_PAIR,
            tiebreakers=(pair_rank, *kickers),
        )

    return HandRank(category=HandCategory.HIGH_CARD, tiebreakers=tuple(ranks_desc))


def evaluate_best_hand_detailed(cards: Sequence[Card]) -> EvaluatedHand:
    """Find the strongest 5-card hand inside a 5- to 7-card pool.

    Args:
        cards: Between :data:`MIN_HAND_SIZE` and :data:`MAX_HAND_SIZE`
            cards, with no duplicates checked here (caller's contract).

    Returns:
        The best :class:`EvaluatedHand` reachable from any 5-card
        subset of ``cards``. Among ties on rank, the first 5-card
        combination encountered by :func:`itertools.combinations` is
        returned; the choice does not affect the rank itself.

    Raises:
        ValueError: If ``cards`` has fewer than :data:`MIN_HAND_SIZE`
            or more than :data:`MAX_HAND_SIZE` cards.
    """
    if not MIN_HAND_SIZE <= len(cards) <= MAX_HAND_SIZE:
        raise ValueError(
            f"evaluate_best_hand_detailed expects {MIN_HAND_SIZE} to "
            f"{MAX_HAND_SIZE} cards, got {len(cards)}",
        )

    if len(cards) == MIN_HAND_SIZE:
        rank = evaluate_five_card_hand(cards)
        return EvaluatedHand(rank=rank, best_five_cards=tuple(cards))

    # Initialise with the first combination so that ``best_*`` are
    # bound from the start; this avoids ``Optional`` types and keeps
    # the loop body symmetric.
    combo_iter = combinations(cards, MIN_HAND_SIZE)
    first_combo = next(combo_iter)
    best_rank = evaluate_five_card_hand(first_combo)
    best_combo: tuple[Card, ...] = first_combo

    for combo in combo_iter:
        rank = evaluate_five_card_hand(combo)
        if rank > best_rank:
            best_rank = rank
            best_combo = combo

    return EvaluatedHand(rank=best_rank, best_five_cards=best_combo)


def _all_same_suit(cards: Sequence[Card]) -> bool:
    """Return ``True`` iff all cards share the same suit.

    The caller of :func:`evaluate_five_card_hand` has already validated
    the length, so this helper does not re-check it.
    """
    first_suit = cards[0].suit
    return all(card.suit is first_suit for card in cards)


def _straight_high_card(ranks_desc: Sequence[int]) -> int | None:
    """Return the high-card rank of a straight, or ``None`` if there is none.

    The wheel (A-2-3-4-5) is recognised: it returns ``5``, since the
    ace plays low and five is the top card of the straight.

    Args:
        ranks_desc: Five rank values in descending order. This helper
            assumes — but does not validate — exactly five entries; it
            is private to :func:`evaluate_five_card_hand`.

    Examples:
        ``[14, 13, 12, 11, 10] → 14`` (ace-high straight)
        ``[9, 8, 7, 6, 5] → 9``
        ``[14, 5, 4, 3, 2] → 5`` (the wheel)
        ``[14, 13, 11, 10, 9] → None`` (gap, no straight)
    """
    unique_desc = sorted(set(ranks_desc), reverse=True)
    if len(unique_desc) != STRAIGHT_LENGTH:
        return None

    if unique_desc[0] - unique_desc[-1] == STRAIGHT_LENGTH - 1:
        return unique_desc[0]

    if unique_desc == [int(Rank.ACE), 5, 4, 3, 2]:
        return 5

    return None
