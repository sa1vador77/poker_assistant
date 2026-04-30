"""Range models: hole combos, hand classes, hand ranges and their operations.

Concepts
--------
* :class:`HoleCombo` — a concrete two-card starting hand (e.g. ``A♠K♠``).
* :class:`HandClass` — an abstract starting hand bucket (e.g. ``"AKs"``).
  Texas Hold'em has exactly 169 hand classes: 13 pairs, 78 suited,
  78 offsuit.
* :class:`HandRange` — a weighted set of hand classes, the standard way
  to describe what a player might be holding.
* :class:`WeightedCombo` — a concrete combo with an explicit weight,
  produced when a range is expanded to enumerable combos.

The module is the foundation of the range subsystem: parser and presets
build on these types, equity and range-inference consume them.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from functools import cache
from itertools import combinations
from typing import Final, Self

from poker_assistant.domain.cards import Card, Rank, Suit, rank_to_label


class ComboShape(StrEnum):
    """Shape of a starting hand."""

    PAIR = "pair"
    SUITED = "suited"
    OFFSUIT = "offsuit"


class _SuitOrder(IntEnum):
    """Stable ordering of suits used to canonicalise combos.

    The specific order is arbitrary but fixed: it lets two cards of the
    same rank be ordered deterministically, which is required so that
    :meth:`HoleCombo.normalized` produces a single canonical form for
    each unordered pair of cards.
    """

    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3


_SUIT_ORDER: dict[Suit, _SuitOrder] = {
    Suit.SPADES: _SuitOrder.SPADES,
    Suit.HEARTS: _SuitOrder.HEARTS,
    Suit.DIAMONDS: _SuitOrder.DIAMONDS,
    Suit.CLUBS: _SuitOrder.CLUBS,
}

_SHAPE_PRIORITY: dict[ComboShape, int] = {
    ComboShape.PAIR: 0,
    ComboShape.SUITED: 1,
    ComboShape.OFFSUIT: 2,
}

# Texas Hold'em rules: each player holds exactly two hole cards and
# the board reveals up to five community cards.
HOLE_CARDS_PER_HAND: Final[int] = 2
MAX_BOARD_CARDS: Final[int] = 5


@dataclass(frozen=True, slots=True)
class HoleCombo:
    """A concrete two-card starting hand in canonical order.

    Invariants:
        * The two cards are distinct.
        * The first card has a rank no lower than the second.
        * For pairs (equal ranks) the suit order is fixed by
          :class:`_SuitOrder`, so the same pair of cards always
          produces the same combo.

    Use :meth:`normalized` to construct a combo from any two cards
    without having to think about the order.
    """

    first: Card
    second: Card

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise ValueError("HoleCombo cannot contain the same card twice")
        if int(self.first.rank) < int(self.second.rank):
            raise ValueError("HoleCombo expects the higher-ranked card first")
        if (
            self.first.rank == self.second.rank
            and _SUIT_ORDER[self.first.suit] >= _SUIT_ORDER[self.second.suit]
        ):
            raise ValueError(
                "HoleCombo with equal ranks expects suits in canonical order",
            )

    @property
    def cards(self) -> tuple[Card, Card]:
        """Return both cards as a 2-tuple in canonical order."""
        return (self.first, self.second)

    @property
    def shape(self) -> ComboShape:
        """Return the shape of this combo (pair, suited, or offsuit)."""
        if self.first.rank == self.second.rank:
            return ComboShape.PAIR
        if self.first.suit == self.second.suit:
            return ComboShape.SUITED
        return ComboShape.OFFSUIT

    @property
    def hand_class(self) -> HandClass:
        """Return the abstract hand class that contains this combo."""
        if self.shape == ComboShape.PAIR:
            return HandClass.pair(self.first.rank)
        if self.shape == ComboShape.SUITED:
            return HandClass.suited(self.first.rank, self.second.rank)
        return HandClass.offsuit(self.first.rank, self.second.rank)

    def conflicts_with(self, dead_cards: Collection[Card]) -> bool:
        """Return ``True`` iff either card of the combo is in ``dead_cards``."""
        return self.first in dead_cards or self.second in dead_cards

    def intersects_combo(self, other: HoleCombo) -> bool:
        """Return ``True`` iff this combo shares any card with ``other``."""
        return self.first in other.cards or self.second in other.cards

    def to_compact_str(self) -> str:
        """Return a compact textual key (e.g. ``"A♠K♠"``)."""
        return f"{self.first}{self.second}"

    @classmethod
    def normalized(cls, card_a: Card, card_b: Card) -> Self:
        """Build a :class:`HoleCombo` from any two cards in canonical order.

        Args:
            card_a: First card; the order in which the cards are passed
                does not matter.
            card_b: Second card.

        Returns:
            A combo whose ``first`` field holds the higher-ranked card
            (or, for pairs, the card whose suit comes first in
            :class:`_SuitOrder`).

        Raises:
            ValueError: If both cards are identical.
        """
        if card_a == card_b:
            raise ValueError("HoleCombo cannot contain the same card twice")

        if int(card_a.rank) > int(card_b.rank):
            return cls(first=card_a, second=card_b)
        if int(card_b.rank) > int(card_a.rank):
            return cls(first=card_b, second=card_a)

        if _SUIT_ORDER[card_a.suit] < _SUIT_ORDER[card_b.suit]:
            return cls(first=card_a, second=card_b)
        return cls(first=card_b, second=card_a)


@dataclass(frozen=True, slots=True)
class WeightedCombo:
    """A concrete combo paired with a non-negative weight.

    Weights are not constrained to ``[0, 1]``: a hand-class weight
    multiplied by an external override may produce values outside that
    interval, and equity calculations only require non-negativity.
    Consumers that need a probability must normalise themselves.
    """

    combo: HoleCombo
    weight: float

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("WeightedCombo weight must be non-negative")


@dataclass(frozen=True, slots=True)
class HandClass:
    """An abstract starting-hand bucket.

    A hand class is identified by its high rank, its low rank, and its
    shape. Pairs always have ``high_rank == low_rank``; suited and
    offsuit classes have ``high_rank > low_rank``.
    """

    high_rank: Rank
    low_rank: Rank
    shape: ComboShape

    def __post_init__(self) -> None:
        if int(self.high_rank) < int(self.low_rank):
            raise ValueError("HandClass expects high_rank >= low_rank")
        if self.shape is ComboShape.PAIR and self.high_rank != self.low_rank:
            raise ValueError("PAIR HandClass must have equal ranks")
        if self.shape is not ComboShape.PAIR and self.high_rank == self.low_rank:
            raise ValueError("SUITED/OFFSUIT HandClass must have different ranks")

    @property
    def is_pair(self) -> bool:
        """Return ``True`` iff this is a pocket pair."""
        return self.shape is ComboShape.PAIR

    @property
    def is_suited(self) -> bool:
        """Return ``True`` iff this is a suited (same-suit) class."""
        return self.shape is ComboShape.SUITED

    @property
    def is_offsuit(self) -> bool:
        """Return ``True`` iff this is an offsuit (mixed-suit) class."""
        return self.shape is ComboShape.OFFSUIT

    def to_label(self) -> str:
        """Return the standard poker label (e.g. ``"AA"``, ``"AKs"``, ``"T9o"``)."""
        high = rank_to_label(self.high_rank)
        low = rank_to_label(self.low_rank)
        if self.shape is ComboShape.PAIR:
            return f"{high}{low}"
        if self.shape is ComboShape.SUITED:
            return f"{high}{low}s"
        return f"{high}{low}o"

    def generate_combos(self) -> list[HoleCombo]:
        """Generate every concrete combo belonging to this class.

        A pair has 6 combos, a suited class has 4, an offsuit class
        has 12. The result is freshly built on each call; if you need
        immutable shared instances, cache them at the call site.
        """
        if self.shape is ComboShape.PAIR:
            return self._generate_pair_combos()
        if self.shape is ComboShape.SUITED:
            return self._generate_suited_combos()
        return self._generate_offsuit_combos()

    def available_combos(self, dead_cards: Collection[Card]) -> list[HoleCombo]:
        """Generate combos that do not conflict with any card in ``dead_cards``.

        ``dead_cards`` is materialised into a :class:`frozenset` once,
        so calling this method in a hot loop with the same dead set
        carries no per-card lookup overhead.
        """
        dead_set = frozenset(dead_cards)
        return [combo for combo in self.generate_combos() if not combo.conflicts_with(dead_set)]

    def identity_key(self) -> tuple[int, int, str]:
        """Return a stable identity tuple suitable for dict and set keys."""
        return (int(self.high_rank), int(self.low_rank), self.shape.value)

    def sort_key(self) -> tuple[int, int, int]:
        """Return a stable sort key (high rank desc, low rank desc, shape priority)."""
        return (
            -int(self.high_rank),
            -int(self.low_rank),
            _SHAPE_PRIORITY[self.shape],
        )

    @classmethod
    def pair(cls, rank: Rank) -> Self:
        """Build the pair hand class for ``rank`` (e.g. ``HandClass.pair(Rank.ACE)``)."""
        return cls(high_rank=rank, low_rank=rank, shape=ComboShape.PAIR)

    @classmethod
    def suited(cls, rank_a: Rank, rank_b: Rank) -> Self:
        """Build the suited hand class for two distinct ranks (order does not matter)."""
        high, low = _ranks_desc(rank_a, rank_b)
        return cls(high_rank=high, low_rank=low, shape=ComboShape.SUITED)

    @classmethod
    def offsuit(cls, rank_a: Rank, rank_b: Rank) -> Self:
        """Build the offsuit hand class for two distinct ranks (order does not matter)."""
        high, low = _ranks_desc(rank_a, rank_b)
        return cls(high_rank=high, low_rank=low, shape=ComboShape.OFFSUIT)

    def _generate_pair_combos(self) -> list[HoleCombo]:
        suited_cards = [Card(rank=self.high_rank, suit=suit) for suit in Suit]
        return [HoleCombo.normalized(a, b) for a, b in combinations(suited_cards, 2)]

    def _generate_suited_combos(self) -> list[HoleCombo]:
        return [
            HoleCombo.normalized(
                Card(rank=self.high_rank, suit=suit),
                Card(rank=self.low_rank, suit=suit),
            )
            for suit in Suit
        ]

    def _generate_offsuit_combos(self) -> list[HoleCombo]:
        return [
            HoleCombo.normalized(
                Card(rank=self.high_rank, suit=high_suit),
                Card(rank=self.low_rank, suit=low_suit),
            )
            for high_suit in Suit
            for low_suit in Suit
            if high_suit is not low_suit
        ]


@dataclass(frozen=True, slots=True)
class RangeItem:
    """A single entry in a hand range: a hand class and its weight in ``[0, 1]``."""

    hand_class: HandClass
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError("RangeItem weight must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class HandRange:
    """A weighted set of hand classes.

    Construction
    ------------
    The direct constructor enforces that ``items`` contains no duplicate
    hand classes; passing duplicates raises :class:`ValueError`. To
    build a range from a sequence that may contain duplicates, use
    :meth:`from_items`, which deduplicates by keeping the maximum
    weight per hand class.

    The single source of truth for "merge two ranges" is
    :meth:`union`; ``union`` always uses the max-weight policy.
    """

    items: tuple[RangeItem, ...]

    def __post_init__(self) -> None:
        seen: set[tuple[int, int, str]] = set()
        for item in self.items:
            key = item.hand_class.identity_key()
            if key in seen:
                raise ValueError(
                    f"Duplicate hand class in range: {item.hand_class.to_label()}",
                )
            seen.add(key)

    @property
    def is_empty(self) -> bool:
        """Return ``True`` iff the range contains no hand classes."""
        return len(self.items) == 0

    @property
    def hand_classes(self) -> tuple[HandClass, ...]:
        """Return the hand classes of this range, without weights."""
        return tuple(item.hand_class for item in self.items)

    def labels(self) -> list[str]:
        """Return the standard poker labels of all classes in the range."""
        return [item.hand_class.to_label() for item in self.items]

    def total_hand_classes(self) -> int:
        """Return the number of distinct hand classes in the range."""
        return len(self.items)

    def total_raw_combos(self) -> int:
        """Return the total number of concrete combos, ignoring dead cards."""
        return sum(len(item.hand_class.generate_combos()) for item in self.items)

    def sorted(self) -> Self:
        """Return a copy of the range with hand classes in canonical order."""
        sorted_items = tuple(sorted(self.items, key=lambda item: item.hand_class.sort_key()))
        return type(self)(items=sorted_items)

    def with_uniform_weight(self, weight: float) -> Self:
        """Return a copy of the range where every hand class has ``weight``."""
        return type(self)(
            items=tuple(
                RangeItem(hand_class=item.hand_class, weight=weight) for item in self.items
            ),
        )

    def without_conflicts(self, dead_cards: Collection[Card]) -> Self:
        """Return a copy keeping only classes that have at least one available combo."""
        dead_set = frozenset(dead_cards)
        kept = [item for item in self.items if item.hand_class.available_combos(dead_set)]
        return type(self)(items=tuple(kept))

    def expand_to_combos(self) -> list[tuple[HoleCombo, float]]:
        """Expand the range to all concrete combos, ignoring dead cards.

        Each result element is a ``(combo, hand_class_weight)`` tuple.
        """
        return [
            (combo, item.weight)
            for item in self.items
            for combo in item.hand_class.generate_combos()
        ]

    def expand_available_combos(
        self,
        dead_cards: Collection[Card],
    ) -> list[tuple[HoleCombo, float]]:
        """Expand the range to combos that do not conflict with ``dead_cards``."""
        dead_set = frozenset(dead_cards)
        return [
            (combo, item.weight)
            for item in self.items
            for combo in item.hand_class.available_combos(dead_set)
        ]

    def expand_available_weighted_combos(
        self,
        dead_cards: Collection[Card],
    ) -> list[WeightedCombo]:
        """Expand the range to :class:`WeightedCombo` objects without dead-card conflicts.

        Combos with non-positive weight are dropped from the result.
        """
        dead_set = frozenset(dead_cards)
        result: list[WeightedCombo] = []
        for item in self.items:
            if item.weight <= 0.0:
                continue
            for combo in item.hand_class.available_combos(dead_set):
                result.append(WeightedCombo(combo=combo, weight=item.weight))
        return result

    def contains(self, hand_class: HandClass) -> bool:
        """Return ``True`` iff ``hand_class`` is in the range."""
        target = hand_class.identity_key()
        return any(item.hand_class.identity_key() == target for item in self.items)

    def combo_count_available(self, dead_cards: Collection[Card]) -> int:
        """Return the number of combos remaining after dead-card filtering."""
        return len(self.expand_available_combos(dead_cards))

    def union(self, *others: HandRange) -> Self:
        """Return the union of this range with ``others`` using max-weight per class.

        If a hand class appears in more than one operand, the resulting
        weight is the maximum of its weights across the operands.
        """
        if not others:
            return self
        merged_items = list(self.items)
        for other in others:
            merged_items.extend(other.items)
        return type(self).from_items(merged_items)

    @classmethod
    def from_hand_classes(cls, hand_classes: Sequence[HandClass], weight: float = 1.0) -> Self:
        """Build a range from hand classes with a uniform weight.

        Args:
            hand_classes: The classes to include. Duplicates are
                deduplicated; the resulting range contains each class
                exactly once.
            weight: Weight assigned to every class.
        """
        items = [RangeItem(hand_class=hand_class, weight=weight) for hand_class in hand_classes]
        return cls.from_items(items)

    @classmethod
    def from_items(cls, items: Iterable[RangeItem]) -> Self:
        """Build a range from items, deduplicating by keeping the max weight per class."""
        by_key: dict[tuple[int, int, str], RangeItem] = {}
        for item in items:
            key = item.hand_class.identity_key()
            existing = by_key.get(key)
            if existing is None or item.weight > existing.weight:
                by_key[key] = item
        deduplicated = sorted(by_key.values(), key=lambda item: item.hand_class.sort_key())
        return cls(items=tuple(deduplicated))

    @classmethod
    def any_two(cls) -> Self:
        """Build the universal range containing all 169 hand classes with weight 1.0."""
        return cls.from_hand_classes(all_hand_classes())


@cache
def all_hand_classes() -> tuple[HandClass, ...]:
    """Return all 169 Texas Hold'em hand classes in canonical order.

    The result is computed once per process and shared across calls.
    """
    ranks_desc = sorted(Rank, reverse=True)
    classes: list[HandClass] = [HandClass.pair(rank) for rank in ranks_desc]
    for high, low in combinations(ranks_desc, 2):
        classes.append(HandClass.suited(high, low))
        classes.append(HandClass.offsuit(high, low))
    classes.sort(key=lambda hc: hc.sort_key())
    return tuple(classes)


def dead_cards_from_known_cards(
    hero_hole_cards: Sequence[Card],
    board_cards: Sequence[Card],
) -> frozenset[Card]:
    """Build the dead-card set from the hero's hole cards and the board.

    Args:
        hero_hole_cards: The hero's two hole cards. Must contain
            exactly two cards.
        board_cards: Up to five community cards.

    Returns:
        A :class:`frozenset` containing every card known to be out of
        the deck. Returning ``frozenset`` (rather than ``set``) makes
        the result safely usable as a key and shareable between callers.

    Raises:
        ValueError: If the input does not satisfy the size constraints
            or contains duplicate cards.
    """
    if len(hero_hole_cards) != HOLE_CARDS_PER_HAND:
        raise ValueError(
            f"dead_cards_from_known_cards expects exactly " f"{HOLE_CARDS_PER_HAND} hero hole cards"
        )
    if len(board_cards) > MAX_BOARD_CARDS:
        raise ValueError(
            f"dead_cards_from_known_cards expects at most " f"{MAX_BOARD_CARDS} board cards"
        )

    combined: list[Card] = [*hero_hole_cards, *board_cards]
    if len(set(combined)) != len(combined):
        raise ValueError("dead_cards_from_known_cards received duplicate cards")
    return frozenset(combined)


def _ranks_desc(left: Rank, right: Rank) -> tuple[Rank, Rank]:
    """Return the two ranks ordered from higher to lower."""
    if int(left) >= int(right):
        return left, right
    return right, left
