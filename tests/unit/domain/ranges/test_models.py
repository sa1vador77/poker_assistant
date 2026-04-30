"""Tests for poker_assistant.domain.ranges.models."""

from __future__ import annotations

import pytest

from poker_assistant.domain.cards import Card, Rank, Suit
from poker_assistant.domain.ranges.models import (
    ComboShape,
    HandClass,
    HandRange,
    HoleCombo,
    RangeItem,
    WeightedCombo,
    all_hand_classes,
    dead_cards_from_known_cards,
)


class TestHoleCombo:
    def test_canonical_order_high_first(self) -> None:
        combo = HoleCombo.normalized(
            Card(rank=Rank.KING, suit=Suit.HEARTS),
            Card(rank=Rank.ACE, suit=Suit.SPADES),
        )
        assert combo.first.rank is Rank.ACE
        assert combo.second.rank is Rank.KING

    def test_normalized_is_order_independent(self) -> None:
        ace = Card(rank=Rank.ACE, suit=Suit.SPADES)
        king = Card(rank=Rank.KING, suit=Suit.SPADES)
        assert HoleCombo.normalized(ace, king) == HoleCombo.normalized(king, ace)

    def test_pair_canonical_suit_order_is_stable(self) -> None:
        spades = Card(rank=Rank.ACE, suit=Suit.SPADES)
        hearts = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        assert HoleCombo.normalized(spades, hearts) == HoleCombo.normalized(hearts, spades)

    def test_rejects_identical_cards(self) -> None:
        card = Card(rank=Rank.ACE, suit=Suit.SPADES)
        with pytest.raises(ValueError, match="same card"):
            HoleCombo.normalized(card, card)

    def test_rejects_low_rank_first(self) -> None:
        with pytest.raises(ValueError, match="higher-ranked card first"):
            HoleCombo(
                first=Card(rank=Rank.TWO, suit=Suit.SPADES),
                second=Card(rank=Rank.ACE, suit=Suit.HEARTS),
            )

    def test_rejects_pair_in_wrong_suit_order(self) -> None:
        # Hearts (1) comes after Spades (0); passing (hearts, spades) directly
        # to the constructor breaks canonicalisation.
        with pytest.raises(ValueError, match="canonical order"):
            HoleCombo(
                first=Card(rank=Rank.ACE, suit=Suit.HEARTS),
                second=Card(rank=Rank.ACE, suit=Suit.SPADES),
            )

    def test_shape_classification(self) -> None:
        ace_spades = Card(rank=Rank.ACE, suit=Suit.SPADES)
        ace_hearts = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        king_spades = Card(rank=Rank.KING, suit=Suit.SPADES)
        king_hearts = Card(rank=Rank.KING, suit=Suit.HEARTS)

        assert HoleCombo.normalized(ace_spades, ace_hearts).shape is ComboShape.PAIR
        assert HoleCombo.normalized(ace_spades, king_spades).shape is ComboShape.SUITED
        assert HoleCombo.normalized(ace_spades, king_hearts).shape is ComboShape.OFFSUIT

    def test_conflicts_with(self) -> None:
        combo = HoleCombo.normalized(
            Card(rank=Rank.ACE, suit=Suit.SPADES),
            Card(rank=Rank.KING, suit=Suit.SPADES),
        )
        assert combo.conflicts_with({Card(rank=Rank.ACE, suit=Suit.SPADES)}) is True
        assert combo.conflicts_with({Card(rank=Rank.QUEEN, suit=Suit.SPADES)}) is False

    def test_to_compact_str_renders_both_cards(self) -> None:
        combo = HoleCombo.normalized(
            Card(rank=Rank.ACE, suit=Suit.SPADES),
            Card(rank=Rank.KING, suit=Suit.SPADES),
        )
        assert combo.to_compact_str() == "A♠K♠"


class TestWeightedCombo:
    def test_accepts_zero_weight(self) -> None:
        combo = HoleCombo.normalized(
            Card(rank=Rank.ACE, suit=Suit.SPADES),
            Card(rank=Rank.KING, suit=Suit.SPADES),
        )
        assert WeightedCombo(combo=combo, weight=0.0).weight == 0.0

    def test_accepts_weight_above_one(self) -> None:
        # Weights are not normalised at construction; this is intentional.
        combo = HoleCombo.normalized(
            Card(rank=Rank.ACE, suit=Suit.SPADES),
            Card(rank=Rank.KING, suit=Suit.SPADES),
        )
        assert WeightedCombo(combo=combo, weight=1.5).weight == 1.5

    def test_rejects_negative_weight(self) -> None:
        combo = HoleCombo.normalized(
            Card(rank=Rank.ACE, suit=Suit.SPADES),
            Card(rank=Rank.KING, suit=Suit.SPADES),
        )
        with pytest.raises(ValueError, match="non-negative"):
            WeightedCombo(combo=combo, weight=-0.1)


class TestHandClass:
    def test_pair_factory(self) -> None:
        hc = HandClass.pair(Rank.ACE)
        assert hc.is_pair
        assert hc.high_rank is Rank.ACE
        assert hc.low_rank is Rank.ACE

    def test_suited_factory_normalises_rank_order(self) -> None:
        hc = HandClass.suited(Rank.KING, Rank.ACE)
        assert hc.is_suited
        assert hc.high_rank is Rank.ACE
        assert hc.low_rank is Rank.KING

    def test_offsuit_factory_normalises_rank_order(self) -> None:
        hc = HandClass.offsuit(Rank.TEN, Rank.JACK)
        assert hc.is_offsuit
        assert hc.high_rank is Rank.JACK
        assert hc.low_rank is Rank.TEN

    @pytest.mark.parametrize(
        ("hc", "expected"),
        [
            (HandClass.pair(Rank.ACE), "AA"),
            (HandClass.pair(Rank.TEN), "TT"),
            (HandClass.suited(Rank.ACE, Rank.KING), "AKs"),
            (HandClass.offsuit(Rank.ACE, Rank.KING), "AKo"),
            (HandClass.suited(Rank.TEN, Rank.NINE), "T9s"),
        ],
    )
    def test_to_label(self, hc: HandClass, expected: str) -> None:
        assert hc.to_label() == expected

    def test_pair_generates_six_combos(self) -> None:
        assert len(HandClass.pair(Rank.ACE).generate_combos()) == 6

    def test_suited_generates_four_combos(self) -> None:
        assert len(HandClass.suited(Rank.ACE, Rank.KING).generate_combos()) == 4

    def test_offsuit_generates_twelve_combos(self) -> None:
        assert len(HandClass.offsuit(Rank.ACE, Rank.KING).generate_combos()) == 12

    def test_generated_combos_are_unique(self) -> None:
        for hc in (
            HandClass.pair(Rank.ACE),
            HandClass.suited(Rank.ACE, Rank.KING),
            HandClass.offsuit(Rank.ACE, Rank.KING),
        ):
            combos = hc.generate_combos()
            assert len(set(combos)) == len(combos)

    def test_available_combos_filters_dead_cards(self) -> None:
        hc = HandClass.suited(Rank.ACE, Rank.KING)
        dead = {Card(rank=Rank.ACE, suit=Suit.SPADES)}
        assert len(hc.available_combos(dead)) == 3

    def test_available_combos_with_no_dead_cards(self) -> None:
        hc = HandClass.suited(Rank.ACE, Rank.KING)
        assert len(hc.available_combos(set())) == 4

    def test_rejects_invalid_pair_with_unequal_ranks(self) -> None:
        with pytest.raises(ValueError, match="PAIR HandClass"):
            HandClass(high_rank=Rank.ACE, low_rank=Rank.KING, shape=ComboShape.PAIR)

    def test_rejects_invalid_suited_with_equal_ranks(self) -> None:
        with pytest.raises(ValueError, match="SUITED/OFFSUIT"):
            HandClass(high_rank=Rank.ACE, low_rank=Rank.ACE, shape=ComboShape.SUITED)

    def test_rejects_high_rank_below_low_rank(self) -> None:
        with pytest.raises(ValueError, match="high_rank >= low_rank"):
            HandClass(high_rank=Rank.TWO, low_rank=Rank.ACE, shape=ComboShape.SUITED)

    def test_identity_key_distinguishes_shapes(self) -> None:
        suited = HandClass.suited(Rank.ACE, Rank.KING).identity_key()
        offsuit = HandClass.offsuit(Rank.ACE, Rank.KING).identity_key()
        assert suited != offsuit


class TestRangeItem:
    def test_default_weight_is_one(self) -> None:
        item = RangeItem(hand_class=HandClass.pair(Rank.ACE))
        assert item.weight == 1.0

    @pytest.mark.parametrize("weight", [-0.1, 1.1, 2.0, -1.0])
    def test_rejects_weight_outside_unit_interval(self, weight: float) -> None:
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            RangeItem(hand_class=HandClass.pair(Rank.ACE), weight=weight)


class TestHandRange:
    def test_constructor_rejects_duplicates(self) -> None:
        item = RangeItem(hand_class=HandClass.pair(Rank.ACE))
        with pytest.raises(ValueError, match="Duplicate hand class"):
            HandRange(items=(item, item))

    def test_from_items_keeps_max_weight(self) -> None:
        items = [
            RangeItem(hand_class=HandClass.pair(Rank.ACE), weight=0.3),
            RangeItem(hand_class=HandClass.pair(Rank.ACE), weight=0.8),
            RangeItem(hand_class=HandClass.pair(Rank.ACE), weight=0.5),
        ]
        result = HandRange.from_items(items)
        assert len(result.items) == 1
        assert result.items[0].weight == 0.8

    def test_from_items_returns_sorted_range(self) -> None:
        items = [
            RangeItem(hand_class=HandClass.pair(Rank.TWO)),
            RangeItem(hand_class=HandClass.pair(Rank.ACE)),
            RangeItem(hand_class=HandClass.pair(Rank.KING)),
        ]
        labels = HandRange.from_items(items).labels()
        assert labels == ["AA", "KK", "22"]

    def test_is_empty(self) -> None:
        assert HandRange(items=()).is_empty is True
        assert HandRange.from_hand_classes([HandClass.pair(Rank.ACE)]).is_empty is False

    def test_total_raw_combos(self) -> None:
        rng = HandRange.from_hand_classes(
            [
                HandClass.pair(Rank.ACE),
                HandClass.suited(Rank.ACE, Rank.KING),
                HandClass.offsuit(Rank.ACE, Rank.KING),
            ],
        )
        assert rng.total_raw_combos() == 6 + 4 + 12

    def test_with_uniform_weight(self) -> None:
        rng = HandRange.from_hand_classes(
            [HandClass.pair(Rank.ACE), HandClass.pair(Rank.KING)],
            weight=1.0,
        )
        weighted = rng.with_uniform_weight(0.5)
        assert all(item.weight == 0.5 for item in weighted.items)

    def test_without_conflicts_drops_class_with_no_available_combos(self) -> None:
        rng = HandRange.from_hand_classes([HandClass.suited(Rank.ACE, Rank.KING)])
        # All four AKs combos are A♠K♠, A♥K♥, A♦K♦, A♣K♣ — kill all four aces.
        dead = {Card(rank=Rank.ACE, suit=suit) for suit in Suit}
        assert rng.without_conflicts(dead).is_empty is True

    def test_without_conflicts_keeps_class_with_partial_blockers(self) -> None:
        rng = HandRange.from_hand_classes([HandClass.suited(Rank.ACE, Rank.KING)])
        dead = {Card(rank=Rank.ACE, suit=Suit.SPADES)}
        assert rng.without_conflicts(dead).total_hand_classes() == 1

    def test_expand_to_combos(self) -> None:
        rng = HandRange.from_hand_classes([HandClass.pair(Rank.ACE)], weight=0.7)
        expanded = rng.expand_to_combos()
        assert len(expanded) == 6
        assert all(weight == 0.7 for _, weight in expanded)

    def test_expand_available_weighted_combos_skips_zero_weight_classes(self) -> None:
        rng = HandRange(
            items=(RangeItem(hand_class=HandClass.pair(Rank.ACE), weight=0.0),),
        )
        assert rng.expand_available_weighted_combos(set()) == []

    def test_expand_available_weighted_combos_drops_blocked_combos(self) -> None:
        rng = HandRange.from_hand_classes([HandClass.pair(Rank.ACE)])
        dead = {Card(rank=Rank.ACE, suit=Suit.SPADES)}
        # Pair of aces has 6 combos; removing one ace blocks combos
        # involving that ace (3 of them), leaving 3.
        result = rng.expand_available_weighted_combos(dead)
        assert len(result) == 3

    def test_contains(self) -> None:
        rng = HandRange.from_hand_classes([HandClass.pair(Rank.ACE)])
        assert rng.contains(HandClass.pair(Rank.ACE)) is True
        assert rng.contains(HandClass.pair(Rank.KING)) is False

    def test_union_uses_max_weight(self) -> None:
        a = HandRange.from_items([RangeItem(HandClass.pair(Rank.ACE), 0.4)])
        b = HandRange.from_items([RangeItem(HandClass.pair(Rank.ACE), 0.9)])
        merged = a.union(b)
        assert merged.items[0].weight == 0.9

    def test_union_combines_distinct_classes(self) -> None:
        a = HandRange.from_hand_classes([HandClass.pair(Rank.ACE)])
        b = HandRange.from_hand_classes([HandClass.pair(Rank.KING)])
        assert a.union(b).total_hand_classes() == 2

    def test_union_with_no_others_returns_self(self) -> None:
        rng = HandRange.from_hand_classes([HandClass.pair(Rank.ACE)])
        assert rng.union() == rng

    def test_any_two_has_169_classes(self) -> None:
        assert HandRange.any_two().total_hand_classes() == 169

    def test_sorted_returns_canonical_order(self) -> None:
        items = (
            RangeItem(hand_class=HandClass.pair(Rank.TWO)),
            RangeItem(hand_class=HandClass.pair(Rank.ACE)),
        )
        rng = HandRange(items=items)
        assert rng.sorted().labels() == ["AA", "22"]


class TestAllHandClasses:
    def test_returns_169_classes(self) -> None:
        assert len(all_hand_classes()) == 169

    def test_contains_each_kind_in_correct_count(self) -> None:
        classes = all_hand_classes()
        pairs = [c for c in classes if c.is_pair]
        suited = [c for c in classes if c.is_suited]
        offsuit = [c for c in classes if c.is_offsuit]
        assert len(pairs) == 13
        assert len(suited) == 78
        assert len(offsuit) == 78

    def test_results_are_cached(self) -> None:
        # The function is wrapped in functools.cache, so successive
        # calls must return the same object.
        assert all_hand_classes() is all_hand_classes()

    def test_no_duplicates(self) -> None:
        keys = {hc.identity_key() for hc in all_hand_classes()}
        assert len(keys) == 169


class TestDeadCardsFromKnownCards:
    def test_combines_hero_and_board(self) -> None:
        hero = [Card(rank=Rank.ACE, suit=Suit.SPADES), Card(rank=Rank.KING, suit=Suit.HEARTS)]
        board = [
            Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
            Card(rank=Rank.JACK, suit=Suit.CLUBS),
            Card(rank=Rank.TEN, suit=Suit.SPADES),
        ]
        result = dead_cards_from_known_cards(hero, board)
        assert len(result) == 5
        assert isinstance(result, frozenset)

    def test_empty_board_is_allowed(self) -> None:
        hero = [Card(rank=Rank.ACE, suit=Suit.SPADES), Card(rank=Rank.KING, suit=Suit.HEARTS)]
        assert len(dead_cards_from_known_cards(hero, [])) == 2

    def test_rejects_wrong_hole_card_count(self) -> None:
        with pytest.raises(ValueError, match="exactly 2 hero hole cards"):
            dead_cards_from_known_cards([Card(rank=Rank.ACE, suit=Suit.SPADES)], [])

    def test_rejects_too_many_board_cards(self) -> None:
        hero = [Card(rank=Rank.ACE, suit=Suit.SPADES), Card(rank=Rank.KING, suit=Suit.HEARTS)]
        board = [Card(rank=rank, suit=Suit.DIAMONDS) for rank in list(Rank)[:6]]
        with pytest.raises(ValueError, match="at most 5 board cards"):
            dead_cards_from_known_cards(hero, board)

    def test_rejects_duplicate_cards(self) -> None:
        ace = Card(rank=Rank.ACE, suit=Suit.SPADES)
        with pytest.raises(ValueError, match="duplicate cards"):
            dead_cards_from_known_cards([ace, ace], [])
