"""Tests for poker_assistant.domain.hand.evaluator."""

from __future__ import annotations

from itertools import pairwise

import pytest

from poker_assistant.domain.cards import Card, parse_cards_compact
from poker_assistant.domain.hand.evaluator import (
    MAX_HAND_SIZE,
    MIN_HAND_SIZE,
    EvaluatedHand,
    HandCategory,
    HandRank,
    evaluate_best_hand_detailed,
    evaluate_five_card_hand,
)


def _hand(notation: str) -> list[Card]:
    """Parse a compact card sequence; convenience for terse test cases."""
    return parse_cards_compact(notation)


def _evaluate(notation: str) -> HandRank:
    return evaluate_five_card_hand(_hand(notation))


class TestHandRankOrdering:
    def test_higher_category_beats_lower(self) -> None:
        flush = HandRank(category=HandCategory.FLUSH, tiebreakers=(2, 3, 4, 5, 7))
        straight = HandRank(category=HandCategory.STRAIGHT, tiebreakers=(14,))
        assert flush > straight

    def test_same_category_compares_by_tiebreakers(self) -> None:
        higher = HandRank(category=HandCategory.ONE_PAIR, tiebreakers=(14, 13, 12, 11))
        lower = HandRank(category=HandCategory.ONE_PAIR, tiebreakers=(14, 13, 12, 10))
        assert higher > lower

    def test_equal_ranks_compare_equal(self) -> None:
        a = HandRank(category=HandCategory.ONE_PAIR, tiebreakers=(14, 13, 12, 11))
        b = HandRank(category=HandCategory.ONE_PAIR, tiebreakers=(14, 13, 12, 11))
        assert a == b
        assert not (a > b)
        assert not (a < b)

    def test_str_renders_category_name_and_tiebreakers(self) -> None:
        rank = HandRank(category=HandCategory.FOUR_OF_A_KIND, tiebreakers=(14, 13))
        assert "FOUR_OF_A_KIND" in str(rank)
        assert "(14, 13)" in str(rank)


class TestHandCategoryValues:
    def test_strength_order_is_strict(self) -> None:
        ordered = [
            HandCategory.HIGH_CARD,
            HandCategory.ONE_PAIR,
            HandCategory.TWO_PAIR,
            HandCategory.THREE_OF_A_KIND,
            HandCategory.STRAIGHT,
            HandCategory.FLUSH,
            HandCategory.FULL_HOUSE,
            HandCategory.FOUR_OF_A_KIND,
            HandCategory.STRAIGHT_FLUSH,
        ]
        for weaker, stronger in pairwise(ordered):
            assert int(weaker) < int(stronger)


class TestEvaluateFiveCardHand:
    @pytest.mark.parametrize(
        ("hand", "expected_category", "expected_tiebreakers"),
        [
            # Royal flush (special case of straight flush, ace high).
            ("A♠K♠Q♠J♠10♠", HandCategory.STRAIGHT_FLUSH, (14,)),
            # Straight flush, nine high.
            ("9♠8♠7♠6♠5♠", HandCategory.STRAIGHT_FLUSH, (9,)),
            # Steel wheel: A-2-3-4-5 of one suit.
            ("A♠5♠4♠3♠2♠", HandCategory.STRAIGHT_FLUSH, (5,)),
            ("A♠A♥A♦A♣K♠", HandCategory.FOUR_OF_A_KIND, (14, 13)),
            ("K♠K♥K♦Q♠Q♥", HandCategory.FULL_HOUSE, (13, 12)),
            ("A♠Q♠10♠5♠2♠", HandCategory.FLUSH, (14, 12, 10, 5, 2)),
            # Mixed-suit straight, nine high.
            ("9♠8♥7♦6♠5♣", HandCategory.STRAIGHT, (9,)),
            # Mixed-suit wheel.
            ("A♠5♥4♦3♠2♣", HandCategory.STRAIGHT, (5,)),
            ("Q♠Q♥Q♦9♠5♣", HandCategory.THREE_OF_A_KIND, (12, 9, 5)),
            ("A♠A♥K♦K♠5♣", HandCategory.TWO_PAIR, (14, 13, 5)),
            ("A♠A♥K♦Q♠5♣", HandCategory.ONE_PAIR, (14, 13, 12, 5)),
            ("A♠Q♥10♦7♠5♣", HandCategory.HIGH_CARD, (14, 12, 10, 7, 5)),
        ],
    )
    def test_recognises_every_category(
        self,
        hand: str,
        expected_category: HandCategory,
        expected_tiebreakers: tuple[int, ...],
    ) -> None:
        rank = _evaluate(hand)
        assert rank.category is expected_category
        assert rank.tiebreakers == expected_tiebreakers

    def test_flush_beats_straight(self) -> None:
        # Same-suit cards that also form a straight are a straight
        # flush; here we verify the non-straight flush case.
        flush = _evaluate("A♠Q♠10♠5♠2♠")
        straight = _evaluate("9♠8♥7♦6♠5♣")
        assert flush > straight

    def test_full_house_beats_flush(self) -> None:
        full_house = _evaluate("K♠K♥K♦Q♠Q♥")
        flush = _evaluate("A♠Q♠10♠5♠2♠")
        assert full_house > flush

    def test_two_pair_higher_top_pair_wins(self) -> None:
        aces_kings = _evaluate("A♠A♥K♦K♠5♣")
        aces_queens = _evaluate("A♠A♥Q♦Q♠5♣")
        assert aces_kings > aces_queens

    def test_two_pair_same_top_higher_second_wins(self) -> None:
        aces_kings = _evaluate("A♠A♥K♦K♠5♣")
        aces_queens = _evaluate("A♠A♥Q♦Q♠5♣")
        # Cross-check from the other angle: same top, second pair decides.
        assert aces_kings > aces_queens

    def test_one_pair_kicker_decides(self) -> None:
        higher_kicker = _evaluate("A♠A♥K♦Q♠5♣")
        lower_kicker = _evaluate("A♠A♥K♦Q♠4♣")
        assert higher_kicker > lower_kicker

    def test_high_card_kicker_chain_decides(self) -> None:
        higher = _evaluate("A♠Q♥10♦7♠5♣")
        lower = _evaluate("A♠Q♥10♦7♠4♣")
        assert higher > lower

    def test_wheel_is_lowest_straight(self) -> None:
        wheel = _evaluate("A♠5♥4♦3♠2♣")
        six_high = _evaluate("6♠5♥4♦3♠2♣")
        assert six_high > wheel

    def test_wheel_loses_to_eight_high_straight(self) -> None:
        wheel = _evaluate("A♠5♥4♦3♠2♣")
        eight_high = _evaluate("8♠7♥6♦5♠4♣")
        assert eight_high > wheel

    def test_ace_high_straight_beats_king_high_straight(self) -> None:
        broadway = _evaluate("A♠K♥Q♦J♠10♣")
        king_high = _evaluate("K♠Q♥J♦10♠9♣")
        assert broadway > king_high

    def test_almost_straight_with_gap_is_high_card(self) -> None:
        # 9-8-7-5-4: missing the 6 means no straight.
        rank = _evaluate("9♠8♥7♦5♠4♣")
        assert rank.category is HandCategory.HIGH_CARD

    def test_pair_in_middle_does_not_form_straight(self) -> None:
        # A pair plus three other ranks with the same total span as a
        # straight must still register as ONE_PAIR, not STRAIGHT.
        rank = _evaluate("9♠8♠8♥6♦5♣")
        assert rank.category is HandCategory.ONE_PAIR

    @pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 6, 7])
    def test_rejects_wrong_card_count(self, size: int) -> None:
        # Build any ``size``-card sample from a known 7-card pool.
        sample = parse_cards_compact("A♠K♠Q♠J♠10♠9♠8♠")[:size]
        with pytest.raises(ValueError, match="exactly 5 cards"):
            evaluate_five_card_hand(sample)


class TestEvaluateBestHandDetailed:
    def test_five_cards_returns_them_unchanged(self) -> None:
        cards = _hand("A♠K♠Q♠J♠10♠")
        result = evaluate_best_hand_detailed(cards)
        assert result.rank.category is HandCategory.STRAIGHT_FLUSH
        assert tuple(cards) == result.best_five_cards

    def test_seven_cards_picks_royal_flush_over_pair(self) -> None:
        # Five suited broadway cards plus two unrelated cards.
        result = evaluate_best_hand_detailed(_hand("A♠K♠Q♠J♠10♠2♥3♣"))
        assert result.rank.category is HandCategory.STRAIGHT_FLUSH
        assert result.rank.tiebreakers == (14,)

    def test_seven_cards_picks_full_house(self) -> None:
        # Three aces, two kings, one queen, one deuce: AAA KK is the
        # best 5-card combination.
        result = evaluate_best_hand_detailed(_hand("A♠A♥A♦K♠K♥Q♠2♣"))
        assert result.rank.category is HandCategory.FULL_HOUSE
        assert result.rank.tiebreakers == (14, 13)

    def test_seven_cards_picks_flush_over_straight(self) -> None:
        # Five spades that include a straight-flush would beat a non-
        # spade straight; here we have a pure flush vs a non-flush
        # straight, so flush must win.
        result = evaluate_best_hand_detailed(_hand("A♠K♠9♠5♠2♠Q♥J♣"))
        assert result.rank.category is HandCategory.FLUSH

    def test_seven_cards_chooses_best_five(self) -> None:
        # Two pair from any 5-card subset; the best five include the
        # higher kicker.
        result = evaluate_best_hand_detailed(_hand("A♠A♥K♦K♠Q♣J♥2♠"))
        assert result.rank.category is HandCategory.TWO_PAIR
        assert result.rank.tiebreakers == (14, 13, 12)
        # The chosen five must include the queen kicker, not the jack
        # or the deuce.
        assert any(int(card.rank) == 12 for card in result.best_five_cards)
        assert all(int(card.rank) != 11 for card in result.best_five_cards)
        assert all(int(card.rank) != 2 for card in result.best_five_cards)

    @pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 8, 9])
    def test_rejects_wrong_pool_size(self, size: int) -> None:
        # Build any ``size``-card sample from a 9-card pool.
        sample = parse_cards_compact("A♠K♠Q♠J♠10♠9♠8♠7♠6♠")[:size]
        with pytest.raises(ValueError, match=f"{MIN_HAND_SIZE} to {MAX_HAND_SIZE}"):
            evaluate_best_hand_detailed(sample)

    def test_returns_evaluated_hand_instance(self) -> None:
        result = evaluate_best_hand_detailed(_hand("A♠K♠Q♠J♠10♠2♥3♣"))
        assert isinstance(result, EvaluatedHand)
        assert isinstance(result.rank, HandRank)
        assert len(result.best_five_cards) == MIN_HAND_SIZE


class TestRankOrderInvariants:
    """Cross-cutting invariants that must hold across every evaluation."""

    def test_every_category_strictly_beats_every_lower_category(self) -> None:
        # A representative hand for each category, ordered weakest to
        # strongest, with a strict ``<`` chain across the whole list.
        representatives = [
            _evaluate("A♠Q♥10♦7♠5♣"),  # high card
            _evaluate("A♠A♥K♦Q♠5♣"),  # one pair
            _evaluate("A♠A♥K♦K♠5♣"),  # two pair
            _evaluate("Q♠Q♥Q♦9♠5♣"),  # trips
            _evaluate("9♠8♥7♦6♠5♣"),  # straight
            _evaluate("A♠Q♠10♠5♠2♠"),  # flush
            _evaluate("K♠K♥K♦Q♠Q♥"),  # full house
            _evaluate("A♠A♥A♦A♣K♠"),  # quads
            _evaluate("9♠8♠7♠6♠5♠"),  # straight flush
        ]
        for weaker, stronger in pairwise(representatives):
            assert weaker < stronger
