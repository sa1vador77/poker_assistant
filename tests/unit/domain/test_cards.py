"""Tests for poker_assistant.domain.cards."""

from __future__ import annotations

import pytest

from poker_assistant.domain.cards import (
    Card,
    Rank,
    Suit,
    SuitOrder,
    cards_are_unique,
    parse_card_token,
    parse_cards_compact,
    rank_to_label,
    suit_from_order,
    suit_order,
)


class TestSuit:
    def test_glyph_round_trip(self) -> None:
        assert Suit("♠") is Suit.SPADES
        assert Suit("♥") is Suit.HEARTS
        assert Suit("♦") is Suit.DIAMONDS
        assert Suit("♣") is Suit.CLUBS

    def test_str_value_is_glyph(self) -> None:
        assert str(Suit.SPADES) == "♠"

    def test_unknown_glyph_raises(self) -> None:
        with pytest.raises(ValueError):
            Suit("X")


class TestSuitOrder:
    def test_order_values_are_zero_through_three(self) -> None:
        # The numeric ordering must be tightly packed [0, 4) so that
        # it can be used as both an array index and a byte slot in the
        # native backend without translation.
        assert {int(member) for member in SuitOrder} == {0, 1, 2, 3}

    @pytest.mark.parametrize(
        ("suit", "expected_order"),
        [
            (Suit.SPADES, SuitOrder.SPADES),
            (Suit.HEARTS, SuitOrder.HEARTS),
            (Suit.DIAMONDS, SuitOrder.DIAMONDS),
            (Suit.CLUBS, SuitOrder.CLUBS),
        ],
    )
    def test_suit_order_maps_to_the_right_member(
        self,
        suit: Suit,
        expected_order: SuitOrder,
    ) -> None:
        assert suit_order(suit) is expected_order

    def test_suit_from_order_inverts_suit_order(self) -> None:
        for suit in Suit:
            assert suit_from_order(suit_order(suit)) is suit

    def test_suit_order_inverts_suit_from_order(self) -> None:
        for order in SuitOrder:
            assert suit_order(suit_from_order(order)) is order

    def test_canonical_order_is_spades_first_clubs_last(self) -> None:
        # This particular order is a contract used by HoleCombo
        # canonicalisation and by the native backend's card encoding.
        # Pinning it here means a future accidental reorder would break
        # this test loudly rather than silently change combo identities.
        assert int(SuitOrder.SPADES) == 0
        assert int(SuitOrder.HEARTS) == 1
        assert int(SuitOrder.DIAMONDS) == 2
        assert int(SuitOrder.CLUBS) == 3


class TestRank:
    def test_int_values_are_poker_strength(self) -> None:
        assert int(Rank.TWO) == 2
        assert int(Rank.ACE) == 14

    def test_ranks_compare_by_strength(self) -> None:
        assert Rank.TWO < Rank.THREE < Rank.JACK < Rank.QUEEN < Rank.KING < Rank.ACE


class TestCard:
    def test_is_hashable(self) -> None:
        card = Card(rank=Rank.ACE, suit=Suit.SPADES)
        assert {card, card} == {card}

    def test_is_immutable(self) -> None:
        card = Card(rank=Rank.ACE, suit=Suit.SPADES)
        with pytest.raises(AttributeError):
            card.rank = Rank.KING  # type: ignore[misc]

    def test_equality_is_value_based(self) -> None:
        a = Card(rank=Rank.ACE, suit=Suit.SPADES)
        b = Card(rank=Rank.ACE, suit=Suit.SPADES)
        assert a == b
        assert a is not b

    @pytest.mark.parametrize(
        ("rank", "suit", "expected"),
        [
            (Rank.ACE, Suit.SPADES, "A♠"),
            (Rank.KING, Suit.HEARTS, "K♥"),
            (Rank.QUEEN, Suit.DIAMONDS, "Q♦"),
            (Rank.JACK, Suit.CLUBS, "J♣"),
            (Rank.TEN, Suit.SPADES, "T♠"),
            (Rank.TWO, Suit.CLUBS, "2♣"),
            (Rank.NINE, Suit.HEARTS, "9♥"),
        ],
    )
    def test_str_renders_short_label(self, rank: Rank, suit: Suit, expected: str) -> None:
        assert str(Card(rank=rank, suit=suit)) == expected


class TestRankToLabel:
    @pytest.mark.parametrize(
        ("rank", "expected"),
        [
            (Rank.TWO, "2"),
            (Rank.NINE, "9"),
            (Rank.TEN, "T"),
            (Rank.JACK, "J"),
            (Rank.QUEEN, "Q"),
            (Rank.KING, "K"),
            (Rank.ACE, "A"),
        ],
    )
    def test_returns_canonical_label(self, rank: Rank, expected: str) -> None:
        assert rank_to_label(rank) == expected

    def test_covers_every_rank(self) -> None:
        for rank in Rank:
            assert isinstance(rank_to_label(rank), str)


class TestParseCardsCompact:
    def test_parses_single_card(self) -> None:
        assert parse_cards_compact("A♠") == [Card(rank=Rank.ACE, suit=Suit.SPADES)]

    def test_parses_three_cards(self) -> None:
        result = parse_cards_compact("7♠3♠J♣")
        assert result == [
            Card(rank=Rank.SEVEN, suit=Suit.SPADES),
            Card(rank=Rank.THREE, suit=Suit.SPADES),
            Card(rank=Rank.JACK, suit=Suit.CLUBS),
        ]

    def test_accepts_two_digit_ten(self) -> None:
        result = parse_cards_compact("5♥10♣6♦")
        assert result == [
            Card(rank=Rank.FIVE, suit=Suit.HEARTS),
            Card(rank=Rank.TEN, suit=Suit.CLUBS),
            Card(rank=Rank.SIX, suit=Suit.DIAMONDS),
        ]

    def test_accepts_t_for_ten(self) -> None:
        assert parse_cards_compact("T♠") == [Card(rank=Rank.TEN, suit=Suit.SPADES)]

    def test_accepts_lower_case_letters(self) -> None:
        assert parse_cards_compact("a♠k♥") == [
            Card(rank=Rank.ACE, suit=Suit.SPADES),
            Card(rank=Rank.KING, suit=Suit.HEARTS),
        ]

    def test_empty_string_returns_empty_list(self) -> None:
        assert parse_cards_compact("") == []

    def test_truncated_after_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="missing suit"):
            parse_cards_compact("A")

    def test_unknown_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown rank"):
            parse_cards_compact("Z♠")

    def test_unknown_suit_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown suit"):
            parse_cards_compact("AX")

    def test_error_does_not_chain_internal_value_error(self) -> None:
        """Parsing errors must not leak the internal ``Suit(...)`` ValueError chain.

        Users see a single, clear ``ValueError`` with our message; the
        ``raise ... from None`` form suppresses the implementation-detail
        cause that would otherwise appear in the traceback.
        """
        with pytest.raises(ValueError) as exc_info:
            parse_cards_compact("AX")
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True


class TestParseCardToken:
    def test_parses_single_card(self) -> None:
        assert parse_card_token("A♠") == Card(rank=Rank.ACE, suit=Suit.SPADES)

    def test_parses_ten_with_two_digits(self) -> None:
        assert parse_card_token("10♦") == Card(rank=Rank.TEN, suit=Suit.DIAMONDS)

    def test_rejects_zero_cards(self) -> None:
        with pytest.raises(ValueError):
            parse_card_token("")

    def test_rejects_more_than_one_card(self) -> None:
        with pytest.raises(ValueError, match="Expected exactly one card"):
            parse_card_token("A♠K♥")


class TestCardsAreUnique:
    def test_empty_collection_is_unique(self) -> None:
        assert cards_are_unique([]) is True

    def test_unique_cards_return_true(self) -> None:
        cards = parse_cards_compact("A♠K♥Q♦")
        assert cards_are_unique(cards) is True

    def test_duplicate_cards_return_false(self) -> None:
        cards = parse_cards_compact("A♠K♥A♠")
        assert cards_are_unique(cards) is False

    def test_accepts_tuple(self) -> None:
        cards = tuple(parse_cards_compact("A♠K♥"))
        assert cards_are_unique(cards) is True

    def test_accepts_set(self) -> None:
        cards = set(parse_cards_compact("A♠K♥A♠"))
        # By construction a set has no duplicates.
        assert cards_are_unique(cards) is True
