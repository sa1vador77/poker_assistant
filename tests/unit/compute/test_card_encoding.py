"""Tests for poker_assistant.compute.card_encoding."""

from __future__ import annotations

import pytest

from poker_assistant.compute.card_encoding import (
    DECK_SIZE,
    RANKS_PER_SUIT,
    decode_card,
    decode_cards,
    encode_card,
    encode_cards,
)
from poker_assistant.domain.cards import Card, Rank, Suit, suit_order


class TestEncodeCard:
    def test_two_of_spades_is_zero(self) -> None:
        # Spades has SuitOrder == 0 and Rank.TWO is the lowest, so the
        # bottom-left of the encoding grid is the canonical "id 0" card.
        assert encode_card(Card(rank=Rank.TWO, suit=Suit.SPADES)) == 0

    def test_ace_of_clubs_is_top(self) -> None:
        # Clubs has SuitOrder == 3 and Rank.ACE is the highest. The
        # top-right of the encoding grid is the last valid id, 51.
        assert encode_card(Card(rank=Rank.ACE, suit=Suit.CLUBS)) == DECK_SIZE - 1

    @pytest.mark.parametrize(
        ("rank", "suit"),
        [
            (Rank.TWO, Suit.SPADES),
            (Rank.ACE, Suit.SPADES),
            (Rank.TWO, Suit.HEARTS),
            (Rank.SEVEN, Suit.DIAMONDS),
            (Rank.KING, Suit.CLUBS),
        ],
    )
    def test_matches_native_decode_formula(self, rank: Rank, suit: Suit) -> None:
        # The native backend uses:
        #   suit = card_id // 13
        #   rank = card_id %% 13 + 2
        # We verify the Python encode produces ids that satisfy these.
        card_id = encode_card(Card(rank=rank, suit=suit))
        assert card_id // RANKS_PER_SUIT == int(suit_order(suit))
        assert card_id % RANKS_PER_SUIT + 2 == int(rank)


class TestDecodeCard:
    def test_zero_decodes_to_two_of_spades(self) -> None:
        assert decode_card(0) == Card(rank=Rank.TWO, suit=Suit.SPADES)

    def test_top_id_decodes_to_ace_of_clubs(self) -> None:
        assert decode_card(DECK_SIZE - 1) == Card(rank=Rank.ACE, suit=Suit.CLUBS)

    @pytest.mark.parametrize("invalid_id", [-1, -100, DECK_SIZE, DECK_SIZE + 1, 1000])
    def test_rejects_out_of_range_ids(self, invalid_id: int) -> None:
        with pytest.raises(ValueError, match=r"\[0, 52\)"):
            decode_card(invalid_id)


class TestRoundTrip:
    def test_encode_then_decode_is_identity_for_every_card(self) -> None:
        # Bijection check: for every (rank, suit) the round-trip
        # produces the same card and a unique id.
        seen_ids: set[int] = set()
        for suit in Suit:
            for rank in Rank:
                card = Card(rank=rank, suit=suit)
                card_id = encode_card(card)
                assert 0 <= card_id < DECK_SIZE
                assert card_id not in seen_ids
                seen_ids.add(card_id)
                assert decode_card(card_id) == card
        # All 52 ids must have been produced exactly once.
        assert seen_ids == set(range(DECK_SIZE))

    def test_decode_then_encode_is_identity_for_every_id(self) -> None:
        for card_id in range(DECK_SIZE):
            assert encode_card(decode_card(card_id)) == card_id


class TestBatchHelpers:
    def test_encode_cards_returns_tuple(self) -> None:
        cards = [
            Card(rank=Rank.ACE, suit=Suit.SPADES),
            Card(rank=Rank.KING, suit=Suit.HEARTS),
        ]
        result = encode_cards(cards)
        assert isinstance(result, tuple)
        assert result == (encode_card(cards[0]), encode_card(cards[1]))

    def test_decode_cards_returns_tuple(self) -> None:
        ids = (0, 25, 51)
        result = decode_cards(ids)
        assert isinstance(result, tuple)
        assert result == tuple(decode_card(card_id) for card_id in ids)

    def test_encode_cards_handles_empty_iterable(self) -> None:
        assert encode_cards([]) == ()

    def test_decode_cards_handles_empty_iterable(self) -> None:
        assert decode_cards([]) == ()

    def test_decode_cards_propagates_errors_on_invalid_id(self) -> None:
        with pytest.raises(ValueError):
            decode_cards([0, 100, 1])

    def test_encode_cards_round_trip_through_decode_cards(self) -> None:
        cards = [
            Card(rank=Rank.TWO, suit=Suit.SPADES),
            Card(rank=Rank.SEVEN, suit=Suit.DIAMONDS),
            Card(rank=Rank.ACE, suit=Suit.CLUBS),
        ]
        assert decode_cards(encode_cards(cards)) == tuple(cards)
