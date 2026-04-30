"""Tests for poker_assistant.domain.board.texture."""

from __future__ import annotations

import pytest

from poker_assistant.domain.board.texture import (
    BoardFacts,
    BoardTexture,
    BoardTextureKind,
    analyze_board_texture,
)
from poker_assistant.domain.cards import Card, Rank, parse_cards_compact


def _board(notation: str) -> list[Card]:
    return parse_cards_compact(notation)


def _texture(notation: str) -> BoardTexture:
    return analyze_board_texture(_board(notation))


class TestBoardSizeValidation:
    @pytest.mark.parametrize("size", [0, 1, 2, 6, 7])
    def test_rejects_wrong_size(self, size: int) -> None:
        sample = parse_cards_compact("A♠K♠Q♠J♠T♠9♠8♠")[:size]
        with pytest.raises(ValueError, match="3 to 5 board cards"):
            analyze_board_texture(sample)

    def test_rejects_duplicate_cards(self) -> None:
        ace = Card(rank=Rank.ACE, suit=parse_cards_compact("A♠")[0].suit)
        with pytest.raises(ValueError, match="duplicate cards"):
            analyze_board_texture([ace, ace, ace])


class TestBoardFacts:
    def test_pairing_flags(self) -> None:
        facts = _texture("A♠A♥K♦").facts
        assert facts.is_paired
        assert not facts.is_double_paired
        assert not facts.is_trips

    def test_double_paired(self) -> None:
        facts = _texture("A♠A♥K♦K♠5♣").facts
        assert facts.is_paired
        assert facts.is_double_paired
        assert not facts.is_trips

    def test_trips(self) -> None:
        facts = _texture("A♠A♥A♦").facts
        assert facts.is_paired
        assert facts.is_trips

    def test_suit_categorisation_is_mutually_exclusive(self) -> None:
        # Each board falls into exactly one of monotone / two-tone /
        # rainbow (when distinct_suits == board_size).
        for notation in ("A♠K♠Q♠", "A♠K♠Q♥", "A♠K♥Q♦", "A♠K♥Q♦J♠T♣"):
            facts = _texture(notation).facts
            categories = [facts.is_monotone, facts.is_two_tone, facts.is_rainbow]
            # At most one of these can be True at the same time when
            # the corresponding size relations hold.
            assert sum(categories) <= 1

    def test_monotone(self) -> None:
        facts = _texture("A♠K♠Q♠").facts
        assert facts.is_monotone
        assert not facts.is_two_tone
        assert facts.max_same_suit == 3

    def test_two_tone_means_exactly_two_suits(self) -> None:
        # The flag tracks the count of distinct suits, not "non-monotone
        # with at least one shared suit". A 4-1 split is two-tone
        # (exactly two suits present), even though the kind is then
        # FOUR_FLUSH because the kind cascade is more specific.
        facts_four_one = _texture("A♠K♠Q♠J♠2♥").facts
        assert facts_four_one.is_two_tone

        facts_three_two = _texture("A♠K♠Q♠J♥2♥").facts
        assert facts_three_two.is_two_tone

        # Three different suits → not two-tone, not monotone, not rainbow
        # (rainbow requires distinct_suits == board_size).
        facts_three_distinct = _texture("A♠K♠Q♥J♥2♦").facts
        assert not facts_three_distinct.is_two_tone
        assert not facts_three_distinct.is_monotone
        assert not facts_three_distinct.is_rainbow

    def test_rainbow(self) -> None:
        facts = _texture("A♠K♥Q♦").facts
        assert facts.is_rainbow

    def test_flush_detection(self) -> None:
        facts_four = _texture("A♠K♠Q♠J♠2♥").facts
        assert facts_four.has_four_flush
        assert not facts_four.has_five_flush

        facts_five = _texture("A♠K♠Q♠J♠T♠").facts
        assert facts_five.has_four_flush
        assert facts_five.has_five_flush

    def test_broadway_count(self) -> None:
        # A K Q are broadway; 9 and 5 are not.
        facts = _texture("A♠K♥Q♦9♠5♣").facts
        assert facts.broadway_count == 3

    def test_high_card_count(self) -> None:
        # A K Q J are "high cards" (>= jack); T is broadway but not high.
        facts = _texture("A♠K♥Q♦J♠T♣").facts
        assert facts.high_card_count == 4

    def test_highest_and_lowest_rank(self) -> None:
        facts = _texture("9♠5♥3♦").facts
        assert facts.highest_rank is Rank.NINE
        assert facts.lowest_rank is Rank.THREE

    def test_max_consecutive_run_recognises_wheel_pattern(self) -> None:
        # A-2-3-4-5 must form a run of length five via the low-ace.
        facts = _texture("A♠5♥4♦3♠2♣").facts
        assert facts.max_consecutive_run == 5
        assert facts.has_straight

    def test_straight_on_board(self) -> None:
        facts = _texture("9♠8♥7♦6♠5♣").facts
        assert facts.has_straight
        assert facts.has_four_to_straight

    def test_four_to_straight_without_straight(self) -> None:
        facts = _texture("9♠8♥7♦6♠2♣").facts
        assert facts.has_four_to_straight
        assert not facts.has_straight

    def test_no_run_no_straight(self) -> None:
        facts = _texture("A♠9♥5♦").facts
        assert facts.max_consecutive_run == 1
        assert not facts.has_straight
        assert not facts.has_four_to_straight


class TestTextureClassification:
    @pytest.mark.parametrize(
        ("notation", "expected"),
        [
            # Flush priorities.
            ("A♠K♠Q♠J♠T♠", BoardTextureKind.FIVE_FLUSH),
            ("A♠A♥K♠Q♠J♠", BoardTextureKind.PAIRED_FOUR_FLUSH),
            ("A♠K♠Q♠J♠2♥", BoardTextureKind.FOUR_FLUSH),
            # Monotone variants.
            ("A♠K♠Q♠", BoardTextureKind.MONOTONE),
            ("A♠A♥K♠Q♠2♠", BoardTextureKind.PAIRED_FOUR_FLUSH),  # four-flush wins
            # Pair structures.
            ("A♠A♥A♦", BoardTextureKind.TRIPS),
            ("A♠A♥K♦K♠5♣", BoardTextureKind.DOUBLE_PAIRED),
            ("9♠9♥8♠7♥", BoardTextureKind.PAIRED_CONNECTED),
            ("A♠A♥3♦", BoardTextureKind.PAIRED),
            # Two-tone connectedness.
            ("9♠8♠7♥", BoardTextureKind.TWO_TONE_CONNECTED),
            ("A♠5♠2♥", BoardTextureKind.TWO_TONE),
            # Rainbow connected.
            ("9♠8♥7♦", BoardTextureKind.VERY_CONNECTED),
            # Dry boards.
            ("A♠7♥2♦", BoardTextureKind.DRY_HIGH),
            ("8♠5♥2♦", BoardTextureKind.DRY),
        ],
    )
    def test_classifies_known_textures(
        self,
        notation: str,
        expected: BoardTextureKind,
    ) -> None:
        assert _texture(notation).kind is expected

    def test_classification_is_a_total_function(self) -> None:
        # Every reachable board produces exactly one kind, never None,
        # never crashes. We sample widely.
        samples = [
            "A♠K♠Q♠",
            "A♠K♥Q♦",
            "A♠K♠Q♥",
            "9♠9♥8♠7♥",
            "A♠A♥K♦K♠5♣",
            "A♠A♥A♦",
            "9♠8♥7♦6♠5♣",
            "8♠5♥2♦",
            "A♠7♥2♦",
            "T♠9♥8♦7♠6♣",
        ]
        for notation in samples:
            kind = _texture(notation).kind
            assert isinstance(kind, BoardTextureKind)


class TestDerivedFlags:
    def test_dry_board_flag(self) -> None:
        # A-7-2 rainbow with no run is the canonical dry high board.
        texture = _texture("A♠7♥2♦")
        assert texture.is_dry
        assert not texture.is_dynamic

    def test_dynamic_board_flag(self) -> None:
        # 9-8-7 rainbow has a run of 3 → very_connected → dynamic.
        texture = _texture("9♠8♥7♦")
        assert texture.is_dynamic
        assert not texture.is_dry

    def test_low_board_flag(self) -> None:
        assert _texture("9♠5♥2♦").is_low_board
        assert not _texture("A♠5♥2♦").is_low_board

    def test_connected_implies_run_within_threshold(self) -> None:
        # 9-8-7 rainbow: span 2, len 3 → connected.
        assert _texture("9♠8♥7♦").is_connected
        # A-9-2: span 12 → not connected.
        assert not _texture("A♠9♥2♦").is_connected

    def test_very_connected_means_run_of_three_or_more(self) -> None:
        assert _texture("9♠8♥7♦").is_very_connected
        # Pair plus an isolated card breaks the run.
        assert not _texture("A♠A♥3♦").is_very_connected


class TestStrEnumProperties:
    def test_kind_values_are_unique(self) -> None:
        values = [kind.value for kind in BoardTextureKind]
        assert len(values) == len(set(values))


class TestBoardTextureStructure:
    def test_facts_and_texture_are_separate(self) -> None:
        texture = _texture("A♠K♠Q♠")
        assert isinstance(texture.facts, BoardFacts)
        assert isinstance(texture, BoardTexture)
        # The classification's flags are not inside BoardFacts.
        assert not hasattr(texture.facts, "is_dry")
        assert not hasattr(texture.facts, "kind")
