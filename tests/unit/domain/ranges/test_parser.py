"""Tests for poker_assistant.domain.ranges.parser."""

from __future__ import annotations

import pytest

from poker_assistant.domain.ranges.parser import (
    RangeParseError,
    parse_range,
    parse_range_token,
)


class TestParseRange:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("AA", ["AA"]),
            ("aa", ["AA"]),
            ("AKs", ["AKs"]),
            ("AKo", ["AKo"]),
            ("AK", ["AKs", "AKo"]),
            ("AA, KK", ["AA", "KK"]),
            (" AA , KK ", ["AA", "KK"]),
            ("77+", ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77"]),
            ("AKs+", ["AKs"]),
            ("A9s+", ["AKs", "AQs", "AJs", "ATs", "A9s"]),
            ("99-66", ["99", "88", "77", "66"]),
            ("66-99", ["99", "88", "77", "66"]),
            ("K9s-K6s", ["K9s", "K8s", "K7s", "K6s"]),
        ],
    )
    def test_expansions(self, text: str, expected: list[str]) -> None:
        assert parse_range(text).labels() == expected

    def test_combines_multiple_tokens(self) -> None:
        result = parse_range("66+,ATs+,KJs+,QJs,JTs,AJo+,KQo")
        assert result.total_hand_classes() == 21

    def test_deduplicates_across_tokens(self) -> None:
        result = parse_range("AA, AA")
        assert result.labels() == ["AA"]

    @pytest.mark.parametrize("bad", ["", "   ", ",,,", " , , "])
    def test_rejects_empty_or_blank(self, bad: str) -> None:
        with pytest.raises(RangeParseError):
            parse_range(bad)

    def test_rejects_unknown_rank(self) -> None:
        with pytest.raises(RangeParseError, match="Unknown rank"):
            parse_range("XY")

    def test_rejects_invalid_shape_marker(self) -> None:
        with pytest.raises(RangeParseError, match="Unknown suitedness"):
            parse_range("AKx")

    def test_rejects_suited_with_equal_ranks(self) -> None:
        with pytest.raises(RangeParseError, match="equal ranks"):
            parse_range("AAs")

    def test_rejects_dash_with_different_shapes(self) -> None:
        with pytest.raises(RangeParseError, match="same shape"):
            parse_range("AKs-AKo")

    def test_rejects_non_pair_dash_with_different_high_ranks(self) -> None:
        with pytest.raises(RangeParseError, match="same high rank"):
            parse_range("AKs-Q9s")

    @pytest.mark.parametrize("bad", ["-", "AA-", "-AA", "AA - "])
    def test_rejects_malformed_dash(self, bad: str) -> None:
        with pytest.raises(RangeParseError):
            parse_range(bad)


class TestParseRangeToken:
    def test_uppercases_input(self) -> None:
        assert parse_range_token("aks").hand_classes[0].to_label() == "AKs"

    def test_strips_whitespace(self) -> None:
        assert parse_range_token("  AA  ").hand_classes[0].to_label() == "AA"

    def test_caches_results(self) -> None:
        a = parse_range_token("AA")
        b = parse_range_token("AA")
        assert a is b

    def test_rejects_empty_token(self) -> None:
        with pytest.raises(RangeParseError, match="Empty range token"):
            parse_range_token("")
