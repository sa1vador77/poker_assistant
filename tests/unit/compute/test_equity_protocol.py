"""Tests for poker_assistant.compute.equity_protocol."""

from __future__ import annotations

import pytest

from poker_assistant.compute.equity_protocol import (
    EquityBackend,
    EquityResult,
)
from poker_assistant.domain.cards import Card
from poker_assistant.domain.ranges import HandRange


class TestEquityResultValidation:
    def test_accepts_valid_rates(self) -> None:
        result = EquityResult(
            hero_equity=0.55,
            win_rate=0.5,
            tie_rate=0.1,
            loss_rate=0.4,
            scenarios_evaluated=1000,
        )
        assert result.hero_equity == 0.55

    def test_accepts_boundary_values(self) -> None:
        EquityResult(
            hero_equity=0.0,
            win_rate=0.0,
            tie_rate=0.0,
            loss_rate=1.0,
            scenarios_evaluated=0,
        )
        EquityResult(
            hero_equity=1.0,
            win_rate=1.0,
            tie_rate=0.0,
            loss_rate=0.0,
            scenarios_evaluated=1,
        )

    @pytest.mark.parametrize("bad_value", [-0.01, 1.01, -1.0, 2.0])
    def test_rejects_hero_equity_outside_unit_interval(self, bad_value: float) -> None:
        with pytest.raises(ValueError, match=r"hero_equity"):
            EquityResult(
                hero_equity=bad_value,
                win_rate=0.5,
                tie_rate=0.0,
                loss_rate=0.5,
                scenarios_evaluated=100,
            )

    @pytest.mark.parametrize("field", ["win_rate", "tie_rate", "loss_rate"])
    def test_rejects_rate_outside_unit_interval(self, field: str) -> None:
        rates: dict[str, float] = {
            "hero_equity": 0.5,
            "win_rate": 0.5,
            "tie_rate": 0.0,
            "loss_rate": 0.5,
        }
        rates[field] = -0.1
        with pytest.raises(ValueError, match=field):
            EquityResult(scenarios_evaluated=100, **rates)

    def test_rejects_negative_scenarios_evaluated(self) -> None:
        with pytest.raises(ValueError, match="scenarios_evaluated"):
            EquityResult(
                hero_equity=0.5,
                win_rate=0.5,
                tie_rate=0.0,
                loss_rate=0.5,
                scenarios_evaluated=-1,
            )

    def test_is_immutable(self) -> None:
        result = EquityResult(
            hero_equity=0.5,
            win_rate=0.5,
            tie_rate=0.0,
            loss_rate=0.5,
            scenarios_evaluated=10,
        )
        with pytest.raises(AttributeError):
            result.hero_equity = 0.9  # type: ignore[misc]


class TestEquityBackendProtocol:
    """Verify that the Protocol is structural and runtime-checkable.

    These tests do not exercise any real backend; they confirm that an
    object implementing all protocol methods is recognised by
    ``isinstance``, while a partial implementation is not. This lets
    test doubles in higher layers safely declare ``EquityBackend`` as
    their interface.
    """

    def test_full_implementation_is_recognised(self) -> None:
        class DummyBackend:
            def supports_exact(self, *, villain_count: int, board_size: int) -> bool:
                return False

            def supports_monte_carlo(self, *, villain_count: int, board_size: int) -> bool:
                return False

            def calculate_exact(
                self,
                *,
                hero_hole_cards: tuple[Card, Card],
                board_cards: tuple[Card, ...],
                villain_ranges: tuple[HandRange, ...],
            ) -> EquityResult:
                raise NotImplementedError

            def calculate_monte_carlo(
                self,
                *,
                hero_hole_cards: tuple[Card, Card],
                board_cards: tuple[Card, ...],
                villain_ranges: tuple[HandRange, ...],
                sample_count: int,
                random_seed: int | None = None,
            ) -> EquityResult:
                raise NotImplementedError

        assert isinstance(DummyBackend(), EquityBackend)

    def test_partial_implementation_is_not_recognised(self) -> None:
        class IncompleteBackend:
            def supports_exact(self, *, villain_count: int, board_size: int) -> bool:
                return False

        assert not isinstance(IncompleteBackend(), EquityBackend)
