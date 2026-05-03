"""Protocol contract for equity backends.

This module defines what it means to *be* an equity backend. The two
concrete implementations — :mod:`poker_assistant.compute.equity_python`
(reference) and :mod:`poker_assistant.compute.equity_native` (fast,
backed by the C++ extension) — both satisfy this protocol. Domain code
that needs equity depends only on the protocol and is therefore
indifferent to which backend is in use at runtime.

Two computation modes are exposed:

* **Exact** — exhaustive enumeration of every reachable showdown,
  weighted by villain combo probabilities. Produces the true equity
  value, not an estimate. Tractable on the river for any number of
  villains; on the turn for up to four villains; on the flop only
  heads-up.
* **Monte Carlo** — random sampling of showdowns. Returns an unbiased
  estimate whose variance shrinks with sample count. Available on
  any street and any villain count up to four.

The two modes do not have separate result types: both produce an
:class:`EquityResult`. Whether the result is exact or estimated is a
property of the *call*, not of the value, so consumers that care can
inspect the source of the result themselves.

Choosing a backend
------------------
The decision layer should ask a backend ``supports_*`` first, then
call the corresponding ``calculate_*`` method only if support is
confirmed. Backends are required to fail loudly (raise
:class:`ValueError`) if a calculate method is called for an
unsupported configuration; the predicates are an optimisation, not a
safety net.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from poker_assistant.domain.cards import Card
from poker_assistant.domain.ranges import HandRange


@dataclass(frozen=True, slots=True)
class EquityResult:
    """Outcome of an equity calculation.

    All probabilities are in ``[0.0, 1.0]`` and the three rates sum to
    ``1.0`` up to floating-point error.

    Attributes:
        hero_equity: ``win_rate + 0.5 * tie_rate``. The standard poker
            equity figure: a tie counts as half a win because the
            chips are split. This is what the decision engine cares
            about most often, and it is precomputed here so callers do
            not all repeat the same arithmetic.
        win_rate: Fraction of weighted scenarios where the hero wins
            outright.
        tie_rate: Fraction of weighted scenarios where the hero ties
            with at least one villain.
        loss_rate: Fraction of weighted scenarios where at least one
            villain beats the hero.
        scenarios_evaluated: For exact calculations, the integer
            number of weighted scenarios summed (rounded). For Monte
            Carlo calculations, the number of successful samples
            drawn. The field exists so callers can sanity-check
            statistical confidence and detect degenerate inputs (e.g.
            a range fully blocked by dead cards would yield zero).
    """

    hero_equity: float
    win_rate: float
    tie_rate: float
    loss_rate: float
    scenarios_evaluated: int

    def __post_init__(self) -> None:
        for name, value in (
            ("hero_equity", self.hero_equity),
            ("win_rate", self.win_rate),
            ("tie_rate", self.tie_rate),
            ("loss_rate", self.loss_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be in [0.0, 1.0], got {value}",
                )
        if self.scenarios_evaluated < 0:
            raise ValueError(
                f"scenarios_evaluated must be non-negative, got {self.scenarios_evaluated}",
            )


@runtime_checkable
class EquityBackend(Protocol):
    """A computational backend that estimates a hero's postflop equity.

    Inputs are expressed in domain terms (:class:`Card`,
    :class:`HandRange`); any encoding to integer card ids or other
    backend-specific representations happens inside the implementation.

    All methods are pure and deterministic given their inputs and any
    seed they expose; backends do not maintain state across calls.
    """

    def supports_exact(
        self,
        *,
        villain_count: int,
        board_size: int,
    ) -> bool:
        """Report whether the backend can solve this spot exactly.

        Args:
            villain_count: Number of villains in the hand (1 or more).
            board_size: Number of board cards (3, 4, or 5).
        """
        ...

    def supports_monte_carlo(
        self,
        *,
        villain_count: int,
        board_size: int,
    ) -> bool:
        """Report whether the backend can sample this spot via Monte Carlo."""
        ...

    def calculate_exact(
        self,
        *,
        hero_hole_cards: tuple[Card, Card],
        board_cards: tuple[Card, ...],
        villain_ranges: tuple[HandRange, ...],
    ) -> EquityResult:
        """Solve the spot exactly via exhaustive enumeration.

        Dead cards are derived inside the backend from
        ``hero_hole_cards`` and ``board_cards`` before each villain
        range is expanded; callers do not need to filter ranges
        beforehand.

        Args:
            hero_hole_cards: The hero's two hole cards.
            board_cards: Three to five community cards.
            villain_ranges: One range per villain. The order is not
                significant for the result, but is preserved by
                implementations to make profiling and debugging
                deterministic.

        Returns:
            The :class:`EquityResult` of the showdown.

        Raises:
            ValueError: If the configuration is not supported (see
                :meth:`supports_exact`), or if the inputs are
                inconsistent (e.g. duplicate cards across hero, board,
                and ranges, or a range that has no available combos
                after dead-card filtering).
        """
        ...

    def calculate_monte_carlo(
        self,
        *,
        hero_hole_cards: tuple[Card, Card],
        board_cards: tuple[Card, ...],
        villain_ranges: tuple[HandRange, ...],
        sample_count: int,
        random_seed: int | None = None,
    ) -> EquityResult:
        """Estimate the spot's equity via Monte Carlo sampling.

        Args:
            hero_hole_cards: The hero's two hole cards.
            board_cards: Three to five community cards.
            villain_ranges: One range per villain.
            sample_count: Target number of successful samples. The
                result's ``scenarios_evaluated`` may be lower if the
                backend's rejection sampling cannot reach the target
                within its retry budget; in practice, with realistic
                ranges, this number is hit exactly.
            random_seed: If provided, the backend uses it to make the
                output deterministic. If ``None``, every call uses a
                fresh source of randomness; two consecutive calls with
                the same arguments will return slightly different
                results.

        Returns:
            The :class:`EquityResult` estimated from the samples.

        Raises:
            ValueError: If the configuration is not supported, if
                inputs are inconsistent, or if ``sample_count`` is
                not positive.
        """
        ...
