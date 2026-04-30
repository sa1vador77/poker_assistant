"""Parser for poker range notation into :class:`HandRange`.

Notation
--------
A range is a comma-separated list of tokens. Each token is one of:

* A pocket pair: ``"AA"``, ``"77"``.
* A suited or offsuit class: ``"AKs"``, ``"T9o"``.
* A two-character class without a shape marker (``"AK"``), which
  expands to both ``"AKs"`` and ``"AKo"``.
* A *plus* token (``"77+"``, ``"AKs+"``): all classes that are at
  least as strong as the base, holding the high rank fixed.
* A *dash* range (``"99-66"``, ``"K9s-K6s"``): all classes between
  the two endpoints inclusive, sharing the same shape (and, for
  non-pairs, the same high rank).

Whitespace around tokens and commas is ignored; rank letters are
case-insensitive. Parsing is cached at the token level so that
repeated calls (e.g. from preset catalogues) are essentially free.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from poker_assistant.domain.cards import Rank
from poker_assistant.domain.ranges.models import HandClass, HandRange

# Ranks ordered from strongest to weakest. Used for both range
# expansion (``77+`` walks down the list from the base rank) and for
# bounds checks in dash ranges.
_RANKS_DESC: tuple[Rank, ...] = (
    Rank.ACE,
    Rank.KING,
    Rank.QUEEN,
    Rank.JACK,
    Rank.TEN,
    Rank.NINE,
    Rank.EIGHT,
    Rank.SEVEN,
    Rank.SIX,
    Rank.FIVE,
    Rank.FOUR,
    Rank.THREE,
    Rank.TWO,
)

# Single-character labels accepted on input. The two-character "10"
# form is not accepted in range notation: standard poker notation
# always uses "T" for the ten in range strings, even when the vision
# pipeline emits "10" for cards on the table.
_RANK_BY_LABEL: dict[str, Rank] = {
    "A": Rank.ACE,
    "K": Rank.KING,
    "Q": Rank.QUEEN,
    "J": Rank.JACK,
    "T": Rank.TEN,
    "9": Rank.NINE,
    "8": Rank.EIGHT,
    "7": Rank.SEVEN,
    "6": Rank.SIX,
    "5": Rank.FIVE,
    "4": Rank.FOUR,
    "3": Rank.THREE,
    "2": Rank.TWO,
}


class RangeParseError(ValueError):
    """Raised when a range string cannot be parsed."""


@dataclass(frozen=True, slots=True)
class ParsedToken:
    """The result of parsing a single range token."""

    token: str
    hand_classes: tuple[HandClass, ...]


def parse_range(text: str) -> HandRange:
    """Parse a comma-separated range expression into a :class:`HandRange`.

    Args:
        text: A non-empty range string, e.g. ``"66+,ATs+,KJs+,QJs"``.

    Returns:
        A :class:`HandRange` with weight 1.0 for every class. Duplicate
        classes across tokens are deduplicated.

    Raises:
        RangeParseError: If the string is empty or any token is malformed.
    """
    if not text or not text.strip():
        raise RangeParseError("Range string is empty")

    tokens = [part.strip() for part in text.split(",") if part.strip()]
    if not tokens:
        raise RangeParseError("Range string contains no valid tokens")

    classes: list[HandClass] = []
    for token in tokens:
        classes.extend(parse_range_token(token).hand_classes)

    return HandRange.from_hand_classes(classes, weight=1.0)


def parse_range_token(token: str) -> ParsedToken:
    """Parse a single range token into a :class:`ParsedToken`.

    The result is cached: repeated calls with the same (normalised)
    token return the same object without re-parsing.

    Args:
        token: One range token, e.g. ``"AKs"``, ``"77+"``, ``"99-66"``.

    Raises:
        RangeParseError: If the token is empty or malformed.
    """
    cleaned = token.strip().upper()
    if not cleaned:
        raise RangeParseError("Empty range token")
    return _parse_token_cached(cleaned)


@lru_cache(maxsize=1024)
def _parse_token_cached(token: str) -> ParsedToken:
    """Parse one already-normalised token. Cached.

    The token must be the result of ``token.strip().upper()`` from the
    public entry point; this internal function does no extra normalisation.
    """
    if "-" in token:
        return ParsedToken(token=token, hand_classes=tuple(_parse_dash_token(token)))
    if token.endswith("+"):
        return ParsedToken(token=token, hand_classes=tuple(_parse_plus_token(token)))
    if len(token) == 2 and not _is_pair_token(token):
        return ParsedToken(token=token, hand_classes=tuple(_parse_unspecified_shape(token)))
    return ParsedToken(token=token, hand_classes=(_parse_single_class(token),))


def _parse_unspecified_shape(token: str) -> list[HandClass]:
    """Expand ``"AK"`` to both ``"AKs"`` and ``"AKo"``."""
    high, low = _ranks_desc(_rank_from_label(token[0]), _rank_from_label(token[1]))
    return [
        HandClass.suited(high, low),
        HandClass.offsuit(high, low),
    ]


def _parse_single_class(token: str) -> HandClass:
    """Parse a single token that names exactly one hand class.

    Accepts pair tokens (``"AA"``, ``"77"``) and 3-character tokens with
    explicit shape (``"AKs"``, ``"T9o"``).
    """
    if _is_pair_token(token):
        return HandClass.pair(_rank_from_label(token[0]))

    if len(token) != 3:
        raise RangeParseError(f"Unsupported hand class token: {token!r}")

    high_rank = _rank_from_label(token[0])
    low_rank = _rank_from_label(token[1])
    if high_rank == low_rank:
        raise RangeParseError(f"Suited/offsuit token cannot have equal ranks: {token!r}")
    high, low = _ranks_desc(high_rank, low_rank)

    shape_marker = token[2]
    if shape_marker == "S":
        return HandClass.suited(high, low)
    if shape_marker == "O":
        return HandClass.offsuit(high, low)
    raise RangeParseError(f"Unknown suitedness marker in token: {token!r}")


def _parse_plus_token(token: str) -> list[HandClass]:
    """Parse a ``"…+"`` token into the list of classes it expands to.

    For pairs, ``"77+"`` means ``77, 88, 99, TT, JJ, QQ, KK, AA``. For
    non-pairs, ``"AKs+"`` means every suited class with the same high
    rank whose low rank is at least the base low rank — for ``AKs+``
    this is just ``AKs`` itself, since no rank is higher than K below A.
    """
    base = token[:-1]
    if not base:
        raise RangeParseError(f"Invalid plus token: {token!r}")

    if _is_pair_token(base):
        start = _rank_from_label(base[0])
        return [HandClass.pair(rank) for rank in _RANKS_DESC if rank >= start]

    base_class = _parse_single_class(base)
    if base_class.is_pair:
        # Defensive: the branch above already handles pair tokens.
        raise RangeParseError(f"Unexpected pair shape in plus token: {token!r}")

    factory = HandClass.suited if base_class.is_suited else HandClass.offsuit
    high = base_class.high_rank
    return [factory(high, low) for low in _RANKS_DESC if base_class.low_rank <= low < high]


def _parse_dash_token(token: str) -> list[HandClass]:
    """Parse an ``"X-Y"`` token into the inclusive range of classes between X and Y.

    Both endpoints must have the same shape. For non-pair endpoints
    they must also share the same high rank: ``"K9s-K6s"`` is valid,
    ``"AKs-Q9s"`` is not.
    """
    parts = token.split("-", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise RangeParseError(f"Invalid dash range token: {token!r}")

    left = _parse_single_class(parts[0].strip())
    right = _parse_single_class(parts[1].strip())

    if left.shape is not right.shape:
        raise RangeParseError(f"Dash range must have same shape on both sides: {token!r}")

    if left.is_pair:
        return _expand_pair_dash_range(left, right)

    if left.high_rank != right.high_rank:
        raise RangeParseError(f"Non-pair dash range must keep same high rank: {token!r}")

    low_min = min(left.low_rank, right.low_rank)
    low_max = max(left.low_rank, right.low_rank)
    factory = HandClass.suited if left.is_suited else HandClass.offsuit
    return [
        factory(left.high_rank, low)
        for low in _RANKS_DESC
        if low_min <= low <= low_max and low != left.high_rank
    ]


def _expand_pair_dash_range(left: HandClass, right: HandClass) -> list[HandClass]:
    """Expand ``"99-66"`` into ``[99, 88, 77, 66]``."""
    rank_min = min(left.high_rank, right.high_rank)
    rank_max = max(left.high_rank, right.high_rank)
    return [HandClass.pair(rank) for rank in _RANKS_DESC if rank_min <= rank <= rank_max]


def _is_pair_token(token: str) -> bool:
    """Return ``True`` iff ``token`` looks like a pocket pair (``"AA"``, ``"77"``)."""
    return len(token) == 2 and token[0] == token[1] and token[0] in _RANK_BY_LABEL


def _rank_from_label(label: str) -> Rank:
    """Map a single uppercase rank character to a :class:`Rank`."""
    rank = _RANK_BY_LABEL.get(label)
    if rank is None:
        raise RangeParseError(f"Unknown rank label: {label!r}")
    return rank


def _ranks_desc(left: Rank, right: Rank) -> tuple[Rank, Rank]:
    """Return the two ranks ordered from higher to lower."""
    if int(left) >= int(right):
        return left, right
    return right, left
