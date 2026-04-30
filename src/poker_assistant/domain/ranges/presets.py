"""Catalog of named poker range presets.

A :class:`RangePresetCatalog` maps :class:`RangePreset` names to
:class:`PresetDefinition` records (name + textual range + description),
and parses the textual range into a :class:`HandRange` on demand.

The default catalogue is exposed via :func:`default_catalog`, which
constructs the catalogue lazily on first use so that simply importing
this module does not pay the parsing cost.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Self

from poker_assistant.domain.ranges.models import HandRange
from poker_assistant.domain.ranges.parser import parse_range


class RangePreset(StrEnum):
    """Named slots for the bundled range presets.

    The catalogue is closed at this layer: each member must have a
    corresponding entry in :data:`DEFAULT_PRESET_DEFINITIONS`. Adding
    a new preset is a single change to both this enum and the
    definitions mapping.
    """

    TIGHT_OPEN = "tight_open"
    LOOSE_OPEN = "loose_open"
    BTN_OPEN = "btn_open"
    SB_LIMP_CALL = "sb_limp_call"
    BB_DEFEND = "bb_defend"
    NITTY_CALL_VS_OPEN = "nitty_call_vs_open"
    LOOSE_CALL_VS_OPEN = "loose_call_vs_open"


@dataclass(frozen=True, slots=True)
class PresetDefinition:
    """A preset: its name, its textual range, and a human description."""

    name: RangePreset
    range_text: str
    description: str

    def __post_init__(self) -> None:
        if not self.range_text.strip():
            raise ValueError(f"range_text for preset {self.name.value!r} must not be empty")
        if not self.description.strip():
            raise ValueError(f"description for preset {self.name.value!r} must not be empty")


class RangePresetCatalog:
    """A registry of preset definitions with on-demand parsing.

    The catalogue parses each preset's range text the first time it is
    requested and caches the result for the lifetime of the catalogue
    instance. Parsing errors propagate as :class:`RangeParseError`.
    """

    def __init__(self, definitions: Iterable[PresetDefinition]) -> None:
        """Initialise the catalogue from an iterable of preset definitions.

        Args:
            definitions: The preset definitions to register. Order is
                preserved and exposed via :meth:`list_presets`.

        Raises:
            ValueError: If ``definitions`` is empty or contains two
                definitions with the same name.
        """
        materialised = list(definitions)
        if not materialised:
            raise ValueError("RangePresetCatalog requires at least one definition")

        by_name: dict[RangePreset, PresetDefinition] = {}
        for definition in materialised:
            if definition.name in by_name:
                raise ValueError(
                    f"Duplicate preset definition for {definition.name.value!r}",
                )
            by_name[definition.name] = definition

        self._definitions: dict[RangePreset, PresetDefinition] = by_name
        self._range_cache: dict[RangePreset, HandRange] = {}

    def has_preset(self, preset: RangePreset) -> bool:
        """Return ``True`` iff the catalogue contains ``preset``."""
        return preset in self._definitions

    def get_definition(self, preset: RangePreset) -> PresetDefinition:
        """Return the full definition for ``preset``.

        Raises:
            KeyError: If ``preset`` is not in the catalogue.
        """
        try:
            return self._definitions[preset]
        except KeyError as exc:
            raise KeyError(f"unknown range preset: {preset.value!r}") from exc

    def get_range(self, preset: RangePreset) -> HandRange:
        """Return the parsed :class:`HandRange` for ``preset``, parsing on first access."""
        cached = self._range_cache.get(preset)
        if cached is not None:
            return cached
        parsed = parse_range(self.get_definition(preset).range_text)
        self._range_cache[preset] = parsed
        return parsed

    def get_ranges(self, presets: Iterable[RangePreset]) -> tuple[HandRange, ...]:
        """Return the parsed ranges for every preset in ``presets``."""
        return tuple(self.get_range(preset) for preset in presets)

    def list_presets(self) -> list[RangePreset]:
        """Return the names of all presets in the catalogue, in registration order."""
        return list(self._definitions)

    def iter_definitions(self) -> tuple[PresetDefinition, ...]:
        """Return all preset definitions, in registration order."""
        return tuple(self._definitions.values())

    def union_of(self, *presets: RangePreset) -> HandRange:
        """Return the union of several presets as a single :class:`HandRange`.

        At least one preset must be passed.
        """
        if not presets:
            raise ValueError("union_of requires at least one preset")
        first, *rest = presets
        return self.get_range(first).union(*(self.get_range(p) for p in rest))

    @classmethod
    def default(cls) -> Self:
        """Build a catalogue from the bundled :data:`DEFAULT_PRESET_DEFINITIONS`."""
        return cls(DEFAULT_PRESET_DEFINITIONS)


# Bundled preset definitions. Each entry is a practical, simplified
# starting-hand range usable as a default in the decision engine; they
# are not meant to be solver-optimal.
DEFAULT_PRESET_DEFINITIONS: tuple[PresetDefinition, ...] = (
    PresetDefinition(
        name=RangePreset.TIGHT_OPEN,
        range_text="66+,ATs+,KJs+,QJs,JTs,AJo+,KQo",
        description=(
            "Tight opening range: strong pairs, strong broadway, " "a few suited connectors."
        ),
    ),
    PresetDefinition(
        name=RangePreset.LOOSE_OPEN,
        range_text="22+,A2s+,K9s+,Q9s+,J9s+,T8s+,98s,87s,76s,65s,ATo+,KTo+,QTo+,JTo",
        description=(
            "Loose opening range: every pair, a wide suited block, "
            "part of offsuit broadway and connectors."
        ),
    ),
    PresetDefinition(
        name=RangePreset.BTN_OPEN,
        range_text=(
            "22+,A2s+,K5s+,Q7s+,J8s+,T8s+,97s+,86s+,76s,65s,54s," "A8o+,K9o+,Q9o+,J9o+,T9o"
        ),
        description="Wide button opening range for 5-max / short-handed play.",
    ),
    PresetDefinition(
        name=RangePreset.SB_LIMP_CALL,
        range_text=("22-99,A2s-A9s,K7s+,Q8s+,J8s+,T8s+,97s+,87s,76s,65s," "A9o-AJo,KTo+,QTo+,JTo"),
        description=(
            "Simplified small-blind limp/call range: many medium-strength "
            "hands, suited hands, and part of broadway."
        ),
    ),
    PresetDefinition(
        name=RangePreset.BB_DEFEND,
        range_text=(
            "22+,A2s+,K5s+,Q7s+,J7s+,T7s+,96s+,86s+,75s+,64s+,54s," "A2o+,K8o+,Q9o+,J9o+,T9o,98o"
        ),
        description=(
            "Simplified big-blind defending range against late-position "
            "opens: wide suited and offsuit coverage."
        ),
    ),
    PresetDefinition(
        name=RangePreset.NITTY_CALL_VS_OPEN,
        range_text="22-JJ,AJs-AQs,KQs,QJs,JTs,T9s,98s,AQo",
        description=(
            "Tight calling range against an open: pairs, strong suited "
            "broadway, a few connectors."
        ),
    ),
    PresetDefinition(
        name=RangePreset.LOOSE_CALL_VS_OPEN,
        range_text=(
            "22-QQ,A2s-AQs,K9s+,Q9s+,J9s+,T8s+,98s,87s,76s,65s,54s," "A9o-AQo,KTo+,QTo+,JTo"
        ),
        description=(
            "Loose calling range against an open: a wide suited block, "
            "pairs, and part of offsuit broadway."
        ),
    ),
)


@cache
def default_catalog() -> RangePresetCatalog:
    """Return the lazily-initialised default :class:`RangePresetCatalog`.

    The catalogue is built once per process and shared across calls.
    """
    return RangePresetCatalog.default()
