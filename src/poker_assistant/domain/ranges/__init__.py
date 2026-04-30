"""Public API of the ranges package."""

from __future__ import annotations

from poker_assistant.domain.ranges.models import (
    HOLE_CARDS_PER_HAND,
    MAX_BOARD_CARDS,
    ComboShape,
    HandClass,
    HandRange,
    HoleCombo,
    RangeItem,
    WeightedCombo,
    all_hand_classes,
    dead_cards_from_known_cards,
)
from poker_assistant.domain.ranges.parser import (
    ParsedToken,
    RangeParseError,
    parse_range,
    parse_range_token,
)
from poker_assistant.domain.ranges.presets import (
    DEFAULT_PRESET_DEFINITIONS,
    PresetDefinition,
    RangePreset,
    RangePresetCatalog,
    default_catalog,
)

__all__ = [
    "DEFAULT_PRESET_DEFINITIONS",
    "HOLE_CARDS_PER_HAND",
    "MAX_BOARD_CARDS",
    "ComboShape",
    "HandClass",
    "HandRange",
    "HoleCombo",
    "ParsedToken",
    "PresetDefinition",
    "RangeItem",
    "RangeParseError",
    "RangePreset",
    "RangePresetCatalog",
    "WeightedCombo",
    "all_hand_classes",
    "dead_cards_from_known_cards",
    "default_catalog",
    "parse_range",
    "parse_range_token",
]
