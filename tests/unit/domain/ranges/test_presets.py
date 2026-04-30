"""Tests for poker_assistant.domain.ranges.presets."""

from __future__ import annotations

import pytest

from poker_assistant.domain.ranges.presets import (
    DEFAULT_PRESET_DEFINITIONS,
    PresetDefinition,
    RangePreset,
    RangePresetCatalog,
    default_catalog,
)


class TestPresetDefinition:
    def test_rejects_empty_range_text(self) -> None:
        with pytest.raises(ValueError, match="range_text"):
            PresetDefinition(
                name=RangePreset.TIGHT_OPEN,
                range_text="   ",
                description="any",
            )

    def test_rejects_empty_description(self) -> None:
        with pytest.raises(ValueError, match="description"):
            PresetDefinition(
                name=RangePreset.TIGHT_OPEN,
                range_text="AA",
                description="   ",
            )


class TestRangePresetCatalog:
    def test_default_catalog_contains_every_preset(self) -> None:
        catalog = RangePresetCatalog.default()
        for preset in RangePreset:
            assert catalog.has_preset(preset)

    def test_default_catalog_parses_every_preset(self) -> None:
        catalog = RangePresetCatalog.default()
        for preset in RangePreset:
            parsed = catalog.get_range(preset)
            assert not parsed.is_empty

    def test_get_range_caches_result(self) -> None:
        catalog = RangePresetCatalog.default()
        first = catalog.get_range(RangePreset.TIGHT_OPEN)
        second = catalog.get_range(RangePreset.TIGHT_OPEN)
        assert first is second

    def test_get_definition_returns_full_record(self) -> None:
        catalog = RangePresetCatalog.default()
        definition = catalog.get_definition(RangePreset.TIGHT_OPEN)
        assert definition.name is RangePreset.TIGHT_OPEN
        assert definition.range_text
        assert definition.description

    def test_get_ranges_returns_tuple_in_input_order(self) -> None:
        catalog = RangePresetCatalog.default()
        order = (RangePreset.LOOSE_OPEN, RangePreset.TIGHT_OPEN)
        ranges = catalog.get_ranges(order)
        assert ranges[0] is catalog.get_range(RangePreset.LOOSE_OPEN)
        assert ranges[1] is catalog.get_range(RangePreset.TIGHT_OPEN)

    def test_list_presets_preserves_registration_order(self) -> None:
        catalog = RangePresetCatalog.default()
        listed = catalog.list_presets()
        expected = [definition.name for definition in DEFAULT_PRESET_DEFINITIONS]
        assert listed == expected

    def test_iter_definitions_matches_list(self) -> None:
        catalog = RangePresetCatalog.default()
        names_from_iter = [d.name for d in catalog.iter_definitions()]
        assert names_from_iter == catalog.list_presets()

    def test_union_of_combines_distinct_classes(self) -> None:
        catalog = RangePresetCatalog.default()
        merged = catalog.union_of(RangePreset.TIGHT_OPEN, RangePreset.LOOSE_OPEN)
        tight = catalog.get_range(RangePreset.TIGHT_OPEN)
        loose = catalog.get_range(RangePreset.LOOSE_OPEN)
        assert merged.total_hand_classes() >= max(
            tight.total_hand_classes(),
            loose.total_hand_classes(),
        )

    def test_union_of_with_no_presets_raises(self) -> None:
        catalog = RangePresetCatalog.default()
        with pytest.raises(ValueError, match="at least one preset"):
            catalog.union_of()

    def test_constructor_rejects_empty_definitions(self) -> None:
        with pytest.raises(ValueError, match="at least one definition"):
            RangePresetCatalog([])

    def test_constructor_rejects_duplicate_definitions(self) -> None:
        definition = DEFAULT_PRESET_DEFINITIONS[0]
        with pytest.raises(ValueError, match="Duplicate"):
            RangePresetCatalog([definition, definition])


class TestDefaultCatalog:
    def test_returns_shared_instance(self) -> None:
        assert default_catalog() is default_catalog()
