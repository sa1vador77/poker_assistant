"""Smoke test: the package imports and exposes a version string."""

from __future__ import annotations

import re

import poker_assistant


def test_package_exposes_version() -> None:
    assert isinstance(poker_assistant.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", poker_assistant.__version__)
