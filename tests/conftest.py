"""Pytest-wide configuration."""

from __future__ import annotations

import logging

from poker_assistant.core.logger import setup_logging


def pytest_configure() -> None:
    """Configure logging once for the whole test session."""
    setup_logging(level=logging.INFO)
