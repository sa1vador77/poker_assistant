"""Tests for the project logging bootstrap."""

from __future__ import annotations

import logging

from poker_assistant.core.logger import setup_logging


def test_setup_logging_replaces_handlers() -> None:
    setup_logging(level=logging.WARNING)
    root = logging.getLogger()
    assert root.level == logging.WARNING
    handlers_after_first = list(root.handlers)
    assert len(handlers_after_first) == 1

    setup_logging(level=logging.DEBUG)
    handlers_after_second = list(root.handlers)
    assert root.level == logging.DEBUG
    assert len(handlers_after_second) == 1
    assert handlers_after_second[0] is not handlers_after_first[0]
