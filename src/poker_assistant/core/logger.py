"""Logging configuration for the project.

The project uses the standard ``logging`` module without third-party wrappers.
Log messages are written in English so that they can be aggregated and
searched in standard observability stacks (ELK, Loki, etc.) without locale
concerns.

Usage
-----
At the start of any entry point (CLI, application bootstrap, test fixture)::

    from poker_assistant.core.logger import setup_logging
    setup_logging()

Inside library modules::

    import logging
    logger = logging.getLogger(__name__)
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging for the project.

    The function is idempotent: calling it more than once replaces existing
    handlers on the root logger so that the most recent invocation wins.
    This matters for tools that may be embedded in a host process which
    has already configured logging differently.

    Args:
        level: Numeric logging level (e.g. ``logging.INFO``).
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root.addHandler(handler)
    root.setLevel(level)
