"""poker_assistant: live poker assistant with a screen overlay."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("poker_assistant")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+local"

__all__ = ["__version__"]