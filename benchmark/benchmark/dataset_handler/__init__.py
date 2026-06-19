"""Dataset handler registry and dispatch.

``discover(root)`` picks the first registered handler that recognizes the tree
and delegates to it. Register additional benchmarks in ``_HANDLERS`` (most
specific first); MalSkillBench is the default fallback.
"""

from __future__ import annotations

from pathlib import Path

from ..models import Unit
from .base import DatasetHandler
from .malskillbench import MalSkillBenchHandler

# Register dataset handlers here, most specific first.
_HANDLERS: list[DatasetHandler] = [MalSkillBenchHandler()]


def get_handler(root: Path) -> DatasetHandler:
    """Return the handler that recognizes ``root`` (defaults to MalSkillBench)."""
    for handler in _HANDLERS:
        if handler.matches(root):
            return handler
    return _HANDLERS[0]


def discover(root: Path) -> list[Unit]:
    """Collect all scannable units under ``root`` via the matching handler."""
    return get_handler(root).discover(root)


__all__ = ["DatasetHandler", "discover", "get_handler"]
