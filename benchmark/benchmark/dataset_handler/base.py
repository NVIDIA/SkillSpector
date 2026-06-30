"""The dataset abstraction.

A *dataset handler* turns an on-disk benchmark tree into a flat list of
``Unit`` objects (each with its resolved ground-truth label) that the runner
can scan. Today only MalSkillBench is implemented; to add another benchmark,
subclass ``DatasetHandler``, implement ``matches``/``discover``, and register
the new handler in ``dataset_handler/__init__.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import Unit


class DatasetHandler(ABC):
    """Interface for turning a dataset tree into scannable units."""

    #: Human-readable handler name (for logs/diagnostics).
    name: str = "dataset"

    @abstractmethod
    def matches(self, root: Path) -> bool:
        """Return True if this handler recognizes the tree rooted at ``root``."""

    @abstractmethod
    def discover(self, root: Path) -> list[Unit]:
        """Collect all scannable units under ``root``."""
