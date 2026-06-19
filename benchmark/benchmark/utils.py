"""Cross-cutting helpers used by more than one harness module.

Deliberately domain-agnostic: anything that knows about MalSkillBench's on-disk
shape belongs in ``dataset_handler``; anything specific to one subsystem belongs
with that subsystem. Only genuinely shared, reusable helpers live here.
"""

from __future__ import annotations

import logging
import os
import pathlib
import warnings

_QUIETED = False


def quiet_logging() -> None:
    """Silence SkillSpector/langchain logging+warnings in a process (once).

    Used both inside a worker (before importing the graph) and as the
    ``ProcessPoolExecutor`` initializer in the parent.
    """
    global _QUIETED
    if _QUIETED:
        return
    warnings.filterwarnings("ignore")
    os.environ.setdefault("SKILLSPECTOR_LOG_LEVEL", "ERROR")
    # langchain/openai use an httpx.AsyncClient whose finalizer runs aclose()
    # after the worker's one-shot event loop has already closed, which asyncio
    # reports as "Task exception was never retrieved: Event loop is closed".
    # It's harmless teardown noise (the scan already returned its result), so
    # drop asyncio's ERROR-level records that clutter the progress bar.
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    _QUIETED = True


def materialized_path(root: pathlib.Path, filename: str) -> pathlib.Path:
    """Resolve `filename` to a path inside `root`, stripping any escape.

    Preserves subdirectories (SkillSpector rules key on path) but drops
    `..`/absolute components so a malicious filename can't write outside root.
    """
    parts = [p for p in pathlib.PurePosixPath(filename).parts if p not in ("", "/", "..")]
    return root.joinpath(*parts) if parts else root / "file"


def rel_id(path: pathlib.Path, anchor: pathlib.Path) -> str:
    """Stable id fragment: path relative to the dataset root when possible."""
    try:
        return str(path.relative_to(anchor))
    except ValueError:
        return path.name


def canonical_label(*parts: str | None) -> str | None:
    """Join non-empty taxonomy parts into a label like 'PI_B14'."""
    kept = [p for p in parts if p]
    return "_".join(kept) if kept else None
