"""runtime_monitor.py - Runtime behavior monitor.

Runtime-execution path removed pending a real OS-level sandbox.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .event_model import EventGraph

logger = logging.getLogger(__name__)


def run_isolated(skill_path: Path, timeout: float = 10.0) -> EventGraph:
    """Run skill in isolated sandbox and collect SecurityEvents.

    NOTE: Direct subprocess execution of untrusted skill code has been removed.
    A full OS-level sandbox is required before untrusted code can be executed.
    """
    logger.info("Runtime execution disabled pending OS-level sandbox: %s", skill_path)
    return EventGraph()


def collect_runtime_capabilities(graph: EventGraph) -> dict[str, bool]:
    """Summarize runtime capabilities observed."""
    caps = {
        "filesystem_read": any(
            e.category == "filesystem" and "read" in e.capability for e in graph.events
        ),
        "filesystem_write": any(
            e.category == "filesystem" and "write" in e.capability for e in graph.events
        ),
        "network": any(e.category in ("network", "outbound", "dns") for e in graph.events),
        "subprocess": any(
            e.category == "processes" and e.action == "execute" for e in graph.events
        ),
        "env": any(e.category == "env" for e in graph.events),
        "executables": any(e.category == "executables" for e in graph.events),
        "package": any(e.category == "package" for e in graph.events),
        "mcp": any(e.category == "mcp" for e in graph.events),
    }
    return caps
