"""Data models shared across the benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePath


@dataclass
class Unit:
    """One unit to scan, with its resolved ground-truth metadata."""

    unit_path: str  # stable id (relative to the Dataset root); the join key
    category: str  # skill | code | prompt
    source_path: str  # on-disk location (dir, or file#index for materialized)
    display_name: str
    is_malicious: bool  # ground truth (always resolvable here)
    label: str | None = None  # fine taxonomy, set only when authoritatively known
    best_guess_label: str | None = None  # heuristic fallback when not confident
    label_source: str = "unknown"  # inventory | classified | field | name | dir | corpus
    attack_vector: str | None = None  # CI | PI | MIXED
    behavior: str | None = None  # B1..B15
    insertion_strategy: str | None = None
    corpus: str | None = None
    sample_type: str | None = None  # GENERATED | WILD | TEST
    # files to materialize into a temp dir before scanning; empty => scan
    # source_path in place (the Skills case).
    materialize: dict[str, str] = field(default_factory=dict)

    @property
    def group(self) -> str:
        """The unit's dataset grouping: the parent directory of its stable id.

        Used by ``--limit`` to cap per grouping. e.g. a skill at
        ``Skills/malware/evil__PI_B2`` groups under ``Skills/malware`` (so
        malware and benign are capped independently); a code/prompt record's
        ``#<index>`` suffix is stripped first, so all records from one source
        file share that file's directory as their group.
        """
        base = self.unit_path.rsplit("#", 1)[0]
        return str(PurePath(base).parent)


@dataclass
class ScanResult:
    """Outcome of scanning a single unit."""

    unit: Unit
    scan_status: str  # ok | error | timeout
    scan_seconds: float = 0.0
    risk_score: int | None = None
    risk_severity: str | None = None
    risk_recommendation: str | None = None
    num_issues: int = 0
    llm_requested: bool = False
    llm_available: bool = False
    error_message: str | None = None
    issues: list[dict] = field(default_factory=list)
    components: list[dict] = field(default_factory=list)
