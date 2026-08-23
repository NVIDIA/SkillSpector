# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact-level evasion signals derived from canonical byte classification."""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass, field

from skillspector.artifacts import ContentKind
from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.models import Finding
from skillspector.python_ast import MAX_PYTHON_AST_SOURCE_CHARS
from skillspector.state import (
    AnalyzerNodeResponse,
    SkillspectorState,
    transitive_remaining_seconds,
)

from .static_runner import MAX_FINDINGS_PER_ANALYZER, MAX_FINDINGS_PER_ARTIFACT

ANALYZER_ID = "artifact_integrity"
_INSTRUCTION_SUFFIXES = (
    ".md",
    ".markdown",
    ".txt",
)
_RUNTIME_CHECK_INTERVAL_CHARS = 4096
_ALLOWED_FORMAT_CHARACTERS = frozenset({"\n", "\r", "\t"})


class _ArtifactIntegrityResourceLimitError(RuntimeError):
    """Stop attacker-controlled work while retaining a deterministic prefix."""

    def __init__(self, reason: LedgerReason, metrics: dict[str, int | float]) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.metrics = metrics


@dataclass
class _ArtifactIntegrityBudget:
    """Enforce one shared deadline and construction-time finding ceilings."""

    state: SkillspectorState
    started_at: float = field(default_factory=time.monotonic)
    initial_allowance: float | None = None
    findings: list[Finding] = field(default_factory=list)
    artifact_findings: dict[str, int] = field(default_factory=dict)

    def check_runtime(self) -> None:
        remaining = transitive_remaining_seconds(self.state)
        if remaining is None:
            return
        if self.initial_allowance is None:
            self.initial_allowance = max(0.0, remaining)
        if remaining <= 0:
            raise _ArtifactIntegrityResourceLimitError(
                LedgerReason.RUNTIME_LIMIT,
                {
                    "observed_seconds": max(0.0, time.monotonic() - self.started_at),
                    "limit_seconds": self.initial_allowance,
                },
            )

    def emit(self, finding: Finding) -> None:
        """Append one finding only after checking both relevant ceilings."""
        self.check_runtime()
        artifact_observed = self.artifact_findings.get(finding.file, 0) + 1
        analyzer_observed = len(self.findings) + 1
        if artifact_observed > MAX_FINDINGS_PER_ARTIFACT:
            raise _ArtifactIntegrityResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": artifact_observed,
                    "limit_findings": MAX_FINDINGS_PER_ARTIFACT,
                },
            )
        if analyzer_observed > MAX_FINDINGS_PER_ANALYZER:
            raise _ArtifactIntegrityResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": analyzer_observed,
                    "limit_findings": MAX_FINDINGS_PER_ANALYZER,
                },
            )
        self.findings.append(finding)
        self.artifact_findings[finding.file] = artifact_observed

    def analyzer_exhausted(self) -> bool:
        """Return whether inspecting another artifact could exceed the cap."""
        return len(self.findings) >= MAX_FINDINGS_PER_ANALYZER


def _text_signals(
    content: str,
    budget: _ArtifactIntegrityBudget,
) -> tuple[float, bool, int | None]:
    """Derive Unicode and NUL signals with cooperative deadline checks.

    Only counters, a three-entry script set, and the first NUL line are kept;
    attacker-controlled text is never copied into match/evidence structures.
    """
    ignored_characters = 0
    mixed_script = False
    token_scripts: set[str] = set()
    line = 1
    first_nul_line: int | None = None

    for index, character in enumerate(content):
        if index % _RUNTIME_CHECK_INTERVAL_CHARS == 0:
            budget.check_runtime()
        category = unicodedata.category(character)
        if character == "\u00ad" or (
            category in {"Cf", "Cc"} and character not in _ALLOWED_FORMAT_CHARACTERS
        ):
            ignored_characters += 1

        if character == "\x00" and first_nul_line is None:
            first_nul_line = line
        if character == "\n":
            line += 1

        if character.isascii() and character.isalpha():
            token_scripts.add("latin")
        elif character.isalpha():
            name = unicodedata.name(character, "")
            if "CYRILLIC" in name:
                token_scripts.add("cyrillic")
            elif "GREEK" in name:
                token_scripts.add("greek")
        elif character.isalnum() or character in {"_", "-"}:
            continue
        else:
            if "latin" in token_scripts and len(token_scripts) > 1:
                mixed_script = True
            token_scripts.clear()

    budget.check_runtime()
    mixed_script = mixed_script or ("latin" in token_scripts and len(token_scripts) > 1)
    density = ignored_characters / len(content) if content else 0.0
    return density, mixed_script, first_nul_line


def _partial_limit_event(
    path: str,
    limit: _ArtifactIntegrityResourceLimitError,
    emitted_finding_ids: list[str] | None = None,
) -> InspectionLedgerEvent:
    """Account one current or unstarted artifact as explicitly partial."""
    return ledger_event(
        analyzer_id=ANALYZER_ID,
        outcome=LedgerOutcome.PARTIAL,
        phase="artifact",
        path=path,
        reason=limit.reason,
        emitted_finding_ids=emitted_finding_ids or (),
        observed_findings=(
            int(limit.metrics["observed_findings"])
            if limit.reason is LedgerReason.OUTPUT_LIMIT
            else None
        ),
        limit_findings=(
            int(limit.metrics["limit_findings"])
            if limit.reason is LedgerReason.OUTPUT_LIMIT
            else None
        ),
        observed_seconds=(
            float(limit.metrics["observed_seconds"])
            if limit.reason is LedgerReason.RUNTIME_LIMIT
            else None
        ),
        limit_seconds=(
            float(limit.metrics["limit_seconds"])
            if limit.reason is LedgerReason.RUNTIME_LIMIT
            else None
        ),
    )


def _finding(
    rule_id: str,
    message: str,
    path: str,
    *,
    severity: str,
    confidence: float,
    line: int = 1,
) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=message,
        severity=severity,
        confidence=confidence,
        file=path,
        start_line=line,
        category="analysis-evasion",
        tags=["artifact-integrity"],
    )


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Emit classification, Unicode, and analysis-ceiling evasion findings."""
    file_cache = state.get("local_file_cache") or state.get("file_cache") or {}
    budget = _ArtifactIntegrityBudget(state)
    inventory: dict[str, object] = {}
    events: list[InspectionLedgerEvent] = []
    terminal_limit: _ArtifactIntegrityResourceLimitError | None = None

    try:
        budget.check_runtime()
        for item in state.get("artifact_inventory") or []:
            budget.check_runtime()
            if isinstance(item, dict):
                inventory[str(item.get("path", ""))] = item
    except _ArtifactIntegrityResourceLimitError as exc:
        terminal_limit = exc

    components = state.get("components") or []
    for path in components:
        if terminal_limit is None and budget.analyzer_exhausted():
            terminal_limit = _ArtifactIntegrityResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": len(budget.findings) + 1,
                    "limit_findings": MAX_FINDINGS_PER_ANALYZER,
                },
            )
        if terminal_limit is not None:
            events.append(_partial_limit_event(path, terminal_limit))
            continue

        raw_artifact = inventory.get(path)
        artifact: dict[str, object] = raw_artifact if isinstance(raw_artifact, dict) else {}
        finding_start = len(budget.findings)
        resource_limit: _ArtifactIntegrityResourceLimitError | None = None
        try:
            budget.check_runtime()
            if artifact.get("misleading_extension"):
                budget.emit(
                    _finding(
                        "AE2",
                        "Artifact content does not match its filename extension",
                        path,
                        severity="MEDIUM",
                        confidence=0.9,
                    )
                )
            content = file_cache.get(path)
            if content is not None:
                normalized_path = path.lower()
                if len(content) > MAX_PYTHON_AST_SOURCE_CHARS and (
                    normalized_path.endswith(_INSTRUCTION_SUFFIXES)
                    or normalized_path.endswith("skill.md")
                ):
                    budget.emit(
                        _finding(
                            "AE5",
                            "Instruction-capable artifact exceeds whole-file semantic analysis limits",
                            path,
                            severity="HIGH",
                            confidence=1.0,
                        )
                    )
                if artifact.get("content_kind") not in {
                    ContentKind.BINARY,
                    ContentKind.OPAQUE,
                }:
                    format_density, mixed_script, first_nul_line = _text_signals(content, budget)
                    if artifact.get("contains_nul") and first_nul_line is not None:
                        budget.emit(
                            _finding(
                                "AE3",
                                "Text artifact contains embedded NUL bytes",
                                path,
                                severity="HIGH",
                                confidence=0.9,
                                line=first_nul_line,
                            )
                        )
                    if format_density >= 0.01 or mixed_script:
                        budget.emit(
                            _finding(
                                "AE4",
                                "Suspicious Unicode normalization or mixed-script content",
                                path,
                                severity="MEDIUM",
                                confidence=0.8,
                            )
                        )
        except _ArtifactIntegrityResourceLimitError as exc:
            resource_limit = exc

        path_findings = budget.findings[finding_start:]
        emitted_ids = [finding.finding_id for finding in path_findings]
        if resource_limit is not None:
            event = _partial_limit_event(path, resource_limit, emitted_ids)
            terminal_limit = resource_limit
        else:
            event = ledger_event(
                analyzer_id=ANALYZER_ID,
                outcome=LedgerOutcome.COMPLETED,
                phase="artifact",
                path=path,
                emitted_finding_ids=emitted_ids,
            )
        events.append(event)
    return {
        "findings": budget.findings,
        "inspection_ledger": events,
        "analyzer_status_events": [analyzer_status_for_events(ANALYZER_ID, events)],
    }
