# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small independent reference model for the risk-scoring contract.

This model deliberately does not import production scoring constants or helpers. A
production change must therefore update this readable specification deliberately,
rather than making differential tests agree through shared implementation details.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReferenceFinding:
    """Only the finding fields that influence the risk score."""

    rule_id: str = "UNKNOWN"
    severity: str = "LOW"
    confidence: float = 0.5
    file: str = "SKILL.md"
    source_identity: str | None = None
    source_url: str | None = None
    source_digest: str | None = None
    evidence: Mapping[str, object] = field(default_factory=dict)


_POINTS = {"CRITICAL": 50, "HIGH": 25, "MEDIUM": 10, "LOW": 5}
_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
_WEIGHTS = (1.0, 0.5, 0.25)
_BANDS = ((81, "CRITICAL"), (51, "HIGH"), (21, "MEDIUM"), (0, "LOW"))
_RECOMMENDATIONS = {
    "CRITICAL": "DO_NOT_INSTALL",
    "HIGH": "DO_NOT_INSTALL",
    "MEDIUM": "CAUTION",
    "LOW": "SAFE",
}


def _confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def _source_scope(item: ReferenceFinding | Mapping[str, object]) -> str:
    for key in ("source_identity", "source_url", "source_digest"):
        value = getattr(item, key, None) if isinstance(item, ReferenceFinding) else item.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    return ""


def _score_floor(finding: ReferenceFinding) -> int:
    if finding.rule_id == "SC8":
        return 51
    if finding.severity.upper() != "CRITICAL":
        return 0
    if finding.evidence.get("activation_state") != "conditional":
        return 0
    if finding.rule_id == "BH2" and finding.evidence.get("proof_status") == "closed":
        return 51
    if finding.rule_id == "BH3":
        return 51
    return 0


def reference_risk_score(
    findings: Sequence[ReferenceFinding],
    has_executable_scripts: bool,
    component_metadata: Sequence[Mapping[str, object]] = (),
) -> tuple[int, str, str]:
    """Evaluate the documented scoring contract without production helpers."""
    executable = {
        (_source_scope(component), str(component.get("path", ""))): bool(
            component.get("executable", False)
        )
        for component in component_metadata
    }
    ordered = sorted(
        findings,
        key=lambda item: (
            item.rule_id or "UNKNOWN",
            _SEVERITY_ORDER.get(item.severity.upper() if item.severity else "LOW", 4),
            -_confidence(item.confidence),
            -int(executable.get((_source_scope(item), item.file), False)),
        ),
    )
    occurrences: dict[str, int] = {}
    score = 0.0
    floor = 0
    for finding in ordered:
        confidence = _confidence(finding.confidence)
        if confidence <= 0.0:
            continue
        floor = max(floor, _score_floor(finding))
        rule_id = finding.rule_id or "UNKNOWN"
        index = occurrences.get(rule_id, 0)
        occurrences[rule_id] = index + 1
        if index >= len(_WEIGHTS):
            continue
        contribution = _POINTS.get(finding.severity.upper(), 5) * _WEIGHTS[index] * confidence
        if has_executable_scripts and executable.get((_source_scope(finding), finding.file), False):
            contribution *= 1.3
        score += contribution

    final = min(100, max(floor, int(score)))
    band = next(label for threshold, label in _BANDS if final >= threshold)
    return final, band, _RECOMMENDATIONS[band]
