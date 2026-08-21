# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision-resistant, occurrence-preserving finding compaction."""

from __future__ import annotations

from dataclasses import replace

from skillspector.logging_config import get_logger
from skillspector.models import Finding

logger = get_logger(__name__)


def _occurrences(finding: Finding) -> list[dict[str, object]]:
    if finding.occurrences:
        return [dict(item) for item in finding.occurrences]
    return [
        {
            "file": finding.file,
            "start_line": finding.start_line,
            "end_line": finding.end_line,
            "source_url": finding.source_url,
            "source_identity": finding.source_identity,
            "source_digest": finding.source_digest,
            "transitive_depth": finding.transitive_depth,
        }
    ]


def _line(value: object, default: int) -> int:
    return value if isinstance(value, int) else default


def _finding_source_scope(finding: Finding) -> str:
    """Return immutable provenance, including occurrence-only compatibility data."""
    direct = finding.source_identity or finding.source_digest or finding.source_url
    if direct:
        return direct
    for occurrence in finding.occurrences:
        candidate = (
            occurrence.get("source_identity")
            or occurrence.get("source_digest")
            or occurrence.get("source_url")
        )
        if candidate:
            return str(candidate)
    return ""


def deduplicate(findings: list[Finding]) -> list[Finding]:
    """Aggregate exact full-match duplicates while preserving every occurrence."""
    groups: dict[tuple[str, str, str], list[Finding]] = {}
    unique_without_match: list[Finding] = []
    for finding in findings:
        fingerprint = finding.fingerprint()
        if fingerprint is None:
            unique_without_match.append(finding)
            continue
        source_scope = _finding_source_scope(finding)
        groups.setdefault((source_scope, finding.rule_id, fingerprint), []).append(finding)

    compacted: list[Finding] = []
    for (_source_scope, _rule_id, fingerprint), group in groups.items():
        representative = max(
            group,
            key=lambda item: (
                item.confidence,
                -item.start_line,
                item.file,
                item.finding_id,
            ),
        )
        occurrences = {
            (
                str(occurrence.get("file", "")),
                _line(occurrence.get("start_line"), 1),
                occurrence.get("end_line"),
                str(occurrence.get("source_identity") or finding.source_identity or ""),
                str(occurrence.get("source_digest") or finding.source_digest or ""),
                str(occurrence.get("source_url") or finding.source_url or ""),
                _line(occurrence.get("transitive_depth"), finding.transitive_depth),
            )
            for finding in group
            for occurrence in _occurrences(finding)
        }
        ordered_occurrences = [
            {
                "file": file,
                "start_line": start,
                "end_line": end,
                **({"source_identity": source_identity} if source_identity else {}),
                **({"source_digest": source_digest} if source_digest else {}),
                **({"source_url": source_url} if source_url else {}),
                **({"transitive_depth": transitive_depth} if transitive_depth else {}),
            }
            for (
                file,
                start,
                end,
                source_identity,
                source_digest,
                source_url,
                transitive_depth,
            ) in sorted(
                occurrences,
                key=lambda item: (
                    item[3],
                    item[4],
                    item[5],
                    item[6],
                    item[0],
                    item[1],
                    _line(item[2], item[1]),
                ),
            )
        ]
        compacted.append(
            replace(
                representative,
                match_fingerprint=fingerprint,
                occurrences=ordered_occurrences,
            )
        )

    compacted.extend(unique_without_match)
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    compacted.sort(
        key=lambda finding: (
            severity_order.get(finding.severity.upper(), 4),
            finding.file,
            finding.start_line,
            finding.rule_id,
        )
    )
    removed = len(findings) - len(compacted)
    if removed:
        logger.info(
            "Deduplication: %d -> %d findings (%d exact duplicates aggregated)",
            len(findings),
            len(compacted),
            removed,
        )
    return compacted
