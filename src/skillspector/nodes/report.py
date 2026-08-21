# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Report node for Skillspector workflow.

Generates SARIF, computes risk score, and produces report_body in terminal/json/markdown/sarif
based on state["output_format"]. Single place for formatting (CLI and REST API reuse).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
from typing import Literal

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from skillspector import __version__ as skillspector_version
from skillspector.inference_usage import sanitize_inference_usage
from skillspector.inspection_ledger import MAX_FINDING_OUTPUT_RECORDS, AnalysisCompleteness
from skillspector.llm_utils import is_llm_available
from skillspector.logging_config import get_logger
from skillspector.models import Finding
from skillspector.nodes.deduplicate import deduplicate
from skillspector.python_ast import clear_python_ast_cache
from skillspector.sarif_models import (
    SARIF_SCHEMA_URI,
    SarifArtifactLocation,
    SarifDriver,
    SarifInvocation,
    SarifLocation,
    SarifLog,
    SarifMessage,
    SarifNotification,
    SarifPhysicalLocation,
    SarifRegion,
    SarifReportingDescriptor,
    SarifResult,
    SarifRun,
    SarifSuppression,
    SarifTool,
    validate_sarif_report,
)
from skillspector.state import SkillspectorState
from skillspector.suppression import Baseline, SuppressedFinding, partition_findings

logger = get_logger(__name__)

# Risk bands (v1 alignment)
_RISK_SEVERITY_BANDS = [(81, "CRITICAL"), (51, "HIGH"), (21, "MEDIUM"), (0, "LOW")]
_RISK_RECOMMENDATION: dict[str, str] = {
    "LOW": "SAFE",
    "MEDIUM": "CAUTION",
    "HIGH": "DO_NOT_INSTALL",
    "CRITICAL": "DO_NOT_INSTALL",
}


# Scanned skill content (and LLM output that quotes it) can carry ANSI escape
# sequences and control bytes (e.g. NUL, ESC) into finding text. Left in, they
# make a rendered report register as binary — GitLab/editors show "download"
# instead of the Markdown, and terminals emit garbled output. Strip them from
# every finding text field so all report formats stay clean UTF-8.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Finding free-text fields that may carry scanned/LLM content.
_SANITIZED_FIELDS = (
    "message",
    "explanation",
    "remediation",
    "finding",
    "context",
    "matched_text",
    "code_snippet",
)

_REPORT_SAFE_SOURCE_ID_RE = re.compile(r"external/[0-9a-f]{64}\Z")


def _clean_text(value: str | None) -> str | None:
    """Strip ANSI escape sequences and disallowed control chars (keep tab/newline)."""
    if not isinstance(value, str):
        return value
    return _CONTROL_RE.sub("", _ANSI_RE.sub("", value))


def _sanitize_finding(finding: Finding) -> Finding:
    """Return a copy of *finding* with control/ANSI bytes stripped from text fields."""
    evidence = {
        _clean_text(str(key)) or "": _clean_text(value) if isinstance(value, str) else value
        for key, value in finding.evidence.items()
    }
    return replace(
        finding,
        message=_clean_text(finding.message) or "",
        explanation=_clean_text(finding.explanation),
        remediation=_clean_text(finding.remediation),
        finding=_clean_text(finding.finding),
        context=_clean_text(finding.context),
        matched_text=_clean_text(finding.matched_text),
        code_snippet=_clean_text(finding.code_snippet),
        evidence=evidence,
    )


def _occurrence_provenance(
    finding: Finding, occurrence: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Return occurrence-level provenance, falling back to the finding scope."""
    occurrence = occurrence or {}
    provenance: dict[str, object] = {}
    for key, fallback in (
        ("source_identity", finding.source_identity),
        ("source_digest", finding.source_digest),
        ("source_url", finding.source_url),
    ):
        value = occurrence.get(key, fallback)
        if isinstance(value, str) and value:
            provenance[key] = value
    depth = occurrence.get("transitive_depth", finding.transitive_depth)
    if isinstance(depth, int) and not isinstance(depth, bool) and depth > 0:
        provenance["transitive_depth"] = depth
    return provenance


def _report_source_identity(provenance: Mapping[str, object]) -> str | None:
    """Return a safe opaque SARIF scope for source provenance."""
    identity = provenance.get("source_identity")
    if isinstance(identity, str) and identity:
        if _REPORT_SAFE_SOURCE_ID_RE.fullmatch(identity):
            return identity
        return f"external/{sha256(identity.encode()).hexdigest()}"
    digest = provenance.get("source_digest")
    source_url = provenance.get("source_url")
    if not digest and not source_url:
        return None
    canonical = json.dumps(
        {"source_digest": digest or "", "source_url": source_url or ""},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"external/{sha256(canonical.encode()).hexdigest()}"


def _sarif_artifact_location(
    finding: Finding, occurrence: Mapping[str, object] | None = None
) -> SarifArtifactLocation:
    """Build a source-scoped SARIF artifact location for one occurrence."""
    occurrence = occurrence or {}
    file_path = str(occurrence.get("file", finding.file)).replace("\\", "/").lstrip("/")
    provenance = _occurrence_provenance(finding, occurrence)
    source_identity = _report_source_identity(provenance)
    if source_identity:
        uri = f"{source_identity}/{file_path}"
    else:
        uri = str(occurrence.get("file", finding.file))
    properties = {
        {
            "source_identity": "sourceIdentity",
            "source_digest": "sourceDigest",
            "source_url": "sourceUrl",
            "transitive_depth": "transitiveDepth",
        }[key]: value
        for key, value in provenance.items()
    }
    return SarifArtifactLocation(uri=uri, properties=properties or None)


def _expand_occurrences(findings: list[Finding]) -> list[Finding]:
    """Expand compacted findings for human/JSON output without losing locations."""
    expanded: list[Finding] = []
    for finding in findings:
        occurrences = finding.occurrences or [
            {
                "file": finding.file,
                "start_line": finding.start_line,
                "end_line": finding.end_line,
            }
        ]
        for occurrence in occurrences:
            start_value = occurrence.get("start_line", finding.start_line)
            start_line = start_value if isinstance(start_value, int) else finding.start_line
            end_value = occurrence.get("end_line")
            end_line = end_value if isinstance(end_value, int) else None
            provenance = _occurrence_provenance(finding, occurrence)
            depth_value = provenance.get("transitive_depth")
            expanded.append(
                replace(
                    finding,
                    file=str(occurrence.get("file", finding.file)),
                    start_line=start_line,
                    end_line=end_line,
                    source_identity=(
                        str(provenance["source_identity"])
                        if "source_identity" in provenance
                        else None
                    ),
                    source_digest=(
                        str(provenance["source_digest"]) if "source_digest" in provenance else None
                    ),
                    source_url=(
                        str(provenance["source_url"]) if "source_url" in provenance else None
                    ),
                    transitive_depth=depth_value if isinstance(depth_value, int) else 0,
                    occurrences=[],
                )
            )
    return expanded


def _finding_output_record_count(findings: Sequence[Finding]) -> int:
    """Count compact findings using their public occurrence-record footprint."""
    return sum(max(1, len(finding.occurrences)) for finding in findings)


def _bounded_report_findings(
    findings: list[Finding],
    *,
    limit: int | None = None,
) -> list[Finding]:
    """Bound compact findings and their occurrence records in severity order."""
    bounded: list[Finding] = []
    remaining = max(0, MAX_FINDING_OUTPUT_RECORDS if limit is None else limit)
    for finding in findings:
        if remaining <= 0:
            break
        occurrences = finding.occurrences
        record_count = max(1, len(occurrences))
        if record_count > remaining:
            if not occurrences:
                break
            bounded.append(replace(finding, occurrences=occurrences[:remaining]))
            remaining = 0
            break
        bounded.append(finding)
        remaining -= record_count
    return bounded


def _bounded_suppressed_findings(
    findings: list[SuppressedFinding],
    *,
    limit: int,
) -> list[SuppressedFinding]:
    """Bound suppressed findings with the same occurrence-record accounting."""
    bounded: list[SuppressedFinding] = []
    remaining = max(0, limit)
    for suppressed in findings:
        if remaining <= 0:
            break
        finding = suppressed.finding
        occurrences = finding.occurrences
        record_count = max(1, len(occurrences))
        if record_count > remaining:
            if not occurrences:
                break
            bounded.append(
                replace(
                    suppressed,
                    finding=replace(finding, occurrences=occurrences[:remaining]),
                )
            )
            remaining = 0
            break
        bounded.append(suppressed)
        remaining -= record_count
    return bounded


def _build_sarif_properties(
    finding: Finding, occurrence: Mapping[str, object] | None = None
) -> dict[str, object] | None:
    """Project selected finding metadata into a SARIF properties dictionary."""
    finding_dict = finding.to_dict()
    metadata: dict[str, object] = {
        "findingId": finding.finding_id,
        "severity": finding_dict["severity"],
        "category": finding_dict["category"],
        "pattern": finding_dict["pattern"],
        "confidence": finding_dict["confidence"],
        "finding": finding_dict["finding"],
        "explanation": finding_dict["explanation"],
        "remediation": finding_dict["remediation"],
        "code_snippet": finding_dict["code_snippet"],
        "intent": finding_dict["intent"],
        "tags": finding_dict["tags"],
        "evidence": finding_dict["evidence"],
    }
    provenance = _occurrence_provenance(finding, occurrence)
    for key, sarif_key in (
        ("source_identity", "sourceIdentity"),
        ("source_digest", "sourceDigest"),
        ("source_url", "sourceUrl"),
        ("transitive_depth", "transitiveDepth"),
    ):
        if key in provenance:
            metadata[sarif_key] = provenance[key]
    cleaned = {key: value for key, value in metadata.items() if value is not None}
    return cleaned or None


def _sanitize_summary_value(value: object) -> object:
    """Return a recursively sanitized copy of structured-summary content."""
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        return [_sanitize_summary_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_summary_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_summary_value(item) for key, item in value.items()}
    return value


def _sanitize_structured_summary(summary: dict[str, object]) -> dict[str, object]:
    """Return a structured summary with control/ANSI bytes stripped from text fields."""
    return {str(key): _sanitize_summary_value(value) for key, value in summary.items()}


def _severity_to_sarif_level(severity: str) -> Literal["error", "warning", "note"]:
    """Map Finding.severity to SARIF result level."""
    return {
        "CRITICAL": "error",
        "HIGH": "error",
        "MEDIUM": "warning",
        "LOW": "note",
    }.get(severity.upper(), "note")  # type: ignore[return-value]


def _summary_display_value(value: object) -> str | None:
    """Format a summary field for terminal / markdown output."""
    if value is None:
        return None
    if isinstance(value, list):
        values = [str(item) for item in value if str(item)]
        return ", ".join(values) if values else None
    text = str(value)
    return text or None


def _structured_summary_notification(summary: dict[str, object]) -> str:
    """Build a note-level SARIF notification message for a structured summary."""
    summary_id = str(summary.get("id") or "SSR")
    message = str(summary.get("message") or "Structured skill summary")
    bits = [f"{summary_id}: {message}"]

    file = _summary_display_value(summary.get("file"))
    if file:
        bits.append(f"file={file}")
    protocol = _summary_display_value(summary.get("protocol"))
    if protocol:
        bits.append(f"protocol={protocol}")
    layout_kind = _summary_display_value(summary.get("layout_kind"))
    if layout_kind:
        bits.append(f"layout={layout_kind}")

    return " | ".join(bits)


_SEVERITY_POINTS: dict[str, int] = {
    "CRITICAL": 50,
    "HIGH": 25,
    "MEDIUM": 10,
    "LOW": 5,
}

# Ranking used only to report the worst finding alongside the normalized verdict.
# Deliberately separate from _SEVERITY_POINTS: those are scoring weights and may be
# retuned, while this is an ordering and must stay stable for consumers.
_SEVERITY_RANK: dict[str, int] = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def _max_issue_severity(findings: Sequence[Finding]) -> str:
    """Highest severity present in the reported issues, or NONE when there are none.

    risk_assessment.severity is a normalized, confidence-weighted verdict: it can read
    LOW while issues[] contains a HIGH finding. That smoothing is intentional, but until
    now it was invisible, so every consumer that wanted to gate on the worst finding had
    to walk issues[] and re-implement this ranking. Reporting it here makes the two
    numbers comparable without changing either.
    """
    worst = 0
    label = "NONE"
    for finding in findings:
        rank = _SEVERITY_RANK.get(str(finding.severity).upper(), 0)
        if rank > worst:
            worst = rank
            label = str(finding.severity).upper()
    return label


_MAX_OCCURRENCES_PER_RULE = 3
_DIMINISHING_WEIGHTS = (1.0, 0.5, 0.25)

# Some findings describe artifacts whose unanalyzed contents can execute. Their
# presence must block installation even when ordinary confidence-weighted,
# per-rule scoring would otherwise keep the aggregate below the CLI threshold.
_RISK_SCORE_FLOORS_BY_RULE_ID = {"SC8": 51}


def _compute_risk_score(
    findings: list[Finding],
    has_executable_scripts: bool,
    component_metadata: list[dict[str, object]] | None = None,
) -> tuple[int, str, str]:
    """
    Compute risk score (0-100), severity band, and recommendation.

    Scoring uses per-rule diminishing returns: the first occurrence of a rule_id
    contributes full points, the second contributes half, and the third contributes
    a quarter. Occurrences beyond the third are ignored for scoring purposes.
    This prevents repeated pattern matches from inflating the score unboundedly.

    Each finding's contribution is also scaled by its confidence value (clamped
    to [0, 1]). Findings with confidence <= 0 are skipped entirely — they do not
    contribute to the score but remain in the reported findings list.

    Within each rule_id bucket, findings are processed in severity-descending
    order so that the highest-severity occurrence always receives the full weight.

    Base points per severity: CRITICAL=50, HIGH=25, MEDIUM=10, LOW=5.
    1.3x multiplier applied only to findings from executable script files;
    findings from documentation files (markdown, text, json, yaml, toml)
    are scored at base weight to avoid punishing security documentation.
    """
    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_findings = sorted(
        findings,
        key=lambda f: (f.rule_id or "UNKNOWN", severity_rank.get((f.severity or "LOW").upper(), 4)),
    )

    def component_source_scope(component: Mapping[str, object]) -> str:
        for key in ("source_identity", "source_url", "source_digest"):
            value = component.get(key)
            if isinstance(value, str) and value:
                return f"{key}:{value}"
        return ""

    def finding_source_scope(finding: Finding) -> str:
        for key, value in (
            ("source_identity", finding.source_identity),
            ("source_url", finding.source_url),
            ("source_digest", finding.source_digest),
        ):
            if value:
                return f"{key}:{value}"
        return ""

    file_executable: dict[tuple[str, str], bool] = {}
    if component_metadata:
        for cm in component_metadata:
            file_executable[(component_source_scope(cm), str(cm.get("path", "")))] = bool(
                cm.get("executable", False)
            )

    rule_occurrence_count: dict[str, int] = {}
    score = 0.0

    for f in sorted_findings:
        confidence = max(0.0, min(1.0, f.confidence))
        if confidence <= 0.0:
            continue

        sev = (f.severity or "LOW").upper()
        base_points = _SEVERITY_POINTS.get(sev, 5)

        rule_id = f.rule_id or "UNKNOWN"
        count = rule_occurrence_count.get(rule_id, 0)
        rule_occurrence_count[rule_id] = count + 1

        if count >= _MAX_OCCURRENCES_PER_RULE:
            continue

        weight = _DIMINISHING_WEIGHTS[count]
        contribution = base_points * weight * confidence

        # Apply 1.3x multiplier only to findings from executable files
        if has_executable_scripts and file_executable.get((finding_source_scope(f), f.file), False):
            contribution *= 1.3

        score += contribution

    score_floor = max(
        (
            _RISK_SCORE_FLOORS_BY_RULE_ID.get(f.rule_id, 0)
            for f in sorted_findings
            if max(0.0, min(1.0, f.confidence)) > 0.0
        ),
        default=0,
    )
    final_score = min(100, max(score_floor, int(score)))

    severity_band = "LOW"
    for threshold, band in _RISK_SEVERITY_BANDS:
        if final_score >= threshold:
            severity_band = band
            break
    recommendation = _RISK_RECOMMENDATION.get(severity_band, "CAUTION")
    return final_score, severity_band, recommendation


def _build_sarif(
    findings: list[Finding],
    suppressed: list[SuppressedFinding] | None = None,
    degraded_notice: str | None = None,
    analysis_completeness: Mapping[str, object] | None = None,
    execution_successful: bool = True,
    structured_summaries: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build one SARIF invocation with canonical inspection notifications."""
    results: list[SarifResult] = []
    seen_rule_ids: dict[str, str] = {}

    for finding in findings:
        if not finding.rule_id or not finding.message:
            continue
        occurrences = finding.occurrences or [
            {
                "file": finding.file,
                "start_line": finding.start_line,
                "end_line": finding.end_line,
            }
        ]
        for occurrence in occurrences:
            start_value = occurrence.get("start_line", finding.start_line)
            start_line = start_value if isinstance(start_value, int) else finding.start_line
            end_value = occurrence.get("end_line")
            end_line = int(end_value) if isinstance(end_value, int) else None
            results.append(
                SarifResult(
                    ruleId=finding.rule_id,
                    message=SarifMessage(text=finding.message),
                    level=_severity_to_sarif_level(finding.severity),
                    properties=_build_sarif_properties(finding, occurrence),
                    locations=[
                        SarifLocation(
                            physicalLocation=SarifPhysicalLocation(
                                artifactLocation=_sarif_artifact_location(finding, occurrence),
                                region=SarifRegion(startLine=start_line, endLine=end_line),
                            )
                        )
                    ],
                )
            )
        if finding.rule_id not in seen_rule_ids:
            seen_rule_ids[finding.rule_id] = finding.message

    # Baseline-suppressed findings are kept in the SARIF for an audit trail, but
    # marked with the `suppressions` property so consumers exclude them from counts.
    for sf in suppressed or []:
        finding = sf.finding
        if not finding.rule_id or not finding.message:
            continue
        occurrences = finding.occurrences or [
            {
                "file": finding.file,
                "start_line": finding.start_line,
                "end_line": finding.end_line,
            }
        ]
        for occurrence in occurrences:
            start_value = occurrence.get("start_line", finding.start_line)
            start_line = start_value if isinstance(start_value, int) else finding.start_line
            end_value = occurrence.get("end_line")
            end_line = int(end_value) if isinstance(end_value, int) else None
            results.append(
                SarifResult(
                    ruleId=finding.rule_id,
                    message=SarifMessage(text=finding.message),
                    level=_severity_to_sarif_level(finding.severity),
                    properties=_build_sarif_properties(finding, occurrence),
                    locations=[
                        SarifLocation(
                            physicalLocation=SarifPhysicalLocation(
                                artifactLocation=_sarif_artifact_location(finding, occurrence),
                                region=SarifRegion(startLine=start_line, endLine=end_line),
                            )
                        )
                    ],
                    suppressions=[SarifSuppression(kind="external", justification=sf.reason)],
                )
            )
        if finding.rule_id not in seen_rule_ids:
            seen_rule_ids[finding.rule_id] = finding.message

    rules = [
        SarifReportingDescriptor(
            id=rule_id,
            shortDescription=SarifMessage(text=description),
        )
        for rule_id, description in sorted(seen_rule_ids.items())
    ]

    completeness = analysis_completeness or {}

    def nonnegative_count(key: str) -> int:
        value = completeness.get(key, 0)
        return max(0, value) if isinstance(value, int) and not isinstance(value, bool) else 0

    raw_ledger_exceptions = completeness.get("ledger_exceptions", [])
    raw_scope_exclusions = completeness.get("scope_exclusions", [])
    raw_limitations = completeness.get("limitations", [])
    ledger_exception_count = (
        sum(1 for item in raw_ledger_exceptions if isinstance(item, Mapping))
        if isinstance(raw_ledger_exceptions, list)
        else 0
    )
    scope_exclusion_count = (
        sum(1 for item in raw_scope_exclusions if isinstance(item, Mapping))
        if isinstance(raw_scope_exclusions, list)
        else 0
    )
    limitation_count = len(raw_limitations) if isinstance(raw_limitations, list) else 0
    fully_inspected = nonnegative_count("fully_inspected_files")
    partially_inspected = nonnegative_count("partially_inspected_files")
    entirely_uninspected = nonnegative_count("entirely_uninspected_files")

    raw_status = completeness.get("status")
    requested_status = (
        raw_status
        if isinstance(raw_status, str) and raw_status in {"complete", "partial", "failed"}
        else None
    )
    is_complete = (
        completeness.get("is_complete", True) is True
        and requested_status in {None, "complete"}
        and execution_successful
        and partially_inspected == 0
        and entirely_uninspected == 0
        and ledger_exception_count == 0
        and limitation_count == 0
    )
    status = (
        "complete"
        if is_complete
        else "failed"
        if not execution_successful or requested_status == "failed"
        else "partial"
    )
    raw_coverage = completeness.get("coverage_percent", 100.0 if is_complete else 0.0)
    coverage = (
        float(raw_coverage)
        if isinstance(raw_coverage, (int, float))
        and not isinstance(raw_coverage, bool)
        and 0.0 <= float(raw_coverage) <= 100.0
        else 0.0
    )
    completeness_projection: dict[str, object] = {
        "isComplete": is_complete,
        "status": status,
        "coveragePercent": coverage,
        "totalComponents": nonnegative_count("total_components"),
        "fullyInspectedFiles": fully_inspected,
        "partiallyInspectedFiles": partially_inspected,
        "entirelyUninspectedFiles": entirely_uninspected,
        # Counts are a deliberately payload-free projection. Detailed, sanitized
        # exceptions remain represented as bounded SARIF notifications below.
        "ledgerExceptionCount": ledger_exception_count,
        "scopeExclusionCount": scope_exclusion_count,
        "limitationCount": limitation_count,
        "notificationRecordLimit": MAX_FINDING_OUTPUT_RECORDS,
        "notificationsTruncated": False,
    }

    notifications: list[SarifNotification] = []
    observed_notifications = 0
    notifications_truncated = False

    def append_notification(notification: SarifNotification) -> None:
        nonlocal observed_notifications, notifications_truncated
        observed_notifications += 1
        if len(notifications) < MAX_FINDING_OUTPUT_RECORDS:
            notifications.append(notification)
        else:
            notifications_truncated = True

    for summary in structured_summaries or []:
        append_notification(
            SarifNotification(
                message=SarifMessage(text=_structured_summary_notification(summary)),
                level="note",
                properties={"kind": "structured_summary"},
            )
        )

    def notification_from_exception(
        exception: Mapping[str, object], level: Literal["error", "warning", "note"]
    ) -> SarifNotification:
        path = str(exception.get("path", ""))
        start_line = exception.get("start_line")
        end_line = exception.get("end_line")
        locations = None
        if path:
            region = (
                SarifRegion(
                    startLine=int(start_line),
                    endLine=int(end_line) if isinstance(end_line, int) else None,
                )
                if isinstance(start_line, int)
                else None
            )
            locations = [
                SarifLocation(
                    physicalLocation=SarifPhysicalLocation(
                        artifactLocation=SarifArtifactLocation(uri=path),
                        region=region,
                    )
                )
            ]
        properties: dict[str, object] = {
            "outcome": str(exception.get("outcome", "")),
            "phase": str(exception.get("phase", "")),
            "reasonCode": str(exception.get("reason_code", "")),
        }
        if exception.get("fatal") is not None:
            properties["fatal"] = bool(exception["fatal"])
        analyzers = exception.get("analyzers")
        if isinstance(analyzers, list):
            properties["analyzers"] = list(analyzers)
        return SarifNotification(
            message=SarifMessage(text=str(exception.get("message", "Inspection exception."))),
            level=level,
            locations=locations,
            properties=properties,
        )

    scope_exclusions = completeness.get("scope_exclusions", [])
    if isinstance(scope_exclusions, list):
        for exception in scope_exclusions:
            if isinstance(exception, Mapping):
                append_notification(notification_from_exception(exception, "note"))
    ledger_exceptions = completeness.get("ledger_exceptions", [])
    if isinstance(ledger_exceptions, list):
        for exception in ledger_exceptions:
            if isinstance(exception, Mapping):
                level: Literal["error", "warning", "note"] = (
                    "error" if exception.get("fatal") else "warning"
                )
                append_notification(notification_from_exception(exception, level))
    limitations = completeness.get("limitations", [])
    if isinstance(limitations, list):
        for limitation in limitations:
            append_notification(
                SarifNotification(
                    message=SarifMessage(text=str(limitation)),
                    level="warning",
                    properties={"kind": "inspection_limitation"},
                )
            )
    if degraded_notice:
        append_notification(
            SarifNotification(
                message=SarifMessage(text=degraded_notice),
                level="warning",
                properties={"kind": "llm_degradation"},
            )
        )
    if notifications_truncated:
        completeness_projection["notificationsTruncated"] = True
        sentinel = SarifNotification(
            message=SarifMessage(text="Inspection notifications reached the output limit."),
            level="warning",
            properties={
                "kind": "inspection_notification_limit",
                "reasonCode": "output_limit",
                "observedRecords": observed_notifications,
                "limitRecords": MAX_FINDING_OUTPUT_RECORDS,
            },
        )
        # Replace one detailed record so the bounded output always carries the
        # explicit truncation fact.
        if notifications:
            notifications[-1] = sentinel
    invocations = [
        SarifInvocation(
            executionSuccessful=execution_successful,
            toolExecutionNotifications=notifications or None,
            properties={"analysisCompleteness": completeness_projection},
        )
    ]

    sarif_log = SarifLog.model_validate(
        {
            "$schema": SARIF_SCHEMA_URI,
            "runs": [
                SarifRun(
                    tool=SarifTool(
                        driver=SarifDriver(
                            name="skillspector",
                            version=skillspector_version,
                            rules=rules if rules else None,
                        )
                    ),
                    results=results,
                    invocations=invocations,
                )
            ],
        }
    )
    rendered = sarif_log.model_dump(mode="json", by_alias=True, exclude_none=True)
    validate_sarif_report(rendered)
    return rendered


def _render_terminal_completeness(
    console: Console,
    completeness: Mapping[str, object],
    execution_successful: bool,
) -> None:
    """Render every public completeness record without leaking internal work IDs."""
    console.print()
    table = Table(title="Inspection Completeness", show_header=False, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Execution", "successful" if execution_successful else "failed")
    table.add_row("Status", str(completeness.get("status", "complete")))
    table.add_row("Coverage", f"{completeness.get('coverage_percent', 100.0)}%")
    table.add_row("Fully inspected", str(completeness.get("fully_inspected_files", 0)))
    table.add_row("Partially inspected", str(completeness.get("partially_inspected_files", 0)))
    table.add_row("Entirely uninspected", str(completeness.get("entirely_uninspected_files", 0)))
    console.print(table)

    def render_rows(title: str, rows: object) -> None:
        if not isinstance(rows, list) or not rows:
            return
        console.print(f"[bold]{escape(title)}[/bold]")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            location = str(row.get("path", ""))
            start_line = row.get("start_line")
            end_line = row.get("end_line")
            if isinstance(start_line, int):
                location += f":{start_line}" + (f"-{end_line}" if end_line else "")
            reason = str(row.get("reason_code", row.get("status", "status")))
            message = str(row.get("message", ""))
            console.print(f"  - {escape(reason)} {escape(location)}: {escape(message)}")

    render_rows("Scope exclusions", completeness.get("scope_exclusions"))
    render_rows("Ledger exceptions", completeness.get("ledger_exceptions"))
    render_rows("Analyzer statuses", completeness.get("analyzer_statuses"))
    limitations = completeness.get("limitations")
    if isinstance(limitations, list) and limitations:
        console.print("[bold]Limitations[/bold]")
        for limitation in limitations:
            console.print(f"  - {escape(str(limitation))}")


def _format_terminal(
    findings: list[Finding],
    component_metadata: list[dict[str, object]],
    manifest: dict[str, object],
    skill_path: str | None,
    risk_score: int,
    risk_severity: str,
    risk_recommendation: str,
    has_executable_scripts: bool,
    use_llm: bool = True,
    llm_call_log: Sequence[Mapping[str, object]] | None = None,
    degraded_notice: str | None = None,
    suppressed: list[SuppressedFinding] | None = None,
    structured_summaries: list[dict[str, object]] | None = None,
    show_suppressed: bool = False,
    analysis_completeness: Mapping[str, object] | None = None,
    execution_successful: bool = True,
) -> str:
    """Generate Rich terminal output and export as string."""
    suppressed = suppressed or []
    console = Console(record=True, force_terminal=True, width=80, file=StringIO())
    skill_name = (manifest.get("name") or "unknown") if manifest else "unknown"
    source = skill_path or ""

    console.print()
    console.print(
        Panel(
            "[bold]SkillSpector Security Report[/bold]",
            subtitle=f"v{skillspector_version}",
        )
    )
    console.print(f"\n[bold]Skill:[/bold] {skill_name}")
    console.print(f"[bold]Source:[/bold] {source}")
    console.print(f"[bold]Scanned:[/bold] {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    severity_colors = {
        "LOW": "green",
        "MEDIUM": "yellow",
        "HIGH": "red",
        "CRITICAL": "bold red",
    }
    color = severity_colors.get(risk_severity, "yellow")
    console.print("\n")
    risk_table = Table(title="Risk Assessment", show_header=False, box=None)
    risk_table.add_column("Metric", style="bold")
    risk_table.add_column("Value")
    risk_table.add_row("Score", f"[{color}]{risk_score}/100[/{color}]")
    risk_table.add_row("Severity", f"[{color}]{risk_severity}[/{color}]")
    risk_table.add_row(
        "Recommendation", f"[{color}]{risk_recommendation.replace('_', ' ')}[/{color}]"
    )
    console.print(risk_table)

    console.print("\n")
    comp_table = Table(title=f"Components ({len(component_metadata)})")
    comp_table.add_column("File", style="cyan")
    comp_table.add_column("Type")
    comp_table.add_column("Lines", justify="right")
    comp_table.add_column("Executable")
    for comp in component_metadata[:15]:
        path = str(comp.get("path", ""))
        typ = str(comp.get("type", ""))
        lines = comp.get("lines", 0)
        exec_flag = comp.get("executable", False)
        exec_marker = "[yellow]Yes[/yellow]" if exec_flag else "No"
        comp_table.add_row(path, typ, str(lines), exec_marker)
    if len(component_metadata) > 15:
        comp_table.add_row(f"... and {len(component_metadata) - 15} more", "", "", "")
    console.print(comp_table)

    effective_degraded_notice = degraded_notice or _llm_degradation_notice(
        use_llm, llm_call_log or []
    )
    if effective_degraded_notice:
        console.print()
        console.print(
            Panel(
                f"[bold]Degraded scan[/bold]\n{effective_degraded_notice}",
                title="[bold red]WARNING[/bold red]",
                border_style="red",
            )
        )

    if findings:
        console.print("\n")
        console.print(f"[bold]Issues ({len(findings)})[/bold]\n")
        severity_icons = {
            "LOW": "[green]LOW[/green]",
            "MEDIUM": "[yellow]MEDIUM[/yellow]",
            "HIGH": "[red]HIGH[/red]",
            "CRITICAL": "[bold red]CRITICAL[/bold red]",
        }
        for f in findings:
            icon = severity_icons.get((f.severity or "LOW").upper(), f.severity)
            console.print(f"  {icon}: {f.rule_id} - {f.message[:60]}...")
            end = f"–{f.end_line}" if f.end_line and f.end_line != f.start_line else ""
            console.print(f"    [dim]Location:[/dim] {f.file}:{f.start_line}{end}")
            if f.source_url:
                console.print(f"    [dim]Source:[/dim] {f.source_url} (depth {f.transitive_depth})")
            console.print(f"    [dim]Confidence:[/dim] {f.confidence:.0%}")
            if f.remediation:
                console.print(f"    [dim]Remediation:[/dim] {(f.remediation or '')[:150]}...")
            if f.evidence:
                rendered = ", ".join(f"{key}={value}" for key, value in sorted(f.evidence.items()))
                console.print(f"    [dim]Evidence:[/dim] {rendered}")
            console.print()
    else:
        console.print("\n[green]No security issues detected.[/green]\n")

    if structured_summaries:
        console.print("\n")
        console.print(f"[bold]Structured Skill Summary ({len(structured_summaries)})[/bold]\n")
        for summary in structured_summaries:
            console.print(
                f"  [cyan]{summary.get('id', 'SSR-1')}[/cyan]: {summary.get('message', '')}"
            )
            file = _summary_display_value(summary.get("file"))
            if file:
                console.print(f"    [dim]File:[/dim] {file}")
            for key, label in (
                ("protocol", "Protocol"),
                ("layout_kind", "Layout"),
                ("declared_tools", "Declared tools"),
                ("workflow_nodes", "Workflow nodes"),
                ("constraints", "Constraints"),
                ("resources", "Resources"),
                ("tags", "Tags"),
            ):
                value = _summary_display_value(summary.get(key))
                if value:
                    console.print(f"    [dim]{label}:[/dim] {value}")
            console.print()

    if suppressed:
        console.print(
            f"[dim]Suppressed by baseline: {len(suppressed)} (not counted toward risk score)[/dim]"
        )
        if show_suppressed:
            for sf in suppressed:
                f = sf.finding
                console.print(
                    f"  [dim]- {f.rule_id} {f.file}:{f.start_line} (reason: {sf.reason})[/dim]"
                )
        else:
            console.print("[dim]Use --show-suppressed to list them.[/dim]")
        console.print()

    _render_terminal_completeness(
        console,
        analysis_completeness or {},
        execution_successful,
    )
    console.print(f"[dim]Executable scripts: {'Yes' if has_executable_scripts else 'No'}[/dim]")
    return console.export_text()


def _llm_runtime_status(
    use_llm: bool, llm_call_log: Sequence[Mapping[str, object]]
) -> tuple[int, int, bool]:
    """Return ``(attempted, succeeded, degraded)`` from the LLM call log.

    ``degraded`` is True when the LLM stage was requested and at least one call
    was attempted, but not every call succeeded: a dropped or throttled batch
    (e.g. a 429) leaves the same coverage gap as a full failure, so a partial
    pass is degraded too, not just a total one.
    """
    attempted = len(llm_call_log)
    succeeded = sum(1 for r in llm_call_log if r.get("ok"))
    degraded = bool(use_llm and attempted > 0 and succeeded < attempted)
    return attempted, succeeded, degraded


def _llm_degradation_notice(
    use_llm: bool, llm_call_log: Sequence[Mapping[str, object]]
) -> str | None:
    """Return a human-readable degraded-scan warning, or None if not degraded."""
    attempted, succeeded, degraded = _llm_runtime_status(use_llm, llm_call_log)
    if not degraded:
        return None
    failed = attempted - succeeded
    return (
        f"LLM analysis was requested but {failed} of {attempted} LLM call(s) failed - "
        "results reflect STATIC analysis only for the affected batch(es)."
    )


def _build_metadata(
    has_executable_scripts: bool,
    use_llm: bool,
    llm_call_log: Sequence[Mapping[str, object]] | None = None,
    inference_usage: Sequence[Mapping[str, object]] | None = None,
    transitive_targets_scanned: int | None = None,
    transitive_bytes_scanned: int | None = None,
    transitive_truncation_reasons: Sequence[str] | None = None,
    llm_execution_enabled: bool | None = None,
) -> dict[str, object]:
    """Build the metadata section shared by all output formats."""
    llm_call_log = llm_call_log or []
    provider_available, llm_error = is_llm_available()
    attempted, succeeded, degraded = _llm_runtime_status(use_llm, llm_call_log)
    # meta_analyzer's own record, independent of whether a DIFFERENT
    # LLM-backed node (a semantic_* analyzer) lost coverage to a dropped
    # batch. A missing record means meta_analyzer never ran (e.g. there were
    # no findings to filter), which is not itself a failure, so it reads as
    # vacuously ok for llm_available (provider/runtime truth). When it does
    # run it always emits exactly one record.
    meta_analyzer_records = [r for r in llm_call_log if r.get("node") == "meta_analyzer"]
    meta_analyzer_ok = all(bool(r.get("ok")) for r in meta_analyzer_records)
    # meta_analysis_applied is stricter: "did meta-analysis actually run"
    # cannot be satisfied vacuously. all([]) is True on an empty list, so
    # meta_analyzer_ok alone is also True when meta_analyzer made no call at
    # all (the no-findings path) - require at least one record, and that
    # record must have succeeded.
    meta_analyzer_succeeded = bool(meta_analyzer_records) and meta_analyzer_ok

    # meta_analysis_applied / llm_available answer different questions.
    # llm_available is provider availability: the binary/credentials were
    # available AND meta_analyzer's own call (if it ran) succeeded - it stays
    # vacuously true when meta_analyzer never ran, because the provider being
    # reachable does not depend on there being findings to filter.
    # meta_analysis_applied is "did meta-analysis itself run", which needs an
    # actual successful meta_analyzer record, not just the absence of a
    # failure. A different analyzer's partial batch loss is a coverage gap,
    # reported separately below via llm_degraded / llm_calls_attempted /
    # llm_calls_succeeded, and must not flip these two fields on its own -
    # that would conflate two independent contracts (meta-analysis ran vs.
    # some coverage was lost) into one boolean.
    execution_enabled = use_llm if llm_execution_enabled is None else llm_execution_enabled
    unavailable_before_execution = bool(use_llm and not execution_enabled)
    meta_analysis_applied = (
        use_llm and execution_enabled and provider_available and meta_analyzer_succeeded
    )

    meta: dict[str, object] = {
        "has_executable_scripts": has_executable_scripts,
        "skillspector_version": skillspector_version,
        "llm_requested": use_llm,
        # llm_available reflects runtime truth: the binary/credentials were
        # available AND meta_analyzer's own call (if it ran) succeeded.
        "llm_available": (
            provider_available and meta_analyzer_ok and not unavailable_before_execution
        ),
        "meta_analysis_applied": meta_analysis_applied,
        # A list (including an empty list) makes observability explicit. Empty
        # means the provider/transport supplied no counters; it is never an
        # estimated zero-cost assertion.
        "inference_usage": sanitize_inference_usage(inference_usage),
    }
    if not meta_analysis_applied:
        meta["filtering_mode"] = "heuristic"
    if use_llm and attempted:
        meta["llm_calls_attempted"] = attempted
        meta["llm_calls_succeeded"] = succeeded
    if unavailable_before_execution:
        meta["llm_error"] = (
            "LLM analysis was requested but unavailable during preflight; "
            "results reflect static analysis only."
        )
    elif degraded:
        meta["llm_degraded"] = True
        reasons = sorted(
            {str(r.get("error")) for r in llm_call_log if not r.get("ok") and r.get("error")}
        )
        detail = f" Reasons: {'; '.join(reasons)}" if reasons else ""
        failed = attempted - succeeded
        meta["llm_error"] = (
            f"LLM analysis was requested but {failed} of {attempted} LLM call(s) failed; "
            f"results reflect static analysis only for the affected batch(es).{detail}"
        )
    elif use_llm and not provider_available:
        meta["llm_error"] = llm_error
    if transitive_targets_scanned is not None:
        meta["transitive_targets_scanned"] = transitive_targets_scanned
    if transitive_bytes_scanned is not None:
        meta["transitive_bytes_scanned"] = transitive_bytes_scanned
    if transitive_truncation_reasons:
        meta["transitive_truncated"] = True
        meta["transitive_truncation_reasons"] = list(transitive_truncation_reasons)
    return meta


def _format_json(
    findings: list[Finding],
    component_metadata: list[dict[str, object]],
    manifest: dict[str, object],
    skill_path: str | None,
    risk_score: int,
    risk_severity: str,
    risk_recommendation: str,
    has_executable_scripts: bool,
    use_llm: bool = True,
    llm_call_log: Sequence[Mapping[str, object]] | None = None,
    inference_usage: Sequence[Mapping[str, object]] | None = None,
    analysis_completeness: Mapping[str, object] | None = None,
    suppressed: list[SuppressedFinding] | None = None,
    execution_successful: bool = True,
    transitive_targets_scanned: int | None = None,
    transitive_bytes_scanned: int | None = None,
    transitive_truncation_reasons: Sequence[str] | None = None,
    structured_summaries: list[dict[str, object]] | None = None,
    llm_execution_enabled: bool | None = None,
) -> str:
    """Generate JSON report string."""
    suppressed = suppressed or []
    skill_name = (manifest.get("name") or "unknown") if manifest else "unknown"
    data: dict[str, object] = {
        "skill": {
            "name": skill_name,
            "source": skill_path or "",
            "scanned_at": datetime.now(UTC).isoformat(),
        },
        "risk_assessment": {
            "score": risk_score,
            "severity": risk_severity,
            "recommendation": risk_recommendation,
            "max_issue_severity": _max_issue_severity(findings),
        },
        "components": [
            {
                "path": c.get("path"),
                "type": c.get("type"),
                "lines": c.get("lines"),
                "executable": c.get("executable"),
                "size_bytes": c.get("size_bytes"),
                "source_url": c.get("source_url"),
                "source_identity": c.get("source_identity"),
                "source_digest": c.get("source_digest"),
            }
            for c in component_metadata
        ],
        "structured_summaries": structured_summaries or [],
        "issues": [f.to_dict() for f in findings],
        "suppressed_count": len(suppressed),
        "suppressed": [sf.to_dict() for sf in suppressed],
        "metadata": _build_metadata(
            has_executable_scripts,
            use_llm,
            llm_call_log,
            inference_usage,
            transitive_targets_scanned,
            transitive_bytes_scanned,
            transitive_truncation_reasons,
            llm_execution_enabled,
        ),
        "execution_successful": execution_successful,
    }
    data["analysis_completeness"] = dict(analysis_completeness or {})
    return json.dumps(data, indent=2)


def _markdown_cell(value: object) -> str:
    """Render dynamic report text safely inside a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_markdown_completeness(
    lines: list[str],
    completeness: Mapping[str, object],
    execution_successful: bool,
) -> None:
    """Append the full public completeness projection to a Markdown report."""
    lines.append("## Inspection Completeness\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Execution | {'successful' if execution_successful else 'failed'} |")
    lines.append(f"| Status | {_markdown_cell(completeness.get('status', 'complete'))} |")
    lines.append(f"| Coverage | {_markdown_cell(completeness.get('coverage_percent', 100.0))}% |")
    lines.append(
        f"| Fully inspected | {_markdown_cell(completeness.get('fully_inspected_files', 0))} |"
    )
    lines.append(
        f"| Partially inspected | {_markdown_cell(completeness.get('partially_inspected_files', 0))} |"
    )
    lines.append(
        f"| Entirely uninspected | {_markdown_cell(completeness.get('entirely_uninspected_files', 0))} |"
    )
    lines.append("")

    def render_rows(title: str, rows: object) -> None:
        if not isinstance(rows, list) or not rows:
            return
        lines.append(f"### {title}\n")
        lines.append("| Reason / Status | Location | Details |")
        lines.append("|-----------------|----------|---------|")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            location = str(row.get("path", ""))
            start_line = row.get("start_line")
            end_line = row.get("end_line")
            if isinstance(start_line, int):
                location += f":{start_line}" + (f"-{end_line}" if end_line else "")
            reason = row.get("reason_code", row.get("status", "status"))
            lines.append(
                f"| {_markdown_cell(reason)} | `{_markdown_cell(location)}` | "
                f"{_markdown_cell(row.get('message', ''))} |"
            )
        lines.append("")

    render_rows("Scope Exclusions", completeness.get("scope_exclusions"))
    render_rows("Ledger Exceptions", completeness.get("ledger_exceptions"))
    render_rows("Analyzer Statuses", completeness.get("analyzer_statuses"))
    limitations = completeness.get("limitations")
    if isinstance(limitations, list) and limitations:
        lines.append("### Limitations\n")
        for limitation in limitations:
            lines.append(f"- {_markdown_cell(limitation)}")
        lines.append("")


def _format_markdown(
    findings: list[Finding],
    component_metadata: list[dict[str, object]],
    manifest: dict[str, object],
    skill_path: str | None,
    risk_score: int,
    risk_severity: str,
    risk_recommendation: str,
    has_executable_scripts: bool,
    use_llm: bool = True,
    llm_call_log: Sequence[Mapping[str, object]] | None = None,
    degraded_notice: str | None = None,
    suppressed: list[SuppressedFinding] | None = None,
    structured_summaries: list[dict[str, object]] | None = None,
    show_suppressed: bool = False,
    analysis_completeness: Mapping[str, object] | None = None,
    execution_successful: bool = True,
) -> str:
    """Generate Markdown report string."""
    suppressed = suppressed or []
    lines: list[str] = []
    skill_name = (manifest.get("name") or "unknown") if manifest else "unknown"
    source = skill_path or ""

    lines.append("# SkillSpector Security Report\n")
    lines.append(f"**Skill:** {skill_name}  ")
    lines.append(f"**Source:** `{source}`  ")
    lines.append(f"**Scanned:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    lines.append("")

    effective_degraded_notice = degraded_notice or _llm_degradation_notice(
        use_llm, llm_call_log or []
    )
    if effective_degraded_notice:
        lines.append(f"> ⚠️ **Degraded scan:** {effective_degraded_notice}")
        lines.append("")

    lines.append("## Risk Assessment\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Score | {risk_score}/100 |")
    lines.append(f"| Severity | {risk_severity} |")
    lines.append(f"| Recommendation | {risk_recommendation.replace('_', ' ')} |")
    lines.append("")

    lines.append(f"## Components ({len(component_metadata)})\n")
    lines.append("| File | Type | Lines | Executable |")
    lines.append("|------|------|-------|------------|")
    for comp in component_metadata:
        path = comp.get("path", "")
        typ = comp.get("type", "")
        line_count = comp.get("lines", 0)
        exec_flag = comp.get("executable", False)
        exec_marker = "Yes" if exec_flag else "No"
        lines.append(f"| `{path}` | {typ} | {line_count} | {exec_marker} |")
    lines.append("")

    if structured_summaries:
        lines.append(f"## Structured Skill Summary ({len(structured_summaries)})\n")
        for summary in structured_summaries:
            lines.append(f"### {summary.get('id', 'SSR-1')}\n")
            lines.append(f"**Message:** {summary.get('message', '')}  ")
            file = _summary_display_value(summary.get("file"))
            if file:
                lines.append(f"**File:** `{file}`  ")
            for key, label in (
                ("protocol", "Protocol"),
                ("layout_kind", "Layout"),
                ("declared_tools", "Declared tools"),
                ("workflow_nodes", "Workflow nodes"),
                ("constraints", "Constraints"),
                ("resources", "Resources"),
                ("tags", "Tags"),
            ):
                value = _summary_display_value(summary.get(key))
                if value:
                    lines.append(f"**{label}:** {value}  ")
            lines.append("")

    lines.append(f"## Issues ({len(findings)})\n")
    if not findings:
        lines.append("No security issues detected.\n")
    else:
        severity_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🔴"}
        for f in findings:
            sev = (f.severity or "LOW").upper()
            emoji = severity_emoji.get(sev, "")
            lines.append(f"### {emoji} {sev}: {f.rule_id}\n")
            end = f"–{f.end_line}" if f.end_line and f.end_line != f.start_line else ""
            lines.append(f"**Location:** `{f.file}:{f.start_line}{end}`  ")
            if f.source_url:
                lines.append(f"**Source:** `{f.source_url}`  ")
                lines.append(f"**Transitive depth:** {f.transitive_depth}  ")
            lines.append(f"**Confidence:** {f.confidence:.0%}  ")
            lines.append("")
            lines.append(f"**Message:** {f.message}")
            lines.append("")
            if f.remediation:
                lines.append(f"**Remediation:** {f.remediation}")
                lines.append("")
            if f.evidence:
                lines.append("**Evidence:**")
                for key, evidence_value in sorted(f.evidence.items()):
                    lines.append(f"- **{key}:** `{evidence_value}`")
                lines.append("")
            lines.append("---\n")

    if suppressed:
        lines.append(f"## Suppressed ({len(suppressed)})\n")
        lines.append(
            "These findings matched the baseline and are **not** counted toward the risk score.\n"
        )
        if show_suppressed:
            lines.append("| Rule | Location | Reason |")
            lines.append("|------|----------|--------|")
            for sf in suppressed:
                f = sf.finding
                reason = sf.reason.replace("|", "\\|")
                lines.append(f"| {f.rule_id} | `{f.file}:{f.start_line}` | {reason} |")
            lines.append("")
        else:
            lines.append("_Run with `--show-suppressed` to list them._\n")

    _render_markdown_completeness(lines, analysis_completeness or {}, execution_successful)
    lines.append("## Metadata\n")
    lines.append(f"- **Executable Scripts:** {'Yes' if has_executable_scripts else 'No'}")
    lines.append(f"\n*Generated by SkillSpector v{skillspector_version}*")
    return "\n".join(lines)


def report(state: SkillspectorState) -> dict[str, object]:
    """Render canonical findings and the exceptional ledger projection.

    Finalization owns completeness derivation. The report node only selects the
    validated finding IDs, applies baseline suppression, and renders all surfaces.
    """
    clear_python_ast_cache(state.get("python_ast_cache_key"))
    raw_findings = state.get("findings")
    if raw_findings is None:
        # Preserve the public node contract for direct callers while ensuring
        # graph executions prefer the pre-meta canonical finding collection.
        raw_findings = state.get("filtered_findings", [])
    findings_by_id = {finding.finding_id: finding for finding in raw_findings}
    # Meta/LLM analysis can enrich canonical objects but cannot remove
    # deterministic findings from primary output.
    selected_findings = list(findings_by_id.values())
    selected_findings = [_sanitize_finding(finding) for finding in selected_findings]

    raw_structured_summaries = state.get("structured_summaries") or []
    structured_summaries = [
        _sanitize_structured_summary(summary)
        for summary in raw_structured_summaries
        if isinstance(summary, dict)
    ]

    empty_completeness: AnalysisCompleteness = {
        "total_components": 0,
        "scanned_components": 0,
        "coverage_percent": 100.0,
        "is_complete": True,
        "status": "complete",
        "execution_successful": True,
        "fully_inspected_files": 0,
        "partially_inspected_files": 0,
        "entirely_uninspected_files": 0,
        "ledger_exceptions": [],
        "scope_exclusions": [],
        "analyzer_statuses": [],
        "limitations": [],
    }
    supplied_completeness = state.get("analysis_completeness")
    analysis_completeness: Mapping[str, object] = (
        supplied_completeness if isinstance(supplied_completeness, Mapping) else empty_completeness
    )
    execution_successful = bool(
        state.get("execution_successful", analysis_completeness.get("execution_successful", True))
    )
    component_metadata = state.get("component_metadata") or []
    file_cache = state.get("local_file_cache") or state.get("file_cache") or {}
    has_executable_scripts = state.get("has_executable_scripts", False)
    manifest = state.get("manifest") or {}
    skill_path = state.get("skill_path")
    output_format = state.get("output_format") or "sarif"
    use_llm = state.get("use_llm", True)
    llm_requested = state.get("llm_requested", use_llm)
    llm_call_log = state.get("llm_call_log") or []
    inference_usage = state.get("inference_usage") or []
    transitive_targets_scanned = state.get("transitive_targets_scanned")
    transitive_bytes_scanned = state.get("transitive_bytes_scanned")
    transitive_truncation_reasons = [
        str(reason)
        for reason in state.get("transitive_truncation_reasons", [])
        if isinstance(reason, str)
    ]
    if transitive_truncation_reasons:
        analysis_completeness = dict(analysis_completeness)
        raw_limitations = analysis_completeness.get("limitations")
        limitations = list(raw_limitations) if isinstance(raw_limitations, list) else []
        limitations.append(
            "Transitive traversal truncated: " + "; ".join(transitive_truncation_reasons)
        )
        analysis_completeness["limitations"] = limitations
        analysis_completeness["is_complete"] = False

    _attempted, _succeeded, degraded = _llm_runtime_status(llm_requested, llm_call_log)
    provider_available, provider_error = is_llm_available()
    has_recorded_failure = any(not r.get("ok") for r in llm_call_log)
    unavailable_before_execution = bool(llm_requested and not use_llm)
    provider_unavailable = bool(
        llm_requested
        and not provider_available
        and (has_recorded_failure or unavailable_before_execution)
    )
    degraded = degraded or provider_unavailable or unavailable_before_execution
    degraded_notice = _llm_degradation_notice(llm_requested, llm_call_log)
    if unavailable_before_execution:
        degraded_notice = (
            "LLM analysis was requested but unavailable during preflight; "
            "results reflect STATIC analysis only."
        )
    elif provider_unavailable and degraded_notice is None:
        degraded_notice = (
            "LLM analysis was requested but the configured provider was unavailable"
            f" ({provider_error or 'unknown reason'}); results may reflect static analysis only."
        )
    if unavailable_before_execution:
        logger.warning(
            "LLM stage unavailable during preflight; report reflects static analysis only"
        )
    elif degraded:
        logger.warning(
            "LLM stage degraded: %d/%d LLM call(s) failed; report reflects static analysis only",
            _attempted - _succeeded,
            _attempted,
        )

    baseline = state.get("baseline")
    show_suppressed = state.get("show_suppressed", False)
    active_findings, suppressed = partition_findings(
        selected_findings,
        baseline if isinstance(baseline, Baseline) else None,
        file_cache=file_cache,
        scanner_version=skillspector_version,
    )
    risk_score, risk_severity, risk_recommendation = _compute_risk_score(
        active_findings, has_executable_scripts, component_metadata
    )
    reported_findings = _bounded_report_findings(deduplicate(active_findings))
    remaining_output_records = max(
        0,
        MAX_FINDING_OUTPUT_RECORDS - _finding_output_record_count(reported_findings),
    )
    suppressed = _bounded_suppressed_findings(
        suppressed,
        limit=remaining_output_records,
    )
    display_findings = _expand_occurrences(reported_findings)
    exceptions = analysis_completeness.get("ledger_exceptions", [])
    fatal_exception = (
        any(
            isinstance(exception, Mapping) and bool(exception.get("fatal"))
            for exception in exceptions
        )
        if isinstance(exceptions, list)
        else False
    )
    entirely_uninspected_value = analysis_completeness.get("entirely_uninspected_files", 0)
    entirely_uninspected = (
        entirely_uninspected_value if isinstance(entirely_uninspected_value, int) else 0
    )
    # Fail closed for any incomplete/degraded scan while preserving the honest
    # score and severity. Transitive traversal limits are an additional source
    # of incomplete coverage.
    incomplete = not bool(analysis_completeness.get("is_complete", True))
    if (
        degraded
        or fatal_exception
        or entirely_uninspected > 0
        or incomplete
        or transitive_truncation_reasons
    ) and risk_recommendation == "SAFE":
        risk_recommendation = "CAUTION"

    sarif_report = _build_sarif(
        reported_findings,
        suppressed,
        degraded_notice=degraded_notice,
        analysis_completeness=analysis_completeness,
        execution_successful=execution_successful,
        structured_summaries=structured_summaries,
    )
    if output_format == "terminal":
        report_body = _format_terminal(
            display_findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
            use_llm=llm_requested,
            llm_call_log=llm_call_log,
            degraded_notice=degraded_notice,
            suppressed=suppressed,
            structured_summaries=structured_summaries,
            show_suppressed=show_suppressed,
            analysis_completeness=analysis_completeness,
            execution_successful=execution_successful,
        )
    elif output_format == "json":
        report_body = _format_json(
            display_findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
            use_llm=llm_requested,
            llm_call_log=llm_call_log,
            inference_usage=inference_usage,
            analysis_completeness=analysis_completeness,
            suppressed=suppressed,
            execution_successful=execution_successful,
            transitive_targets_scanned=(
                int(transitive_targets_scanned)
                if isinstance(transitive_targets_scanned, int)
                else None
            ),
            transitive_bytes_scanned=(
                int(transitive_bytes_scanned) if isinstance(transitive_bytes_scanned, int) else None
            ),
            transitive_truncation_reasons=transitive_truncation_reasons,
            structured_summaries=structured_summaries,
            llm_execution_enabled=use_llm,
        )
    elif output_format == "markdown":
        report_body = _format_markdown(
            display_findings,
            component_metadata,
            manifest,
            skill_path,
            risk_score,
            risk_severity,
            risk_recommendation,
            has_executable_scripts,
            use_llm=llm_requested,
            llm_call_log=llm_call_log,
            degraded_notice=degraded_notice,
            suppressed=suppressed,
            structured_summaries=structured_summaries,
            show_suppressed=show_suppressed,
            analysis_completeness=analysis_completeness,
            execution_successful=execution_successful,
        )
    else:
        report_body = json.dumps(sarif_report, indent=2)

    logger.debug(
        "Report generated: format=%s, findings_count=%d, suppressed_count=%d",
        output_format,
        len(active_findings),
        len(suppressed),
    )
    return {
        "sarif_report": sarif_report,
        "risk_score": risk_score,
        "risk_severity": risk_severity,
        "risk_recommendation": risk_recommendation,
        "report_body": report_body,
        "filtered_findings": reported_findings,
        "suppressed_findings": suppressed,
        "execution_successful": execution_successful,
        "transitive_targets_scanned": transitive_targets_scanned,
        "transitive_bytes_scanned": transitive_bytes_scanned,
        "transitive_truncated": bool(transitive_truncation_reasons),
        "transitive_truncation_reasons": transitive_truncation_reasons,
    }
