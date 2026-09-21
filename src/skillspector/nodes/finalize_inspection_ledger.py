# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-node adapter for canonical inspection-ledger finalization."""

from __future__ import annotations

from collections.abc import Mapping

from skillspector.artifacts import ContentKind
from skillspector.inspection_ledger import (
    MAX_FINDING_OUTPUT_RECORDS,
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    analyzer_status_for_events,
    finalize_ledger,
    ledger_event,
)
from skillspector.models import Finding
from skillspector.nodes.analyzers import ANALYZER_MODULES
from skillspector.semantic_runtime import (
    has_semantic_runtime_event,
    semantic_runtime_intent,
    semantic_runtime_ledger_event,
)
from skillspector.state import SkillspectorState


def _has_only_format_limitations(
    inventory_item: Mapping[str, object] | None,
    events: list[Mapping[str, object]],
) -> bool:
    """Recognize unsupported content without hiding other coverage failures."""
    if not inventory_item or str(inventory_item.get("content_kind")) not in {
        ContentKind.BINARY,
        ContentKind.OPAQUE,
    }:
        return False
    if str(inventory_item.get("disposition")) not in {"partial", "out_of_scope"}:
        return False
    format_reasons = {LedgerReason.BINARY_CONTENT, LedgerReason.OPAQUE_CONTENT}
    inventory_reason = inventory_item.get("reason")
    if inventory_reason is not None and str(inventory_reason) not in format_reasons:
        return False
    # Classifying an asset is insufficient: require explicit ledger evidence.
    # Missing/unknown reasons, skipped work, failed work and mixed limitations
    # must continue to produce AE1. Completed analyzers do not erase exceptions.
    exceptions = [event for event in events if event.get("outcome") != "completed"]
    return (
        bool(exceptions)
        and not any(event.get("fatal") for event in events)
        and all(
            str(event.get("outcome")) in {"partial", "out_of_scope"}
            and str(event.get("reason_code")) in format_reasons
            for event in exceptions
        )
    )


def _reference_coverage_findings(
    state: SkillspectorState,
) -> list[Finding]:
    """Create AE1 only for canonical resolved targets with incomplete disposition."""
    raw_references = state.get("artifact_references") or []
    inventory: dict[str, Mapping[str, object]] = {}
    duplicate_inventory_paths: set[str] = set()
    for item in state.get("artifact_inventory") or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", ""))
        if path in inventory:
            duplicate_inventory_paths.add(path)
        inventory[path] = item
    events_by_path: dict[str, list[Mapping[str, object]]] = {}
    for event in state.get("inspection_ledger") or []:
        if not isinstance(event, Mapping):
            continue
        events_by_path.setdefault(str(event.get("path", "")), []).append(event)
    findings: list[Finding] = []
    seen_locations: set[tuple[str, int, str]] = set()
    for reference in raw_references:
        if not isinstance(reference, dict):
            continue
        status = str(reference.get("status", ""))
        if status != "resolved":
            continue
        target = reference.get("target_path")
        target_path = str(target) if target else ""
        inventory_item = inventory.get(target_path)
        disposition = str(inventory_item.get("disposition", "")) if inventory_item else ""
        target_events = events_by_path.get(target_path, [])
        exceptional = {str(event.get("outcome", "")) for event in target_events}
        final_disposition = (
            "failed"
            if "failed" in exceptional
            else "partial"
            if "partial" in exceptional
            else "out_of_scope"
            if "out_of_scope" in exceptional
            else disposition
        )
        if final_disposition not in {"partial", "failed", "out_of_scope"}:
            continue
        if target_path not in duplicate_inventory_paths and _has_only_format_limitations(
            inventory_item, target_events
        ):
            continue
        line_value = reference.get("line", 1)
        source_path = str(reference.get("source_path", "SKILL.md"))
        line = line_value if isinstance(line_value, int) else 1
        # A Markdown label and destination may resolve to the same artifact.
        # Findings identify source lines, so emit that coverage gap only once.
        location = (source_path, line, target_path)
        if location in seen_locations:
            continue
        seen_locations.add(location)
        evidence = str(reference.get("evidence", ""))[:160]
        findings.append(
            Finding(
                rule_id="AE1",
                message="Referenced artifact was not completely inspected",
                severity="HIGH",
                confidence=1.0,
                file=source_path,
                start_line=line,
                category="analysis-evasion",
                tags=["coverage", "reference", f"target-disposition:{final_disposition}"],
                finding=f"{target_path} ({final_disposition})"[:200],
                code_snippet=evidence,
                matched_text=target_path,
                remediation=(
                    "Make the referenced artifact locally available and fully analyzable, "
                    "or remove the reference."
                ),
            )
        )
    return findings


def _size_coverage_findings(
    state: SkillspectorState,
    covered_paths: set[str],
) -> list[Finding]:
    """Create AE7 for artifacts truncated by the per-file size cap.

    A file too large to fully analyze must not be able to produce a
    zero-finding report: the unreviewed region is itself the finding.
    Scoped to the per-file cap (``size_limit``); aggregate budget
    exhaustion already fails closed through the completeness projection.
    Paths already reported by AE1 (referenced artifacts) are skipped.
    """
    findings: list[Finding] = []
    for item in state.get("artifact_inventory") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("disposition", "")) != "partial":
            continue
        if str(item.get("reason", "")) != LedgerReason.SIZE_LIMIT.value:
            continue
        if str(item.get("path", "")) in covered_paths:
            continue
        path = str(item.get("path", ""))
        size_bytes = item.get("size_bytes", 0)
        findings.append(
            Finding(
                rule_id="AE7",
                message=(
                    "File exceeds the analyzable size limit; trailing content was not inspected"
                ),
                severity="HIGH",
                confidence=1.0,
                file=path,
                start_line=1,
                category="analysis-evasion",
                tags=["coverage", "size-limit"],
                finding=f"{path} ({size_bytes} bytes, partially inspected)"[:200],
                matched_text=path,
                remediation=(
                    "Keep analyzable files under the per-file size limit, or "
                    "split oversized content so every byte can be inspected."
                ),
            )
        )
    return findings


def finalize_inspection_ledger(state: SkillspectorState) -> dict[str, object]:
    """Validate full internal facts and derive the public completeness projection."""
    reference_findings = _reference_coverage_findings(state)
    size_findings = _size_coverage_findings(
        state,
        covered_paths={str(finding.matched_text or "") for finding in reference_findings},
    )
    coverage_findings = [*reference_findings, *size_findings]
    # Work IDs are scoped to analyzer, source file and line range. Distinct
    # targets on one line must share a terminal row with all emitted findings.
    coverage_ids_by_line: dict[tuple[str, int | None], list[str]] = {}
    for finding in coverage_findings:
        coverage_ids_by_line.setdefault((finding.file, finding.start_line), []).append(
            finding.finding_id
        )
    reference_events: list[InspectionLedgerEvent] = [
        ledger_event(
            outcome=LedgerOutcome.COMPLETED,
            phase="reference",
            analyzer_id="reference_coverage",
            path=path,
            start_line=line,
            end_line=line,
            emitted_finding_ids=finding_ids,
        )
        for (path, line), finding_ids in coverage_ids_by_line.items()
    ]
    merged_state = dict(state)
    all_findings = [*(state.get("findings") or []), *coverage_findings]
    output_events: list[InspectionLedgerEvent] = []
    finding_output_records = sum(max(1, len(finding.occurrences)) for finding in all_findings)
    if finding_output_records > MAX_FINDING_OUTPUT_RECORDS:
        output_events.append(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="finding_output",
                path=next(
                    (finding.file for finding in all_findings if finding.occurrences),
                    all_findings[MAX_FINDING_OUTPUT_RECORDS].file
                    if len(all_findings) > MAX_FINDING_OUTPUT_RECORDS
                    else "SKILL.md",
                ),
                reason=LedgerReason.OUTPUT_LIMIT,
                observed_findings=finding_output_records,
                limit_findings=MAX_FINDING_OUTPUT_RECORDS,
            )
        )
    merged_state["findings"] = all_findings
    merged_state["effective_finding_ids"] = [
        *(state.get("effective_finding_ids") or []),
        *(finding.finding_id for finding in coverage_findings),
    ]
    llm_requested, llm_enabled = semantic_runtime_intent(merged_state)
    runtime_event = semantic_runtime_ledger_event(
        requested=llm_requested,
        enabled=llm_enabled,
        result=merged_state,
        discovered_modules=ANALYZER_MODULES,
    )
    runtime_events = (
        [runtime_event]
        if runtime_event is not None
        and not has_semantic_runtime_event(state.get("inspection_ledger") or [], runtime_event)
        else []
    )
    merged_state["inspection_ledger"] = [
        *(state.get("inspection_ledger") or []),
        *reference_events,
        *output_events,
        *runtime_events,
    ]
    reference_statuses = (
        [analyzer_status_for_events("reference_coverage", reference_events)]
        if reference_events
        else []
    )
    merged_state["analyzer_status_events"] = [
        *(state.get("analyzer_status_events") or []),
        *reference_statuses,
    ]
    completeness, effective_finding_ids = finalize_ledger(merged_state)
    if coverage_findings and completeness["status"] == "complete":
        completeness["status"] = "partial"
        completeness["is_complete"] = False
        limitations = completeness.setdefault("limitations", [])
        limitations.append("One or more artifacts were not completely inspected.")
    return {
        "analysis_completeness": completeness,
        "execution_successful": completeness["execution_successful"],
        "findings": coverage_findings,
        "effective_finding_ids": effective_finding_ids,
        "inspection_ledger": [*reference_events, *output_events, *runtime_events],
        "analyzer_status_events": reference_statuses,
    }
