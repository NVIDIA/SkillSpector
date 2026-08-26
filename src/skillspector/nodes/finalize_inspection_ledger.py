# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-node adapter for canonical inspection-ledger finalization."""

from __future__ import annotations

from collections.abc import Mapping

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
from skillspector.state import SkillspectorState


def _reference_coverage_findings(
    state: SkillspectorState,
) -> list[Finding]:
    """Create AE1 only for canonical resolved targets with incomplete disposition."""
    raw_references = state.get("artifact_references") or []
    inventory: dict[str, Mapping[str, object]] = {
        str(item.get("path", "")): item
        for item in state.get("artifact_inventory") or []
        if isinstance(item, dict)
    }
    exceptional_outcomes: dict[str, set[str]] = {}
    for event in state.get("inspection_ledger") or []:
        if not isinstance(event, Mapping):
            continue
        outcome = str(event.get("outcome", ""))
        if outcome in {"partial", "failed", "out_of_scope"}:
            exceptional_outcomes.setdefault(str(event.get("path", "")), set()).add(outcome)
    findings: list[Finding] = []
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
        exceptional = exceptional_outcomes.get(target_path, set())
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
        line_value = reference.get("line", 1)
        evidence = str(reference.get("evidence", ""))[:160]
        findings.append(
            Finding(
                rule_id="AE1",
                message="Referenced artifact was not completely inspected",
                severity="HIGH",
                confidence=1.0,
                file=str(reference.get("source_path", "SKILL.md")),
                start_line=line_value if isinstance(line_value, int) else 1,
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


def finalize_inspection_ledger(state: SkillspectorState) -> dict[str, object]:
    """Validate full internal facts and derive the public completeness projection."""
    reference_findings = _reference_coverage_findings(state)
    reference_events: list[InspectionLedgerEvent] = [
        ledger_event(
            outcome=LedgerOutcome.COMPLETED,
            phase="reference",
            analyzer_id="reference_coverage",
            path=finding.file,
            start_line=finding.start_line,
            end_line=finding.start_line,
            emitted_finding_ids=[finding.finding_id],
        )
        for finding in reference_findings
    ]
    merged_state = dict(state)
    all_findings = [*(state.get("findings") or []), *reference_findings]
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
        *(finding.finding_id for finding in reference_findings),
    ]
    merged_state["inspection_ledger"] = [
        *(state.get("inspection_ledger") or []),
        *reference_events,
        *output_events,
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
    if reference_findings and completeness["status"] == "complete":
        completeness["status"] = "partial"
        completeness["is_complete"] = False
        limitations = completeness.setdefault("limitations", [])
        limitations.append("One or more referenced artifacts were not completely inspected.")
    return {
        "analysis_completeness": completeness,
        "execution_successful": completeness["execution_successful"],
        "findings": reference_findings,
        "effective_finding_ids": effective_finding_ids,
        "inspection_ledger": [*reference_events, *output_events],
        "analyzer_status_events": reference_statuses,
    }
