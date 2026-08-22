# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for canonical inspection-ledger finalization."""

from __future__ import annotations

import json

import pytest

import skillspector.inspection_ledger as inspection_ledger_module
import skillspector.nodes.finalize_inspection_ledger as finalizer_module
import skillspector.nodes.report as report_module
import skillspector.state as state_module
from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    analyzer_status_event,
    finalize_ledger,
    guard_analyzer_node,
    inspection_work_id,
    ledger_event,
)
from skillspector.models import Finding
from skillspector.nodes.finalize_inspection_ledger import finalize_inspection_ledger
from skillspector.nodes.report import report
from skillspector.state import AnalyzerNodeResponse, SkillspectorState


def _target(work_id: str, path: str) -> dict[str, str | int | None]:
    return {"work_id": work_id, "path": path, "start_line": None, "end_line": None}


def test_completed_work_is_covered_and_resolves_emitted_finding_ids() -> None:
    finding = Finding(rule_id="AST1", message="unsafe call", file="run.py")
    work_id = inspection_work_id("behavioral_ast", "run.py", None, None)
    state: SkillspectorState = {
        "components": ["run.py"],
        "findings": [finding],
        "effective_finding_ids": [finding.finding_id],
        "inspection_ledger": [
            ledger_event(
                outcome=LedgerOutcome.COMPLETED,
                phase="behavioral",
                analyzer_id="behavioral_ast",
                path="run.py",
                emitted_finding_ids=[finding.finding_id],
            )
        ],
        "analyzer_status_events": [
            analyzer_status_event(
                analyzer_id="behavioral_ast",
                status="completed",
                planned_work=[_target(work_id, "run.py")],
            )
        ],
    }

    completeness, effective_ids = finalize_ledger(state)

    assert completeness["execution_successful"] is True
    assert completeness["coverage_percent"] == 100.0
    assert completeness["ledger_exceptions"] == []
    assert effective_ids == [finding.finding_id]


def test_missing_terminal_row_becomes_fatal_unaccounted_work() -> None:
    work_id = inspection_work_id("behavioral_ast", "broken.py", None, None)

    result = finalize_inspection_ledger(
        {
            "components": ["broken.py"],
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [
                analyzer_status_event(
                    analyzer_id="behavioral_ast",
                    status="failed",
                    planned_work=[_target(work_id, "broken.py")],
                )
            ],
        }
    )

    exception = result["analysis_completeness"]["ledger_exceptions"][0]
    assert exception["reason_code"] == LedgerReason.UNACCOUNTED_WORK
    assert exception["path"] == "broken.py"
    assert exception["fatal"] is True
    assert result["execution_successful"] is False


def test_unknown_emitted_finding_id_is_fatal_accounting_error() -> None:
    work_id = inspection_work_id("behavioral_ast", "run.py", None, None)
    completeness, _ = finalize_ledger(
        {
            "components": ["run.py"],
            "findings": [],
            "inspection_ledger": [
                ledger_event(
                    outcome=LedgerOutcome.COMPLETED,
                    phase="behavioral",
                    analyzer_id="behavioral_ast",
                    path="run.py",
                    emitted_finding_ids=["finding-missing"],
                )
            ],
            "analyzer_status_events": [
                analyzer_status_event(
                    analyzer_id="behavioral_ast",
                    status="completed",
                    planned_work=[_target(work_id, "run.py")],
                )
            ],
        }
    )

    exception = completeness["ledger_exceptions"][0]
    assert exception["reason_code"] == LedgerReason.FINDING_ACCOUNTING_ERROR
    assert exception["fatal"] is True


def test_meta_failure_preserves_primary_coverage_but_fails_execution() -> None:
    finding = Finding(rule_id="P1", message="unsafe", file="SKILL.md")
    producer_work = inspection_work_id("prompt_injection", "SKILL.md", None, None)
    meta_work = inspection_work_id("meta_analyzer", "SKILL.md", None, None)
    completeness, effective_ids = finalize_ledger(
        {
            "components": ["SKILL.md"],
            "findings": [finding],
            "effective_finding_ids": [finding.finding_id],
            "inspection_ledger": [
                ledger_event(
                    outcome=LedgerOutcome.COMPLETED,
                    phase="static",
                    analyzer_id="prompt_injection",
                    path="SKILL.md",
                    emitted_finding_ids=[finding.finding_id],
                ),
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    phase="meta",
                    analyzer_id="meta_analyzer",
                    reason=LedgerReason.LLM_BATCH_FAILED,
                    path="SKILL.md",
                    input_finding_ids=[finding.finding_id],
                    emitted_finding_ids=[finding.finding_id],
                ),
            ],
            "analyzer_status_events": [
                analyzer_status_event(
                    analyzer_id="prompt_injection",
                    status="completed",
                    planned_work=[_target(producer_work, "SKILL.md")],
                ),
                analyzer_status_event(
                    analyzer_id="meta_analyzer",
                    status="failed",
                    planned_work=[_target(meta_work, "SKILL.md")],
                ),
            ],
        }
    )

    assert completeness["coverage_percent"] == 100.0
    assert completeness["is_complete"] is False
    assert completeness["execution_successful"] is False
    assert effective_ids == [finding.finding_id]


def test_skipped_meta_event_that_drops_findings_is_a_fatal_accounting_error() -> None:
    """Finalization rejects malformed skipped meta rows that bypass the factory."""
    finding = Finding(rule_id="P1", message="unsafe", file="SKILL.md")
    skipped_meta = ledger_event(
        outcome=LedgerOutcome.SKIPPED,
        phase="meta",
        analyzer_id="meta_analyzer",
        reason=LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID,
        path="SKILL.md",
        input_finding_ids=[finding.finding_id],
        emitted_finding_ids=[finding.finding_id],
    )
    skipped_meta["emitted_finding_ids"] = []

    completeness, _ = finalize_ledger(
        {
            "components": ["SKILL.md"],
            "findings": [finding],
            "inspection_ledger": [skipped_meta],
        }
    )

    assert completeness["execution_successful"] is False
    assert completeness["ledger_exceptions"][0]["reason_code"] == (
        LedgerReason.FINDING_ACCOUNTING_ERROR
    )
    assert completeness["ledger_exceptions"][0]["fatal"] is True


def test_json_round_trip_keeps_failed_ledger_work_fatal() -> None:
    """Deserialized StrEnum values must retain failure semantics."""
    state = json.loads(
        json.dumps(
            {
                "components": ["SKILL.md"],
                "inspection_ledger": [
                    ledger_event(
                        outcome=LedgerOutcome.FAILED,
                        phase="cache",
                        analyzer_id="cache_reader",
                        reason=LedgerReason.READ_ERROR,
                        path="SKILL.md",
                    )
                ],
                "analyzer_status_events": [
                    analyzer_status_event(analyzer_id="cache_reader", status="failed")
                ],
            }
        )
    )

    completeness, _ = finalize_ledger(state)

    assert completeness["execution_successful"] is False
    assert completeness["ledger_exceptions"][0]["outcome"] == LedgerOutcome.FAILED
    assert completeness["ledger_exceptions"][0]["fatal"] is True


def test_scope_exclusion_does_not_reduce_requested_coverage() -> None:
    completeness, _ = finalize_ledger(
        {
            "components": ["SKILL.md"],
            "inspection_ledger": [
                ledger_event(
                    outcome=LedgerOutcome.OUT_OF_SCOPE,
                    record_type=LedgerRecordType.SCOPE_BOUNDARY,
                    phase="discovery",
                    reason=LedgerReason.EXCLUDED_DIRECTORY,
                    path="node_modules/",
                )
            ],
            "analyzer_status_events": [],
        }
    )
    assert completeness["coverage_percent"] == 100.0
    assert completeness["is_complete"] is True


@pytest.mark.parametrize(
    ("disposition", "referenced", "expected_total", "expected_counts"),
    [
        ("partial", False, 1, (0, 1, 0)),
        ("failed", False, 1, (0, 0, 1)),
        ("out_of_scope", True, 1, (0, 0, 1)),
        ("out_of_scope", False, 0, (0, 0, 0)),
    ],
)
def test_inventory_disposition_takes_precedence_over_completed_analyzer_work(
    disposition: str,
    referenced: bool,
    expected_total: int,
    expected_counts: tuple[int, int, int],
) -> None:
    """Opaque inventory facts cannot be promoted by downstream completion."""
    path = "assets/target.bin"
    work_id = inspection_work_id("artifact_integrity", path, None, None)
    references = (
        [
            {
                "source_path": "SKILL.md",
                "line": 1,
                "column": 1,
                "evidence": path,
                "target_path": path,
                "status": "resolved",
                "disposition": disposition,
            }
        ]
        if referenced
        else []
    )

    completeness, _ = finalize_ledger(
        {
            "components": [path],
            "findings": [],
            "artifact_inventory": [
                {
                    "path": path,
                    "disposition": disposition,
                    "content_kind": "opaque",
                }
            ],
            "artifact_references": references,
            "inspection_ledger": [
                ledger_event(
                    outcome=LedgerOutcome.COMPLETED,
                    phase="static",
                    analyzer_id="artifact_integrity",
                    path=path,
                )
            ],
            "analyzer_status_events": [
                analyzer_status_event(
                    analyzer_id="artifact_integrity",
                    status="completed",
                    planned_work=[_target(work_id, path)],
                )
            ],
        }
    )

    assert completeness["total_components"] == expected_total
    assert (
        completeness["fully_inspected_files"],
        completeness["partially_inspected_files"],
        completeness["entirely_uninspected_files"],
    ) == expected_counts
    assert completeness["coverage_percent"] == (100.0 if expected_total == 0 else 0.0)


def test_omitted_partial_inventory_row_remains_entirely_uninspected() -> None:
    """A partial disposition does not imply that any omitted bytes were read."""
    completeness, _ = finalize_ledger(
        {
            "components": [],
            "findings": [],
            "artifact_inventory": [
                {
                    "path": "omitted.txt",
                    "disposition": "partial",
                    "content_kind": "opaque",
                }
            ],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        }
    )

    assert completeness["total_components"] == 1
    assert completeness["fully_inspected_files"] == 0
    assert completeness["partially_inspected_files"] == 0
    assert completeness["entirely_uninspected_files"] == 1
    assert completeness["coverage_percent"] == 0.0


def test_workflow_ledger_reducer_caps_all_producers_with_partial_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(state_module, "MAX_INSPECTION_LEDGER_EVENTS", 2)
    events = [
        ledger_event(
            outcome=LedgerOutcome.COMPLETED,
            record_type=LedgerRecordType.SYSTEM,
            phase="test",
            path=f"file-{index}.txt",
        )
        for index in range(4)
    ]

    bounded = state_module.merge_inspection_ledger(events[:1], events[1:3])
    bounded = state_module.merge_inspection_ledger(bounded, events[3:])

    assert len(bounded) == 2
    assert bounded[-1]["phase"] == "ledger_output"
    assert bounded[-1]["reason_code"] == LedgerReason.OUTPUT_LIMIT
    assert bounded[-1]["observed_records"] == 4
    completeness, _ = finalize_ledger(
        {
            "components": ["file-0.txt"],
            "findings": [],
            "inspection_ledger": bounded,
            "analyzer_status_events": [],
        }
    )
    assert completeness["status"] == "partial"
    assert completeness["execution_successful"] is True


def test_workflow_ledger_reducer_recaps_oversized_preloaded_sentinel_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(state_module, "MAX_INSPECTION_LEDGER_EVENTS", 2)
    events = [
        ledger_event(
            outcome=LedgerOutcome.COMPLETED,
            record_type=LedgerRecordType.SYSTEM,
            phase="test",
            path=f"file-{index}.txt",
        )
        for index in range(4)
    ]
    preloaded = [
        *events[:2],
        ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            record_type=LedgerRecordType.SYSTEM,
            phase="ledger_output",
            path="file-2.txt",
            reason=LedgerReason.OUTPUT_LIMIT,
            observed_records=3,
            limit_records=2,
        ),
    ]

    bounded = state_module.merge_inspection_ledger(preloaded, events[2:])

    assert len(bounded) == 2
    assert bounded[0] == events[0]
    assert bounded[-1]["phase"] == "ledger_output"
    assert bounded[-1]["reason_code"] == LedgerReason.OUTPUT_LIMIT
    assert bounded[-1]["observed_records"] == 5
    assert bounded[-1]["limit_records"] == 2


def test_finding_projection_is_globally_bounded_and_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inspection_ledger_module, "MAX_EFFECTIVE_FINDINGS", 2)
    monkeypatch.setattr(finalizer_module, "MAX_FINDING_OUTPUT_RECORDS", 2)
    monkeypatch.setattr(report_module, "MAX_FINDING_OUTPUT_RECORDS", 2)
    findings = [
        Finding(rule_id=f"T{index}", message="bounded", file=f"file-{index}.txt")
        for index in range(3)
    ]

    result = finalize_inspection_ledger(
        {
            "components": [finding.file for finding in findings],
            "findings": findings,
            "effective_finding_ids": [finding.finding_id for finding in findings],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        }
    )

    assert len(result["effective_finding_ids"]) == 2
    assert any(
        event.get("phase") == "finding_output"
        and event.get("reason_code") == LedgerReason.OUTPUT_LIMIT
        for event in result["inspection_ledger"]
    )
    assert result["analysis_completeness"]["status"] == "partial"
    rendered = report(
        {
            "output_format": "json",
            "findings": findings,
            "effective_finding_ids": result["effective_finding_ids"],
            "analysis_completeness": result["analysis_completeness"],
            "execution_successful": result["execution_successful"],
            "component_metadata": [],
            "manifest": {},
            "use_llm": False,
        }
    )
    assert len(rendered["filtered_findings"]) == 2


def test_occurrence_projection_uses_the_same_global_record_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finalizer_module, "MAX_FINDING_OUTPUT_RECORDS", 2)
    monkeypatch.setattr(report_module, "MAX_FINDING_OUTPUT_RECORDS", 2)
    finding = Finding(
        rule_id="T1",
        message="bounded occurrences",
        file="a.txt",
        matched_text="same",
        occurrences=[
            {"file": "a.txt", "start_line": 1, "end_line": 1},
            {"file": "b.txt", "start_line": 1, "end_line": 1},
            {"file": "c.txt", "start_line": 1, "end_line": 1},
        ],
    )
    finalized = finalize_inspection_ledger(
        {
            "components": ["a.txt", "b.txt", "c.txt"],
            "findings": [finding],
            "effective_finding_ids": [finding.finding_id],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        }
    )
    rendered = report(
        {
            "output_format": "json",
            "findings": [finding],
            "analysis_completeness": finalized["analysis_completeness"],
            "execution_successful": finalized["execution_successful"],
            "component_metadata": [],
            "manifest": {},
            "use_llm": False,
        }
    )

    assert finalized["analysis_completeness"]["status"] == "partial"
    assert len(rendered["filtered_findings"]) == 1
    assert len(rendered["filtered_findings"][0].occurrences) == 2


def test_healthy_uninstrumented_analyzer_is_not_falsely_unaccounted() -> None:
    """A completed legacy analyzer with no work rows remains compatible with !150."""
    completeness, _ = finalize_ledger(
        {
            "components": ["SKILL.md"],
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [
                analyzer_status_event(
                    analyzer_id="legacy_healthy_analyzer",
                    status="completed",
                )
            ],
        }
    )

    assert completeness["execution_successful"] is True
    assert completeness["ledger_exceptions"] == []


def test_overlapping_analyzer_work_is_not_falsely_unaccounted() -> None:
    """Overlapping ranges from separate analyzers retain distinct terminal work."""
    first_work = inspection_work_id("semantic_a", "scripts/check.py", 1, 100)
    second_work = inspection_work_id("semantic_b", "scripts/check.py", 1, 100)

    completeness, _ = finalize_ledger(
        {
            "components": ["scripts/check.py"],
            "findings": [],
            "inspection_ledger": [
                ledger_event(
                    outcome=LedgerOutcome.COMPLETED,
                    phase="semantic",
                    analyzer_id="semantic_a",
                    path="scripts/check.py",
                    start_line=1,
                    end_line=100,
                ),
                ledger_event(
                    outcome=LedgerOutcome.COMPLETED,
                    phase="semantic",
                    analyzer_id="semantic_b",
                    path="scripts/check.py",
                    start_line=1,
                    end_line=100,
                ),
            ],
            "analyzer_status_events": [
                analyzer_status_event(
                    analyzer_id="semantic_a",
                    status="completed",
                    planned_work=[_target(first_work, "scripts/check.py")],
                ),
                analyzer_status_event(
                    analyzer_id="semantic_b",
                    status="completed",
                    planned_work=[_target(second_work, "scripts/check.py")],
                ),
            ],
        }
    )

    assert completeness["execution_successful"] is True
    assert completeness["ledger_exceptions"] == []


def test_resolved_partial_reference_produces_one_canonically_counted_ae1() -> None:
    result = finalize_inspection_ledger(
        {
            "components": ["SKILL.md", "assets/blob.bin"],
            "findings": [],
            "effective_finding_ids": [],
            "artifact_inventory": [
                {
                    "path": "assets/blob.bin",
                    "disposition": "partial",
                    "content_kind": "binary",
                }
            ],
            "artifact_references": [
                {
                    "source_path": "SKILL.md",
                    "line": 4,
                    "column": 8,
                    "evidence": "Read [the blob](assets/blob.bin).",
                    "target_path": "assets/blob.bin",
                    "status": "resolved",
                    "disposition": "partial",
                }
            ],
            "inspection_ledger": [
                ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    record_type=LedgerRecordType.SYSTEM,
                    phase="cache",
                    path="assets/blob.bin",
                    reason=LedgerReason.OPAQUE_CONTENT,
                )
            ],
            "analyzer_status_events": [],
        }
    )

    assert [finding.rule_id for finding in result["findings"]] == ["AE1"]
    assert len(result["effective_finding_ids"]) == 1
    completeness = result["analysis_completeness"]
    assert completeness["findings_before_filtering"] == 1
    assert completeness["findings_after_filtering"] == 1
    assert completeness["is_complete"] is False


@pytest.mark.parametrize("use_llm", [False, True])
@pytest.mark.parametrize(
    ("disposition", "outcome", "reason", "expected_ae1"),
    [
        ("analyzed", LedgerOutcome.COMPLETED, None, False),
        ("partial", LedgerOutcome.PARTIAL, LedgerReason.SIZE_LIMIT, True),
        ("failed", LedgerOutcome.FAILED, LedgerReason.READ_ERROR, True),
        ("out_of_scope", LedgerOutcome.OUT_OF_SCOPE, LedgerReason.BINARY_CONTENT, True),
    ],
)
def test_resolved_reference_ae1_disposition_matrix_is_llm_independent(
    use_llm: bool,
    disposition: str,
    outcome: LedgerOutcome,
    reason: LedgerReason | None,
    expected_ae1: bool,
) -> None:
    result = finalize_inspection_ledger(
        {
            "components": ["SKILL.md", "assets/target.bin"],
            "findings": [],
            "effective_finding_ids": [],
            "use_llm": use_llm,
            "artifact_inventory": [
                {
                    "path": "assets/target.bin",
                    "disposition": disposition,
                    "content_kind": "binary" if disposition == "out_of_scope" else "text",
                }
            ],
            "artifact_references": [
                {
                    "source_path": "SKILL.md",
                    "line": 7,
                    "column": 9,
                    "evidence": "Inspect [the target](assets/target.bin)." + "x" * 200,
                    "target_path": "assets/target.bin",
                    "status": "resolved",
                    "disposition": disposition,
                }
            ],
            "inspection_ledger": [
                ledger_event(
                    outcome=outcome,
                    record_type=(
                        LedgerRecordType.SCOPE_BOUNDARY
                        if outcome is LedgerOutcome.OUT_OF_SCOPE
                        else LedgerRecordType.SYSTEM
                    ),
                    phase="cache",
                    path="assets/target.bin",
                    reason=reason,
                )
            ],
            "analyzer_status_events": [],
        }
    )

    ae1 = [finding for finding in result["findings"] if finding.rule_id == "AE1"]
    assert bool(ae1) is expected_ae1
    if expected_ae1:
        assert len(ae1) == 1
        assert ae1[0].file == "SKILL.md"
        assert ae1[0].start_line == 7
        assert ae1[0].matched_text == "assets/target.bin"
        assert f"target-disposition:{disposition}" in ae1[0].tags
        assert len(ae1[0].code_snippet or "") <= 160
        assert result["analysis_completeness"]["findings_before_filtering"] == 1
        assert result["analysis_completeness"]["findings_after_filtering"] == 1
        assert len(result["effective_finding_ids"]) == 1
    else:
        assert result["effective_finding_ids"] == []


@pytest.mark.parametrize("status", ["missing", "ambiguous", "rejected"])
def test_unresolved_reference_does_not_synthesize_ae1(status: str) -> None:
    result = finalize_inspection_ledger(
        {
            "components": ["SKILL.md"],
            "findings": [],
            "effective_finding_ids": [],
            "artifact_inventory": [],
            "artifact_references": [
                {
                    "source_path": "SKILL.md",
                    "line": 2,
                    "column": 1,
                    "evidence": "missing.md",
                    "target_path": None,
                    "status": status,
                    "disposition": "partial",
                }
            ],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        }
    )

    assert result["findings"] == []
    assert result["effective_finding_ids"] == []


def test_guard_analyzer_node_converts_unexpected_exception_to_fatal_facts() -> None:
    def broken_node(_state: SkillspectorState) -> AnalyzerNodeResponse:
        raise RuntimeError("provider detail must remain private")

    guarded = guard_analyzer_node("broken_analyzer", broken_node)
    result = guarded({"components": ["a.py"]})

    assert result["findings"] == []
    assert result["inspection_ledger"][0]["reason_code"] == LedgerReason.ANALYZER_RUNTIME_ERROR
    assert result["inspection_ledger"][0]["error_class"] == "RuntimeError"
    assert "provider detail" not in result["inspection_ledger"][0]["message"]
    assert result["analyzer_status_events"][0]["status"] == "failed"
