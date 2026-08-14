# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for canonical inspection-ledger finalization."""

from __future__ import annotations

import json

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
