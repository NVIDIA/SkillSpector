# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical completeness projections for every report surface."""

from __future__ import annotations

import json

import pytest

from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    analyzer_status_for_events,
    finalize_ledger,
    ledger_event,
)
from skillspector.models import Finding
from skillspector.nodes.report import report
from skillspector.sarif_models import validate_sarif_report
from skillspector.state import SkillspectorState


def _state_with_two_ledger_exceptions(output_format: str) -> SkillspectorState:
    finding = Finding(rule_id="AST1", message="unsafe call", file="clean.py")
    exceptions = [
        {
            "outcome": "failed",
            "phase": "cache",
            "reason_code": "read_error",
            "message": "File content could not be read.",
            "path": "a.py",
            "start_line": None,
            "end_line": None,
            "analyzers": ["behavioral_ast"],
            "fatal": True,
        },
        {
            "outcome": "skipped",
            "phase": "behavioral",
            "reason_code": "syntax_error",
            "message": "Python source could not be parsed.",
            "path": "b.py",
            "start_line": None,
            "end_line": None,
            "analyzers": ["behavioral_ast"],
            "fatal": False,
        },
    ]
    return {
        "output_format": output_format,
        "analysis_completeness": {
            "total_components": 2,
            "scanned_components": 0,
            "coverage_percent": 0.0,
            "is_complete": False,
            "execution_successful": False,
            "fully_inspected_files": 0,
            "partially_inspected_files": 1,
            "entirely_uninspected_files": 1,
            "ledger_exceptions": exceptions,
            "scope_exclusions": [],
            "analyzer_statuses": [],
            "limitations": [],
            "findings_before_filtering": 1,
            "findings_after_filtering": 1,
        },
        "execution_successful": False,
        "inspection_ledger": [
            {
                "work_id": "work-completed",
                "record_type": "work_item",
                "outcome": "completed",
                "phase": "behavioral",
                "path": "clean.py",
                "start_line": None,
                "end_line": None,
                "analyzer_id": "behavioral_ast",
                "input_finding_ids": [],
                "emitted_finding_ids": [finding.finding_id],
            }
        ],
        "findings": [finding],
        "effective_finding_ids": [finding.finding_id],
        "component_metadata": [],
        "manifest": {"name": "test"},
        "use_llm": False,
    }


@pytest.mark.parametrize("output_format", ["json", "terminal", "markdown", "sarif"])
def test_every_format_preserves_exceptions_but_omits_completed_rows(
    output_format: str,
) -> None:
    state = _state_with_two_ledger_exceptions(output_format)
    result = report(state)

    assert result["execution_successful"] is False
    if output_format == "json":
        payload = json.loads(result["report_body"])
        assert payload["execution_successful"] is False
        assert len(payload["analysis_completeness"]["ledger_exceptions"]) == 2
        assert payload["issues"][0]["finding_id"] == state["findings"][0].finding_id
    elif output_format == "sarif":
        validate_sarif_report(result["sarif_report"])
        run = result["sarif_report"]["runs"][0]
        notifications = run["invocations"][0]["toolExecutionNotifications"]
        assert len(notifications) == 2
        assert run["results"][0]["properties"]["findingId"] == state["findings"][0].finding_id
        assert run["invocations"][0]["executionSuccessful"] is False
    else:
        assert "read_error" in result["report_body"]
        assert "syntax_error" in result["report_body"]

    assert "work-completed" not in result["report_body"]


def test_fatal_omission_floors_safe_recommendation_without_changing_score() -> None:
    state = _state_with_two_ledger_exceptions("json")
    state["findings"] = []
    state["effective_finding_ids"] = []
    result = report(state)

    assert result["risk_score"] == 0
    assert result["risk_recommendation"] == "CAUTION"


def test_unscanned_executable_content_is_successful_but_incomplete() -> None:
    event = ledger_event(
        analyzer_id="dependency_source_coverage",
        outcome=LedgerOutcome.PARTIAL,
        phase="static",
        path="docs/setup.md",
        start_line=3,
        end_line=5,
        reason=LedgerReason.UNSCANNED_EXECUTABLE_CONTENT,
    )

    completeness, effective_ids = finalize_ledger(
        {
            "components": ["docs/setup.md"],
            "findings": [],
            "effective_finding_ids": [],
            "inspection_ledger": [event],
            "analyzer_status_events": [
                analyzer_status_for_events("dependency_source_coverage", [event])
            ],
            "artifact_inventory": [],
        }
    )

    assert effective_ids == []
    assert completeness["execution_successful"] is True
    assert completeness["is_complete"] is False
    assert completeness["status"] == "partial"
    assert completeness["ledger_exceptions"] == [
        {
            "outcome": LedgerOutcome.PARTIAL,
            "phase": "static",
            "reason_code": LedgerReason.UNSCANNED_EXECUTABLE_CONTENT,
            "message": "Executable content was identified but is not inspected for dependency-source changes.",
            "path": "docs/setup.md",
            "start_line": 3,
            "end_line": 5,
            "fatal": False,
            "analyzers": ["dependency_source_coverage"],
        }
    ]
