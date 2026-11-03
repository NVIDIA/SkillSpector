# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider outages block execution without inventing artifact evasion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillspector.inspection_ledger import (
    MAX_INSPECTION_LEDGER_EVENTS,
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.models import Finding
from skillspector.nodes.finalize_inspection_ledger import (
    _reference_coverage_findings,
    finalize_inspection_ledger,
)
from skillspector.semantic_runtime import (
    CANONICAL_SEMANTIC_ANALYZER_IDS,
    semantic_runtime_ledger_event,
)
from skillspector.state import SkillspectorState, llm_call_record


@pytest.fixture
def _unavailable_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep report availability probes offline; inspection and reporting stay real."""
    monkeypatch.setattr(
        "skillspector.nodes.report.is_llm_available",
        lambda **kwargs: (False, "Transient LLM provider failure"),
    )


def _provider_outage_state(target: str = "scripts/helper.py") -> SkillspectorState:
    events = [
        ledger_event(
            outcome=LedgerOutcome.FAILED,
            phase="semantic",
            analyzer_id=analyzer,
            path=target,
            start_line=1,
            end_line=8,
            reason=LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED,
            error_class="InternalServerError",
        )
        for analyzer in sorted(CANONICAL_SEMANTIC_ANALYZER_IDS)
    ]
    return {
        "components": [target],
        "findings": [],
        "effective_finding_ids": [],
        "artifact_inventory": [{"path": target, "disposition": "analyzed", "content_kind": "text"}],
        "artifact_references": [
            {
                "source_path": "SKILL.md",
                "line": 4,
                "target_path": target,
                "status": "resolved",
                "disposition": "analyzed",
            }
        ],
        "inspection_ledger": events,
        "analyzer_status_events": [
            analyzer_status_for_events(str(event["analyzer_id"]), [event]) for event in events
        ],
        "use_llm": True,
        "llm_requested": True,
        "llm_call_log": [
            llm_call_record(analyzer, ok=False)
            for analyzer in sorted(CANONICAL_SEMANTIC_ANALYZER_IDS)
        ],
        "output_format": "json",
    }


def _finalized_state(state: SkillspectorState) -> SkillspectorState:
    updates = finalize_inspection_ledger(state)
    return {
        **state,
        **updates,
        "findings": [*state["findings"], *updates["findings"]],
        "inspection_ledger": [*state["inspection_ledger"], *updates["inspection_ledger"]],
        "analyzer_status_events": [
            *state["analyzer_status_events"],
            *updates["analyzer_status_events"],
        ],
    }


@pytest.mark.parametrize("target", ["scripts/helper.py", "SKILL.md"])
@pytest.mark.parametrize("preexisting_runtime", [False, True])
@pytest.mark.parametrize("source_identity", [None, "source-a"])
def test_provider_only_reference_failure_remains_failed_without_ae1(
    target: str, preexisting_runtime: bool, source_identity: str | None
) -> None:
    state = _provider_outage_state(target)
    if source_identity is not None:
        for rows in (
            state["inspection_ledger"],
            state["analyzer_status_events"],
            state["llm_call_log"],
        ):
            for row in rows:
                row["source_identity"] = source_identity
    if preexisting_runtime:
        event = semantic_runtime_ledger_event(
            requested=True, enabled=True, result=state, discovered_modules={}
        )
        assert event is not None
        state["inspection_ledger"].append(event)

    finalized = _finalized_state(state)

    assert finalized["findings"] == []
    completeness = finalized["analysis_completeness"]
    assert finalized["execution_successful"] is False
    assert completeness["is_complete"] is False
    assert completeness["status"] == "failed"
    assert completeness["coverage_percent"] < 100
    assert completeness["findings_before_filtering"] == 0
    assert any(
        row["reason_code"] == "llm_connection_retries_exhausted" and row["fatal"] is True
        for row in completeness["ledger_exceptions"]
    )


def test_provider_outage_preserves_real_finding_and_json_blocked_summary(
    _unavailable_provider: None,
) -> None:
    from skillspector.nodes.report import report

    state = _provider_outage_state()
    finding = Finding(
        rule_id="AST1", message="unsafe call", severity="HIGH", file="scripts/helper.py"
    )
    producer = ledger_event(
        outcome=LedgerOutcome.COMPLETED,
        phase="behavioral",
        analyzer_id="behavioral_ast",
        path="scripts/helper.py",
        emitted_finding_ids=[finding.finding_id],
    )
    meta = ledger_event(
        outcome=LedgerOutcome.FAILED,
        phase="meta",
        analyzer_id="meta_analyzer",
        path="scripts/helper.py",
        reason=LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED,
        input_finding_ids=[finding.finding_id],
        emitted_finding_ids=[finding.finding_id],
    )
    state["findings"] = [finding]
    state["effective_finding_ids"] = [finding.finding_id]
    state["inspection_ledger"].extend([producer, meta])
    state["analyzer_status_events"].extend(
        [
            analyzer_status_for_events("behavioral_ast", [producer]),
            analyzer_status_for_events("meta_analyzer", [meta]),
        ]
    )
    state["llm_call_log"].append(llm_call_record("meta_analyzer", ok=False))

    finalized = _finalized_state(state)
    payload = json.loads(report(finalized)["report_body"])

    assert [issue["id"] for issue in payload["issues"]] == ["AST1"]
    assert finalized["effective_finding_ids"] == [finding.finding_id]
    assert payload["execution_successful"] is False
    assert payload["analysis_completeness"]["is_complete"] is False
    assert payload["risk_assessment"]["recommendation"] == "CAUTION"


@pytest.mark.parametrize(
    "mutation",
    [
        "parser",
        "refusal",
        "invalid_response",
        "unknown_reason",
        "missing_reason",
        "missing_terminal",
        "contradictory_status",
        "inventory_failure",
        "inventory_partial",
        "duplicate_inventory",
        "malformed_row",
        "ledger_limit",
        "ledger_size",
        "runtime_unaccounted_analyzer",
        "runtime_missing_call",
        "runtime_unrelated_failure",
        "runtime_no_statuses",
        "runtime_wrong_source",
        "runtime_wrong_event_source",
        "runtime_malformed_source",
        "runtime_wrong_stage",
        "nonsemantic_provider_label",
        "missing_inventory_kind",
    ],
)
def test_provider_reason_cannot_hide_other_or_unproven_reference_limitations(mutation: str) -> None:
    state = _provider_outage_state("SKILL.md")
    events = state["inspection_ledger"]
    if mutation in {"parser", "refusal", "invalid_response", "unknown_reason", "missing_reason"}:
        events[0]["reason_code"] = {
            "parser": "static_parse_limit",
            "refusal": "llm_response_refused",
            "invalid_response": "llm_structured_response_invalid",
            "unknown_reason": "unknown_future_failure",
            "missing_reason": None,
        }[mutation]
    elif mutation == "missing_terminal":
        events.pop()
    elif mutation == "contradictory_status":
        state["analyzer_status_events"][0]["status"] = "completed"
    elif mutation in {"inventory_failure", "inventory_partial"}:
        item = state["artifact_inventory"][0]
        item["disposition"] = "failed" if mutation == "inventory_failure" else "partial"
        item["reason"] = "read_error" if mutation == "inventory_failure" else "size_limit"
        state["artifact_references"][0]["disposition"] = item["disposition"]
    elif mutation == "duplicate_inventory":
        state["artifact_inventory"].append(dict(state["artifact_inventory"][0]))
    elif mutation == "malformed_row":
        events.append({"outcome": "failed"})
    elif mutation == "nonsemantic_provider_label":
        events[0]["phase"] = "cache"
    elif mutation == "missing_inventory_kind":
        state["artifact_inventory"][0].pop("content_kind")
    elif mutation == "ledger_limit":
        events.append(
            ledger_event(
                outcome=LedgerOutcome.PARTIAL,
                record_type=LedgerRecordType.SYSTEM,
                phase="ledger_output",
                path="output",
                reason=LedgerReason.OUTPUT_LIMIT,
                observed_records=MAX_INSPECTION_LEDGER_EVENTS + 1,
                limit_records=MAX_INSPECTION_LEDGER_EVENTS,
            )
        )
    elif mutation == "ledger_size":
        events.extend(dict(events[0]) for _ in range(MAX_INSPECTION_LEDGER_EVENTS))
    else:
        event = semantic_runtime_ledger_event(
            requested=True, enabled=True, result=state, discovered_modules={}
        )
        assert event is not None
        events.append(event)
        if mutation == "runtime_unaccounted_analyzer":
            state["analyzer_status_events"].pop()
        elif mutation == "runtime_missing_call":
            state["llm_call_log"].pop()
        elif mutation == "runtime_unrelated_failure":
            state["llm_call_log"].append(llm_call_record("other_analyzer", ok=False))
        elif mutation == "runtime_no_statuses":
            state["analyzer_status_events"] = None
        elif mutation == "runtime_wrong_source":
            state["llm_call_log"][0]["source_identity"] = "different-source"
        elif mutation == "runtime_wrong_event_source":
            event["source_identity"] = "different-source"
        elif mutation == "runtime_malformed_source":
            state["llm_call_log"][0]["source_identity"] = []
        elif mutation == "runtime_wrong_stage":
            event["stage"] = "unknown_stage"

    assert [finding.rule_id for finding in _reference_coverage_findings(state)] == ["AE1"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_analyzer",
        "missing_call",
        "unknown_call",
        "wrong_source",
        "malformed_call",
        "malformed_error",
        "missing_error",
    ],
)
def test_first_finalization_keeps_ae1_for_unexplained_runtime_failure(mutation: str) -> None:
    """The runtime event derived later cannot be hidden by reference-pass ordering."""
    state = _provider_outage_state("SKILL.md")
    if mutation == "missing_analyzer":
        state["analyzer_status_events"].pop()
    elif mutation == "missing_call":
        state["llm_call_log"].pop()
    elif mutation == "unknown_call":
        state["llm_call_log"].append(llm_call_record("unknown_analyzer", ok=False))
    elif mutation == "wrong_source":
        state["llm_call_log"][0]["source_identity"] = "different-source"
    elif mutation == "malformed_error":
        state["llm_call_log"][0]["error"] = {}
    elif mutation == "missing_error":
        state["llm_call_log"][0].pop("error")
    else:
        state["llm_call_log"][0] = {"node": "unknown_analyzer"}

    assert [finding.rule_id for finding in finalize_inspection_ledger(state)["findings"]] == ["AE1"]


@pytest.mark.parametrize("strict", [False, True])
def test_provider_outage_cli_writes_incomplete_report_and_exits_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, strict: bool, _unavailable_provider: None
) -> None:
    from typer.testing import CliRunner

    from skillspector.cli import app
    from skillspector.nodes.report import report

    (tmp_path / "SKILL.md").write_text("# Example", encoding="utf-8")
    finalized = _finalized_state(_provider_outage_state())
    result = {**finalized, **report(finalized)}
    monkeypatch.setattr("skillspector.cli.graph.invoke", lambda state, config: result)
    monkeypatch.setattr(
        "skillspector.cli.graph.stream",
        lambda state, config, stream_mode: iter([{"report": result}]),
    )
    output = tmp_path / "report.json"
    args = ["scan", str(tmp_path), "--format", "json", "--output", str(output)]
    if strict:
        args.append("--fail-on-incomplete")

    invocation = CliRunner().invoke(app, args)

    assert invocation.exit_code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["issues"] == []
    assert payload["analysis_completeness"]["is_complete"] is False
    assert payload["risk_assessment"]["score"] == 0
    assert payload["risk_assessment"]["recommendation"] == "CAUTION"
