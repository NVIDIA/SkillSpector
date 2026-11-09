# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fragment failures retain one terminal original-line result and all failure facts."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

import skillspector.llm_analyzer_base as lab
import skillspector.nodes.meta_analyzer as meta
from skillspector.inspection_ledger import LedgerReason, finalize_ledger
from skillspector.models import Finding


@pytest.mark.parametrize("phase", ["semantic", "meta"])
@pytest.mark.parametrize("successful_finding", [False, True])
@pytest.mark.parametrize(
    "reasons",
    [
        (LedgerReason.RUNTIME_LIMIT, LedgerReason.RUNTIME_LIMIT),
        (
            LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID,
            LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID,
        ),
        (LedgerReason.RUNTIME_LIMIT, LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID),
        (LedgerReason.LLM_BATCH_FAILED, LedgerReason.RUNTIME_LIMIT),
    ],
)
def test_two_failed_fragments_have_unique_terminal_accounting(phase, successful_finding, reasons):
    finding = Finding(
        rule_id="SYNTHETIC",
        message="Retained concern",
        severity="HIGH",
        file="generated.js",
        start_line=1,
    )
    batches = [
        lab.Batch(
            file_path="generated.js",
            content="x" * 10,
            start_line=1,
            end_line=1,
            findings=[finding] if phase == "meta" else [],
            source_start_column=left,
            source_end_column=right,
            source_line_length=40,
        )
        for left, right in [(0, 10), (10, 20), (20, 30), (30, 40)]
    ]
    outcome = lab.BatchExecutionResult(
        successful=[
            (batch, [finding] if successful_finding and phase == "semantic" else [])
            for batch in batches[2:]
        ],
        failures=[
            lab.BatchFailure(batch, f"SyntheticFailure{index}", reason)
            for index, (batch, reason) in enumerate(zip(batches[:2], reasons, strict=True))
        ],
    )
    if phase == "meta":
        events, status = meta._meta_ledger_response(batches, outcome, [finding])
    else:
        # One successful sibling reports the concern, without duplicating its origin.
        if successful_finding:
            outcome.successful[1] = (batches[3], [])
        events, status = lab.ledger_events_for_batches("generated", outcome)
    findings = [finding] if phase == "meta" or successful_finding else []
    completeness, _ = finalize_ledger(
        {
            "components": ["generated.js"],
            "findings": findings,
            "inspection_ledger": events,
            "analyzer_status_events": [status],
            "meta_review_required": phase == "meta",
        }
    )
    assert not any(
        row["reason_code"] == LedgerReason.UNACCOUNTED_WORK
        for row in completeness["ledger_exceptions"]
    )
    terminal = [event for event in events if event["record_type"] == "work_item"]
    assert len({event["work_id"] for event in terminal}) == len(terminal)
    assert len(terminal) == 1
    assert (
        sum(target["work_id"] == terminal[0]["work_id"] for target in status["planned_work"]) == 1
    )
    expected_fatal = LedgerReason.LLM_BATCH_FAILED in reasons
    assert completeness["execution_successful"] is not expected_fatal
    assert status["status"] == ("failed" if expected_fatal else "degraded")
    assert set(reasons) <= {row["reason_code"] for row in completeness["ledger_exceptions"]}
    assert {"SyntheticFailure0", "SyntheticFailure1"} <= {
        row.get("error_class") for row in completeness["ledger_exceptions"]
    }
    if findings:
        assert any(finding.finding_id in row["emitted_finding_ids"] for row in events)


@pytest.mark.parametrize("phase", ["semantic", "meta"])
def test_real_fragment_aggregation_keeps_runtime_failures_partial(monkeypatch, phase):
    finding = Finding(
        rule_id="SYNTHETIC",
        message="Retained concern",
        severity="HIGH",
        file="generated.js",
        start_line=1,
    )
    cls = meta.LLMMetaAnalyzer if phase == "meta" else lab.LLMAnalyzerBase
    analyzer = cls.__new__(cls)
    analyzer.base_prompt = (
        meta.PER_FILE_ANALYSIS_PROMPT if phase == "meta" else "Review supplied source"
    )
    analyzer._input_budget = 16000
    analyzer._token_counter = lambda text: len(text.encode())
    analyzer._timeout = None
    analyzer._usage_collector = MagicMock()
    analyzer._usage_collector.snapshot.return_value = []
    text = "x" * 100000
    batches = analyzer.get_batches(
        ["generated.js"], {"generated.js": text}, [finding] if phase == "meta" else []
    )
    assert len(batches) > 3
    successes = {batches[0].source_start_column, batches[1].source_start_column}

    def invoke(batch, prompt, **kwargs):
        if batch.source_start_column not in successes:
            raise lab.LLMRuntimeLimitError("synthetic deadline reached")
        return batch, [finding] if phase == "semantic" and batch.source_start_column == 0 else []

    if phase == "meta":

        async def ainvoke(batch, prompt, **kwargs):
            return invoke(batch, prompt, **kwargs)

        monkeypatch.setattr(analyzer, "_ainvoke_batch_with_retries", ainvoke)
        monkeypatch.setattr(meta, "LLMMetaAnalyzer", lambda **kwargs: analyzer)
        response = meta.meta_analyzer(
            {
                "components": ["generated.js"],
                "file_cache": {"generated.js": text},
                "findings": [finding],
                "manifest": {},
                "model_config": {"default": "synthetic"},
            }
        )
        events, status = response["inspection_ledger"], response["analyzer_status_events"][0]
        findings = response["findings"]
        outcome = analyzer._last_batch_outcome
    else:
        monkeypatch.setattr(analyzer, "_invoke_batch_with_retries", invoke)
        outcome = analyzer.run_batches_detailed(batches)
        events, status = lab.ledger_events_for_batches("generated", outcome)
        findings = [finding]
    assert len(outcome.failures) > 1
    completeness, _ = finalize_ledger(
        {
            "components": ["generated.js"],
            "findings": findings,
            "inspection_ledger": events,
            "analyzer_status_events": [status],
            "meta_review_required": phase == "meta",
        }
    )
    assert status["status"] == "degraded"
    assert completeness["execution_successful"] is True
    assert not any(
        row["reason_code"] == LedgerReason.UNACCOUNTED_WORK
        for row in completeness["ledger_exceptions"]
    )
    assert len(status["planned_work"]) == 1
    assert any(finding.finding_id in event["emitted_finding_ids"] for event in events)


def _recovery_fixture(phase="semantic"):
    finding = Finding(
        rule_id="SYNTHETIC_RECOVERY",
        message="Original HIGH concern must survive partial recovery",
        severity="HIGH",
        file="generated.js",
        start_line=1,
    )
    batches = [
        lab.Batch(
            file_path=finding.file,
            content="x" * 10,
            start_line=1,
            end_line=1,
            findings=[finding] if phase == "meta" else [],
            source_start_column=left,
            source_end_column=left + 10,
            source_line_length=40,
        )
        for left in (0, 10, 20, 30)
    ]
    return finding, batches


def _assert_incomplete_recovery_line(events, status, findings, *, phase, fatal):
    completeness, _ = finalize_ledger(
        {
            "components": ["generated.js"],
            "findings": findings,
            "inspection_ledger": events,
            "analyzer_status_events": [status],
            "meta_review_required": phase == "meta",
        }
    )
    terminal = [event for event in events if event["record_type"] == "work_item"]
    assert len(terminal) == 1
    planned_ids = [target["work_id"] for target in status["planned_work"]]
    expected_ids = [
        event["work_id"] for event in events if event["record_type"] != "scope_boundary"
    ]
    # A preserved fatal SYSTEM fact is additional accounting, not a second
    # terminal original-line WORK_ITEM or proof of successful source coverage.
    assert sorted(planned_ids) == sorted(expected_ids)
    assert len(planned_ids) == len(set(planned_ids))
    assert planned_ids.count(terminal[0]["work_id"]) == 1
    assert (terminal[0]["start_line"], terminal[0]["end_line"]) == (1, 1)
    assert terminal[0]["outcome"] != "completed"
    assert completeness["is_complete"] is False
    assert completeness["execution_successful"] is not fatal
    assert status["status"] == ("failed" if fatal else "degraded")
    assert not any(
        row["reason_code"] == LedgerReason.UNACCOUNTED_WORK
        for row in completeness["ledger_exceptions"]
    )
    return completeness


@pytest.mark.parametrize("provider_failure", [False, True])
def test_generic_partial_recovery_preserves_high_without_completing_fragment_line(
    monkeypatch, provider_failure
):
    finding, batches = _recovery_fixture()
    analyzer = lab.LLMAnalyzerBase.__new__(lab.LLMAnalyzerBase)
    analyzer.base_prompt = "Review the original source."
    analyzer._timeout = None

    def invoke(batch, prompt, **kwargs):
        if batch.source_start_column == 0:
            return lab._RecoveredBatch(
                batch,
                [finding],
                False,
                RuntimeError("Synthetic provider failure") if provider_failure else None,
            )
        return batch, []

    monkeypatch.setattr(analyzer, "_invoke_batch_with_retries", invoke)
    outcome = analyzer.run_batches_detailed(batches)
    assert len(outcome.successful) == 3
    assert outcome.recovered == [(batches[0], [finding])]
    assert bool(outcome.failures) is provider_failure
    assert bool(outcome.warnings) is not provider_failure
    projected, _, partial = lab._project_line_fragments(
        outcome.successful, [*outcome.failures, *outcome.warnings], outcome.recovered
    )
    assert projected == [], "Recovered values do not establish successful source coverage"
    assert partial[0][1] == [finding]
    events, status = lab.ledger_events_for_batches("generated", outcome)
    completeness = _assert_incomplete_recovery_line(
        events, status, [finding], phase="semantic", fatal=provider_failure
    )
    expected = (
        LedgerReason.LLM_BATCH_FAILED
        if provider_failure
        else LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED
    )
    assert expected in {row["reason_code"] for row in completeness["ledger_exceptions"]}
    assert any(finding.finding_id in event["emitted_finding_ids"] for event in events)
    system_failures = [
        event
        for event in events
        if event["record_type"] == "system" and event["outcome"] == "failed"
    ]
    assert bool(system_failures) is provider_failure
    if provider_failure:
        assert any(event.get("error_class") == "RuntimeError" for event in system_failures)


@pytest.mark.parametrize("phase", ["semantic", "meta"])
def test_recovered_warning_and_runtime_gap_retain_both_incomplete_reason_facts(phase):
    finding, batches = _recovery_fixture(phase)
    outcome = lab.BatchExecutionResult(
        successful=[(batch, []) for batch in batches[2:]],
        recovered=[(batches[0], [finding] if phase == "semantic" else [])],
        warnings=[
            lab.BatchFailure(
                batches[0], "UnverifiedAssessment", LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED
            )
        ],
        failures=[lab.BatchFailure(batches[1], "LLMRuntimeLimitError", LedgerReason.RUNTIME_LIMIT)],
    )
    if phase == "meta":
        events, status = meta._meta_ledger_response(batches, outcome, [finding])
    else:
        events, status = lab.ledger_events_for_batches("generated", outcome)
    completeness = _assert_incomplete_recovery_line(
        events, status, [finding], phase=phase, fatal=False
    )
    assert {LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED, LedgerReason.RUNTIME_LIMIT} <= {
        row["reason_code"] for row in completeness["ledger_exceptions"]
    }
    assert any(
        row.get("error_class") == "LLMRuntimeLimitError"
        for row in completeness["ledger_exceptions"]
    )
    assert not any(event["outcome"] == "failed" for event in events)


@pytest.mark.parametrize("provider_failure", [False, True])
def test_meta_reconciliation_canonicalizes_recovered_values_and_terminal_evidence(
    provider_failure,
):
    finding, batches = _recovery_fixture("meta")
    returned = replace(batches[0])
    values = [{"finding_id": finding.finding_id, "is_vulnerability": False}]
    terminal = lab.BatchFailure(
        returned,
        "RuntimeError" if provider_failure else "UnverifiedAssessment",
        LedgerReason.LLM_BATCH_FAILED
        if provider_failure
        else LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED,
    )
    outcome = lab.BatchExecutionResult(
        successful=[(replace(batch), []) for batch in batches[1:]],
        recovered=[(returned, values)],
        failures=[terminal] if provider_failure else [],
        warnings=[] if provider_failure else [terminal],
    )
    canonical = meta._account_for_missing_meta_batches(batches, outcome)
    assert canonical.recovered[0][0] is batches[0]
    assert canonical.recovered[0][1] is values
    assert all(
        returned_batch is batch
        for (returned_batch, _), batch in zip(canonical.successful, batches[1:], strict=True)
    )
    terminal_rows = [*canonical.failures, *canonical.warnings]
    assert len(terminal_rows) == 1
    assert terminal_rows[0].batch is batches[0]
    assert terminal_rows[0].reason is terminal.reason
    assert terminal_rows[0].error_class == terminal.error_class
    assert not any(row.error_class == "MissingBatchResult" for row in terminal_rows)
    assert outcome.recovered[0][0] is returned, "Reconciliation must not mutate returned evidence"


@pytest.mark.parametrize("provider_failure", [False, True])
def test_real_meta_partial_fragment_recovery_cannot_disagree_away_original_high(
    monkeypatch, provider_failure
):
    finding, _ = _recovery_fixture("meta")
    analyzer = meta.LLMMetaAnalyzer.__new__(meta.LLMMetaAnalyzer)
    analyzer.base_prompt = meta.PER_FILE_ANALYSIS_PROMPT
    analyzer._input_budget = 16000
    analyzer._token_counter = lambda text: len(text.encode())
    analyzer._timeout = None
    analyzer._pack_files = False
    analyzer._usage_collector = MagicMock()
    analyzer._usage_collector.snapshot.return_value = []
    text = "x" * 100000
    seen = []

    async def invoke(batch, prompt, **kwargs):
        seen.append((batch.source_start_column, batch.source_end_column))
        assessment = {
            "finding_id": finding.finding_id,
            "pattern_id": finding.rule_id,
            "start_line": 1,
            "is_vulnerability": False,
            "confidence": 0.99,
            "intent": "benign",
            "impact": "low",
            "_file": finding.file,
        }
        if batch.source_start_column == 0:
            return lab._RecoveredBatch(
                batch,
                [assessment],
                False,
                RuntimeError("Synthetic provider failure") if provider_failure else None,
            )
        return batch, [assessment]

    monkeypatch.setattr(analyzer, "_ainvoke_batch_with_retries", invoke)
    monkeypatch.setattr(meta, "LLMMetaAnalyzer", lambda **kwargs: analyzer)
    response = meta.meta_analyzer(
        {
            "components": [finding.file],
            "file_cache": {finding.file: text},
            "findings": [finding],
            "manifest": {},
            "model_config": {"default": "synthetic"},
        }
    )
    assert len(seen) > 2 and len(seen) == len(set(seen))
    assert any(left == 0 for left, _ in seen)
    assert len(analyzer._last_batch_outcome.recovered) == 1
    assert bool(analyzer._last_batch_outcome.failures) is provider_failure
    assert bool(analyzer._last_batch_outcome.warnings) is not provider_failure
    assert finding.finding_id in response["effective_finding_ids"]
    retained = next(item for item in response["findings"] if item.finding_id == finding.finding_id)
    assert retained.severity == "HIGH" and retained.message == finding.message
    assert retained.evidence["llm_review_outcome"] == "failed"
    assert "llm-unconfirmed" not in retained.tags
    assert response["llm_call_log"][0]["ok"] is not provider_failure
    status = response["analyzer_status_events"][0]
    completeness = _assert_incomplete_recovery_line(
        response["inspection_ledger"],
        status,
        response["findings"],
        phase="meta",
        fatal=provider_failure,
    )
    assert not any(
        row.get("error_class") == "MissingBatchResult" for row in completeness["ledger_exceptions"]
    )
