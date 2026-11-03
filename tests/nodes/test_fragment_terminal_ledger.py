# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fragment failures retain one terminal original-line result and all failure facts."""

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
    assert len(status["planned_work"]) == 1
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
