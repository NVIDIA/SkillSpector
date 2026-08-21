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

"""Tests for the meta_analyzer node.

Covers ``LLMMetaAnalyzer`` filtering and partial-batch-failure handling, plus
the LLM-call telemetry and fail-closed construction that drive the report's
degradation signal.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason, finalize_ledger
from skillspector.llm_analyzer_base import Batch, BatchExecutionResult, BatchFailure
from skillspector.models import Finding
from skillspector.nodes.meta_analyzer import (
    LLMMetaAnalyzer,
    _meta_ledger_response,
    meta_analyzer,
)
from skillspector.state import SkillspectorState

MOCK_PATCH_TARGET = "skillspector.llm_analyzer_base.get_chat_model"


def _mock_get_chat_model(*_args, **_kwargs):
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()
    return mock_llm


def _analyzer() -> LLMMetaAnalyzer:
    # Skip __init__ so no LLM client / API key is needed; apply_filter is pure.
    return LLMMetaAnalyzer.__new__(LLMMetaAnalyzer)


def _finding(
    rule_id: str,
    start_line: int,
    end_line: int | None = None,
    severity: str = "CRITICAL",
) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=f"static finding {rule_id}",
        severity=severity,
        confidence=0.9,
        file="requirements.txt",
        start_line=start_line,
        end_line=end_line,
    )


def _llm_item(rule_id: str, start_line: int, **kw: object) -> dict[str, object]:
    item: dict[str, object] = {
        "pattern_id": rule_id,
        "is_vulnerability": True,
        "confidence": 1.0,
        "start_line": start_line,
        "_file": "requirements.txt",
    }
    item.update(kw)
    return item


def test_confirmed_finding_kept_when_model_returns_end_line() -> None:
    """Regression: a static finding with end_line=None must still match a
    confirmation whose end_line is populated (e.g. end_line == start_line, as
    some models return). Previously these confirmed findings were silently
    dropped. See issue #67."""
    findings = [_finding("SC4", 4), _finding("SC4", 5)]
    items = [_llm_item("SC4", 4, end_line=4), _llm_item("SC4", 5, end_line=5)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert {f.start_line for f in kept} == {4, 5}
    assert len(kept) == 2


def test_rejected_finding_still_dropped() -> None:
    """The end_line-agnostic fallback must not resurrect findings the LLM
    rejected (is_vulnerability=False)."""
    findings = [_finding("SC4", 4, severity="MEDIUM")]
    items = [_llm_item("SC4", 4, end_line=4, is_vulnerability=False)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert kept == []


def test_low_confidence_finding_dropped() -> None:
    """Confirmations below the confidence threshold are not kept."""
    findings = [_finding("SC4", 4, severity="MEDIUM")]
    items = [_llm_item("SC4", 4, end_line=4, confidence=0.3)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert kept == []


def test_exact_end_line_match_still_works() -> None:
    """Existing behaviour: when both sides carry the same concrete end_line,
    the finding is kept (no regression from the new fallback)."""
    findings = [_finding("AST1", 21, end_line=21)]
    items = [_llm_item("AST1", 21, end_line=21)]
    batch = Batch(file_path="requirements.txt", content="", findings=findings)

    kept = _analyzer().apply_filter(findings, [(batch, items)])

    assert len(kept) == 1
    assert kept[0].rule_id == "AST1"


def _confirm(pattern_id: str, file: str, start_line: int) -> dict[str, object]:
    """LLM item confirming a finding, as parse_response would emit it."""
    return {
        "pattern_id": pattern_id,
        "is_vulnerability": True,
        "confidence": 0.9,
        "explanation": "confirmed by llm",
        "remediation": "fix it",
        "_file": file,
        "start_line": start_line,
        "end_line": None,
    }


def _lineage_finding(finding_id: str, file: str, start_line: int) -> Finding:
    """Build a finding with an explicit ID for ledger-lineage assertions."""
    return Finding(
        rule_id=finding_id.upper(),
        message=f"static finding {finding_id}",
        finding_id=finding_id,
        severity="MEDIUM",
        confidence=0.9,
        file=file,
        start_line=start_line,
    )


class TestMetaLedgerResponse:
    """Direct contract tests for meta-analysis finding lineage."""

    def test_mixed_batches_distinguish_retained_and_filtered_findings(self) -> None:
        retained = _lineage_finding("retained", "complete.py", 1)
        filtered = _lineage_finding("filtered", "complete.py", 2)
        passed_through = _lineage_finding("passed-through", "failed.py", 3)
        completed_batch = Batch(
            file_path="complete.py",
            content="complete",
            findings=[retained, filtered],
        )
        failed_batch = Batch(
            file_path="failed.py",
            content="failed",
            findings=[passed_through],
        )

        events, status = _meta_ledger_response(
            [completed_batch, failed_batch],
            BatchExecutionResult(
                successful=[(completed_batch, [])],
                failures=[BatchFailure(batch=failed_batch, error_class="TimeoutError")],
            ),
            [retained, passed_through],
        )

        completed, failed = events
        assert completed["outcome"] == "completed"
        assert completed["input_finding_ids"] == ["retained", "filtered"]
        assert completed["emitted_finding_ids"] == ["retained"]
        assert failed["outcome"] == "failed"
        assert failed["input_finding_ids"] == ["passed-through"]
        assert failed["emitted_finding_ids"] == ["passed-through"]
        assert failed["reason_code"] == "llm_batch_failed"
        assert failed["error_class"] == "TimeoutError"
        assert status["status"] == "failed"
        assert status["planned_work"] == [
            {
                "work_id": event["work_id"],
                "path": event["path"],
                "start_line": event["start_line"],
                "end_line": event["end_line"],
            }
            for event in events
        ]

    def test_connection_failure_remains_fatal(self) -> None:
        failed = _lineage_finding("failed", "failed.py", 3)
        failed_batch = Batch(file_path="failed.py", content="failed", findings=[failed])

        events, status = _meta_ledger_response(
            [failed_batch],
            BatchExecutionResult(
                failures=[
                    BatchFailure(
                        batch=failed_batch,
                        error_class="APIConnectionError",
                        reason=LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED,
                    )
                ]
            ),
            [failed],
        )

        assert events[0]["outcome"] is LedgerOutcome.FAILED
        assert events[0]["reason_code"] == LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED
        assert events[0]["message"] == "LLM connection failed after bounded retries."
        assert status["status"] == "failed"

        completeness, _ = finalize_ledger(
            {
                "components": ["failed.py"],
                "findings": [failed],
                "effective_finding_ids": [failed.finding_id],
                "inspection_ledger": events,
                "analyzer_status_events": [status],
            }
        )

        assert completeness["execution_successful"] is False
        assert completeness["ledger_exceptions"][0]["fatal"] is True

    def test_structured_response_failure_is_nonfatal_and_degraded(self) -> None:
        failed = _lineage_finding("failed", "failed.py", 3)
        failed_batch = Batch(file_path="failed.py", content="failed", findings=[failed])

        events, status = _meta_ledger_response(
            [failed_batch],
            BatchExecutionResult(
                failures=[
                    BatchFailure(
                        batch=failed_batch,
                        error_class="ValidationError",
                        reason=LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID,
                    )
                ]
            ),
            [failed],
        )

        assert events[0]["outcome"] is LedgerOutcome.SKIPPED
        assert events[0]["input_finding_ids"] == [failed.finding_id]
        assert events[0]["emitted_finding_ids"] == [failed.finding_id]
        assert status["status"] == "degraded"

        completeness, _ = finalize_ledger(
            {
                "components": ["failed.py"],
                "findings": [failed],
                "effective_finding_ids": [failed.finding_id],
                "inspection_ledger": events,
                "analyzer_status_events": [status],
            }
        )

        assert completeness["execution_successful"] is True
        assert completeness["is_complete"] is False
        assert completeness["ledger_exceptions"][0]["fatal"] is False

    def test_overlapping_batches_do_not_reaccount_completed_finding(self) -> None:
        shared = _lineage_finding("shared", "complete.py", 1)
        failed_only = _lineage_finding("failed-only", "failed.py", 2)
        completed_batch = Batch(file_path="complete.py", content="complete", findings=[shared])
        failed_batch = Batch(
            file_path="failed.py",
            content="failed",
            findings=[shared, failed_only],
        )

        events, _ = _meta_ledger_response(
            [completed_batch, failed_batch],
            BatchExecutionResult(
                successful=[(completed_batch, [])],
                failures=[BatchFailure(batch=failed_batch, error_class="ProviderError")],
            ),
            [shared, failed_only],
        )

        completed, failed = events
        assert completed["input_finding_ids"] == ["shared"]
        assert completed["emitted_finding_ids"] == ["shared"]
        assert failed["input_finding_ids"] == ["failed-only"]
        assert failed["emitted_finding_ids"] == ["failed-only"]

    def test_fully_accounted_failed_batch_does_not_degrade_meta_status(self) -> None:
        shared = _lineage_finding("shared", "complete.py", 1)
        completed_batch = Batch(file_path="complete.py", content="complete", findings=[shared])
        failed_batch = Batch(file_path="complete.py", content="retry", findings=[shared])

        events, status = _meta_ledger_response(
            [completed_batch, failed_batch],
            BatchExecutionResult(
                successful=[(completed_batch, [])],
                failures=[BatchFailure(batch=failed_batch, error_class="ProviderError")],
            ),
            [shared],
        )

        assert len(events) == 1
        assert events[0]["outcome"] == "completed"
        assert status["status"] == "completed"

    def test_empty_failed_batch_does_not_degrade_meta_status(self) -> None:
        empty_batch = Batch(file_path="empty.py", content="empty", findings=[])

        events, status = _meta_ledger_response(
            [empty_batch],
            BatchExecutionResult(
                failures=[BatchFailure(batch=empty_batch, error_class="ProviderError")]
            ),
            [],
        )

        assert events == []
        assert status["status"] == "completed"

    def test_failed_batch_passes_all_findings_through_when_none_are_retained(self) -> None:
        first = _lineage_finding("first", "failed.py", 1)
        second = _lineage_finding("second", "failed.py", 2)
        failed_batch = Batch(
            file_path="failed.py",
            content="failed",
            findings=[first, second],
        )

        events, status = _meta_ledger_response(
            [failed_batch],
            BatchExecutionResult(
                failures=[BatchFailure(batch=failed_batch, error_class="ConnectionError")]
            ),
            [],
        )

        assert events[0]["input_finding_ids"] == ["first", "second"]
        assert events[0]["emitted_finding_ids"] == ["first", "second"]
        assert status["status"] == "failed"


@patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
class TestMetaAnalyzerPartialBatchFailure:
    def _state(self, findings: list[Finding]) -> dict[str, object]:
        return {
            "findings": findings,
            "use_llm": True,
            "file_cache": {"a.py": "code a", "b.py": "code b"},
            "manifest": {},
            "model_config": {},
        }

    def test_unanalysed_findings_survive_a_failed_batch(self) -> None:
        """Findings whose batch failed are kept (no verdict != rejection)."""
        f_confirmed = Finding(rule_id="R1", message="m", file="a.py", start_line=1)
        f_rejected = Finding(rule_id="R2", message="m", file="a.py", start_line=5)
        f_unseen = Finding(rule_id="R1", message="m", file="b.py", start_line=3)

        batch_a = Batch(file_path="a.py", content="code a", findings=[f_confirmed, f_rejected])
        batch_b = Batch(file_path="b.py", content="code b", findings=[f_unseen])

        # batch_b never returned (timeout/429): only batch_a's verdicts exist,
        # and the LLM confirmed R1 but stayed silent on R2 (= rejection).
        partial_results = [(batch_a, [_confirm("R1", "a.py", 1)])]

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch_a, batch_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=partial_results,
            ),
        ):
            result = meta_analyzer(self._state([f_confirmed, f_rejected, f_unseen]))

        filtered = result["findings"]
        kept = {(f.file, f.rule_id) for f in filtered}

        # the real filter still applies to the batch that came back
        assert ("a.py", "R1") in kept
        assert ("a.py", "R2") not in kept
        # the finding the LLM never saw must NOT be silently dropped
        assert ("b.py", "R1") in kept
        assert result["effective_finding_ids"] == [
            f_confirmed.finding_id,
            f_unseen.finding_id,
        ]
        assert result["analyzer_status_events"][0]["status"] == "failed"
        assert "filtered_findings" not in result

        confirmed = next(f for f in filtered if f.file == "a.py")
        assert confirmed.explanation == "confirmed by llm"

    def test_selection_does_not_persist_filtered_findings(self) -> None:
        finding = _lineage_finding("retained", "a.py", 1)
        state = self._state([finding])
        state["use_llm"] = False

        result = meta_analyzer(state)

        assert [returned.finding_id for returned in result["findings"]] == [finding.finding_id]
        assert result["effective_finding_ids"] == [finding.finding_id]
        assert "filtered_findings" not in result

    def test_all_batches_failed_keeps_everything_via_fallback(self) -> None:
        f1 = Finding(rule_id="R1", message="m", file="a.py", start_line=1)
        f2 = Finding(rule_id="R2", message="m", file="b.py", start_line=2)
        batch_a = Batch(file_path="a.py", content="code a", findings=[f1])
        batch_b = Batch(file_path="b.py", content="code b", findings=[f2])

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch_a, batch_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = meta_analyzer(self._state([f1, f2]))

        kept = {(f.file, f.rule_id) for f in result["findings"]}
        assert kept == {("a.py", "R1"), ("b.py", "R2")}
        assert "filtered_findings" not in result

    def test_reconstructed_partial_result_uses_canonical_batch_and_finding_ids(self) -> None:
        rejected = _lineage_finding("rejected", "a.py", 1)
        unseen = _lineage_finding("unseen", "b.py", 2)
        submitted_a = Batch(file_path="a.py", content="code a", findings=[rejected])
        submitted_b = Batch(file_path="b.py", content="code b", findings=[unseen])
        returned_a = Batch(
            file_path="a.py",
            content="code a",
            findings=[_lineage_finding("rejected", "a.py", 1)],
        )

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[submitted_a, submitted_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=[(returned_a, [])],
            ),
        ):
            result = meta_analyzer(self._state([rejected, unseen]))

        assert result["effective_finding_ids"] == [unseen.finding_id]
        assert [finding.finding_id for finding in result["findings"]] == [unseen.finding_id]
        assert result["analyzer_status_events"][0]["status"] == "failed"

    def test_duplicate_return_does_not_account_for_a_missing_batch(self) -> None:
        confirmed = _lineage_finding("confirmed", "a.py", 1)
        unseen = _lineage_finding("unseen", "b.py", 2)
        submitted_a = Batch(file_path="a.py", content="code a", findings=[confirmed])
        submitted_b = Batch(file_path="b.py", content="code b", findings=[unseen])
        returned_a = Batch(
            file_path="a.py",
            content="code a",
            findings=[_lineage_finding("confirmed", "a.py", 1)],
        )

        # A malformed/custom executor can return the same batch twice while
        # omitting another submitted batch. The missing batch must still use
        # fallback filtering; matching result-list lengths is insufficient.
        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[submitted_a, submitted_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=[
                    (returned_a, [_confirm("CONFIRMED", "a.py", 1)]),
                    (returned_a, [_confirm("CONFIRMED", "a.py", 1)]),
                ],
            ),
        ):
            result = meta_analyzer(self._state([confirmed, unseen]))

        assert [finding.finding_id for finding in result["findings"]] == [
            confirmed.finding_id,
            unseen.finding_id,
        ]
        assert result["analyzer_status_events"][0]["status"] == "failed"

    def test_empty_meta_batches_are_not_submitted(self) -> None:
        finding = _lineage_finding("retained", "a.py", 1)
        empty_batch = Batch(file_path="a.py", content="context", findings=[])
        finding_batch = Batch(file_path="a.py", content="finding", findings=[finding])

        with (
            patch.object(
                LLMMetaAnalyzer,
                "get_batches",
                return_value=[empty_batch, finding_batch],
            ),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=[(finding_batch, [_confirm("RETAINED", "a.py", 1)])],
            ) as arun_batches,
        ):
            result = meta_analyzer(self._state([finding]))

        assert arun_batches.await_args.args[0] == [finding_batch]
        assert result["effective_finding_ids"] == [finding.finding_id]

    def test_no_failures_keeps_strict_confirm_or_drop(self) -> None:
        """When every batch returns, unconfirmed findings are dropped as before."""
        f_confirmed = Finding(rule_id="R1", message="m", file="a.py", start_line=1)
        f_rejected = Finding(rule_id="R2", message="m", file="b.py", start_line=2)
        batch_a = Batch(file_path="a.py", content="code a", findings=[f_confirmed])
        batch_b = Batch(file_path="b.py", content="code b", findings=[f_rejected])

        full_results = [
            (batch_a, [_confirm("R1", "a.py", 1)]),
            (batch_b, []),
        ]

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch_a, batch_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=full_results,
            ),
        ):
            result = meta_analyzer(self._state([f_confirmed, f_rejected]))

        kept = {(f.file, f.rule_id) for f in result["findings"]}
        assert kept == {("a.py", "R1")}

    def test_effective_ids_follow_meta_batch_emission_order(self) -> None:
        a_pattern = _lineage_finding("pattern-a", "a.py", 1)
        b_pattern = _lineage_finding("pattern-b", "b.py", 2)
        a_entity = _lineage_finding("entity-a", "a.py", 3)
        b_entity = _lineage_finding("entity-b", "b.py", 4)
        batch_a = Batch(file_path="a.py", content="code a", findings=[a_pattern, a_entity])
        batch_b = Batch(file_path="b.py", content="code b", findings=[b_pattern, b_entity])
        findings = [a_pattern, b_pattern, a_entity, b_entity]

        with (
            patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch_a, batch_b]),
            patch.object(
                LLMMetaAnalyzer,
                "arun_batches",
                new_callable=AsyncMock,
                return_value=[
                    (batch_a, [_confirm("PATTERN-A", "a.py", 1), _confirm("ENTITY-A", "a.py", 3)]),
                    (batch_b, [_confirm("PATTERN-B", "b.py", 2), _confirm("ENTITY-B", "b.py", 4)]),
                ],
            ),
        ):
            result = meta_analyzer(self._state(findings))

        assert result["effective_finding_ids"] == [
            "pattern-a",
            "entity-a",
            "pattern-b",
            "entity-b",
        ]
        completeness, effective_ids = finalize_ledger(
            {
                "components": ["a.py", "b.py"],
                "findings": result["findings"],
                "effective_finding_ids": result["effective_finding_ids"],
                "inspection_ledger": result["inspection_ledger"],
                "analyzer_status_events": result["analyzer_status_events"],
            }
        )

        assert effective_ids == result["effective_finding_ids"]
        assert completeness["execution_successful"] is True
        assert completeness["ledger_exceptions"] == []


def test_local_only_high_finding_never_constructs_llm_analyzer() -> None:
    finding = Finding(
        rule_id="SC9",
        message="concealed executable",
        finding_id="local-finding",
        severity="HIGH",
        file=".hidden.docx!/payload.sh",
        tags=[],
        evidence={"outer_path": ".hidden.docx"},
    )
    state = {
        "findings": [finding],
        "file_cache": {"SKILL.md": "sentinel-visible-content"},
        "component_metadata": [{"path": ".hidden.docx!/payload.sh", "local_only": True}],
        "use_llm": True,
    }

    with patch("skillspector.nodes.meta_analyzer.LLMMetaAnalyzer") as analyzer_cls:
        result = meta_analyzer(state)

    analyzer_cls.assert_not_called()
    assert result["effective_finding_ids"] == ["local-finding"]
    assert result["findings"][0].severity == "HIGH"


# ---------------------------------------------------------------------------
# LLM-call telemetry + fail-closed construction (drives the report's
# degradation signal).
# ---------------------------------------------------------------------------


def _degr_finding(rule_id: str = "P1", severity: str = "HIGH") -> Finding:
    return Finding(
        rule_id=rule_id,
        message="test",
        severity=severity,
        confidence=0.8,
        file="SKILL.md",
        start_line=1,
    )


def _degr_state(**overrides: object) -> SkillspectorState:
    state: SkillspectorState = {
        "findings": [_degr_finding()],
        "use_llm": True,
        "file_cache": {"SKILL.md": "# Skill"},
        "manifest": {},
        "model_config": {},
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_records_ok_true_on_success() -> None:
    finding = _degr_finding()
    batch = Batch(file_path="SKILL.md", content="# Skill", findings=[finding])
    with (
        patch("skillspector.llm_analyzer_base.get_chat_model", return_value=MagicMock()),
        patch.object(LLMMetaAnalyzer, "get_batches", return_value=[batch]),
        patch(
            "skillspector.nodes.meta_analyzer.LLMMetaAnalyzer.arun_batches",
            new_callable=AsyncMock,
            return_value=[(batch, [])],
        ),
    ):
        result = meta_analyzer(_degr_state(findings=[finding]))
    assert result["llm_call_log"] == [{"node": "meta_analyzer", "ok": True, "error": None}]
    assert "filtered_findings" not in result


def test_construction_failure_is_caught_not_raised() -> None:
    """Regression: the chat model is constructed INSIDE the try, so a construction
    failure degrades (records ok=False, preserves findings) instead of crashing
    the whole graph."""
    with patch(
        "skillspector.llm_analyzer_base.get_chat_model",
        side_effect=RuntimeError("provider construction failed"),
    ):
        result = meta_analyzer(_degr_state())  # must not raise
    # Findings are preserved via the fallback path...
    assert len(result["findings"]) == 1
    assert "filtered_findings" not in result
    # ...and the failure is recorded so the report can flag degradation.
    log = result["llm_call_log"]
    assert log[0]["node"] == "meta_analyzer"
    assert log[0]["ok"] is False
    assert "provider construction failed" in log[0]["error"]
    status = result["analyzer_status_events"][0]
    assert status["status"] == "unavailable"
    assert "reason_code" not in status


def test_credential_error_propagates_instead_of_being_labelled_unavailable() -> None:
    """Only actual credential failures propagate; provider failures have no guessed cause."""
    with patch(
        "skillspector.llm_analyzer_base.get_chat_model",
        side_effect=ValueError("No LLM API key configured."),
    ):
        try:
            meta_analyzer(_degr_state())
        except ValueError as error:
            assert "API key" in str(error)
        else:
            raise AssertionError("credential failure must not be reported as unavailable")


def test_use_llm_false_records_nothing() -> None:
    result = meta_analyzer(_degr_state(use_llm=False))
    assert "llm_call_log" not in result
    assert "filtered_findings" not in result


def test_no_findings_records_nothing() -> None:
    result = meta_analyzer(_degr_state(findings=[]))
    assert "llm_call_log" not in result
    assert "filtered_findings" not in result
