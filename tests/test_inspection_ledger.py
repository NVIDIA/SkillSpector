# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for inspection-ledger event factories."""

import pytest

from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    analyzer_status_for_events,
    inspection_work_id,
    ledger_event,
    outcome_for_llm_batch_failure,
)


def test_completed_ledger_event_references_findings_without_copying_them() -> None:
    """Completed producer rows keep only emitted finding IDs."""
    event = ledger_event(
        outcome=LedgerOutcome.COMPLETED,
        phase="behavioral",
        analyzer_id="behavioral_ast",
        path="scripts/install.py",
        emitted_finding_ids=["finding-a", "finding-b"],
    )

    assert event["work_id"] == inspection_work_id(
        "behavioral_ast", "scripts/install.py", None, None
    )
    assert event["input_finding_ids"] == []
    assert event["emitted_finding_ids"] == ["finding-a", "finding-b"]
    assert "reason_code" not in event
    assert "fatal" not in event


def test_inspection_work_id_separates_analyzers_and_overlapping_chunks() -> None:
    """Distinct analyzer work and overlapping chunks cannot collide in the ledger."""
    first_chunk = inspection_work_id("semantic_a", "scripts/check.py", 1, 100)
    overlapping_chunk = inspection_work_id("semantic_a", "scripts/check.py", 51, 150)
    other_analyzer = inspection_work_id("semantic_b", "scripts/check.py", 1, 100)

    assert len({first_chunk, overlapping_chunk, other_analyzer}) == 3


def test_analyzer_status_for_events_summarizes_terminal_work() -> None:
    """The shared helper exposes only planned work and its aggregate outcome."""
    event = ledger_event(
        outcome=LedgerOutcome.SKIPPED,
        analyzer_id="static_test",
        phase="static",
        path="evals/evals.json",
        reason=LedgerReason.EVAL_DATASET,
    )

    status = analyzer_status_for_events("static_test", [event])

    assert status == {
        "analyzer_id": "static_test",
        "status": "degraded",
        "planned_work": [
            {
                "work_id": event["work_id"],
                "path": "evals/evals.json",
                "start_line": None,
                "end_line": None,
            }
        ],
    }


@pytest.mark.parametrize(
    ("reason", "expected_outcome"),
    [
        (LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID, LedgerOutcome.SKIPPED),
        (LedgerReason.LLM_BATCH_FAILED, LedgerOutcome.FAILED),
        (LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED, LedgerOutcome.FAILED),
        (LedgerReason.LLM_RATE_LIMIT_RETRIES_EXHAUSTED, LedgerOutcome.FAILED),
    ],
)
def test_outcome_for_llm_batch_failure_preserves_failure_policy(
    reason: LedgerReason, expected_outcome: LedgerOutcome
) -> None:
    """Only exhausted malformed structured responses are non-fatal."""
    assert outcome_for_llm_batch_failure(reason) is expected_outcome


def test_failed_producer_ledger_event_cannot_reference_findings() -> None:
    """Failed producers do not claim findings they did not successfully emit."""
    with pytest.raises(ValueError, match="cannot reference findings"):
        ledger_event(
            outcome=LedgerOutcome.FAILED,
            phase="cache",
            reason=LedgerReason.READ_ERROR,
            path="scripts/install.py",
            emitted_finding_ids=["finding-a"],
        )


def test_completed_meta_event_emits_a_subset_of_its_inputs() -> None:
    """Completed meta work can retain only the findings it confirms."""
    event = ledger_event(
        outcome=LedgerOutcome.COMPLETED,
        phase="meta",
        analyzer_id="meta_analyzer",
        path="SKILL.md",
        input_finding_ids=["finding-a", "finding-b"],
        emitted_finding_ids=["finding-a"],
    )

    assert event["input_finding_ids"] == ["finding-a", "finding-b"]
    assert event["emitted_finding_ids"] == ["finding-a"]


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (LedgerOutcome.FAILED, LedgerReason.LLM_BATCH_FAILED),
        (LedgerOutcome.SKIPPED, LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID),
    ],
)
def test_unprocessed_meta_event_must_pass_every_input_through(
    outcome: LedgerOutcome, reason: LedgerReason
) -> None:
    """Failed or skipped meta work is fail-closed and preserves every input ID."""
    event = ledger_event(
        outcome=outcome,
        phase="meta",
        analyzer_id="meta_analyzer",
        reason=reason,
        path="SKILL.md",
        input_finding_ids=["finding-a", "finding-b"],
        emitted_finding_ids=["finding-a", "finding-b"],
    )

    assert event["emitted_finding_ids"] == event["input_finding_ids"]


def test_ledger_event_rejects_absolute_paths() -> None:
    """Ledger paths are always report-safe relative POSIX paths."""
    with pytest.raises(ValueError, match="relative POSIX path"):
        ledger_event(
            outcome=LedgerOutcome.FAILED,
            phase="cache",
            reason=LedgerReason.READ_ERROR,
            path="/private/tmp/secret.py",
        )


def test_failed_event_includes_sanitized_failure_metadata_only_when_provided() -> None:
    """Failure metadata is structured and absent unless a producer supplies it."""
    event = ledger_event(
        outcome=LedgerOutcome.FAILED,
        phase="cache",
        reason=LedgerReason.READ_ERROR,
        path="scripts/install.py",
        error_class="PermissionError",
        stage="read",
    )

    assert event["error_class"] == "PermissionError"
    assert event["stage"] == "read"
