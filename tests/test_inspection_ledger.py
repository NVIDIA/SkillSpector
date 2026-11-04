# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for inspection-ledger event factories."""

import pytest

from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    ParserCause,
    analyzer_status_for_events,
    inspection_work_id,
    ledger_event,
    outcome_for_llm_batch_failure,
    resource_measurements,
    sanitize_parser_diagnostic,
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


def test_analyzer_status_for_scope_boundaries_has_no_planned_work() -> None:
    """Scope-boundary records are not work assigned to the analyzer."""
    event = ledger_event(
        outcome=LedgerOutcome.OUT_OF_SCOPE,
        record_type=LedgerRecordType.SCOPE_BOUNDARY,
        phase="static",
        analyzer_id="static_test",
        path="assets/font.ttf",
        reason=LedgerReason.BINARY_CONTENT,
    )

    status = analyzer_status_for_events("static_test", [event])

    assert status == {
        "analyzer_id": "static_test",
        "status": "not_applicable",
        "planned_work": [],
        "reason_code": LedgerReason.NO_APPLICABLE_FILES,
        "message": "No files matched this analyzer's applicability contract.",
    }


@pytest.mark.parametrize(
    ("reason", "expected_outcome"),
    [
        (LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID, LedgerOutcome.SKIPPED),
        (LedgerReason.LLM_BATCH_FAILED, LedgerOutcome.FAILED),
        (LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED, LedgerOutcome.FAILED),
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


@pytest.mark.parametrize(
    "unit", ["characters", "bytes", "findings", "artifacts", "depth", "records"]
)
@pytest.mark.parametrize("value", [0, (1 << 53) + 1, (1 << 63) - 1])
def test_resource_counters_preserve_zero_and_exact_integers(unit: str, value: int) -> None:
    measurements = {f"observed_{unit}": value, f"limit_{unit}": value}
    assert resource_measurements(measurements) == measurements


@pytest.mark.parametrize(
    "unit", ["characters", "bytes", "findings", "artifacts", "depth", "records"]
)
@pytest.mark.parametrize(
    "value", [True, False, -1, 1.5, "12", float("nan"), float("inf"), 1 << 4096]
)
def test_resource_counters_reject_unsafe_or_inexact_values(unit: str, value: object) -> None:
    assert resource_measurements({f"observed_{unit}": value, f"limit_{unit}": value}) == {}


@pytest.mark.parametrize("value", [0, 0.0, 0.25, 1e308])
def test_resource_seconds_preserve_finite_values_and_missing_side(value: int | float) -> None:
    assert resource_measurements({"observed_seconds": value}) == {"observed_seconds": value}


@pytest.mark.parametrize(
    "value", [True, False, -0.1, "0.1", float("nan"), float("inf"), float("-inf")]
)
def test_resource_seconds_reject_invalid_values(value: object) -> None:
    assert resource_measurements({"observed_seconds": value, "limit_seconds": value}) == {}


def test_resource_measurements_never_copy_stage_or_arbitrary_payloads() -> None:
    assert resource_measurements(
        {"observed_bytes": 0, "limit_bytes": 4, "stage": "private-stage", "payload": "secret"}
    ) == {"observed_bytes": 0, "limit_bytes": 4}


def _parser_diagnostic() -> dict[str, object]:
    return {
        "cause": "resource_limit",
        "check": "command_word_unquoted_characters",
        "source_start_offset": 2,
        "source_end_offset": 4099,
        "start_line": 2,
        "end_line": 2,
        "unchecked_scope": "command_word",
        "location_precision": "source_span",
        "observed_is_lower_bound": True,
        "full_input_checked": False,
        "observed_characters": 4097,
        "limit_characters": 4096,
    }


def test_parser_diagnostics_allow_only_known_identifiers_and_safe_measurements() -> None:
    original = _parser_diagnostic()
    original.update(
        {"command": "private payload", "stage": "private stage", "observed_bytes": True}
    )
    projected = sanitize_parser_diagnostic(original)
    assert projected is not None
    assert projected["cause"] is ParserCause.RESOURCE_LIMIT
    assert projected["observed_characters"] == 4097
    assert projected["observed_is_lower_bound"] is True
    assert "command" not in projected
    assert "stage" not in projected
    assert "observed_bytes" not in projected


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("cause", "untrusted cause"),
        ("check", "untrusted check"),
        ("unchecked_scope", "untrusted scope"),
        ("location_precision", []),
        ("source_start_offset", True),
        ("source_start_offset", -1),
        ("source_end_offset", 1),
        ("start_line", 0),
        ("end_line", 1),
        ("observed_is_lower_bound", 1),
        ("full_input_checked", "true"),
    ],
)
def test_parser_diagnostics_reject_invalid_envelopes(key: str, value: object) -> None:
    diagnostic = _parser_diagnostic()
    diagnostic[key] = value
    assert sanitize_parser_diagnostic(diagnostic) is None


@pytest.mark.parametrize(
    "precision,checked", [("source_span", False), ("enclosing_expression", True)]
)
def test_malformed_diagnostic_requires_complete_exact_source_evidence(
    precision: str, checked: bool
) -> None:
    diagnostic = _parser_diagnostic()
    diagnostic.update(
        {"cause": "malformed_input", "location_precision": precision, "full_input_checked": checked}
    )
    projected = sanitize_parser_diagnostic(diagnostic)
    assert projected is not None
    assert projected["cause"] is ParserCause.BOUNDARY_UNRESOLVED
