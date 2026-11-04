# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for inspection-ledger event factories."""

from copy import deepcopy

import pytest

from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    LedgerRecordType,
    ParserCause,
    analyzer_status_for_events,
    finalize_ledger,
    format_recovery_policy_accepted,
    inspection_work_id,
    ledger_event,
    outcome_for_llm_batch_failure,
    resource_measurements,
    sanitize_parser_diagnostic,
)


def _format_policy_completeness() -> dict:
    """A real canonical ledger projection, not a manually asserted acceptance flag."""
    from skillspector.semantic_runtime import required_semantic_analyzer_ids

    events = [
        ledger_event(
            analyzer_id=analyzer,
            outcome=(
                LedgerOutcome.PARTIAL
                if analyzer == "semantic_security_discovery"
                else LedgerOutcome.COMPLETED
            ),
            phase="semantic",
            path="SKILL.md",
            reason=(
                LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED
                if analyzer == "semantic_security_discovery"
                else None
            ),
        )
        for analyzer in sorted(required_semantic_analyzer_ids({}))
    ]
    result, _ = finalize_ledger(
        {
            "use_llm": True,
            "components": ["SKILL.md"],
            "inspection_ledger": events,
            "analyzer_status_events": [
                analyzer_status_for_events(event["analyzer_id"], [event]) for event in events
            ],
            "llm_call_log": [
                {"node": event["analyzer_id"], "ok": True, "error": None} for event in events
            ],
        }
    )
    return dict(result)


def test_format_policy_retains_unknown_assessment_and_accurate_coverage() -> None:
    result = _format_policy_completeness()
    assert format_recovery_policy_accepted(result)
    assert result["execution_successful"] is True
    assert result["is_complete"] is False
    assert result["status"] == "partial"
    assert result["coverage_percent"] == 0.0
    assert result["partially_inspected_files"] == 1
    assert result["warnings"][0]["assessment_complete"] is False


@pytest.mark.parametrize(
    "mutation",
    [None, "fatal", "phase", "analyzers", "error_class", "path", "line", "source", "reason"],
)
def test_format_policy_preserves_only_valid_missing_reference_caveats(mutation: str | None) -> None:
    result = _format_policy_completeness()
    row = {
        "outcome": "partial",
        "phase": "reference_resolution",
        "fatal": False,
        "reason_code": "reference_missing",
        "path": "SKILL.md",
        "start_line": 3,
        "end_line": 3,
    }
    changes = {
        "fatal": ("fatal", True),
        "phase": ("phase", "static"),
        "analyzers": ("analyzers", ["static_patterns_tool_misuse"]),
        "error_class": ("error_class", "RuntimeError"),
        "path": ("path", "../outside.py"),
        "line": ("start_line", True),
        "source": ("source_identity", "external/other"),
        "reason": ("reason_code", "reference_unscanned"),
    }
    if mutation is not None:
        key, value = changes[mutation]
        row[key] = value
    result["ledger_exceptions"].append(row)
    assert format_recovery_policy_accepted(result) is (mutation is None)
    assert result["is_complete"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_gap",
        "warning_path",
        "missing_warning",
        "fake_completion",
        "unaccounted",
        "boolean_counter",
        "extra_limitation",
        "duplicate_status",
        "duplicate_warning",
        "legacy_error",
        "wrong_source",
        "scope_gap",
        "missing_semantic_status",
        "fake_static_analyzer",
        "invalid_line",
        "invalid_phase",
        "invalid_scope_reason",
        "wrong_status_reason",
    ],
)
def test_format_policy_projection_rejects_mismatched_or_other_gaps(mutation: str) -> None:
    result = deepcopy(_format_policy_completeness())
    if mutation == "extra_gap":
        result["ledger_exceptions"].append(
            {
                **result["ledger_exceptions"][0],
                "reason_code": LedgerReason.LLM_RESPONSE_REFUSED,
            }
        )
    elif mutation == "warning_path":
        result["warnings"][0]["path"] = "other.py"
    elif mutation == "missing_warning":
        result["warnings"] = []
    elif mutation == "fake_completion":
        result["is_complete"] = True
    elif mutation == "unaccounted":
        result["analyzer_statuses"][0]["unaccounted"] = 1
    elif mutation == "boolean_counter":
        result["analyzer_statuses"][0]["failed"] = False
    elif mutation == "extra_limitation":
        result["limitations"].append("Static parsing did not finish.")
    elif mutation == "duplicate_status":
        result["analyzer_statuses"].append(dict(result["analyzer_statuses"][0]))
    elif mutation == "duplicate_warning":
        result["warnings"].append(dict(result["warnings"][0]))
    elif mutation == "legacy_error":
        result["errors"] = ["other failure"]
    elif mutation == "wrong_source":
        result["warnings"][0]["source_identity"] = "external/other"
    elif mutation == "scope_gap":
        result["scope_exclusions"] = [
            {
                "outcome": "out_of_scope",
                "fatal": False,
                "reason_code": "user_exclusion",
            }
        ]
    elif mutation == "missing_semantic_status":
        result["analyzer_statuses"] = [
            status
            for status in result["analyzer_statuses"]
            if status["analyzer_id"] != "semantic_quality_policy"
        ]
    elif mutation == "fake_static_analyzer":
        for row in [result["warnings"][0], result["ledger_exceptions"][0]]:
            row["analyzers"] = ["static_patterns_agent_snooping"]
    elif mutation == "invalid_line":
        result["ledger_exceptions"][0]["start_line"] = []
    elif mutation == "invalid_phase":
        result["ledger_exceptions"][0]["phase"] = []
    elif mutation == "invalid_scope_reason":
        result["scope_exclusions"] = [
            {
                "outcome": "out_of_scope",
                "fatal": False,
                "reason_code": [],
            }
        ]
    else:
        next(status for status in result["analyzer_statuses"] if status["status"] == "degraded")[
            "reason_code"
        ] = "llm_response_refused"
    assert not format_recovery_policy_accepted(result)


@pytest.mark.parametrize("bad_field", [None, "path", "analyzers", "start_line"])
def test_only_canonical_correlated_runtime_summary_is_accepted(bad_field: str | None) -> None:
    result = _format_policy_completeness()
    row = {
        "outcome": "partial",
        "phase": "semantic_runtime",
        "fatal": False,
        "reason_code": "semantic_runtime_incomplete",
        "path": "SKILL.md",
        "start_line": None,
        "end_line": None,
        "message": "Semantic review is incomplete.",
    }
    if bad_field is not None:
        row[bad_field] = {"path": "other.py", "analyzers": ["other"], "start_line": 1}[bad_field]
    result["ledger_exceptions"].append(row)
    assert format_recovery_policy_accepted(result) is (bad_field is None)


def test_format_policy_requires_every_source_to_keep_its_semantic_statuses() -> None:
    result = _format_policy_completeness()
    source = "external/child-digest"
    for row in [*result["ledger_exceptions"], *result["warnings"]]:
        row["path"] = source + "/" + row["path"]
        row["source_identity"] = source
    for status in result["analyzer_statuses"]:
        status["source_identity"] = source
    assert format_recovery_policy_accepted(result)
    next(
        status
        for status in result["analyzer_statuses"]
        if status["analyzer_id"] == "semantic_quality_policy"
    )["source_identity"] = "external/different"
    assert not format_recovery_policy_accepted(result)


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
