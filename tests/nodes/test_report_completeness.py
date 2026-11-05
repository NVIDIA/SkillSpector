# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-content omissions are bounded, additive, and format-local."""

from __future__ import annotations

import importlib
import json
from copy import deepcopy
from itertools import repeat

import pytest
from markdown_it import MarkdownIt

from skillspector.models import Finding
from skillspector.nodes.report import prepare_report, report, serialize_report
from skillspector.report_completeness import (
    MAX_OMISSION_REASON_CHARS,
    MAX_OMISSION_REASONS,
    MAX_OMITTED_COUNT,
    build_report_completeness,
    merge_report_completeness,
    report_completeness_notice,
)
from skillspector.sarif_models import validate_sarif_report
from skillspector.suppression import Baseline, SuppressionRule

report_module = importlib.import_module("skillspector.nodes.report")
FORMATS = ("json", "sarif", "terminal", "markdown")
COMPLETE = {
    "is_complete": True,
    "omission_reasons": [],
    "omitted_records": 0,
    "omitted_skills": 0,
}


def _assert_projection(result: dict[str, object], fmt: str, expected: dict) -> None:
    assert result["report_completeness"] == expected
    if fmt == "json":
        assert json.loads(result["report_body"])["report_completeness"] == expected
    elif fmt == "sarif":
        sarif = json.loads(result["report_body"])
        assert sarif == result["sarif_report"]
        properties = sarif["runs"][0]["invocations"][0]["properties"]
        assert properties["reportCompleteness"] == expected
        validate_sarif_report(sarif)
    else:
        body = result["report_body"]
        if fmt == "markdown":
            body = MarkdownIt().enable("table").render(body)
        assert report_completeness_notice(expected) in body


def _occurrences(rule_id: str, count: int) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=f"{rule_id} finding",
        severity="HIGH",
        confidence=1.0,
        file=f"{rule_id}.py",
        occurrences=[
            {"file": f"{rule_id}.py", "start_line": index + 1, "end_line": index + 1}
            for index in range(count)
        ],
    )


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("show_suppressed", [False, True])
def test_full_projections_and_normal_layout_differences_are_complete(
    fmt: str, show_suppressed: bool
) -> None:
    active = _occurrences("ACTIVE", 2)
    suppressed = _occurrences("SUPPRESSED", 1)
    state = {
        "findings": [active, suppressed],
        "use_llm": False,
        "show_suppressed": show_suppressed,
        "baseline": Baseline(rules=[SuppressionRule(rule_id="SUPPRESSED", reason="accepted")]),
        "structured_summaries": [
            {
                "id": "SSR-1",
                "message": "Structured bundle detected",
                "file": "bundle.json",
                "protocol": "AISOP V1",
                "declared_tools": ["calendar"],
            }
        ],
        "inference_usage": [
            {
                "node": "meta_analyzer",
                "provider": "openai",
                "model": "safe-model",
                "request_kind": "structured_output",
                "usage_source": "provider_response",
                "model_source": "requested_model",
                "total_tokens": 10,
            }
        ],
    }
    prepared = prepare_report(state)
    original = deepcopy(prepared)

    result = serialize_report(prepared, fmt)

    _assert_projection(result, fmt, COMPLETE)
    assert prepared.report_completeness == COMPLETE
    assert prepared.public_fields["report_completeness"] == COMPLETE
    assert len(prepared.display_findings) == 2
    assert len(prepared.suppressed) == 1
    assert prepared == original
    legacy = report({**state, "output_format": fmt})
    assert set(legacy) == {*prepared.public_fields, "report_body", "sarif_report"}
    assert legacy["report_completeness"] == COMPLETE
    assert (
        legacy["sarif_report"]["runs"][0]["invocations"][0]["properties"][
            "reportCompleteness"
        ]
        == COMPLETE
    )


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("limit", [0, 2, 3, 4, 5])
def test_canonical_finding_and_suppression_omissions_seed_all_formats(
    monkeypatch: pytest.MonkeyPatch, fmt: str, limit: int
) -> None:
    state = {
        "findings": [_occurrences("ACTIVE", 3), _occurrences("SUPPRESSED", 2)],
        "use_llm": False,
        "baseline": Baseline(rules=[SuppressionRule(rule_id="SUPPRESSED", reason="accepted")]),
    }
    full = prepare_report(state)
    monkeypatch.setattr(report_module, "MAX_FINDING_OUTPUT_RECORDS", limit)
    prepared = prepare_report(state)
    original = deepcopy(prepared)
    reasons = []
    if limit < 3:
        reasons.append("finding_record_limit")
    if limit < 5:
        reasons.append("suppression_record_limit")
    expected = {
        "is_complete": limit == 5,
        "omission_reasons": reasons,
        "omitted_records": 5 - limit,
        "omitted_skills": 0,
    }

    result = serialize_report(prepared, fmt)

    _assert_projection(result, fmt, expected)
    assert prepared.report_completeness == expected
    assert prepared.public_fields["report_completeness"] == expected
    assert prepared.report_completeness is not prepared.public_fields["report_completeness"]
    assert prepared.public_fields["active_findings"] == full.public_fields["active_findings"]
    assert prepared.analysis_completeness == full.analysis_completeness
    assert prepared.execution_successful == full.execution_successful
    assert (
        prepared.risk_score,
        prepared.risk_severity,
        prepared.risk_recommendation,
    ) == (full.risk_score, full.risk_severity, full.risk_recommendation)
    assert prepared == original


@pytest.mark.parametrize("fmt", FORMATS)
def test_incomplete_analysis_can_have_complete_report_content(fmt: str) -> None:
    analysis = {
        "is_complete": False,
        "status": "failed",
        "execution_successful": False,
        "coverage_percent": 50.0,
        "total_components": 2,
        "fully_inspected_files": 1,
        "partially_inspected_files": 0,
        "entirely_uninspected_files": 1,
        "ledger_exceptions": [
            {
                "path": "broken.zip",
                "phase": "nested_artifact_inspection",
                "outcome": "failed",
                "reason_code": "archive_truncated",
                "message": "ZIP-compatible content is truncated.",
                "fatal": True,
            }
        ],
        "scope_exclusions": [],
        "limitations": [],
    }
    prepared = prepare_report(
        {
            "findings": [],
            "use_llm": False,
            "analysis_completeness": analysis,
            "execution_successful": False,
        }
    )
    original = deepcopy(prepared)

    result = serialize_report(prepared, fmt)

    _assert_projection(result, fmt, COMPLETE)
    assert result["analysis_completeness"] == analysis
    assert result["execution_successful"] is False
    assert (result["risk_score"], result["risk_severity"], result["risk_recommendation"]) == (
        0,
        "LOW",
        "CAUTION",
    )
    assert prepared == original


@pytest.mark.parametrize("limit", [0, 1, 2, 3])
def test_sarif_notification_fallback_is_local_and_counts_replaced_detail(
    monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    prepared = prepare_report(
        {
            "findings": [],
            "use_llm": False,
            "structured_summaries": [
                {"id": f"SSR-{index}", "message": "Structured bundle", "file": "bundle.json"}
                for index in range(3)
            ],
        }
    )
    original = deepcopy(prepared)
    monkeypatch.setattr(report_module, "MAX_FINDING_OUTPUT_RECORDS", limit)

    result = serialize_report(prepared, "sarif")

    omitted = 3 - max(0, limit - 1) if limit < 3 else 0
    expected = build_report_completeness(
        omission_reasons=["notification_record_limit"] if omitted else [],
        omitted_records=omitted,
    )
    _assert_projection(result, "sarif", expected)
    invocation = result["sarif_report"]["runs"][0]["invocations"][0]
    assert invocation["properties"]["analysisCompleteness"]["isComplete"] is True
    assert invocation["properties"]["analysisCompleteness"]["notificationsTruncated"] is (
        limit < 3
    )
    assert len(invocation.get("toolExecutionNotifications", [])) <= limit
    assert result["risk_recommendation"] == "SAFE"
    for fmt in ("json", "terminal", "markdown", "json"):
        other = serialize_report(prepared, fmt)
        _assert_projection(other, fmt, COMPLETE)
        if fmt == "json":
            assert len(json.loads(other["report_body"])["structured_summaries"]) == 3
    assert prepared == original
    assert prepared.report_completeness == COMPLETE


@pytest.mark.parametrize("fmt", FORMATS)
def test_serializers_return_independent_mutable_completeness_copies(fmt: str) -> None:
    prepared = prepare_report({"findings": [], "use_llm": False})
    original = deepcopy(prepared)
    first = serialize_report(prepared, fmt)
    second = serialize_report(prepared, fmt)

    first["report_completeness"]["is_complete"] = False
    first["report_completeness"]["omission_reasons"].append("local_output_limit")
    first["report_completeness"]["omitted_records"] = 1
    if fmt == "sarif":
        nested = first["sarif_report"]["runs"][0]["invocations"][0]["properties"][
            "reportCompleteness"
        ]
        assert nested == COMPLETE
        nested["omission_reasons"].append("another_local_limit")

    assert prepared == original
    assert second["report_completeness"] == COMPLETE
    _assert_projection(serialize_report(prepared, fmt), fmt, COMPLETE)


def test_merge_combines_disjoint_retention_and_local_omissions_without_mutation() -> None:
    child = build_report_completeness(
        omission_reasons=["finding_record_limit"], omitted_records=2
    )
    retention = build_report_completeness(
        omission_reasons=["prepared_payload_limit"], omitted_skills=3
    )
    original = deepcopy([child, retention])

    merged = merge_report_completeness(
        child,
        retention,
        omission_reasons=["finding_record_limit", "serialized_output_limit"],
        omitted_records=4,
        omitted_skills=1,
    )

    assert merged == {
        "is_complete": False,
        "omission_reasons": [
            "finding_record_limit",
            "prepared_payload_limit",
            "serialized_output_limit",
        ],
        "omitted_records": 6,
        "omitted_skills": 4,
    }
    assert [child, retention] == original
    merged["omission_reasons"].clear()
    assert [child, retention] == original
    assert merge_report_completeness() == COMPLETE
    assert merge_report_completeness({"is_complete": False})["is_complete"] is False


@pytest.mark.parametrize("count", [-1, True, False, 1.5, "100", None])
def test_counts_accept_only_nonnegative_integers(count: object) -> None:
    result = build_report_completeness(omitted_records=count, omitted_skills=count)
    assert result == COMPLETE


def test_counts_saturate_and_unknown_omission_counts_do_not_claim_completeness() -> None:
    huge = 10**1000
    child = build_report_completeness(omitted_records=huge, omitted_skills=huge)
    assert child["omitted_records"] == child["omitted_skills"] == MAX_OMITTED_COUNT
    merged = merge_report_completeness(child, child, omitted_records=1, omitted_skills=1)
    assert merged["omitted_records"] == merged["omitted_skills"] == MAX_OMITTED_COUNT
    assert merged["is_complete"] is False
    assert build_report_completeness(is_complete=False) == {**COMPLETE, "is_complete": False}
    assert build_report_completeness(omission_reasons=["unknown_count"])["is_complete"] is False


@pytest.mark.parametrize("size", [MAX_OMISSION_REASONS, MAX_OMISSION_REASONS + 1])
def test_reason_metadata_is_bounded_with_an_explicit_overflow_code(size: int) -> None:
    result = build_report_completeness(omission_reasons=(f"limit_{i}" for i in range(size)))
    assert len(result["omission_reasons"]) == MAX_OMISSION_REASONS
    assert ("omission_reason_limit" in result["omission_reasons"]) is (
        size > MAX_OMISSION_REASONS
    )
    assert all(len(reason) <= MAX_OMISSION_REASON_CHARS for reason in result["omission_reasons"])
    assert result["is_complete"] is False
    assert merge_report_completeness(result)["is_complete"] is False


def test_reason_ingestion_has_bounded_work_and_never_exposes_untrusted_text() -> None:
    bounded = build_report_completeness(omission_reasons=repeat("duplicate"))
    assert bounded["omission_reasons"] == ["duplicate", "omission_reason_limit"]
    result = build_report_completeness(
        omission_reasons=[
            "local_limit",
            "local_limit",
            "\x1b[31mhttps://user:private-secret@example.invalid/?token=secret",
            "x" * (MAX_OMISSION_REASON_CHARS + 1),
        ]
    )
    assert result["omission_reasons"] == ["local_limit", "unknown_omission"]
    notice = report_completeness_notice(result)
    assert "private-secret" not in notice
    assert "\x1b" not in notice
    assert "Report completeness: incomplete" in notice
