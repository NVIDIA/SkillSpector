# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation preserves the established report evidence and policy contracts."""

from __future__ import annotations

import importlib
import json
import re
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest
from markdown_it import MarkdownIt

from skillspector.models import Finding
from skillspector.nodes.report import PreparedReport, prepare_report, report, serialize_report
from skillspector.sarif_models import validate_sarif_report
from skillspector.state import SkillspectorState, llm_call_record
from skillspector.suppression import Baseline, SuppressionRule

report_module = importlib.import_module("skillspector.nodes.report")
FORMATS = ("terminal", "json", "markdown", "sarif")


@pytest.fixture(autouse=True)
def available_provider(monkeypatch: pytest.MonkeyPatch) -> Mock:
    probe = Mock(return_value=(True, None))
    monkeypatch.setattr(report_module, "is_llm_available", probe)
    return probe


def _without_timestamps(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_timestamps(item) for key, item in value.items() if key != "scanned_at"
        }
    if isinstance(value, list):
        return [_without_timestamps(item) for item in value]
    if isinstance(value, str):
        # Terminal/Markdown put the scan timestamp in prose, not metadata.
        return re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", "<timestamp>", value)
    return value


def _body(result: dict[str, object], fmt: str) -> object:
    body = result["report_body"]
    return _without_timestamps(json.loads(body) if fmt in {"json", "sarif"} else body)


def _visible_body(result: dict[str, object], fmt: str) -> str:
    body = result["report_body"]
    return MarkdownIt().enable("table").render(body) if fmt == "markdown" else body


def _assert_parity(state: SkillspectorState, fmt: str) -> tuple[PreparedReport, dict[str, object]]:
    original = deepcopy(state)
    legacy = report({**deepcopy(state), "output_format": fmt})
    prepared = prepare_report(deepcopy(state))
    before = deepcopy(prepared)
    rendered = serialize_report(prepared, fmt)

    assert set(rendered) == {*prepared.public_fields, "report_body"} | (
        {"sarif_report"} if fmt == "sarif" else set()
    )
    assert {key: legacy[key] for key in prepared.public_fields} == prepared.public_fields
    assert {key: rendered[key] for key in prepared.public_fields} == prepared.public_fields
    assert _body(rendered, fmt) == _body(legacy, fmt)
    sarif = serialize_report(prepared, "sarif")["sarif_report"]
    assert sarif == legacy["sarif_report"]
    validate_sarif_report(sarif)
    assert prepared == before
    assert state == original
    return prepared, rendered


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("show_suppressed", [False, True])
def test_suppression_preserves_active_effective_findings_and_risk(
    fmt: str, show_suppressed: bool
) -> None:
    first = Finding(
        rule_id="P1",
        finding_id="first",
        message="Instruction override",
        severity="HIGH",
        confidence=1.0,
        file="SKILL.md",
        start_line=2,
        matched_text="ignore previous instructions",
    )
    second = replace(first, finding_id="second", file="notes.md", start_line=7)
    suppressed = Finding(
        rule_id="BH2",
        finding_id="suppressed",
        message="Blocked behavior",
        severity="CRITICAL",
        confidence=1.0,
    )
    state: SkillspectorState = {
        "findings": [first, second, suppressed],
        # A reduced meta result must not erase canonical deterministic evidence.
        "filtered_findings": [first],
        "effective_finding_ids": [first.finding_id, second.finding_id],
        "use_llm": False,
        "baseline": Baseline(rules=[SuppressionRule(rule_id="BH2", reason="reviewed")]),
        "show_suppressed": show_suppressed,
    }

    prepared, rendered = _assert_parity(state, fmt)

    assert [f.finding_id for f in prepared.public_fields["active_findings"]] == [
        "first",
        "second",
    ]
    assert len(prepared.reported_findings) == 1
    assert len(prepared.display_findings) == 2
    assert [f.finding_id for f in prepared.display_findings] == ["first", "second"]
    assert prepared.suppressed[0].finding.finding_id == "suppressed"
    assert prepared.suppressed[0].reason == "reviewed"
    assert (prepared.risk_score, prepared.risk_severity, prepared.risk_recommendation) == (
        37,  # Two HIGH occurrences: int(25 * (1 + 0.5)).
        "MEDIUM",
        "CAUTION",
    )
    if fmt == "json":
        payload = json.loads(rendered["report_body"])
        assert [issue["id"] for issue in payload["issues"]] == ["P1", "P1"]
        assert payload["suppressed_count"] == 1
        assert payload["suppressed"][0]["suppression_reason"] == "reviewed"
    elif fmt == "sarif":
        results = rendered["sarif_report"]["runs"][0]["results"]
        blocked = next(row for row in results if row["ruleId"] == "BH2")
        assert blocked["suppressions"][0]["justification"] == "reviewed"


@pytest.mark.parametrize("fmt", FORMATS)
def test_structured_summaries_and_sanitized_inference_provenance(fmt: str) -> None:
    usage = {
        "node": "meta_analyzer",
        "request_kind": "structured_output",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "model_source": "provider_response",
        "usage_source": "provider_response",
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "total_tokens": 168,
    }
    state: SkillspectorState = {
        "findings": [],
        "use_llm": True,
        "llm_call_log": [llm_call_record("meta_analyzer", ok=True)],
        "structured_summaries": [
            {
                "id": "SSR-1",
                "message": "Structured bundle\x1b[31m detected\x00",
                "file": "bundle.aisop.json",
                "protocol": "AISOP V1",
                "layout_kind": "structured",
                "declared_tools": ["calendar", "search"],
                "workflow_nodes": ["system", "user"],
                "constraints": ["query"],
                "resources": ["docs"],
                "tags": ["AISOP"],
            }
        ],
        "inference_usage": [{**usage, "secret": "private-usage-sentinel"}],
        "llm_provenance": {
            "provider": {"configured_adapter": "anthropic", "resolved_adapter": "anthropic"},
            "analyzers": [{"analyzer_id": "meta_analyzer", "model": "claude-sonnet-4-6"}],
            "endpoint": "https://private-endpoint-sentinel.invalid",
        },
    }

    prepared, rendered = _assert_parity(state, fmt)

    assert prepared.structured_summaries[0]["message"] == "Structured bundle detected"
    assert prepared.inference_usage == state["inference_usage"]
    assert prepared.llm_provenance == state["llm_provenance"]
    visible = _visible_body(rendered, fmt)
    assert "SSR-1" in visible
    assert "private-usage-sentinel" not in visible
    assert "private-endpoint-sentinel" not in visible
    assert "\x1b" not in rendered["report_body"]
    assert "\x00" not in rendered["report_body"]
    if fmt == "json":
        payload = json.loads(rendered["report_body"])
        assert payload["structured_summaries"] == prepared.structured_summaries
        assert payload["issues"] == []
        assert payload["metadata"]["inference_usage"] == [usage]
        assert payload["metadata"]["llm_provenance"]["provider"]["effective_adapter"] == (
            "anthropic"
        )


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("status", ["complete", "partial", "failed"])
def test_analysis_completeness_and_execution_keep_honest_risk(fmt: str, status: str) -> None:
    complete = status == "complete"
    successful = status != "failed"
    completeness = {
        "total_components": 2,
        "scanned_components": 2 if complete else 1,
        "coverage_percent": 100.0 if complete else 50.0,
        "is_complete": complete,
        "status": status,
        "execution_successful": successful,
        "fully_inspected_files": 2 if complete else 1,
        "partially_inspected_files": 0,
        "entirely_uninspected_files": 0 if complete else 1,
        "ledger_exceptions": []
        if complete
        else [
            {
                "path": "outer.zip!/broken.zip",
                "phase": "nested_artifact_inspection",
                "outcome": "failed" if not successful else "skipped",
                "reason_code": "archive_truncated",
                "message": "ZIP-compatible content is truncated.",
                "fatal": not successful,
            }
        ],
        "scope_exclusions": [],
        "limitations": [],
    }
    state: SkillspectorState = {
        "findings": [],
        "use_llm": False,
        "analysis_completeness": completeness,
        "execution_successful": successful,
    }

    prepared, rendered = _assert_parity(state, fmt)

    assert prepared.analysis_completeness == completeness
    assert prepared.execution_successful is successful
    assert prepared.risk_score == 0
    assert prepared.risk_severity == "LOW"
    assert prepared.risk_recommendation == ("SAFE" if complete else "CAUTION")
    if fmt == "json":
        payload = json.loads(rendered["report_body"])
        assert payload["analysis_completeness"] == completeness
        assert payload["execution_successful"] is successful
    elif fmt == "sarif":
        invocation = rendered["sarif_report"]["runs"][0]["invocations"][0]
        assert invocation["executionSuccessful"] is successful
        projection = invocation["properties"]["analysisCompleteness"]
        assert projection["isComplete"] is complete
        assert projection["coveragePercent"] == completeness["coverage_percent"]
        assert projection["ledgerExceptionCount"] == len(completeness["ledger_exceptions"])


@pytest.mark.parametrize("fmt", FORMATS)
def test_occurrence_spans_preserve_source_and_archive_provenance(fmt: str) -> None:
    source = {
        "source_identity": "external/" + "a" * 64,
        "source_digest": "b" * 64,
        "source_url": "https://example.invalid/package.zip",
        "transitive_depth": 1,
        "surface": "bundled_script",
    }
    locations = [
        {
            "file": "outer.zip!/inner.zip!/run.py",
            "start_line": 2,
            "end_line": 2,
            "start_column": 0,
            "end_column": 28,
        },
        {
            "file": "notes.md",
            "start_line": 1,
            "end_line": 2,
            "start_column": 8,
            "end_column": 12,
        },
        {"file": "legacy.md", "start_line": 4, "end_line": 4},
    ]
    finding = Finding(
        rule_id="P1",
        message="Instruction override",
        severity="HIGH",
        confidence=1.0,
        file=locations[0]["file"],
        start_line=2,
        start_column=0,
        end_column=28,
        occurrences=[{**location, **source} for location in locations],
        **source,
    )
    archive = {
        "outer_path": "outer.zip",
        "nested_path": "inner.zip!/run.py",
        "container_type": "zip",
        "container_ancestry": ["zip", "zip"],
        "container_depth": 2,
    }
    state: SkillspectorState = {
        "findings": [finding],
        "use_llm": False,
        "component_metadata": [
            {
                "path": locations[0]["file"],
                "type": "python",
                "lines": 3,
                "executable": True,
                "size_bytes": 42,
                **archive,
                **source,
            },
            {"path": "notes!/run.py", "type": "python", "executable": True},
        ],
    }

    prepared, rendered = _assert_parity(state, fmt)

    assert [f.to_dict()["location"] for f in prepared.display_findings] == locations
    assert {f.source_identity for f in prepared.display_findings} == {source["source_identity"]}
    assert {f.source_digest for f in prepared.display_findings} == {source["source_digest"]}
    if fmt == "json":
        payload = json.loads(rendered["report_body"])
        assert [issue["location"] for issue in payload["issues"]] == locations
        assert {issue["match_fingerprint"] for issue in payload["issues"]} == {
            finding.fingerprint()
        }
        member, physical = payload["components"]
        assert {key: member[key] for key in archive} == archive
        assert {key: member[key] for key in source if key.startswith("source_")} == {
            key: value for key, value in source.items() if key.startswith("source_")
        }
        assert not any(key in physical for key in archive)
    elif fmt == "sarif":
        run = rendered["sarif_report"]["runs"][0]
        assert run["columnKind"] == "unicodeCodePoints"
        physical = [row["locations"][0]["physicalLocation"] for row in run["results"]]
        assert [row["region"] for row in physical] == [
            {"startLine": 2, "endLine": 2, "startColumn": 1, "endColumn": 29},
            {"startLine": 1, "endLine": 2, "startColumn": 9, "endColumn": 13},
            {"startLine": 4, "endLine": 4},
        ]
        assert all(
            row["artifactLocation"]["uri"].startswith(source["source_identity"] + "/")
            for row in physical
        )


@pytest.mark.parametrize("fmt", FORMATS)
def test_sanitation_preserves_evidence_types_and_original_findings(fmt: str) -> None:
    url = "https://parity-user:parity-password@example.invalid/?token=parity-token"
    dirty = "\x1b[31mReview\x1b[0m\x00 " + url
    finding = Finding(
        rule_id="E2",
        message=dirty,
        severity="HIGH",
        file="setup.sh",
        pattern=dirty,
        finding=dirty,
        explanation=dirty,
        remediation=dirty,
        context=dirty,
        matched_text=dirty,
        code_snippet=dirty,
        evidence={"dirty\x00key": [{url: [url, None, True, 42, 1.25]}]},
    )
    state: SkillspectorState = {"findings": [finding], "use_llm": False}

    prepared, rendered = _assert_parity(state, fmt)

    cleaned = prepared.public_fields["active_findings"][0]
    values = next(iter(cleaned.evidence["dirtykey"][0].values()))
    assert values[1:] == [None, True, 42, 1.25]
    assert [type(value) for value in values[1:]] == [type(None), bool, int, float]
    serialized = json.dumps(cleaned.to_dict()) + _visible_body(rendered, fmt)
    for secret in ("parity-user", "parity-password", "parity-token", "\\u001b", "\\u0000"):
        assert secret not in serialized
    assert "\x1b" not in rendered["report_body"]
    assert "\x00" not in rendered["report_body"]
    assert finding.message == dirty
    assert "dirty\x00key" in finding.evidence


@pytest.mark.parametrize("fmt", FORMATS)
def test_bounded_report_retains_full_active_findings_and_risk(
    monkeypatch: pytest.MonkeyPatch, fmt: str
) -> None:
    first = Finding(
        rule_id="P1",
        finding_id="first",
        message="Instruction override",
        severity="HIGH",
        confidence=1.0,
        file="SKILL.md",
        start_line=2,
        matched_text="ignore previous instructions",
    )
    findings = [first, replace(first, finding_id="second", file="notes.md", start_line=7)]
    monkeypatch.setattr(report_module, "MAX_FINDING_OUTPUT_RECORDS", 1)

    prepared, _ = _assert_parity({"findings": findings, "use_llm": False}, fmt)

    assert prepared.public_fields["active_findings"] == findings
    assert len(prepared.reported_findings) == 1
    assert len(prepared.reported_findings[0].occurrences) == 1
    assert len(prepared.display_findings) == 1
    assert prepared.risk_score == 37


@pytest.mark.parametrize("requested", [False, True])
def test_static_and_captured_preflight_reports_never_resolve_provider(
    monkeypatch: pytest.MonkeyPatch, requested: bool
) -> None:
    def unexpected_probe(**kwargs: object) -> None:
        pytest.fail("Static/captured preflight result must not re-probe the provider")

    monkeypatch.setattr(report_module, "is_llm_available", unexpected_probe)
    state: SkillspectorState = {
        "findings": [],
        "use_llm": False,
        "llm_requested": requested,
        "llm_preflight_error": "Captured provider configuration failure",
    }
    prepared = prepare_report(state)
    assert prepared.provider_available is False
    for fmt in FORMATS:
        legacy = report({**deepcopy(state), "output_format": fmt})
        rendered = serialize_report(prepared, fmt)
        assert _body(rendered, fmt) == _body(legacy, fmt)
        if fmt == "json":
            metadata = json.loads(rendered["report_body"])["metadata"]
            assert metadata["llm_available"] is False
            if requested:
                assert "Captured provider configuration failure" in metadata["llm_error"]


@pytest.mark.parametrize("provider_available", [False, True])
def test_provider_resolution_is_captured_once_not_repeated_by_serialization(
    monkeypatch: pytest.MonkeyPatch, available_provider: Mock, provider_available: bool
) -> None:
    available_provider.return_value = (
        provider_available,
        None if provider_available else "Provider unavailable",
    )
    state: SkillspectorState = {
        "findings": [Finding(rule_id="P1", message="Retained finding")],
        "use_llm": True,
        "llm_requested": True,
        "llm_call_log": [llm_call_record("meta_analyzer", ok=True)],
    }
    legacy = {fmt: report({**deepcopy(state), "output_format": fmt}) for fmt in FORMATS}
    available_provider.reset_mock()
    prepared = prepare_report(state)
    available_provider.assert_called_once()
    assert prepared.provider_available is provider_available
    before = deepcopy(prepared)

    def unexpected_work(*args: object, **kwargs: object) -> None:
        pytest.fail("Serialization repeated preparation, suppression, or provider resolution")

    monkeypatch.setattr(report_module, "is_llm_available", unexpected_work)
    monkeypatch.setattr(report_module, "prepare_report", unexpected_work)
    monkeypatch.setattr(report_module, "partition_findings", unexpected_work)
    monkeypatch.setattr(report_module, "_compute_risk_score", unexpected_work)
    for fmt in (*FORMATS, *reversed(FORMATS), "json"):
        rendered = serialize_report(prepared, fmt)
        assert _body(rendered, fmt) == _body(legacy[fmt], fmt)
        assert rendered["filtered_findings"] == prepared.reported_findings
        if fmt == "json":
            metadata = json.loads(rendered["report_body"])["metadata"]
            assert metadata["llm_available"] is provider_available
    assert prepared == before
