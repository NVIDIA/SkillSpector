# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-graph public-output contracts for direct SC10 configuration evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillspector.graph import graph

_SKILL = "---\nname: helper\ndescription: Formats ordinary text.\n---\n# Helper\nFormats text.\n"
_SENTINELS = ("alice", "supersecret", "querysecret", "fragmentsecret")
_NONCANONICAL_NPMRC = (
    "registry=https://alice:supersecret@packages.example.invalid/private"
    "?token=querysecret&channel=stable#fragmentsecret\n"
)
_CANONICAL_NPMRC = "registry=https://registry.npmjs.org/\n"
_EXPECTED_SC10 = {
    "rule": "SC10",
    "severity": "HIGH",
    "ecosystem": "npm",
    "surface": ".npmrc",
    "operation": "replace",
    "scope": "global",
    "destination": "https://packages.example.invalid/REDACTED_PATH",
    "destination_status": "resolved",
    "path": ".npmrc",
    "start_line": 1,
    "end_line": 1,
    "confidence": 1.0,
    "category": "supply-chain",
    "matched_text": "https://packages.example.invalid/REDACTED_PATH",
    "evidence": {
        "ecosystem": "npm",
        "surface": ".npmrc",
        "operation": "replace",
        "scope": "global",
        "destination": "https://packages.example.invalid/REDACTED_PATH",
        "destination_status": "resolved",
    },
}


_DIRECT_CONFIGURATION_CASES = [
    pytest.param(
        _NONCANONICAL_NPMRC,
        _EXPECTED_SC10,
        id="credential-bearing-noncanonical-npmrc",
    )
]
_CANONICAL_DEFAULT_CASES = [
    pytest.param(
        _CANONICAL_NPMRC,
        id="canonical-npm-default",
    )
]


def _write_skill(root: Path, npmrc: str) -> Path:
    (root / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    (root / ".npmrc").write_text(npmrc, encoding="utf-8")
    return root


def _scan(root: Path, output_format: str) -> dict[str, object]:
    return graph.invoke({"skill_path": str(root), "output_format": output_format, "use_llm": False})


def _normalized_sc10(result: dict[str, object]) -> list[dict[str, object]]:
    findings = result["filtered_findings"]
    assert isinstance(findings, list)
    normalized = []
    for finding in findings:
        if finding.rule_id != "SC10":
            continue
        evidence = finding.evidence
        normalized.append(
            {
                "rule": finding.rule_id,
                "severity": finding.severity,
                "ecosystem": evidence["ecosystem"],
                "surface": evidence["surface"],
                "operation": evidence["operation"],
                "scope": evidence["scope"],
                "destination": evidence["destination"],
                "destination_status": evidence["destination_status"],
                "path": finding.file,
                "start_line": finding.start_line,
                "end_line": finding.end_line,
                "confidence": finding.confidence,
                "category": finding.category,
                "matched_text": finding.matched_text,
                "evidence": evidence,
            }
        )
    return normalized


def _sc10_json_issue(report: dict[str, object]) -> dict[str, object]:
    return next(issue for issue in report["issues"] if issue["id"] == "SC10")


def _sc10_sarif_result(report: dict[str, object]) -> dict[str, object]:
    return next(item for item in report["runs"][0]["results"] if item["ruleId"] == "SC10")


@pytest.mark.parametrize(("npmrc", "expected"), _DIRECT_CONFIGURATION_CASES)
def test_noncanonical_npmrc_has_one_redacted_sc10_across_public_outputs(
    tmp_path: Path, npmrc: str, expected: dict[str, object]
) -> None:
    """Direct registry configuration must be a structured, redacted SC10 finding."""
    root = _write_skill(tmp_path, npmrc)
    results = {
        output_format: _scan(root, output_format)
        for output_format in (
            "terminal",
            "json",
            "markdown",
            "sarif",
        )
    }

    assert _normalized_sc10(results["json"]) == [expected]
    for output_format, result in results.items():
        completeness = result["analysis_completeness"]
        assert isinstance(completeness, dict)
        assert result["execution_successful"] is True
        assert completeness["is_complete"] is True
        assert completeness["status"] == "complete"

        serialized = result["report_body"]
        assert isinstance(serialized, str)
        assert "SC10" in serialized
        assert all(sentinel not in serialized for sentinel in _SENTINELS)
        if output_format == "terminal":
            assert "REDACTED" in serialized
            assert "packages.example.invalid" in serialized
            assert "REDACTED_PATH" in serialized
        elif output_format == "markdown":
            assert expected["destination"] in serialized

    json_report = json.loads(results["json"]["report_body"])
    assert _sc10_json_issue(json_report)["evidence"]["destination"] == expected["destination"]

    sarif_report = json.loads(results["sarif"]["report_body"])
    assert (
        _sc10_sarif_result(sarif_report)["properties"]["evidence"]["destination"]
        == expected["destination"]
    )


@pytest.mark.parametrize("npmrc", _CANONICAL_DEFAULT_CASES)
def test_canonical_npm_registry_is_safe_without_sc10(tmp_path: Path, npmrc: str) -> None:
    """The default npm registry remains a complete SAFE result once SC10 is active."""
    result = _scan(_write_skill(tmp_path, npmrc), "json")
    completeness = result["analysis_completeness"]
    assert isinstance(completeness, dict)
    analyzer_status = next(
        status
        for status in result["analyzer_status_events"]
        if status["analyzer_id"] == "dependency_sources"
    )
    assert analyzer_status["status"] == "completed"
    planned_work = analyzer_status["planned_work"]
    assert len(planned_work) == 1
    assert planned_work[0]["path"] == ".npmrc"
    assert planned_work[0]["start_line"] == 1
    assert planned_work[0]["end_line"] == 2
    completed_npmrc_events = [
        event
        for event in result["inspection_ledger"]
        if event["record_type"] == "work_item"
        and event["analyzer_id"] == "dependency_sources"
        and event["path"] == ".npmrc"
        and event["outcome"] == "completed"
    ]
    assert len(completed_npmrc_events) == 1
    assert completed_npmrc_events[0]["work_id"] == planned_work[0]["work_id"]
    assert _normalized_sc10(result) == []
    assert result["risk_recommendation"] == "SAFE"
    assert result["execution_successful"] is True
    assert completeness["is_complete"] is True
    assert completeness["status"] == "complete"
