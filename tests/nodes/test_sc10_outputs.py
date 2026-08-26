# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-graph public-output contracts for direct SC10 configuration evidence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from skillspector.graph import graph
from skillspector.sarif_models import validate_sarif_report

_SKILL = "---\nname: helper\ndescription: Formats ordinary text.\n---\n# Helper\nFormats text.\n"
_SENTINELS = ("alice", "supersecret", "querysecret", "fragmentsecret")
_NONCANONICAL_NPMRC = (
    "registry=https://alice:supersecret@packages.example.invalid/private"
    "?token=querysecret&channel=stable#fragmentsecret\n"
)
_CANONICAL_NPMRC = "registry=https://registry.npmjs.org/\n"
_TASK9_GRAPH_SENTINELS = {
    "https": "task9-graph-https-secret",
    "ssh": "task9-graph-ssh-secret",
    "scp": "task9-graph-scp-secret",
    "query": "task9-graph-query-secret",
    "fragment": "task9-graph-fragment-secret",
    "assignment": "task9-graph-assignment-secret",
    "heredoc": "task9-graph-heredoc-secret",
    "generated_config": "task9-graph-generated-secret",
    "miss": "task9-graph-miss-secret",
    "partial": "task9-graph-partial-secret",
}
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


def _write_task9_credential_skill(root: Path) -> Path:
    sentinels = _TASK9_GRAPH_SENTINELS
    (root / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    (root / ".npmrc").write_text(
        "registry=https://user:"
        f"{sentinels['https']}@https.example.invalid/private"
        f"?token={sentinels['query']}#{sentinels['fragment']}\n",
        encoding="utf-8",
    )
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "setup.sh").write_text(
        "#!/bin/sh\n"
        "REGISTRY=https://user:"
        f"{sentinels['assignment']}@assignment.example.invalid/private\n"
        'npm config set registry "$REGISTRY"\n'
        "npm config set registry ssh://git:"
        f"{sentinels['ssh']}@ssh.example.invalid/org/repo.git\n"
        f"npm config set registry {sentinels['scp']}@scp.example.invalid:org/repo.git\n"
        "writer >pip.conf <<'EOF'\n"
        "[global]\n"
        "index-url=https://user:"
        f"{sentinels['heredoc']}@heredoc.example.invalid/private\n"
        "EOF\n"
        "writer >.npmrc <<'EOF'\n"
        "registry=https://user:"
        f"{sentinels['generated_config']}@generated.example.invalid/private\n"
        "EOF\n",
        encoding="utf-8",
    )
    (scripts / "partial.sh").write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' 'config set registry https://user:"
        f"{sentinels['partial']}@partial.example.invalid/private' | xargs npm\n",
        encoding="utf-8",
    )
    docs = root / "docs"
    docs.mkdir()
    (docs / "install.md").write_text(
        "```bash\n"
        "npm config set registry https://user:"
        f"{sentinels['miss']}@miss.example.invalid/private\n"
        "```\n",
        encoding="utf-8",
    )
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


@pytest.mark.parametrize("output_format", ["terminal", "json", "markdown", "sarif"])
def test_unique_credential_carriers_never_cross_real_graph_output_boundaries(
    tmp_path: Path,
    output_format: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = _write_task9_credential_skill(tmp_path)

    with caplog.at_level(logging.DEBUG, logger="skillspector"):
        result = _scan(root, output_format)

    sc10_findings = [finding for finding in result["findings"] if finding.rule_id == "SC10"]
    assert sc10_findings
    public_finding_fields = [
        {
            "message": finding.message,
            "matched_text": finding.matched_text,
            "evidence": finding.evidence,
        }
        for finding in sc10_findings
    ]
    public_artifacts = (
        str(result["findings"]),
        str(result["filtered_findings"]),
        str(result["analysis_completeness"]),
        str(public_finding_fields),
        result["report_body"],
        json.dumps(result["sarif_report"], sort_keys=True),
        caplog.text,
    )
    for carrier, sentinel in _TASK9_GRAPH_SENTINELS.items():
        assert all(sentinel not in artifact for artifact in public_artifacts), carrier

    assert result["execution_successful"] is True
    completeness = result["analysis_completeness"]
    assert isinstance(completeness, dict)
    assert completeness["status"] == "partial"
    assert any(
        exception["reason_code"] == "dependency_source_unsupported_semantics"
        and exception["path"] == "scripts/partial.sh"
        for exception in completeness["ledger_exceptions"]
    )
    if output_format == "json":
        payload = json.loads(result["report_body"])
        assert any(issue["id"] == "SC10" for issue in payload["issues"])
    if output_format == "sarif":
        validate_sarif_report(json.loads(result["report_body"]))
