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

"""Tests for report-output sanitization (ANSI / control-byte stripping)."""

from __future__ import annotations

import json

import pytest

from skillspector.models import Finding
from skillspector.nodes.report import _clean_text, _sanitize_finding, report
from skillspector.sarif_models import validate_sarif_report
from skillspector.state import SkillspectorState
from skillspector.suppression import Baseline, SuppressionRule
from skillspector.url_redaction import MAX_REDACTION_NODES


def _dirty_finding() -> Finding:
    return Finding(
        rule_id="E2",
        message="creds \x1b[31mleak\x1b[0m here\x00",
        severity="HIGH",
        confidence=0.9,
        file="a/SKILL.md",
        start_line=5,
        remediation="redact \x1b[1mnow\x1b[0m",
        context="line with \x07 bell and \x1b[0m reset",
    )


def test_clean_text_strips_ansi_and_control_keeps_readable() -> None:
    assert _clean_text("a\x1b[31mb\x1b[0mc\x00d") == "abcd"
    # Tabs and newlines are preserved.
    assert _clean_text("a\tb\nc") == "a\tb\nc"
    # Emoji / multibyte UTF-8 is untouched.
    assert _clean_text("🔴 HIGH") == "🔴 HIGH"
    # Non-strings pass through.
    assert _clean_text(None) is None


def test_sanitize_finding_cleans_text_fields_only() -> None:
    cleaned = _sanitize_finding(_dirty_finding())
    assert "\x1b" not in cleaned.message and "\x00" not in cleaned.message
    assert "leak" in cleaned.message and "here" in cleaned.message
    assert "\x1b" not in (cleaned.remediation or "")
    assert "\x07" not in (cleaned.context or "")
    # Non-text fields are unchanged.
    assert cleaned.rule_id == "E2"
    assert cleaned.start_line == 5


@pytest.mark.parametrize("fmt", ["markdown", "json", "sarif", "terminal"])
def test_report_emits_clean_utf8_for_all_formats(fmt: str) -> None:
    """No ANSI/control bytes leak into any report format."""
    state: SkillspectorState = {
        "filtered_findings": [_dirty_finding()],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": fmt,
    }
    body = report(state)["report_body"]
    assert "\x00" not in body, f"NUL leaked into {fmt}"
    assert "\x1b" not in body, f"ESC leaked into {fmt}"
    # The readable content survives the sanitization.
    assert "leak" in body and "here" in body


def _credential_bearing_finding(sentinel: str) -> Finding:
    raw_url = f"https://user:{sentinel}@packages.example.invalid/private?token={sentinel}"
    return Finding(
        rule_id="SC10",
        message=f"Dependency source points to {raw_url}",
        severity="HIGH",
        confidence=0.95,
        file="pip.conf",
        start_line=2,
        end_line=2,
        category="supply_chain",
        finding=f"index-url = {raw_url}",
        explanation=f"The configured source is {raw_url}",
        remediation=f"Replace {raw_url}",
        context=f"index-url = {raw_url}",
        matched_text=f"index-url = {raw_url}",
        source_url=raw_url,
        evidence={
            "destination": raw_url,
            "nested_untrusted": {"credential": raw_url},
            "history": [raw_url, {"credential": raw_url}],
        },
        occurrences=[
            {
                "file": "pip.conf",
                "start_line": 2,
                "end_line": 2,
                "source_url": raw_url,
                "untrusted": {"credential": raw_url},
            }
        ],
    )


_TASK9_REPORT_SENTINELS = {
    "https": "task9-report-https-secret",
    "ssh": "task9-report-ssh-secret",
    "scp": "task9-report-scp-secret",
    "query": "task9-report-query-secret",
    "fragment": "task9-report-fragment-secret",
    "assignment": "task9-report-assignment-secret",
    "heredoc": "task9-report-heredoc-secret",
    "generated_config": "task9-report-generated-secret",
    "finding": "task9-report-finding-secret",
    "exception": "task9-report-exception-secret",
    "provider_batch": "task9-report-provider-batch-secret",
}


def _task9_unique_credential_finding() -> Finding:
    sentinels = _TASK9_REPORT_SENTINELS
    return Finding(
        rule_id="SC10",
        message=(
            "Dependency source points to https://user:"
            f"{sentinels['https']}@https.example.invalid/private"
        ),
        severity="HIGH",
        confidence=1.0,
        file="scripts/setup.sh",
        start_line=1,
        category="supply-chain",
        finding=(f"registry=ssh://git:{sentinels['ssh']}@ssh.example.invalid/org/repo.git"),
        explanation=f"source {sentinels['scp']}@scp.example.invalid:org/repo.git",
        remediation=(f"replace https://query.example.invalid/private?token={sentinels['query']}"),
        context=(f"source https://fragment.example.invalid/private#{sentinels['fragment']}"),
        matched_text=(
            f"REGISTRY=https://user:{sentinels['assignment']}@assignment.example.invalid/private"
        ),
        source_url=(f"https://user:{sentinels['heredoc']}@heredoc.example.invalid/private"),
        evidence={
            "destination": (
                f"https://user:{sentinels['generated_config']}@generated.example.invalid/private"
            ),
            "destination_status": "resolved",
            "ecosystem": "npm",
            "operation": "set",
            "scope": "global",
            "surface": "generated-config",
        },
        occurrences=[
            {
                "file": "scripts/setup.sh",
                "start_line": 1,
                "end_line": 1,
                "source_url": (
                    f"https://user:{sentinels['finding']}@finding.example.invalid/private"
                ),
            }
        ],
    )


@pytest.mark.parametrize("fmt", ["terminal", "json", "markdown", "sarif"])
def test_report_redacts_credentials_across_every_public_artifact(fmt: str) -> None:
    sentinel = "task7-public-output-secret"
    raw_url = f"https://user:{sentinel}@packages.example.invalid/private?token={sentinel}"
    finding = _credential_bearing_finding(sentinel)
    state: SkillspectorState = {
        "findings": [finding],
        "component_metadata": [
            {
                "path": "pip.conf",
                "type": "text",
                "lines": 2,
                "executable": False,
                "size_bytes": 100,
                "source_url": raw_url,
                "untrusted": {"credential": raw_url},
            }
        ],
        "has_executable_scripts": False,
        "manifest": {"name": f"source {raw_url}"},
        "skill_path": raw_url,
        "output_format": fmt,
        "use_llm": True,
        "llm_call_log": [
            {"node": "meta_analyzer", "ok": False, "error": f"provider failed at {raw_url}"}
        ],
        "analysis_completeness": {
            "is_complete": False,
            "status": "partial",
            "execution_successful": True,
            "ledger_exceptions": [
                {
                    "path": "pip.conf",
                    "message": f"redaction failed at {raw_url}",
                    "fatal": False,
                }
            ],
            "limitations": [f"provider failure at {raw_url}"],
        },
    }

    result = report(state)
    body = result["report_body"]

    assert sentinel not in body
    assert sentinel not in json.dumps(result["sarif_report"], sort_keys=True)
    assert sentinel not in str(result["filtered_findings"])
    assert finding.message.endswith(raw_url), "report sanitization must not mutate canonical state"
    if fmt == "json":
        payload = json.loads(body)
        evidence = payload["issues"][0]["evidence"]
        assert isinstance(evidence, dict)
        assert evidence == {}
    if fmt == "sarif":
        payload = json.loads(body)
        validate_sarif_report(payload)
        evidence = payload["runs"][0]["results"][0]["properties"]["evidence"]
        assert isinstance(evidence, dict)
        assert evidence == {}


@pytest.mark.parametrize("fmt", ["terminal", "json", "markdown", "sarif"])
def test_unique_credential_carriers_never_cross_any_report_boundary(
    fmt: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinels = _TASK9_REPORT_SENTINELS
    finding = _task9_unique_credential_finding()
    exception_url = f"https://user:{sentinels['exception']}@exception.example.invalid/private"
    provider_url = f"https://user:{sentinels['provider_batch']}@provider.example.invalid/private"
    state: SkillspectorState = {
        "findings": [finding],
        "component_metadata": [],
        "has_executable_scripts": True,
        "manifest": {},
        "skill_path": None,
        "output_format": fmt,
        "use_llm": True,
        "llm_call_log": [
            {"node": "meta_analyzer", "ok": False, "error": f"provider failed at {provider_url}"}
        ],
        "analysis_completeness": {
            "is_complete": False,
            "status": "partial",
            "execution_successful": True,
            "ledger_exceptions": [
                {
                    "path": "scripts/setup.sh",
                    "message": f"dependency source incomplete at {exception_url}",
                    "fatal": False,
                }
            ],
        },
    }

    with caplog.at_level("DEBUG", logger="skillspector"):
        result = report(state)

    public_artifacts = (
        result["report_body"],
        json.dumps(result["sarif_report"], sort_keys=True),
        str(result["findings"]),
        str(result["filtered_findings"]),
        caplog.text,
    )
    for carrier, sentinel in sentinels.items():
        assert all(sentinel not in artifact for artifact in public_artifacts), carrier

    assert finding.message.endswith(f"{sentinels['https']}@https.example.invalid/private"), (
        "report sanitization must not mutate canonical findings"
    )
    [public_finding] = result["findings"]
    assert public_finding is not finding
    assert (
        public_finding.finding_id,
        public_finding.rule_id,
        public_finding.severity,
        public_finding.confidence,
        public_finding.file,
        public_finding.start_line,
        public_finding.end_line,
    ) == (
        finding.finding_id,
        finding.rule_id,
        finding.severity,
        finding.confidence,
        finding.file,
        finding.start_line,
        finding.end_line,
    )
    if fmt == "json":
        payload = json.loads(result["report_body"])
        assert payload["issues"][0]["evidence"]["destination"] == (
            "https://generated.example.invalid/REDACTED_PATH"
        )
    if fmt == "sarif":
        payload = json.loads(result["report_body"])
        validate_sarif_report(payload)


def test_report_evidence_with_arbitrary_top_level_key_fails_closed() -> None:
    sentinel = "task7-arbitrary-evidence-secret"
    raw_url = f"https://user:{sentinel}@packages.example.invalid/private"
    finding = Finding(
        rule_id="SC10",
        message="dependency source replacement",
        evidence={"destination": raw_url, "attacker_key": raw_url},
    )

    sanitized = _sanitize_finding(finding)

    assert sanitized.evidence == {}
    assert sentinel not in str(sanitized.evidence)


def test_report_evidence_depth_exhaustion_fails_closed() -> None:
    nested: object = "nested"
    for _ in range(32):
        nested = [nested]
    finding = Finding(
        rule_id="SC9",
        message="concealed artifact",
        evidence={"concealment_reasons": nested},
    )

    sanitized = _sanitize_finding(finding)

    assert sanitized.evidence == {}


def test_report_evidence_node_exhaustion_fails_closed() -> None:
    finding = Finding(
        rule_id="SC9",
        message="concealed artifact",
        evidence={"concealment_reasons": ["ordinary"] * MAX_REDACTION_NODES},
    )

    sanitized = _sanitize_finding(finding)

    assert sanitized.evidence == {}


def test_report_known_evidence_schema_preserves_list_and_string_types() -> None:
    sentinel = "task7-known-evidence-secret"
    raw_url = f"https://user:{sentinel}@packages.example.invalid/private"
    finding = Finding(
        rule_id="SC9",
        message="concealed dependency source",
        evidence={
            "destination": raw_url,
            "concealment_reasons": [raw_url],
            "container_depth": 2,
            "local_only": True,
        },
    )

    sanitized = _sanitize_finding(finding)

    assert type(sanitized.evidence) is dict
    assert isinstance(sanitized.evidence["destination"], str)
    assert isinstance(sanitized.evidence["concealment_reasons"], list)
    assert all(isinstance(item, str) for item in sanitized.evidence["concealment_reasons"])
    assert sanitized.evidence["container_depth"] == 2
    assert sanitized.evidence["local_only"] is True
    assert sentinel not in str(sanitized.evidence)


def test_report_baseline_score_and_recommendation_use_canonical_pre_redaction_finding() -> None:
    sentinel = "task7-baseline-secret"
    finding = _credential_bearing_finding(sentinel)
    finding.source_url = None
    baseline = Baseline(
        rules=[
            SuppressionRule(
                rule_id="SC10",
                message=f"*{sentinel}*",
                reason="accepted deterministic finding",
            )
        ]
    )
    state: SkillspectorState = {
        "findings": [finding],
        "file_cache": {"pip.conf": finding.matched_text or ""},
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "json",
        "baseline": baseline,
    }

    result = report(state)

    assert result["risk_score"] == 0
    assert result["risk_severity"] == "LOW"
    assert result["risk_recommendation"] == "SAFE"
    assert result["filtered_findings"] == []
    assert len(result["suppressed_findings"]) == 1
    assert sentinel not in result["report_body"]
    assert finding.message.endswith(sentinel)
