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

"""Tests for SARIF security-severity on results and rule descriptors."""

from __future__ import annotations

import pytest

from skillspector.models import Finding
from skillspector.nodes.report import _build_sarif, _severity_to_security_severity
from skillspector.sarif_models import validate_sarif_report
from skillspector.suppression import SuppressedFinding


def _make_finding(rule_id: str = "P5", severity: str = "CRITICAL", **kwargs) -> Finding:
    defaults = {
        "message": "Malicious instruction",
        "confidence": 0.9,
        "file": "SKILL.md",
        "start_line": 1,
        "end_line": 1,
        "remediation": "Remove the instruction",
        "tags": ["malicious"],
        "context": "context",
        "matched_text": "match",
        "category": "prompt_injection",
        "pattern": "P5",
        "finding": "snippet",
        "explanation": "explain",
        "code_snippet": "code",
        "intent": None,
    }
    defaults.update(kwargs)
    return Finding(rule_id=rule_id, severity=severity, **defaults)


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        ("CRITICAL", "9.5"),
        ("HIGH", "8.0"),
        ("MEDIUM", "5.5"),
        ("LOW", "3.0"),
        ("critical", "9.5"),
        ("UNKNOWN", None),
        ("", None),
    ],
)
def test_severity_to_security_severity_mapping(severity: str, expected: str | None) -> None:
    """Each known severity maps inside its GitHub code scanning band."""
    assert _severity_to_security_severity(severity) == expected


def test_result_carries_security_severity() -> None:
    """A CRITICAL result keeps level error and gains security-severity 9.5."""
    sarif = _build_sarif([_make_finding(severity="CRITICAL")])
    result = sarif["runs"][0]["results"][0]
    assert result["level"] == "error"
    assert result["properties"]["security-severity"] == "9.5"
    assert result["properties"]["severity"] == "CRITICAL"


def test_high_and_critical_distinguishable() -> None:
    """HIGH and CRITICAL share level error but carry different security-severity."""
    sarif = _build_sarif(
        [
            _make_finding(rule_id="P5", severity="CRITICAL"),
            _make_finding(rule_id="E2", severity="HIGH"),
        ]
    )
    by_rule = {r["ruleId"]: r for r in sarif["runs"][0]["results"]}
    assert by_rule["P5"]["level"] == by_rule["E2"]["level"] == "error"
    assert by_rule["P5"]["properties"]["security-severity"] == "9.5"
    assert by_rule["E2"]["properties"]["security-severity"] == "8.0"


def test_unknown_severity_omits_security_severity() -> None:
    """Unknown severities emit no security-severity rather than a wrong score."""
    sarif = _build_sarif([_make_finding(severity="UNKNOWN")])
    result = sarif["runs"][0]["results"][0]
    assert "security-severity" not in result["properties"]


def test_rule_descriptor_carries_security_severity_for_single_severity_rule() -> None:
    """A rule with one severity across all findings is annotated."""
    sarif = _build_sarif(
        [
            _make_finding(rule_id="P5", severity="HIGH"),
            _make_finding(rule_id="P5", severity="HIGH", file="other.py"),
        ]
    )
    rules = {r["id"]: r for r in sarif["runs"][0]["tool"]["driver"]["rules"]}
    assert rules["P5"]["properties"]["security-severity"] == "8.0"


def test_rule_descriptor_omits_security_severity_for_mixed_severity_rule() -> None:
    """A rule with mixed severities stays unannotated instead of mislabeling."""
    sarif = _build_sarif(
        [
            _make_finding(rule_id="P5", severity="CRITICAL"),
            _make_finding(rule_id="P5", severity="LOW"),
        ]
    )
    rules = {r["id"]: r for r in sarif["runs"][0]["tool"]["driver"]["rules"]}
    assert "properties" not in rules["P5"]


def test_suppressed_finding_carries_security_severity() -> None:
    """Baseline-suppressed findings keep the annotation for audit consumers."""
    finding = _make_finding(severity="MEDIUM")
    sarif = _build_sarif([], suppressed=[SuppressedFinding(finding=finding, reason="baseline")])
    result = sarif["runs"][0]["results"][0]
    assert result["suppressions"][0]["kind"] == "external"
    assert result["properties"]["security-severity"] == "5.5"


def test_security_severity_report_validates() -> None:
    """Annotated output still satisfies the SARIF 2.1.0 model."""
    sarif = _build_sarif(
        [
            _make_finding(rule_id="P5", severity="CRITICAL"),
            _make_finding(rule_id="E2", severity="HIGH"),
            _make_finding(rule_id="M1", severity="MEDIUM"),
            _make_finding(rule_id="L1", severity="LOW"),
        ]
    )
    validate_sarif_report(sarif)
    assert sarif["runs"][0]["results"][0]["properties"]["security-severity"] == "9.5"
