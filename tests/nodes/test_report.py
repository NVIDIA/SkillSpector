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

"""Unit tests for the report node (risk scoring, output_format, report_body)."""

from __future__ import annotations

import json

from skillspector.models import Finding
from skillspector.nodes.report import report
from skillspector.state import SkillspectorState


def _finding(rule_id: str, severity: str = "LOW", message: str = "test") -> Finding:
    return Finding(
        rule_id=rule_id,
        message=message,
        severity=severity,
        confidence=0.8,
        file="SKILL.md",
        start_line=1,
    )


def test_report_empty_findings_zero_risk() -> None:
    """No findings yields risk_score 0, risk_severity LOW, risk_recommendation SAFE."""
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": "/tmp/skill",
        "output_format": "sarif",
    }
    result = report(state)
    assert result["risk_score"] == 0
    assert result["risk_severity"] == "LOW"
    assert result["risk_recommendation"] == "SAFE"
    assert "report_body" in result
    assert "sarif_report" in result


def test_report_critical_finding_high_score() -> None:
    """One CRITICAL finding yields score 50, severity MEDIUM (band 21-50), CAUTION."""
    state: SkillspectorState = {
        "filtered_findings": [_finding("P5", "CRITICAL")],
        "component_metadata": [
            {
                "path": "SKILL.md",
                "type": "markdown",
                "lines": 10,
                "executable": False,
                "size_bytes": 100,
            }
        ],
        "has_executable_scripts": False,
        "manifest": {"name": "test"},
        "skill_path": "/tmp/skill",
        "output_format": "json",
    }
    result = report(state)
    assert result["risk_score"] == 50
    assert result["risk_severity"] == "MEDIUM"  # band: 21-50
    assert result["risk_recommendation"] == "CAUTION"


def test_report_high_severity_do_not_install() -> None:
    """Score >= 51 yields severity HIGH and DO_NOT_INSTALL."""
    state: SkillspectorState = {
        "filtered_findings": [
            _finding("P5", "CRITICAL"),
            _finding("E2", "LOW"),
        ],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "json",
    }
    result = report(state)
    # 50 + 5 = 55 => HIGH band
    assert result["risk_score"] == 55
    assert result["risk_severity"] == "HIGH"
    assert result["risk_recommendation"] == "DO_NOT_INSTALL"


def test_report_executable_scripts_multiplier() -> None:
    """has_executable_scripts applies 1.3x to risk score (capped at 100)."""
    # 2 HIGH = 50, * 1.3 = 65
    state: SkillspectorState = {
        "filtered_findings": [
            _finding("E2", "HIGH"),
            _finding("PE3", "HIGH"),
        ],
        "component_metadata": [
            {"path": "run.py", "type": "python", "lines": 5, "executable": True, "size_bytes": 200}
        ],
        "has_executable_scripts": True,
        "manifest": {},
        "skill_path": "/tmp/skill",
        "output_format": "json",
    }
    result = report(state)
    assert result["risk_score"] == 65
    assert result["risk_severity"] == "HIGH"
    assert result["risk_recommendation"] == "DO_NOT_INSTALL"


def test_report_output_format_json() -> None:
    """output_format json produces report_body as valid JSON with skill, risk_assessment, components, issues."""
    state: SkillspectorState = {
        "filtered_findings": [_finding("P1", "HIGH")],
        "component_metadata": [
            {"path": "a.md", "type": "markdown", "lines": 1, "executable": False, "size_bytes": 10}
        ],
        "has_executable_scripts": False,
        "manifest": {"name": "my-skill"},
        "skill_path": "/path/to/skill",
        "output_format": "json",
    }
    result = report(state)
    body = result["report_body"]
    data = json.loads(body)
    assert data["skill"]["name"] == "my-skill"
    assert "risk_assessment" in data
    assert "score" in data["risk_assessment"]
    assert "severity" in data["risk_assessment"]
    assert "recommendation" in data["risk_assessment"]
    assert "components" in data
    assert "issues" in data
    assert len(data["issues"]) == 1
    assert data["issues"][0]["id"] == "P1"


def test_report_output_format_markdown() -> None:
    """output_format markdown produces report_body with # SkillSpector and ## Risk Assessment."""
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "markdown",
    }
    result = report(state)
    body = result["report_body"]
    assert "# SkillSpector Security Report" in body
    assert "## Risk Assessment" in body
    assert "## Components" in body
    assert "## Issues" in body


def test_report_output_format_terminal() -> None:
    """output_format terminal produces report_body with SkillSpector and Risk Assessment."""
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {"name": "cli-test"},
        "skill_path": "/foo",
        "output_format": "terminal",
    }
    result = report(state)
    body = result["report_body"]
    assert "SkillSpector" in body
    assert "Risk Assessment" in body
    assert "cli-test" in body


def test_report_output_format_sarif() -> None:
    """output_format sarif (default) produces report_body as JSON SARIF string."""
    state: SkillspectorState = {
        "filtered_findings": [_finding("E2", "HIGH", "env harvest")],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "sarif",
    }
    result = report(state)
    body = result["report_body"]
    data = json.loads(body)
    assert "runs" in data
    assert data.get("$schema") or "runs" in data


def test_report_default_output_format_is_sarif() -> None:
    """When output_format is missing, report uses sarif and report_body is JSON."""
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
    }
    result = report(state)
    body = result["report_body"]
    json.loads(body)
    assert "sarif_report" in result
