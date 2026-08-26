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

import pytest

from skillspector.models import Finding
from skillspector.nodes.report import (
    _DIMINISHING_WEIGHTS,
    _MAX_OCCURRENCES_PER_RULE,
    _SEVERITY_POINTS,
    _compute_risk_score,
    report,
)
from skillspector.sarif_models import validate_sarif_report
from skillspector.state import SkillspectorState, llm_call_record
from skillspector.suppression import Baseline, SuppressionRule


def _finding(
    rule_id: str,
    severity: str = "LOW",
    message: str = "test",
    confidence: float = 1.0,
    file: str = "SKILL.md",
) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=message,
        severity=severity,
        confidence=confidence,
        file=file,
        start_line=1,
    )


def _structured_summary(
    *,
    message: str = "Structured structured bundle detected (AISOP V1)",
    file: str = "bundle.aisop.json",
) -> dict[str, object]:
    return {
        "id": "SSR-1",
        "message": message,
        "file": file,
        "protocol": "AISOP V1",
        "layout_kind": "structured",
        "declared_tools": ["calendar", "search"],
        "workflow_nodes": ["system", "user"],
        "constraints": ["query"],
        "resources": ["docs"],
        "tags": ["AISOP", "AISP", "structured-skill"],
    }


# --- Risk score computation tests ---


class TestComputeRiskScoreBasic:
    """Tests for basic scoring behavior with single findings."""

    def test_empty_findings_yields_zero(self) -> None:
        score, band, rec = _compute_risk_score([], False)
        assert score == 0
        assert band == "LOW"
        assert rec == "SAFE"

    @pytest.mark.parametrize(
        "severity,expected_points",
        [
            ("CRITICAL", 50),
            ("HIGH", 25),
            ("MEDIUM", 10),
            ("LOW", 5),
        ],
    )
    def test_single_finding_full_confidence_scores_base_points(
        self, severity: str, expected_points: int
    ) -> None:
        findings = [_finding("R1", severity, confidence=1.0)]
        score, _, _ = _compute_risk_score(findings, False)
        assert score == expected_points

    def test_single_finding_partial_confidence_scales_score(self) -> None:
        findings = [_finding("R1", "HIGH", confidence=0.5)]
        score, _, _ = _compute_risk_score(findings, False)
        assert score == 12  # 25 * 1.0 * 0.5 = 12.5 -> int(12.5) = 12

    def test_shipped_bytecode_enforces_blocking_risk_floor(self) -> None:
        findings = [_finding("SC8", "HIGH", confidence=0.95, file="payload.pyc")]
        score, band, recommendation = _compute_risk_score(findings, False)
        assert score == 51
        assert band == "HIGH"
        assert recommendation == "DO_NOT_INSTALL"

    def test_sc10_high_finding_remains_advisory_caution(self) -> None:
        findings = [_finding("SC10", "HIGH", confidence=1.0, file=".npmrc")]

        score, band, recommendation = _compute_risk_score(findings, False)

        assert score == 25
        assert band == "MEDIUM"
        assert recommendation == "CAUTION"

    def test_unknown_severity_defaults_to_low_points(self) -> None:
        f = _finding("R1", "LOW")
        f.severity = ""
        score, _, _ = _compute_risk_score([f], False)
        assert score == 5


class TestComputeRiskScoreDiminishingReturns:
    """Tests for per-rule diminishing returns logic."""

    def test_same_rule_twice_second_scores_half(self) -> None:
        findings = [
            _finding("TM1", "MEDIUM", confidence=1.0),
            _finding("TM1", "MEDIUM", confidence=1.0),
        ]
        score, _, _ = _compute_risk_score(findings, False)
        # 10*1.0 + 10*0.5 = 15
        assert score == 15

    def test_same_rule_three_times_third_scores_quarter(self) -> None:
        findings = [
            _finding("TM1", "MEDIUM", confidence=1.0),
            _finding("TM1", "MEDIUM", confidence=1.0),
            _finding("TM1", "MEDIUM", confidence=1.0),
        ]
        score, _, _ = _compute_risk_score(findings, False)
        # 10*1.0 + 10*0.5 + 10*0.25 = 17.5 -> 17
        assert score == 17

    def test_same_rule_beyond_cap_contributes_zero(self) -> None:
        findings = [_finding("TM1", "MEDIUM", confidence=1.0) for _ in range(10)]
        score, _, _ = _compute_risk_score(findings, False)
        # Only first 3 count: 10*1.0 + 10*0.5 + 10*0.25 = 17.5 -> 17
        assert score == 17

    def test_different_rules_each_score_independently(self) -> None:
        findings = [
            _finding("TM1", "MEDIUM", confidence=1.0),
            _finding("EA2", "MEDIUM", confidence=1.0),
            _finding("SQP1", "MEDIUM", confidence=1.0),
        ]
        score, _, _ = _compute_risk_score(findings, False)
        # Each is first occurrence: 10*1.0 + 10*1.0 + 10*1.0 = 30
        assert score == 30

    def test_mixed_rules_diminishing_applies_per_rule(self) -> None:
        findings = [
            _finding("TM1", "MEDIUM", confidence=1.0),
            _finding("TM1", "MEDIUM", confidence=1.0),
            _finding("EA2", "HIGH", confidence=1.0),
            _finding("EA2", "HIGH", confidence=1.0),
        ]
        score, _, _ = _compute_risk_score(findings, False)
        # TM1: 10*1.0 + 10*0.5 = 15
        # EA2: 25*1.0 + 25*0.5 = 37.5
        # Total: 52.5 -> 52
        assert score == 52


class TestComputeRiskScoreExecutableMultiplier:
    """Tests for the executable scripts multiplier."""

    def test_executable_multiplier_applies(self) -> None:
        findings = [_finding("R1", "HIGH", confidence=1.0, file="run.py")]
        component_metadata = [{"path": "run.py", "executable": True}]
        score, _, _ = _compute_risk_score(findings, True, component_metadata)
        # 25 * 1.3 = 32.5 -> 32
        assert score == 32

    def test_executable_multiplier_caps_at_100(self) -> None:
        findings = [
            _finding("C1", "CRITICAL", confidence=1.0),
            _finding("C2", "CRITICAL", confidence=1.0),
            _finding("C3", "CRITICAL", confidence=1.0),
        ]
        score, _, _ = _compute_risk_score(findings, True)
        # 50 + 50 + 50 = 150, * 1.3 = 195, capped at 100
        assert score == 100

    def test_executable_lookup_is_scoped_for_same_path_in_two_sources(self) -> None:
        first_identity = f"external/{'a' * 64}"
        second_identity = f"external/{'b' * 64}"
        first = _finding("R1", "HIGH", confidence=1.0, file="run.py")
        first.source_identity = first_identity
        first.source_digest = f"sha256:{'c' * 64}"
        first.source_url = "https://github.com/org/shared"
        second = _finding("R1", "HIGH", confidence=1.0, file="run.py")
        second.source_identity = second_identity
        second.source_digest = f"sha256:{'d' * 64}"
        second.source_url = "https://github.com/org/shared"
        metadata = [
            {"path": "run.py", "source_identity": first_identity, "executable": False},
            {"path": "run.py", "source_identity": second_identity, "executable": True},
        ]

        assert _compute_risk_score([first], True, metadata)[0] == 25
        assert _compute_risk_score([second], True, metadata)[0] == 32


class TestComputeRiskScoreEdgeCases:
    """Tests for edge cases identified in code review."""

    def test_zero_confidence_finding_does_not_consume_weight_slot(self) -> None:
        """A finding with confidence=0 should be skipped entirely."""
        findings = [
            _finding("TM1", "HIGH", confidence=0.0),
            _finding("TM1", "HIGH", confidence=1.0),
        ]
        score, _, _ = _compute_risk_score(findings, False)
        # Zero-confidence skipped, second TM1 is first real occurrence: 25*1.0*1.0 = 25
        assert score == 25

    def test_negative_confidence_clamped_to_zero_and_skipped(self) -> None:
        findings = [_finding("R1", "HIGH", confidence=-0.5)]
        score, _, _ = _compute_risk_score(findings, False)
        assert score == 0

    def test_confidence_above_one_clamped(self) -> None:
        findings = [_finding("R1", "HIGH", confidence=1.5)]
        score, _, _ = _compute_risk_score(findings, False)
        # Clamped to 1.0: 25 * 1.0 * 1.0 = 25
        assert score == 25

    def test_none_rule_id_bucketed_as_unknown(self) -> None:
        """Findings with empty/None rule_id all share one bucket."""
        f1 = _finding("", "MEDIUM", confidence=1.0)
        f1.rule_id = ""
        f2 = _finding("", "MEDIUM", confidence=1.0)
        f2.rule_id = ""
        score, _, _ = _compute_risk_score([f1, f2], False)
        # Both go to "UNKNOWN" bucket: 10*1.0 + 10*0.5 = 15
        assert score == 15

    def test_same_rule_mixed_severities(self) -> None:
        """Same rule_id with different severities still uses per-rule diminishing."""
        findings = [
            _finding("TM1", "CRITICAL", confidence=1.0),
            _finding("TM1", "LOW", confidence=1.0),
        ]
        score, _, _ = _compute_risk_score(findings, False)
        # First TM1: 50*1.0, second TM1: 5*0.5 = 2.5 -> total 52.5 -> 52
        assert score == 52

    def test_same_rule_low_before_critical_sorted_correctly(self) -> None:
        """LOW before CRITICAL in input order must still score as if CRITICAL came first.

        Without severity sorting, LOW gets the full weight (5*1.0=5) and CRITICAL
        gets the diminished weight (50*0.5=25), yielding 30. With sorting, CRITICAL
        gets full weight (50*1.0=50) and LOW gets diminished (5*0.5=2.5), yielding 52.
        """
        findings = [
            _finding("TM1", "LOW", confidence=1.0),
            _finding("TM1", "CRITICAL", confidence=1.0),
        ]
        score, _, _ = _compute_risk_score(findings, False)
        # Sorted: CRITICAL first (50*1.0) + LOW second (5*0.5=2.5) = 52.5 -> 52
        assert score == 52

    def test_exact_band_boundary_21_is_medium(self) -> None:
        findings = [
            _finding("R1", "MEDIUM", confidence=1.0),
            _finding("R2", "MEDIUM", confidence=1.0),
            _finding("R3", "LOW", confidence=0.2),
        ]
        score, band, _ = _compute_risk_score(findings, False)
        # 10 + 10 + 5*1.0*0.2 = 21
        assert score == 21
        assert band == "MEDIUM"

    def test_exact_band_boundary_20_is_low(self) -> None:
        findings = [
            _finding("R1", "MEDIUM", confidence=1.0),
            _finding("R2", "MEDIUM", confidence=1.0),
        ]
        score, band, _ = _compute_risk_score(findings, False)
        # 10 + 10 = 20
        assert score == 20
        assert band == "LOW"


class TestComputeRiskScoreBands:
    """Tests for severity band assignment."""

    def test_score_0_to_20_is_low(self) -> None:
        findings = [_finding("R1", "MEDIUM", confidence=1.0)]
        score, band, rec = _compute_risk_score(findings, False)
        assert score == 10
        assert band == "LOW"
        assert rec == "SAFE"

    def test_score_21_to_50_is_medium(self) -> None:
        findings = [
            _finding("R1", "HIGH", confidence=1.0),
            _finding("R2", "LOW", confidence=1.0),
        ]
        score, band, rec = _compute_risk_score(findings, False)
        # 25 + 5 = 30
        assert score == 30
        assert band == "MEDIUM"
        assert rec == "CAUTION"

    def test_score_51_to_80_is_high(self) -> None:
        findings = [
            _finding("R1", "CRITICAL", confidence=1.0),
            _finding("R2", "MEDIUM", confidence=1.0),
        ]
        score, band, rec = _compute_risk_score(findings, False)
        # 50 + 10 = 60
        assert score == 60
        assert band == "HIGH"
        assert rec == "DO_NOT_INSTALL"

    def test_score_81_plus_is_critical(self) -> None:
        findings = [
            _finding("R1", "CRITICAL", confidence=1.0),
            _finding("R2", "CRITICAL", confidence=1.0),
        ]
        score, band, rec = _compute_risk_score(findings, False)
        # 50 + 50 = 100
        assert score == 100
        assert band == "CRITICAL"
        assert rec == "DO_NOT_INSTALL"


class TestComputeRiskScoreRealWorldScenarios:
    """Tests simulating real-world scanning scenarios from issue #134."""

    def test_multi_file_skill_same_rule_does_not_saturate(self) -> None:
        """A skill using subprocess in 10 files should NOT hit 100."""
        findings = [
            _finding("TM1", "MEDIUM", confidence=0.5, file=f"step{i}.py") for i in range(10)
        ]
        score, band, _ = _compute_risk_score(findings, False)
        # Only 3 count: 10*1.0*0.5 + 10*0.5*0.5 + 10*0.25*0.5 = 5 + 2.5 + 1.25 = 8.75 -> 8
        assert score == 8
        assert band == "LOW"

    def test_diverse_rules_still_accumulate_meaningfully(self) -> None:
        """Different genuine vulnerabilities should still produce a high score."""
        findings = [
            _finding("RCE1", "CRITICAL", confidence=0.9),
            _finding("SQLI", "CRITICAL", confidence=0.85),
            _finding("XSS", "HIGH", confidence=0.9),
            _finding("SSRF", "HIGH", confidence=0.8),
        ]
        score, band, _ = _compute_risk_score(findings, False)
        # RCE1: 50*1.0*0.9 = 45
        # SQLI: 50*1.0*0.85 = 42.5
        # XSS: 25*1.0*0.9 = 22.5
        # SSRF: 25*1.0*0.8 = 20
        # Total: 130 -> capped at 100
        assert score == 100
        assert band == "CRITICAL"

    def test_single_critical_vulnerability_scores_appropriately(self) -> None:
        """One genuine CRITICAL should register strongly."""
        findings = [_finding("RCE1", "CRITICAL", confidence=0.95)]
        score, band, _ = _compute_risk_score(findings, False)
        # 50 * 1.0 * 0.95 = 47.5 -> 47
        assert score == 47
        assert band == "MEDIUM"

    def test_constants_are_consistent(self) -> None:
        """Verify module-level constants are in expected ranges."""
        assert _MAX_OCCURRENCES_PER_RULE == len(_DIMINISHING_WEIGHTS)
        assert all(0 < w <= 1.0 for w in _DIMINISHING_WEIGHTS)
        assert _DIMINISHING_WEIGHTS[0] >= _DIMINISHING_WEIGHTS[-1]
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            assert sev in _SEVERITY_POINTS


# --- Report node integration tests ---


class TestReportNode:
    """Tests for the full report() node function."""

    def test_report_empty_findings_zero_risk(self) -> None:
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

    def test_report_critical_finding_medium_band(self) -> None:
        """One CRITICAL finding at confidence 1.0 yields score 50, MEDIUM band."""
        state: SkillspectorState = {
            "filtered_findings": [_finding("P5", "CRITICAL", confidence=1.0)],
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
        assert result["risk_severity"] == "MEDIUM"
        assert result["risk_recommendation"] == "CAUTION"

    def test_report_high_severity_do_not_install(self) -> None:
        """Score >= 51 yields severity HIGH and DO_NOT_INSTALL."""
        state: SkillspectorState = {
            "filtered_findings": [
                _finding("P5", "CRITICAL", confidence=1.0),
                _finding("E2", "MEDIUM", confidence=1.0),
            ],
            "component_metadata": [],
            "has_executable_scripts": False,
            "manifest": {},
            "skill_path": None,
            "output_format": "json",
        }
        result = report(state)
        # 50 + 10 = 60 => HIGH band
        assert result["risk_score"] == 60
        assert result["risk_severity"] == "HIGH"
        assert result["risk_recommendation"] == "DO_NOT_INSTALL"

    def test_report_executable_scripts_multiplier(self) -> None:
        """has_executable_scripts applies 1.3x to risk score."""
        state: SkillspectorState = {
            "filtered_findings": [
                _finding("E2", "HIGH", confidence=1.0, file="run.py"),
                _finding("PE3", "HIGH", confidence=1.0, file="run.py"),
            ],
            "component_metadata": [
                {
                    "path": "run.py",
                    "type": "python",
                    "lines": 5,
                    "executable": True,
                    "size_bytes": 200,
                }
            ],
            "has_executable_scripts": True,
            "manifest": {},
            "skill_path": "/tmp/skill",
            "output_format": "json",
        }
        result = report(state)
        # (25 + 25) * 1.3 = 65
        assert result["risk_score"] == 65
        assert result["risk_severity"] == "HIGH"
        assert result["risk_recommendation"] == "DO_NOT_INSTALL"

    def test_report_output_format_json(self) -> None:
        """output_format json produces valid JSON with expected structure."""
        state: SkillspectorState = {
            "filtered_findings": [_finding("P1", "HIGH", confidence=1.0)],
            "structured_summaries": [_structured_summary()],
            "component_metadata": [
                {
                    "path": "a.md",
                    "type": "markdown",
                    "lines": 1,
                    "executable": False,
                    "size_bytes": 10,
                }
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
        assert "structured_summaries" in data
        assert len(data["structured_summaries"]) == 1
        assert data["structured_summaries"][0]["id"] == "SSR-1"
        assert "issues" in data
        assert len(data["issues"]) == 1
        assert data["issues"][0]["id"] == "P1"

    def test_report_output_format_markdown(self) -> None:
        """output_format markdown produces expected headings."""
        state: SkillspectorState = {
            "filtered_findings": [],
            "structured_summaries": [_structured_summary()],
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
        assert "## Structured Skill Summary (1)" in body
        assert "## Issues" in body

    def test_report_markdown_lists_nonfatal_llm_validation_exception(self) -> None:
        """A non-fatal structured-output failure remains visible in the report."""
        state: SkillspectorState = {
            "filtered_findings": [],
            "component_metadata": [],
            "has_executable_scripts": False,
            "manifest": {},
            "skill_path": None,
            "output_format": "markdown",
            "execution_successful": True,
            "analysis_completeness": {
                "coverage_percent": 0.0,
                "fully_inspected_files": 0,
                "partially_inspected_files": 0,
                "entirely_uninspected_files": 1,
                "is_complete": False,
                "execution_successful": True,
                "ledger_exceptions": [
                    {
                        "reason_code": "llm_structured_response_invalid",
                        "path": "SKILL.md",
                        "message": "LLM returned a malformed structured response after bounded retries.",
                        "fatal": False,
                    }
                ],
                "scope_exclusions": [],
                "analyzer_statuses": [
                    {
                        "analyzer_id": "semantic_quality_policy",
                        "status": "degraded",
                        "planned_work": [],
                    }
                ],
                "limitations": ["Analyzer semantic_quality_policy status: degraded."],
            },
        }

        body = report(state)["report_body"]

        assert "| Execution | successful |" in body
        assert "### Ledger Exceptions" in body
        assert "llm_structured_response_invalid" in body
        assert "`SKILL.md`" in body
        assert "### Analyzer Statuses" in body
        assert "### Limitations" in body

    def test_report_output_format_terminal(self) -> None:
        """output_format terminal produces Rich-formatted output."""
        state: SkillspectorState = {
            "filtered_findings": [],
            "structured_summaries": [_structured_summary()],
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
        assert "Structured Skill Summary" in body

    def test_report_output_format_sarif(self) -> None:
        """output_format sarif produces valid SARIF JSON."""
        state: SkillspectorState = {
            "structured_summaries": [_structured_summary()],
            "filtered_findings": [],
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
        run = data["runs"][0]
        assert run["results"] == []
        assert "invocations" in run
        notifications = run["invocations"][0]["toolExecutionNotifications"]
        assert notifications[0]["level"] == "note"
        assert "SSR-1" in notifications[0]["message"]["text"]

    def test_report_json_structured_summary_survives_llm_mode(self) -> None:
        """A structured-only scan keeps SSR-1 visible when use_llm is true."""
        state: SkillspectorState = {
            "filtered_findings": [],
            "structured_summaries": [_structured_summary()],
            "component_metadata": [],
            "has_executable_scripts": False,
            "manifest": {},
            "skill_path": None,
            "output_format": "json",
            "use_llm": True,
            "llm_call_log": [],
        }
        result = report(state)
        assert result["risk_score"] == 0
        assert result["risk_recommendation"] == "SAFE"
        assert "structured_summaries" not in result
        data = json.loads(result["report_body"])
        assert data["issues"] == []
        assert data["structured_summaries"][0]["id"] == "SSR-1"

    def test_report_output_format_sarif_includes_finding_properties(self) -> None:
        finding = _finding("E2", "HIGH", "env harvest", confidence=0.85, file="tool.py")
        finding.category = "environment"
        finding.pattern = r"os\.environ"
        finding.finding = "TOKEN lookup"
        finding.explanation = "Environment-derived secret access"
        finding.remediation = "Drop env var usage"
        finding.code_snippet = "os.environ['TOKEN']"
        finding.intent = "secret_exfiltration"
        finding.tags = ["env", "secret"]
        state: SkillspectorState = {
            "filtered_findings": [finding],
            "component_metadata": [],
            "has_executable_scripts": False,
            "manifest": {},
            "skill_path": None,
            "output_format": "sarif",
        }
        result = report(state)
        result_row = result["sarif_report"]["runs"][0]["results"][0]
        assert result_row["properties"]["severity"] == "HIGH"
        assert result_row["properties"]["category"] == "environment"
        assert result_row["properties"]["pattern"] == r"os\.environ"
        assert result_row["properties"]["confidence"] == 0.85
        assert result_row["properties"]["finding"] == "TOKEN lookup"
        assert result_row["properties"]["explanation"] == "Environment-derived secret access"
        assert result_row["properties"]["remediation"] == "Drop env var usage"
        assert result_row["properties"]["code_snippet"] == "os.environ['TOKEN']"
        assert result_row["properties"]["intent"] == "secret_exfiltration"
        assert result_row["properties"]["tags"] == ["env", "secret"]

    @pytest.mark.parametrize("output_format", ["terminal", "json", "markdown", "sarif"])
    def test_report_preserves_structured_evidence(self, output_format: str) -> None:
        finding = _finding(
            "SC9",
            "HIGH",
            "concealed executable",
            confidence=1.0,
            file="archive.docx!/payload.sh",
        )
        finding.evidence = {
            "outer_path": "archive.docx",
            "nested_path": "payload.sh",
        }
        state: SkillspectorState = {
            "filtered_findings": [finding],
            "component_metadata": [],
            "has_executable_scripts": False,
            "manifest": {},
            "output_format": output_format,
        }

        result = report(state)

        if output_format == "json":
            issue = json.loads(result["report_body"])["issues"][0]
            assert issue["evidence"] == finding.evidence
        elif output_format == "sarif":
            row = json.loads(result["report_body"])["runs"][0]["results"][0]
            assert row["properties"]["evidence"] == finding.evidence
        else:
            assert "outer_path" in result["report_body"]
            assert "archive.docx" in result["report_body"]

    @pytest.mark.parametrize("output_format", ["terminal", "json", "markdown", "sarif"])
    def test_report_preserves_nested_inspection_limit_reason_and_path(
        self, output_format: str
    ) -> None:
        completeness = {
            "total_components": 1,
            "scanned_components": 0,
            "coverage_percent": 0.0,
            "is_complete": False,
            "execution_successful": True,
            "fully_inspected_files": 0,
            "partially_inspected_files": 0,
            "entirely_uninspected_files": 1,
            "ledger_exceptions": [
                {
                    "reason_code": "archive_member_limit",
                    "path": "outer.zip!/nested.zip",
                    "message": "Cumulative archive member limit was reached.",
                    "fatal": False,
                }
            ],
            "scope_exclusions": [],
            "analyzer_statuses": [],
            "limitations": [],
        }
        state: SkillspectorState = {
            "filtered_findings": [],
            "component_metadata": [],
            "has_executable_scripts": False,
            "manifest": {},
            "output_format": output_format,
            "analysis_completeness": completeness,  # type: ignore[typeddict-item]
        }

        body = report(state)["report_body"]

        assert "archive_member_limit" in body
        assert "outer.zip!/nested.zip" in body

    def test_report_default_output_format_is_sarif(self) -> None:
        """When output_format is missing, report uses sarif."""
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

    def test_report_surfaces_transitive_provenance(self) -> None:
        finding = _finding("T1", "HIGH", "child issue", file="dep.py")
        finding.source_url = "https://user:task7-source-secret@github.com/org/dep"
        finding.transitive_depth = 2
        state: SkillspectorState = {
            "filtered_findings": [finding],
            "component_metadata": [],
            "has_executable_scripts": False,
            "manifest": {},
            "output_format": "markdown",
        }

        markdown = report(state)["report_body"]
        assert "task7-source-secret" not in markdown
        assert "https://github.com/REDACTED_PATH" in markdown
        assert "Transitive depth:** 2" in markdown

        state["output_format"] = "sarif"
        sarif = report(state)["sarif_report"]
        properties = sarif["runs"][0]["results"][0]["properties"]
        assert properties["sourceUrl"] == "https://github.com/REDACTED_PATH"
        assert properties["transitiveDepth"] == 2

        state["output_format"] = "terminal"
        terminal = report(state)["report_body"]
        assert "task7-source-secret" not in terminal
        assert "https://github.com/REDACTED_PATH" in terminal

    def test_report_keeps_same_path_from_distinct_immutable_sources(self) -> None:
        shared_url = "https://github.com/org/shared"
        findings: list[Finding] = []
        for identity, digest in (
            (f"external/{'a' * 64}", f"sha256:{'c' * 64}"),
            (f"external/{'b' * 64}", f"sha256:{'d' * 64}"),
        ):
            finding = Finding(
                rule_id="TM1",
                message="same evidence",
                severity="HIGH",
                confidence=1.0,
                file="tool.py",
                start_line=7,
                matched_text="subprocess.run(cmd, shell=True)",
                source_url=shared_url,
                source_identity=identity,
                source_digest=digest,
                transitive_depth=1,
            )
            findings.append(finding)
        state: SkillspectorState = {
            "findings": findings,
            "component_metadata": [],
            "has_executable_scripts": False,
            "manifest": {},
            "output_format": "json",
        }

        result = report(state)
        body = json.loads(result["report_body"])

        assert len(result["filtered_findings"]) == 2
        assert {issue["source_identity"] for issue in body["issues"]} == {
            f"external/{'a' * 64}",
            f"external/{'b' * 64}",
        }
        assert {issue["occurrences"][0]["source_digest"] for issue in body["issues"]} == {
            f"sha256:{'c' * 64}",
            f"sha256:{'d' * 64}",
        }

    def test_report_dedup_affects_score_only_not_report_output(self) -> None:
        """Deduplication reduces score but all affected files appear in the report."""
        duplicated = [
            Finding(
                rule_id="TM1",
                message="shell injection",
                severity="HIGH",
                confidence=0.8,
                file=f"step{i}.py",
                start_line=10,
                matched_text="subprocess.run(cmd, shell=True)",
            )
            for i in range(4)
        ]
        state: SkillspectorState = {
            "filtered_findings": duplicated,
            "component_metadata": [],
            "has_executable_scripts": False,
            "manifest": {"name": "multi-file"},
            "skill_path": "/tmp/skill",
            "output_format": "json",
        }
        result = report(state)
        body = json.loads(result["report_body"])
        reported_files = {issue["location"]["file"] for issue in body["issues"]}
        assert reported_files == {"step0.py", "step1.py", "step2.py", "step3.py"}
        assert len(body["issues"]) == 4
        assert result["risk_score"] < 4 * 25


def test_report_baseline_suppresses_finding_and_lowers_score() -> None:
    """A baseline-suppressed CRITICAL finding does not count toward the risk score."""
    baseline = Baseline(rules=[SuppressionRule(rule_id="P5", reason="false positive")])
    suppressed_finding = _finding("P5", "CRITICAL", confidence=1.0)
    suppressed_finding.category = "critical_path"
    suppressed_finding.pattern = r"exec\("
    suppressed_finding.finding = "exec call"
    suppressed_finding.explanation = "Dynamic execution remains reachable"
    suppressed_finding.remediation = "Drop suspicious logic"
    suppressed_finding.code_snippet = "exec(payload)"
    suppressed_finding.intent = "command_execution"
    suppressed_finding.tags = ["critical", "injection"]
    state: SkillspectorState = {
        "filtered_findings": [suppressed_finding],
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
    # Suppressed findings stay in SARIF but are marked with `suppressions`
    # (audit trail) so consumers exclude them from counts.
    sarif_results = result["sarif_report"]["runs"][0]["results"]
    assert len(sarif_results) == 1
    suppressed_result = sarif_results[0]
    assert suppressed_result["suppressions"][0]["kind"] == "external"
    assert suppressed_result["suppressions"][0]["justification"] == "false positive"
    assert suppressed_result["properties"]["severity"] == "CRITICAL"
    assert suppressed_result["properties"]["category"] == "critical_path"
    assert suppressed_result["properties"]["pattern"] == r"exec\("
    assert suppressed_result["properties"]["confidence"] == 1.0
    assert suppressed_result["properties"]["finding"] == "exec call"
    assert suppressed_result["properties"]["explanation"] == "Dynamic execution remains reachable"
    assert suppressed_result["properties"]["remediation"] == "Drop suspicious logic"
    assert suppressed_result["properties"]["code_snippet"] == "exec(payload)"
    assert suppressed_result["properties"]["intent"] == "command_execution"
    assert suppressed_result["properties"]["tags"] == ["critical", "injection"]
    assert len(result["suppressed_findings"]) == 1


def test_report_baseline_keeps_unmatched_finding() -> None:
    """Findings not matched by the baseline are kept and scored normally."""
    baseline = Baseline(rules=[SuppressionRule(rule_id="SQP-1", reason="nit")])
    state: SkillspectorState = {
        "filtered_findings": [_finding("P5", "CRITICAL"), _finding("SQP-1", "MEDIUM")],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "json",
        "baseline": baseline,
    }
    result = report(state)
    assert result["risk_score"] == 50  # only the CRITICAL counts
    assert len(result["suppressed_findings"]) == 1


def test_report_json_reports_worst_issue_severity() -> None:
    """max_issue_severity names the worst finding even when the verdict normalizes it away.

    A single HIGH finding scores below the HIGH band, so risk_assessment.severity reads
    LOW. Before this field, a consumer reading the verdict got the opposite of what the
    findings said, with nothing in the report flagging the disagreement.
    """
    state: SkillspectorState = {
        "filtered_findings": [_finding("PE3", "HIGH")],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "json",
    }
    data = json.loads(report(state)["report_body"])
    assert data["risk_assessment"]["max_issue_severity"] == "HIGH"
    assert data["issues"][0]["severity"] == "HIGH"


def test_report_json_worst_issue_severity_is_none_without_findings() -> None:
    """With no issues the field says NONE, not the empty string or a bare LOW."""
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "json",
    }
    data = json.loads(report(state)["report_body"])
    assert data["risk_assessment"]["max_issue_severity"] == "NONE"


def test_report_json_worst_issue_severity_ignores_suppressed() -> None:
    """A suppressed finding is not a reported issue, so it must not set the maximum."""
    baseline = Baseline(rules=[SuppressionRule(rule_id="P5", reason="fp")])
    state: SkillspectorState = {
        "filtered_findings": [_finding("P5", "CRITICAL"), _finding("PE3", "MEDIUM")],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "json",
        "baseline": baseline,
    }
    data = json.loads(report(state)["report_body"])
    assert data["risk_assessment"]["max_issue_severity"] == "MEDIUM"


def test_report_json_includes_suppressed_section() -> None:
    """JSON output reports suppressed_count and a suppressed array."""
    baseline = Baseline(rules=[SuppressionRule(rule_id="P5", reason="fp")])
    state: SkillspectorState = {
        "filtered_findings": [_finding("P5", "CRITICAL")],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "json",
        "baseline": baseline,
    }
    data = json.loads(report(state)["report_body"])
    assert data["suppressed_count"] == 1
    assert data["issues"] == []
    assert data["suppressed"][0]["suppression_reason"] == "fp"


@pytest.mark.parametrize("output_format", ["terminal", "json", "markdown", "sarif"])
def test_report_shares_record_budget_with_suppressed_occurrences(
    monkeypatch: pytest.MonkeyPatch,
    output_format: str,
) -> None:
    monkeypatch.setattr("skillspector.nodes.report.MAX_FINDING_OUTPUT_RECORDS", 4)
    active = Finding(
        rule_id="ACTIVE_BOUND",
        message="active issue",
        severity="CRITICAL",
        confidence=1.0,
        file="active-a.py",
        occurrences=[
            {"file": "active-a.py", "start_line": 1, "end_line": 1},
            {"file": "active-b.py", "start_line": 2, "end_line": 2},
        ],
    )
    suppressed = Finding(
        rule_id="SUPPRESSED_BOUND",
        message="suppressed issue",
        severity="HIGH",
        confidence=1.0,
        file="suppressed-a.py",
        occurrences=[
            {"file": "suppressed-a.py", "start_line": 1, "end_line": 1},
            {"file": "suppressed-b.py", "start_line": 2, "end_line": 2},
            {"file": "suppressed-c.py", "start_line": 3, "end_line": 3},
        ],
    )
    state: SkillspectorState = {
        "findings": [active, suppressed],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": output_format,
        "baseline": Baseline(
            rules=[SuppressionRule(rule_id="SUPPRESSED_BOUND", reason="accepted")]
        ),
        "show_suppressed": True,
    }

    result = report(state)
    active_records = sum(
        max(1, len(finding.occurrences)) for finding in result["filtered_findings"]
    )
    suppressed_records = sum(
        max(1, len(item.finding.occurrences)) for item in result["suppressed_findings"]
    )

    assert result["risk_score"] == 50
    assert active_records == 2
    assert suppressed_records == 2
    assert active_records + suppressed_records == 4
    assert len(result["suppressed_findings"][0].finding.occurrences) == 2
    assert len(result["sarif_report"]["runs"][0]["results"]) <= 4

    body = result["report_body"]
    if output_format == "json":
        payload = json.loads(body)
        serialized_records = sum(len(issue["occurrences"]) for issue in payload["issues"])
        serialized_records += sum(len(item["occurrences"]) for item in payload["suppressed"])
        assert serialized_records == 4
    elif output_format == "sarif":
        assert len(json.loads(body)["runs"][0]["results"]) <= 4
    else:
        assert body.count("ACTIVE_BOUND") == 2
        assert body.count("SUPPRESSED_BOUND") == 1


def test_report_scores_all_active_findings_before_output_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillspector.nodes.report.MAX_FINDING_OUTPUT_RECORDS", 1)
    findings = [
        _finding("CRITICAL_BEFORE_BOUND", "CRITICAL"),
        _finding("HIGH_BEFORE_BOUND", "HIGH"),
    ]
    state: SkillspectorState = {
        "findings": findings,
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "json",
    }

    result = report(state)

    assert result["risk_score"] == 75
    assert [finding.rule_id for finding in result["filtered_findings"]] == ["CRITICAL_BEFORE_BOUND"]


def test_report_markdown_show_suppressed_lists_rows() -> None:
    """Markdown lists suppressed findings only when show_suppressed is set."""
    baseline = Baseline(rules=[SuppressionRule(rule_id="P5", reason="fp")])
    base_state: SkillspectorState = {
        "filtered_findings": [_finding("P5", "CRITICAL")],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "markdown",
        "baseline": baseline,
    }
    hidden = report({**base_state})["report_body"]
    assert "## Suppressed (1)" in hidden
    assert "--show-suppressed" in hidden

    shown = report({**base_state, "show_suppressed": True})["report_body"]
    assert "## Suppressed (1)" in shown
    assert "fp" in shown


def test_report_no_baseline_unchanged() -> None:
    """Without a baseline, scoring is unchanged and nothing is suppressed."""
    state: SkillspectorState = {
        "filtered_findings": [_finding("P5", "CRITICAL")],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "json",
    }
    result = report(state)
    assert result["risk_score"] == 50
    assert result["suppressed_findings"] == []


# ---------------------------------------------------------------------------
# LLM degradation signal (use_llm requested but every LLM call failed)
# ---------------------------------------------------------------------------


def _meta_from_json_report(state: SkillspectorState) -> dict:
    """Run the report node in JSON mode and return the metadata block."""
    return json.loads(report(state)["report_body"])["metadata"]


def test_report_llm_degraded_when_all_calls_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """use_llm requested + every semantic-analyzer call failed -> llm_degraded True.

    llm_available/meta_analysis_applied are about the provider and
    meta_analyzer's OWN call specifically (see the meta_analysis_applied
    tests below); none of these three failures is a meta_analyzer record,
    so those two fields stay True here and the failure surfaces only via
    llm_degraded / llm_calls_attempted / llm_calls_succeeded / llm_error.
    """
    # Pre-flight reports available (binary/creds present); the failure is at runtime.
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [
            llm_call_record("semantic_security_discovery", ok=False, error="claude empty stdout"),
            llm_call_record("semantic_developer_intent", ok=False, error="claude empty stdout"),
            llm_call_record("semantic_quality_policy", ok=False, error="boom"),
        ],
    }
    meta = _meta_from_json_report(state)
    assert meta["llm_requested"] is True
    assert meta["llm_available"] is True  # provider ok; meta_analyzer never ran/failed
    assert meta["llm_degraded"] is True
    assert meta["llm_calls_attempted"] == 3
    assert meta["llm_calls_succeeded"] == 0
    # Distinct error reasons are surfaced (deduped).
    assert "claude empty stdout" in meta["llm_error"]
    assert "static analysis only" in meta["llm_error"]


def test_report_degraded_when_some_calls_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dropped/throttled batch degrades the scan even though other calls succeeded.

    A rate-limited provider can 429 one batch (e.g. the security-discovery
    analyzer) while the rest of the fan-out succeeds; that is still a coverage
    gap and must not read as a clean, fully-analyzed scan.
    """
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [
            llm_call_record("semantic_security_discovery", ok=True),
            llm_call_record("semantic_quality_policy", ok=False, error="boom"),
        ],
    }
    meta = _meta_from_json_report(state)
    # Neither record is meta_analyzer, so llm_available/meta_analysis_applied
    # are untouched by this coverage gap; llm_degraded is the signal for it.
    assert meta["llm_available"] is True
    assert meta["llm_degraded"] is True
    assert meta["llm_calls_attempted"] == 2
    assert meta["llm_calls_succeeded"] == 1
    assert "1 of 2" in meta["llm_error"]


def test_report_meta_analysis_applied_survives_other_analyzer_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """meta_analyzer succeeding is independent of a different analyzer's coverage loss.

    Regression for the reviewed gap: a semantic analyzer dropping a batch
    forced meta_analysis_applied=False and llm_available=False even though
    meta_analyzer's own call succeeded in full, which misstated two
    independent contracts (meta-analysis ran vs. some coverage was lost) as
    one boolean. Matches the reported 3/4 scenario: 3 calls succeed
    (including meta_analyzer), 1 semantic-analyzer batch is dropped.
    """
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [
            llm_call_record("semantic_security_discovery", ok=False, error="429 rate limited"),
            llm_call_record("semantic_developer_intent", ok=True),
            llm_call_record("semantic_quality_policy", ok=True),
            llm_call_record("meta_analyzer", ok=True),
        ],
    }
    meta = _meta_from_json_report(state)
    assert meta["meta_analysis_applied"] is True
    assert meta["llm_available"] is True
    assert "filtering_mode" not in meta
    # The lost coverage is still visible, just not through these two fields.
    assert meta["llm_degraded"] is True
    assert meta["llm_calls_attempted"] == 4
    assert meta["llm_calls_succeeded"] == 3


def test_report_meta_analysis_not_applied_when_meta_analyzer_itself_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """meta_analyzer's own failure still zeros meta_analysis_applied/llm_available.

    This is the other half of the independent-contracts fix: the two fields
    are not blind to meta_analyzer - they just ignore everyone ELSE.
    """
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [
            llm_call_record("semantic_security_discovery", ok=True),
            llm_call_record("meta_analyzer", ok=False, error="claude empty stdout"),
        ],
    }
    meta = _meta_from_json_report(state)
    assert meta["meta_analysis_applied"] is False
    assert meta["llm_available"] is False
    assert meta["filtering_mode"] == "heuristic"


def test_report_meta_analysis_not_applied_when_no_meta_analyzer_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty meta_analyzer_records list must not vacuously satisfy meta_analysis_applied.

    meta_analyzer never runs when there are no findings to filter (it
    short-circuits to not_applicable), so its records list is empty and
    all([]) is True; a plain all(...) check with no non-empty guard reports
    meta_analysis_applied=True with no meta-analyzer call ever made.
    meta_analysis_applied now requires at least one record and all of them ok.
    llm_available stays True: provider availability is a separate contract
    from whether meta_analyzer had anything to do.
    """
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [],
    }
    meta = _meta_from_json_report(state)
    assert meta["meta_analysis_applied"] is False
    assert meta["llm_available"] is True
    assert meta["filtering_mode"] == "heuristic"


def test_report_not_degraded_when_no_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """use_llm True but no LLM calls attempted (e.g. empty skill) -> not degraded."""
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [],
    }
    meta = _meta_from_json_report(state)
    assert meta["llm_available"] is True
    assert "llm_degraded" not in meta
    assert "llm_calls_attempted" not in meta


def test_json_report_exposes_only_sanitized_provider_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [llm_call_record("meta_analyzer", ok=True)],
        "inference_usage": [
            {
                "node": "meta_analyzer",
                "request_kind": "structured_output",
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "model_source": "provider_response",
                "usage_source": "provider_response",
                "prompt_tokens": 123,
                "completion_tokens": 45,
                "total_tokens": 168,
                "secret": "not serialized",
            }
        ],
    }

    meta = _meta_from_json_report(state)

    assert meta["inference_usage"] == [
        {
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
    ]


def test_report_no_llm_failures_not_counted_as_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """use_llm False -> failures (if any) never mark the scan degraded."""
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": False,
        "llm_call_log": [llm_call_record("meta_analyzer", ok=False, error="boom")],
    }
    meta = _meta_from_json_report(state)
    assert "llm_degraded" not in meta


def test_report_terminal_shows_degraded_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Terminal output surfaces a visible degraded-scan warning."""
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {"name": "t"},
        "output_format": "terminal",
        "use_llm": True,
        "llm_call_log": [llm_call_record("semantic_quality_policy", ok=False, error="boom")],
    }
    body = report(state)["report_body"]
    assert "Degraded scan" in body
    assert "STATIC analysis only" in body


def test_report_markdown_shows_degraded_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markdown output surfaces a visible degraded-scan warning."""
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "markdown",
        "use_llm": True,
        "llm_call_log": [llm_call_record("meta_analyzer", ok=False, error="boom")],
    }
    body = report(state)["report_body"]
    assert "Degraded scan" in body


def test_report_sarif_carries_degradation_notification() -> None:
    """The default SARIF output surfaces degradation via a tool-execution notification."""
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "sarif",
        "use_llm": True,
        "llm_call_log": [
            llm_call_record("semantic_security_discovery", ok=False, error="claude empty stdout"),
        ],
    }
    result = report(state)
    run = result["sarif_report"]["runs"][0]
    assert "invocations" in run
    invocation = run["invocations"][0]
    assert invocation["executionSuccessful"] is True  # scan completed; LLM sub-stage degraded
    notification = invocation["toolExecutionNotifications"][0]
    assert notification["level"] == "warning"
    assert "STATIC analysis only" in notification["message"]["text"]
    # The serialized report_body carries it too, and the doc stays schema-valid.
    body = json.loads(result["report_body"])
    assert body["runs"][0]["invocations"][0]["toolExecutionNotifications"]
    validate_sarif_report(result["sarif_report"])


def test_report_sarif_has_successful_invocation_when_not_degraded() -> None:
    """SARIF always records the single canonical inspection invocation."""
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "sarif",
        "use_llm": True,
        "llm_call_log": [llm_call_record("semantic_security_discovery", ok=True)],
    }
    result = report(state)
    invocations = result["sarif_report"]["runs"][0]["invocations"]
    assert len(invocations) == 1
    assert invocations[0]["executionSuccessful"] is True


def test_report_sarif_projects_complete_analysis_completeness() -> None:
    """SARIF consumers can distinguish a complete scan without parsing prose."""
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "sarif",
        "analysis_completeness": {  # type: ignore[typeddict-item]
            "total_components": 2,
            "coverage_percent": 100.0,
            "is_complete": True,
            "status": "complete",
            "fully_inspected_files": 2,
            "partially_inspected_files": 0,
            "entirely_uninspected_files": 0,
            "ledger_exceptions": [],
            "scope_exclusions": [],
            "limitations": [],
        },
    }

    result = report(state)
    invocation = result["sarif_report"]["runs"][0]["invocations"][0]
    projected = invocation["properties"]["analysisCompleteness"]

    assert projected == {
        "isComplete": True,
        "status": "complete",
        "coveragePercent": 100.0,
        "totalComponents": 2,
        "fullyInspectedFiles": 2,
        "partiallyInspectedFiles": 0,
        "entirelyUninspectedFiles": 0,
        "ledgerExceptionCount": 0,
        "scopeExclusionCount": 0,
        "limitationCount": 0,
        "notificationRecordLimit": 10_000,
        "notificationsTruncated": False,
    }
    assert (
        json.loads(result["report_body"])["runs"][0]["invocations"][0]["properties"]
        == invocation["properties"]
    )
    validate_sarif_report(result["sarif_report"])


def test_report_sarif_projects_incomplete_analysis_without_internal_payloads() -> None:
    """The fail-closed SARIF property bag is bounded to safe canonical fields."""
    secret = "provider-response-must-not-leak"
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "sarif",
        "analysis_completeness": {  # type: ignore[typeddict-item]
            "total_components": 2,
            "coverage_percent": 50.0,
            "is_complete": False,
            "status": "partial",
            "fully_inspected_files": 1,
            "partially_inspected_files": 0,
            "entirely_uninspected_files": 1,
            "ledger_exceptions": [
                {
                    "outcome": "skipped",
                    "phase": "nested_artifact_inspection",
                    "reason_code": "archive_malformed",
                    "message": "ZIP-compatible content is malformed.",
                    "path": "broken.zip",
                    "fatal": False,
                    "provider_payload": secret,
                }
            ],
            "scope_exclusions": [],
            "limitations": [],
            "internal_debug": secret,
        },
    }

    result = report(state)
    invocation = result["sarif_report"]["runs"][0]["invocations"][0]
    projected = invocation["properties"]["analysisCompleteness"]

    assert projected["isComplete"] is False
    assert projected["status"] == "partial"
    assert projected["coveragePercent"] == 50.0
    assert projected["fullyInspectedFiles"] == 1
    assert projected["entirelyUninspectedFiles"] == 1
    assert projected["ledgerExceptionCount"] == 1
    assert invocation["toolExecutionNotifications"][0]["properties"]["reasonCode"] == (
        "archive_malformed"
    )
    assert secret not in result["report_body"]
    validate_sarif_report(result["sarif_report"])


def test_report_sarif_bounds_completeness_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillspector.nodes.report.MAX_FINDING_OUTPUT_RECORDS", 2)
    exceptions = [
        {
            "outcome": "partial",
            "phase": "test",
            "reason_code": "size_limit",
            "message": "Inspection reached a size limit.",
            "path": f"file-{index}.txt",
            "fatal": False,
        }
        for index in range(4)
    ]
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "sarif",
        "analysis_completeness": {  # type: ignore[typeddict-item]
            "total_components": 4,
            "coverage_percent": 0.0,
            "is_complete": False,
            "status": "partial",
            "fully_inspected_files": 0,
            "partially_inspected_files": 4,
            "entirely_uninspected_files": 0,
            "ledger_exceptions": exceptions,
            "scope_exclusions": [],
            "limitations": [],
        },
    }

    invocation = report(state)["sarif_report"]["runs"][0]["invocations"][0]
    notifications = invocation["toolExecutionNotifications"]
    projected = invocation["properties"]["analysisCompleteness"]

    assert len(notifications) == 2
    assert notifications[-1]["properties"]["reasonCode"] == "output_limit"
    assert notifications[-1]["properties"]["observedRecords"] == 4
    assert projected["ledgerExceptionCount"] == 4
    assert projected["notificationRecordLimit"] == 2
    assert projected["notificationsTruncated"] is True


# ---------------------------------------------------------------------------
# Fail-closed: a degraded deep scan must not be able to report SAFE
# ---------------------------------------------------------------------------


def test_degraded_scan_floors_recommendation_at_caution() -> None:
    """No findings would normally be SAFE; a degraded LLM stage forces CAUTION."""
    state: SkillspectorState = {
        "filtered_findings": [],  # static score 0 -> would be SAFE
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [llm_call_record("semantic_security_discovery", ok=False, error="boom")],
    }
    result = report(state)
    assert result["risk_score"] == 0  # score is left honest
    assert result["risk_recommendation"] == "CAUTION"  # but never SAFE when degraded


def test_unavailable_provider_floors_recommendation_even_with_success_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider truth wins when swallowed batch failures produced false success records."""
    monkeypatch.setattr(
        "skillspector.nodes.report.is_llm_available",
        lambda: (False, "codex binary not found"),
    )
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [
            llm_call_record("semantic_developer_intent", ok=True),
            llm_call_record("semantic_quality_policy", ok=True),
            llm_call_record("semantic_security_discovery", ok=False, error="binary missing"),
        ],
    }

    result = report(state)
    assert result["risk_recommendation"] == "CAUTION"
    payload = json.loads(result["report_body"])
    assert payload["risk_assessment"]["recommendation"] == "CAUTION"
    assert payload["metadata"]["llm_available"] is False


def test_partial_llm_failure_also_floors_recommendation_at_caution() -> None:
    """A rate-limited provider dropping one batch must not read as a clean scan.

    Matches the reported failure: llm_calls_attempted=4, llm_calls_succeeded=3
    (one batch 429'd and was dropped), yet the report emitted a plain SAFE
    verdict because only an all-calls-failed scan was treated as degraded.
    """
    state: SkillspectorState = {
        "filtered_findings": [],  # static score 0 -> would be SAFE
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [
            llm_call_record("semantic_security_discovery", ok=False, error="429 rate limited"),
            llm_call_record("semantic_developer_intent", ok=True),
            llm_call_record("semantic_quality_policy", ok=True),
            llm_call_record("meta_analyzer", ok=True),
        ],
    }
    result = report(state)
    assert result["risk_score"] == 0  # score is left honest
    assert result["risk_recommendation"] == "CAUTION"  # never SAFE on a partial pass


def test_analyzer_partial_batch_failure_flows_through_to_report_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: an analyzer-level partial batch failure reaches the report as degraded.

    Regression for the reviewed gap: llm_call_log records were built with
    ``ok=bool(outcome.successful) or not outcome.failures``, so a batch
    dropping while a sibling batch in the same analyzer succeeded still
    recorded ok=True and the report never saw the coverage loss (the exact
    two-file/one-429 case ``test_partial_batch_failure_records_llm_failure``
    in test_semantic_developer_intent.py now pins). This drives the real
    ``semantic_developer_intent`` node through a mocked partial-batch outcome
    and feeds its ACTUAL llm_call_log output into report(), rather than
    hand-constructing the log the way the report-only tests above do.
    """
    from unittest.mock import MagicMock, patch

    from skillspector.llm_analyzer_base import (
        BatchExecutionResult,
        BatchFailure,
        LLMAnalyzerBase,
    )
    from skillspector.nodes.analyzers.semantic_developer_intent import node as di_node

    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))

    def _mock_get_chat_model(*_args: object, **_kwargs: object) -> MagicMock:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        return mock_llm

    async def partially_succeeds(self: LLMAnalyzerBase, batches: list, **_kwargs: object) -> list:
        successful = [(batches[0], [])]
        self._last_batch_outcome = BatchExecutionResult(
            successful=successful,
            failures=[BatchFailure(batches[1], "TimeoutError")],
        )
        return successful

    with (
        patch("skillspector.llm_analyzer_base.get_chat_model", _mock_get_chat_model),
        patch.object(LLMAnalyzerBase, "arun_batches", partially_succeeds),
    ):
        analyzer_result = di_node({"file_cache": {"first.py": "print(1)", "second.py": "print(2)"}})

    # The analyzer's own record reflects the dropped batch...
    assert analyzer_result["llm_call_log"] == [
        {"node": "semantic_developer_intent", "ok": False, "error": None}
    ]

    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": analyzer_result["llm_call_log"],
    }
    result = report(state)
    meta = json.loads(result["report_body"])["metadata"]

    # ...and the report-level verdict reflects it too: never a plain SAFE on
    # a multi-batch analyzer that silently lost coverage.
    assert meta["llm_degraded"] is True
    assert result["risk_recommendation"] == "CAUTION"


def test_non_degraded_clean_scan_stays_safe() -> None:
    """Without degradation, a clean scan still reports SAFE (no over-flooring)."""
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [llm_call_record("semantic_security_discovery", ok=True)],
    }
    result = report(state)
    assert result["risk_recommendation"] == "SAFE"


def test_degraded_scan_does_not_downgrade_a_blocking_verdict() -> None:
    """A degraded scan that is already DO_NOT_INSTALL stays blocking (floor only lifts SAFE)."""
    state: SkillspectorState = {
        "filtered_findings": [_finding("P5", "CRITICAL"), _finding("P6", "CRITICAL")],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [llm_call_record("meta_analyzer", ok=False, error="boom")],
    }
    result = report(state)
    assert result["risk_recommendation"] == "DO_NOT_INSTALL"


def test_report_executable_scripts_multiplier() -> None:
    """1.3x multiplier applied only to findings from executable files."""
    # 2 HIGH findings in run.py = 2 × 25 × 1.3 = 65 (float-based accumulation)
    state: SkillspectorState = {
        "filtered_findings": [
            _finding("E2", "HIGH", file="run.py"),
            _finding("PE3", "HIGH", file="run.py"),
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


def test_report_doc_findings_no_multiplier() -> None:
    """Findings from non-executable files (markdown/docs) are not multiplied."""
    # 2 HIGH in SKILL.md (non-executable) = 2 × 25 = 50 (no 1.3x)
    state: SkillspectorState = {
        "filtered_findings": [
            _finding("P1", "HIGH", file="SKILL.md"),
            _finding("P2", "HIGH", file="SKILL.md"),
        ],
        "component_metadata": [
            {
                "path": "SKILL.md",
                "type": "markdown",
                "lines": 10,
                "executable": False,
                "size_bytes": 500,
            },
            {"path": "run.py", "type": "python", "lines": 5, "executable": True, "size_bytes": 200},
        ],
        "has_executable_scripts": True,
        "manifest": {},
        "skill_path": "/tmp/skill",
        "output_format": "json",
    }
    result = report(state)
    # Without the multiplier: 2 HIGH = 50, not 65
    assert result["risk_score"] == 50
    assert result["risk_severity"] == "MEDIUM"


def test_report_sarif_preserves_high_vs_critical_severity() -> None:
    """HIGH and CRITICAL both map to SARIF error, but properties keep the exact severity."""
    state: SkillspectorState = {
        "filtered_findings": [
            _finding("R1", "HIGH", message="high finding", file="high.py"),
            _finding("R2", "CRITICAL", message="critical finding", file="critical.py"),
        ],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "skill_path": None,
        "output_format": "sarif",
    }
    results = report(state)["sarif_report"]["runs"][0]["results"]
    by_rule = {item["ruleId"]: item for item in results}
    assert by_rule["R1"]["level"] == "error"
    assert by_rule["R2"]["level"] == "error"
    assert by_rule["R1"]["properties"]["severity"] == "HIGH"
    assert by_rule["R2"]["properties"]["severity"] == "CRITICAL"
