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
from skillspector.state import SkillspectorState
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
        findings = [_finding("R1", "HIGH", confidence=1.0)]
        score, _, _ = _compute_risk_score(findings, True)
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
                _finding("E2", "HIGH", confidence=1.0),
                _finding("PE3", "HIGH", confidence=1.0),
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
        assert "issues" in data
        assert len(data["issues"]) == 1
        assert data["issues"][0]["id"] == "P1"

    def test_report_output_format_markdown(self) -> None:
        """output_format markdown produces expected headings."""
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

    def test_report_output_format_terminal(self) -> None:
        """output_format terminal produces Rich-formatted output."""
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

    def test_report_output_format_sarif(self) -> None:
        """output_format sarif produces valid SARIF JSON."""
        state: SkillspectorState = {
            "filtered_findings": [_finding("E2", "HIGH", "env harvest", confidence=1.0)],
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
    state: SkillspectorState = {
        "filtered_findings": [_finding("P5", "CRITICAL")],
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
    assert sarif_results[0]["suppressions"][0]["kind"] == "external"
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
