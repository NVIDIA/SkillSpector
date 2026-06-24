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

"""Tests for analysis_completeness field in report output."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from skillspector.models import Finding
from skillspector.nodes.report import _build_analysis_completeness, report


def _make_finding(**kwargs) -> Finding:
    defaults = {
        "rule_id": "PE3",
        "message": "Credential Access",
        "severity": "HIGH",
        "confidence": 0.9,
        "file": "tool.py",
        "start_line": 1,
        "end_line": 1,
        "remediation": "Remove",
        "tags": ["test"],
        "context": "ctx",
        "matched_text": "match",
        "category": "priv_esc",
        "pattern": "PE3",
        "finding": "snippet",
        "explanation": "explain",
        "code_snippet": "code",
        "intent": None,
    }
    defaults.update(kwargs)
    return Finding(**defaults)


class TestBuildAnalysisCompleteness:
    """_build_analysis_completeness produces correct coverage metadata."""

    def test_full_coverage_complete(self) -> None:
        components = ["a.py", "b.py"]
        file_cache = {"a.py": "code", "b.py": "code"}
        findings = [_make_finding()]
        with patch("skillspector.nodes.report.is_llm_available", return_value=(True, None)):
            result = _build_analysis_completeness(
                components,
                file_cache,
                use_llm=True,
                findings_pre_filter=findings,
                findings_post_filter=findings,
            )
        assert result["total_components"] == 2
        assert result["scanned_components"] == 2
        assert result["coverage_percent"] == 100.0
        assert result["llm_analysis"] == "applied"
        assert result["is_complete"] is True
        assert result["limitations"] is None

    def test_partial_coverage_reports_skipped(self) -> None:
        components = ["a.py", "b.py", "c.py"]
        file_cache = {"a.py": "code"}
        with patch("skillspector.nodes.report.is_llm_available", return_value=(True, None)):
            result = _build_analysis_completeness(
                components,
                file_cache,
                use_llm=True,
                findings_pre_filter=[],
                findings_post_filter=[],
            )
        assert result["total_components"] == 3
        assert result["scanned_components"] == 1
        assert result["coverage_percent"] == pytest.approx(33.3, abs=0.1)
        assert result["is_complete"] is False
        assert any("2 component(s)" in lim for lim in result["limitations"])

    def test_llm_unavailable_noted(self) -> None:
        with patch(
            "skillspector.nodes.report.is_llm_available",
            return_value=(False, "OPENAI_API_KEY not set"),
        ):
            result = _build_analysis_completeness(
                ["a.py"],
                {"a.py": "code"},
                use_llm=True,
                findings_pre_filter=[],
                findings_post_filter=[],
            )
        assert result["llm_analysis"] == "skipped"
        assert result["is_complete"] is False
        assert any("LLM meta-analysis unavailable" in lim for lim in result["limitations"])

    def test_llm_disabled_noted(self) -> None:
        with patch("skillspector.nodes.report.is_llm_available", return_value=(True, None)):
            result = _build_analysis_completeness(
                ["a.py"],
                {"a.py": "code"},
                use_llm=False,
                findings_pre_filter=[],
                findings_post_filter=[],
            )
        assert result["llm_analysis"] == "skipped"
        assert result["is_complete"] is False
        assert any("--no-llm" in lim for lim in result["limitations"])

    def test_findings_filtered_noted(self) -> None:
        pre = [_make_finding(), _make_finding(), _make_finding()]
        post = [_make_finding()]
        with patch("skillspector.nodes.report.is_llm_available", return_value=(True, None)):
            result = _build_analysis_completeness(
                ["a.py"],
                {"a.py": "code"},
                use_llm=True,
                findings_pre_filter=pre,
                findings_post_filter=post,
            )
        assert result["findings_before_filtering"] == 3
        assert result["findings_after_filtering"] == 1
        assert any("2 finding(s) filtered" in lim for lim in result["limitations"])

    def test_empty_components_gives_100_coverage(self) -> None:
        with patch("skillspector.nodes.report.is_llm_available", return_value=(True, None)):
            result = _build_analysis_completeness(
                [],
                {},
                use_llm=True,
                findings_pre_filter=[],
                findings_post_filter=[],
            )
        assert result["coverage_percent"] == 100.0
        assert result["total_components"] == 0


class TestCompletenessInJsonReport:
    """analysis_completeness field appears in JSON report output."""

    @patch("skillspector.nodes.report.is_llm_available", return_value=(True, None))
    def test_json_report_includes_completeness(self, _mock_llm) -> None:
        state = {
            "findings": [_make_finding()],
            "filtered_findings": [_make_finding()],
            "components": ["tool.py"],
            "file_cache": {"tool.py": "import os"},
            "component_metadata": [{"path": "tool.py", "type": "python", "lines": 1}],
            "has_executable_scripts": False,
            "manifest": {"name": "test-skill"},
            "skill_path": "/tmp/skill",
            "output_format": "json",
            "use_llm": True,
        }
        result = report(state)
        body = json.loads(result["report_body"])
        assert "analysis_completeness" in body
        assert body["analysis_completeness"]["total_components"] == 1
        assert body["analysis_completeness"]["scanned_components"] == 1
        assert body["analysis_completeness"]["coverage_percent"] == 100.0

    @patch("skillspector.nodes.report.is_llm_available", return_value=(True, None))
    def test_sarif_format_does_not_include_completeness(self, _mock_llm) -> None:
        state = {
            "findings": [_make_finding()],
            "filtered_findings": [_make_finding()],
            "components": ["tool.py"],
            "file_cache": {"tool.py": "import os"},
            "component_metadata": [],
            "has_executable_scripts": False,
            "manifest": {},
            "skill_path": None,
            "output_format": "sarif",
            "use_llm": True,
        }
        result = report(state)
        body = json.loads(result["report_body"])
        assert "analysis_completeness" not in body
        assert "$schema" in body
