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

"""Tests for the Skillspector LangGraph workflow."""

import json
from pathlib import Path

import pytest

from skillspector.graph import graph


def test_graph_invoke_with_output_format_json(tmp_path: Path) -> None:
    """Invoking with output_format=json yields report_body as valid JSON with skill and risk_assessment."""
    (tmp_path / "SKILL.md").write_text("---\nname: test\n---\n# Hi", encoding="utf-8")
    result = graph.invoke(
        {
            "skill_path": str(tmp_path),
            "output_format": "json",
            "use_llm": False,
        }
    )
    body = result.get("report_body", "")
    assert body
    data = json.loads(body)
    assert "skill" in data
    assert "risk_assessment" in data
    assert "score" in data["risk_assessment"]
    assert "components" in data


def test_graph_invoke_returns_findings_and_report(tmp_path: Path) -> None:
    """Graph runs to completion; returns findings, SARIF report, report_body, risk_score."""
    result = graph.invoke({"skill_path": str(tmp_path), "use_llm": False})

    assert "findings" in result
    assert isinstance(result["findings"], list)
    assert "sarif_report" in result
    assert "risk_score" in result
    assert "report_body" in result
    assert result["risk_score"] >= 0
    assert isinstance(result["report_body"], str)


def test_graph_invalid_skill_path_raises() -> None:
    """Invalid skill_path raises instead of returning a clean low-risk report."""
    with pytest.raises(ValueError, match="not an existing directory"):
        graph.invoke(
            {
                "skill_path": "/nonexistent/path/xyz",
                "output_format": "json",
                "use_llm": False,
            }
        )
