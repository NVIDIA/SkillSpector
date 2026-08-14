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


def test_graph_excludes_valid_oms_signature_from_static_findings(tmp_path: Path) -> None:
    """A real OMS signature remains inventoried without producing scan findings."""
    fixture = Path(__file__).parents[1] / "fixtures" / "oms" / "mcore-split-pr.skill.oms.sig"
    (tmp_path / "SKILL.md").write_text("---\nname: signed\n---\n# Signed\n", encoding="utf-8")
    (tmp_path / "skill.oms.sig").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    result = graph.invoke(
        {
            "skill_path": str(tmp_path),
            "output_format": "json",
            "use_llm": False,
        }
    )

    report = json.loads(result["report_body"])
    signature_component = next(
        component for component in report["components"] if component["path"] == "skill.oms.sig"
    )
    assert signature_component["type"] == "oms_signature"
    assert report["analysis_completeness"]["coverage_percent"] == 100.0
    assert report["analysis_completeness"]["scope_exclusions"] == [
        {
            "outcome": "out_of_scope",
            "phase": "discovery",
            "reason_code": "oms_signature",
            "message": "Recognized OMS signature metadata is excluded from content analysis.",
            "path": "skill.oms.sig",
            "start_line": None,
            "end_line": None,
            "fatal": False,
        }
    ]
    assert report["analysis_completeness"]["ledger_exceptions"] == []
    assert report["analysis_completeness"]["execution_successful"] is True
    assert "skill.oms.sig" not in result["components"]
    assert "skill.oms.sig" not in result["file_cache"]
    assert not any(
        event["path"] == "skill.oms.sig" and event["outcome"] == "failed"
        for event in result["inspection_ledger"]
    )
    assert all(finding.file != "skill.oms.sig" for finding in result["findings"])
    assert all(issue["file"] != "skill.oms.sig" for issue in report["issues"])


@pytest.mark.parametrize("output_format", ["terminal", "markdown", "sarif"])
def test_graph_reports_oms_scope_exclusion_in_every_non_json_format(
    tmp_path: Path, output_format: str
) -> None:
    """OMS scope exclusions remain visible in every user-facing report format."""
    fixture = Path(__file__).parents[1] / "fixtures" / "oms" / "mcore-split-pr.skill.oms.sig"
    (tmp_path / "SKILL.md").write_text("---\nname: signed\n---\n# Signed\n", encoding="utf-8")
    (tmp_path / "skill.oms.sig").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    result = graph.invoke(
        {
            "skill_path": str(tmp_path),
            "output_format": output_format,
            "use_llm": False,
        }
    )

    scope_exclusion = result["analysis_completeness"]["scope_exclusions"]
    assert scope_exclusion[0]["path"] == "skill.oms.sig"
    assert scope_exclusion[0]["reason_code"] == "oms_signature"

    if output_format == "sarif":
        notifications = result["sarif_report"]["runs"][0]["invocations"][0][
            "toolExecutionNotifications"
        ]
        notification = next(
            item for item in notifications if item["properties"]["reasonCode"] == "oms_signature"
        )
        assert notification["level"] == "note"
        assert notification["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == (
            "skill.oms.sig"
        )
    else:
        expected_heading = "Scope exclusions" if output_format == "terminal" else "Scope Exclusions"
        assert expected_heading in result["report_body"]
        assert "oms_signature" in result["report_body"]
        assert "skill.oms.sig" in result["report_body"]


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


def test_graph_surfaces_degraded_llm_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: use_llm requested but every LLM call fails.

    Proves (a) the operator.add reducer accumulates llm_call_log across the
    parallel analyzer fan-out AND the meta node, (b) the graph completes
    instead of crashing (regression guard for meta_analyzer constructing its
    chat model outside the try/except), and (c) the report flags the
    degraded, static-only scan in every surface.
    """
    (tmp_path / "SKILL.md").write_text(
        "---\nname: demo\ndescription: reads files\n---\n# Demo\n", encoding="utf-8"
    )
    # os.system gives a static finding so meta_analyzer also runs (and is exercised).
    (tmp_path / "run.py").write_text("import os\nos.system('ls')\n", encoding="utf-8")

    def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("simulated LLM transport failure")

    class FailingTP4Analyzer:
        """Simulate an attempted TP4 request failing after analyzer setup."""

        @property
        def inference_usage(self) -> list[object]:
            return []

        def __init__(self, _model: str) -> None:
            pass

        def run_batches_detailed(self, _batches: object) -> object:
            raise RuntimeError("simulated LLM transport failure")

    # Semantic analyzers and meta_analyzer fail while constructing their shared
    # transport. TP4's analyzer construction is deliberately not an attempted
    # LLM call, so fail it at batch execution to assert its ledger projection.
    monkeypatch.setattr("skillspector.llm_analyzer_base.get_chat_model", boom)
    monkeypatch.setattr(
        "skillspector.nodes.analyzers.mcp_tool_poisoning._TP4Analyzer", FailingTP4Analyzer
    )

    result = graph.invoke({"skill_path": str(tmp_path), "use_llm": True, "output_format": "json"})

    log = result["llm_call_log"]
    assert log, "expected LLM telemetry records"
    assert all(r["ok"] is False for r in log), log
    nodes = {r["node"] for r in log}
    # The three semantic analyzers always attempt; meta_analyzer runs because the
    # static finding above gives it work (and must be caught, not crash).
    assert {
        "semantic_security_discovery",
        "semantic_developer_intent",
        "semantic_quality_policy",
        "meta_analyzer",
        "mcp_tool_poisoning",
    } <= nodes

    meta = json.loads(result["report_body"])["metadata"]
    assert meta["llm_available"] is False
    assert meta["llm_degraded"] is True
    assert meta["llm_calls_succeeded"] == 0
    assert result["execution_successful"] is False

    notification = result["sarif_report"]["runs"][0]["invocations"][0][
        "toolExecutionNotifications"
    ][0]
    assert notification["level"] == "error"
