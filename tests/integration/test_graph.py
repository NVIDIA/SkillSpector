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
from importlib import import_module
from pathlib import Path

import pytest

from skillspector.graph import create_graph, graph


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


@pytest.mark.parametrize("output_format", ["terminal", "json", "markdown", "sarif"])
@pytest.mark.parametrize(
    ("script", "expected_destination"),
    [
        (
            "MARKER=1 NPM_CONFIG_REGISTRY=https://packages.example.invalid\n",
            "https://packages.example.invalid",
        ),
        (
            "export MARKER=1 PIP_INDEX_URL=https://packages.example.invalid/simple\n",
            "https://packages.example.invalid/simple",
        ),
        (
            "export MARKER=1 PIP_EXTRA_INDEX_URL=https://packages.example.invalid/simple\n",
            "https://packages.example.invalid/simple",
        ),
        (
            "export MARKER=1 CARGO_REGISTRIES_PRIVATE_INDEX="
            "sparse+https://packages.example.invalid/index\n",
            "sparse+https://packages.example.invalid/index",
        ),
        (
            "env MARKER=1 npm config set registry https://packages.example.invalid\n",
            "https://packages.example.invalid",
        ),
        (
            "sudo -E npm config set registry https://packages.example.invalid\n",
            "https://packages.example.invalid",
        ),
        (
            "command -- npm config set registry https://packages.example.invalid\n",
            "https://packages.example.invalid",
        ),
        (
            "( npm config set registry https://packages.example.invalid )\n",
            "https://packages.example.invalid",
        ),
        (
            """cat > .npmrc <<END$OF
registry=https://packages.example.invalid
END$OF
""",
            "https://packages.example.invalid",
        ),
    ],
)
def test_graph_reports_wrapped_dependency_source_changes_in_every_format(
    tmp_path: Path, output_format: str, script: str, expected_destination: str
) -> None:
    """SC10 survives the complete static graph and every public report format."""
    (tmp_path / "SKILL.md").write_text(
        "---\nname: dependency-source-test\n---\n# Dependency Source Test\n",
        encoding="utf-8",
    )
    (tmp_path / "setup.sh").write_text(script, encoding="utf-8")

    result = graph.invoke(
        {
            "skill_path": str(tmp_path),
            "output_format": output_format,
            "use_llm": False,
        }
    )

    finding = next(item for item in result["findings"] if item.rule_id == "SC10")
    assert finding.severity == "HIGH"
    assert finding.evidence["destination"] == expected_destination
    rendered = (
        json.dumps(result["sarif_report"]) if output_format == "sarif" else result["report_body"]
    )
    assert "SC10" in rendered
    assert "packages.example.invalid" in rendered


@pytest.mark.parametrize("output_format", ["terminal", "json", "markdown", "sarif"])
@pytest.mark.parametrize(
    "script",
    [
        "NPM_CONFIG_REGISTRY=https://registry.npmjs.org/ MARKER=1\n",
        '"npm config set registry https://packages.example.invalid"\n',
        """cat <<END$OF
npm config set registry https://packages.example.invalid
END$OF
""",
    ],
)
def test_graph_keeps_reviewed_canonical_and_inert_forms_clear(
    tmp_path: Path, output_format: str, script: str
) -> None:
    """Reviewed negative forms remain clear in every public report format."""
    (tmp_path / "SKILL.md").write_text(
        "---\nname: dependency-source-negative-test\n---\n# Dependency Source Negative Test\n",
        encoding="utf-8",
    )
    (tmp_path / "setup.sh").write_text(script, encoding="utf-8")

    result = graph.invoke(
        {
            "skill_path": str(tmp_path),
            "output_format": output_format,
            "use_llm": False,
        }
    )

    assert all(item.rule_id != "SC10" for item in result["findings"])
    rendered = (
        json.dumps(result["sarif_report"]) if output_format == "sarif" else result["report_body"]
    )
    assert "SC10" not in rendered


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

        def __init__(self, _model: str, **_kwargs: object) -> None:
            pass

        async def arun_batches_detailed(self, _batches: object) -> object:
            raise RuntimeError("simulated LLM transport failure")

    # Semantic analyzers and meta_analyzer fail while constructing their shared
    # transport. TP4's analyzer construction is deliberately not an attempted
    # LLM call, so fail it at batch execution to assert its ledger projection.
    monkeypatch.setattr("skillspector.llm_analyzer_base.get_chat_model", boom)
    monkeypatch.setattr(
        "skillspector.nodes.analyzers.mcp_tool_poisoning._TP4Analyzer", FailingTP4Analyzer
    )

    # Build after configuring availability so the mocked semantic transports
    # are exercised even when the test environment has no provider credentials.
    monkeypatch.setattr(
        import_module("skillspector.graph"), "is_llm_available", lambda: (True, None)
    )
    result = create_graph().invoke(
        {"skill_path": str(tmp_path), "use_llm": True, "output_format": "json"}
    )

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
