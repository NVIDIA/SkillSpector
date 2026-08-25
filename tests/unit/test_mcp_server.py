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

"""Tests for the MCP server wrapper (run_scan core + scan_skill tool)."""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from skillspector import mcp_server
from skillspector.graph import graph as workflow_graph
from skillspector.mcp_server import _llm_runtime_accounting, run_scan
from skillspector.models import Finding
from skillspector.nodes.report import report
from skillspector.providers import reset_provider, use_provider
from skillspector.suppression import SuppressedFinding


def _write_skill(tmp_path: Path, body: str = "# Safe skill") -> Path:
    (tmp_path / "SKILL.md").write_text(f"---\nname: mcp-test\n---\n{body}", encoding="utf-8")
    return tmp_path


async def test_run_scan_returns_structured_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_scan returns a JSON-serialisable verdict with the expected shape."""
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))
    _write_skill(tmp_path)

    result = await run_scan(str(tmp_path), use_llm=False, output_format="json")

    assert result["target"] == str(tmp_path)
    assert isinstance(result["risk_score"], int)
    assert 0 <= result["risk_score"] <= 100
    assert isinstance(result["findings"], list)
    assert isinstance(result["safe_to_install"], bool)
    assert result["safe_to_install"] == (result["risk_score"] <= 50)
    assert result["report"]  # non-empty rendered report


async def test_run_scan_llm_accounting_is_honest_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requesting the LLM with no credentials must report it as not used."""
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))
    _write_skill(tmp_path)

    result = await run_scan(str(tmp_path), use_llm=True, output_format="json")

    assert result["llm_requested"] is True
    assert result["llm_available"] is False
    assert result["llm_used"] is False
    assert result["scan_mode"] == "static-only"


def _complete_zero_risk_graph_result() -> dict[str, object]:
    """Return a complete graph verdict suitable for MCP safety-predicate tests."""
    return {
        "filtered_findings": [],
        "risk_score": 0,
        "risk_severity": "LOW",
        "risk_recommendation": "SAFE",
        "execution_successful": True,
        "analysis_completeness": {
            "is_complete": True,
            "status": "complete",
            "entirely_uninspected_files": 0,
        },
        "report_body": "{}",
    }


def test_empty_llm_telemetry_is_not_a_complete_requested_pass() -> None:
    """Empty telemetry cannot prove that a requested semantic pass ran."""
    assert _llm_runtime_accounting(enabled=True, result={"llm_call_log": []}) == (False, False)


def test_all_not_applicable_semantic_statuses_are_complete_without_llm_use() -> None:
    """A real no-work pass is complete only when every semantic node says so."""
    result = {
        "llm_call_log": [],
        "analyzer_status_events": [
            {"analyzer_id": "semantic_security_discovery", "status": "not_applicable"},
            {"analyzer_id": "semantic_developer_intent", "status": "not_applicable"},
            {"analyzer_id": "semantic_quality_policy", "status": "not_applicable"},
        ],
    }

    assert _llm_runtime_accounting(enabled=True, result=result) == (False, True)


def test_effective_findings_require_successful_meta_analyzer_telemetry() -> None:
    """Meta-analysis is required when effective findings gave it work to do."""
    result = {
        "effective_finding_ids": ["finding-1"],
        "analyzer_status_events": [
            {"analyzer_id": "semantic_security_discovery", "status": "completed"},
            {"analyzer_id": "semantic_developer_intent", "status": "completed"},
            {"analyzer_id": "semantic_quality_policy", "status": "completed"},
        ],
        "llm_call_log": [
            {"node": "semantic_security_discovery", "ok": True, "error": None},
            {"node": "semantic_developer_intent", "ok": True, "error": None},
            {"node": "semantic_quality_policy", "ok": True, "error": None},
        ],
    }

    assert _llm_runtime_accounting(enabled=True, result=result) == (True, False)


def test_completed_semantic_analyzer_requires_its_own_successful_telemetry() -> None:
    """A completed semantic status cannot stand in for its missing call record."""
    result = {
        "effective_finding_ids": ["finding-1"],
        "analyzer_status_events": [
            {"analyzer_id": "semantic_security_discovery", "status": "completed"},
            {"analyzer_id": "semantic_developer_intent", "status": "completed"},
            {"analyzer_id": "semantic_quality_policy", "status": "completed"},
        ],
        "llm_call_log": [
            {"node": "semantic_security_discovery", "ok": True, "error": None},
            {"node": "semantic_quality_policy", "ok": True, "error": None},
            {"node": "meta_analyzer", "ok": True, "error": None},
        ],
    }

    assert _llm_runtime_accounting(enabled=True, result=result) == (True, False)


def _fully_accounted_semantic_result() -> dict[str, object]:
    """Return hand-authored complete semantic telemetry for validation tests."""
    return {
        "effective_finding_ids": ["finding-1"],
        "analyzer_status_events": [
            {"analyzer_id": "semantic_security_discovery", "status": "completed"},
            {"analyzer_id": "semantic_developer_intent", "status": "completed"},
            {"analyzer_id": "semantic_quality_policy", "status": "completed"},
        ],
        "llm_call_log": [
            {"node": "semantic_security_discovery", "ok": True, "error": None},
            {"node": "semantic_developer_intent", "ok": True, "error": None},
            {"node": "semantic_quality_policy", "ok": True, "error": None},
            {"node": "meta_analyzer", "ok": True, "error": None},
        ],
    }


@pytest.mark.parametrize(
    "malformed_status",
    [
        {"analyzer_id": [], "status": "completed"},
        {"analyzer_id": {}, "status": "completed"},
        {"status": "completed"},
        {"analyzer_id": "semantic_security_discovery"},
        {"analyzer_id": "semantic_security_discovery", "status": None},
        {"analyzer_id": "semantic_security_discovery", "status": []},
        [],
    ],
    ids=[
        "list-analyzer-id",
        "mapping-analyzer-id",
        "missing-analyzer-id",
        "missing-status",
        "none-status",
        "list-status",
        "non-mapping-event",
    ],
)
def test_malformed_analyzer_status_event_is_incomplete_without_crashing(
    malformed_status: object,
) -> None:
    """Malformed global or semantic status evidence cannot be silently discarded."""
    result = _fully_accounted_semantic_result()
    result["analyzer_status_events"].append(malformed_status)  # type: ignore[index]

    assert _llm_runtime_accounting(enabled=True, result=result) == (True, False)


def test_duplicate_semantic_status_evidence_is_incomplete() -> None:
    """Each semantic analyzer must produce exactly one terminal status."""
    result = _fully_accounted_semantic_result()
    result["analyzer_status_events"].append(  # type: ignore[index]
        {"analyzer_id": "semantic_security_discovery", "status": "completed"}
    )

    assert _llm_runtime_accounting(enabled=True, result=result) == (True, False)


def test_not_applicable_status_cannot_have_successful_semantic_telemetry() -> None:
    """A no-work terminal status conflicts with a successful call for that node."""
    result = _fully_accounted_semantic_result()
    result["analyzer_status_events"][0]["status"] = "not_applicable"  # type: ignore[index]

    assert _llm_runtime_accounting(enabled=True, result=result) == (True, False)


async def test_all_not_applicable_semantic_pass_remains_install_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit no-work statuses are complete without claiming an LLM call."""
    graph_result = _complete_zero_risk_graph_result()
    graph_result["llm_call_log"] = []
    graph_result["analyzer_status_events"] = [
        {"analyzer_id": "semantic_security_discovery", "status": "not_applicable"},
        {"analyzer_id": "semantic_developer_intent", "status": "not_applicable"},
        {"analyzer_id": "semantic_quality_policy", "status": "not_applicable"},
    ]
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    monkeypatch.setattr(mcp_server.graph, "ainvoke", AsyncMock(return_value=graph_result))

    verdict = await run_scan("fixture", use_llm=True, output_format="json")

    assert verdict["llm_used"] is False
    assert verdict["scan_mode"] == "static-only"
    assert verdict["safe_to_install"] is True
    assert verdict["recommendation"] == "SAFE"


def test_static_only_graph_keeps_semantic_nodes_without_constructing_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Static graph execution skips semantic model construction but keeps nodes wired."""
    capability_probes = 0

    class _CapabilityProvider:
        DEFAULT_MODEL = "test-model"
        SLOT_DEFAULTS = {"meta_analyzer": "test-model"}

        def get_context_length(self, model: str) -> int | None:
            return 4096 if model == "test-model" else None

        def get_max_output_tokens(self, model: str) -> int | None:
            return 128 if model == "test-model" else None

        def resolve_model(self, slot: str = "default") -> str:
            del slot
            return "test-model"

        def resolve_credentials(self) -> tuple[str, str | None] | None:
            return None

        def create_chat_model(
            self,
            model: str,
            *,
            max_tokens: int,
            timeout: float | None = 120,
        ) -> object:
            nonlocal capability_probes
            del model, max_tokens, timeout
            capability_probes += 1
            return object()

    semantic_model_factory = MagicMock(
        side_effect=AssertionError("static-only semantic nodes must not construct a chat model")
    )
    monkeypatch.setattr("skillspector.llm_analyzer_base.get_chat_model", semantic_model_factory)
    token = use_provider(_CapabilityProvider())
    _write_skill(tmp_path)
    try:
        result = workflow_graph.invoke(
            {"skill_path": str(tmp_path), "use_llm": False, "output_format": "json"}
        )
    finally:
        reset_provider(token)

    assert capability_probes > 0
    semantic_model_factory.assert_not_called()
    assert json.loads(result["report_body"])["metadata"]["llm_available"] is True
    semantic_statuses = {
        status["analyzer_id"]: status["status"]
        for status in result["analyzer_status_events"]
        if status["analyzer_id"].startswith("semantic_")  # type: ignore[index]
    }
    assert semantic_statuses == {
        "semantic_security_discovery": "disabled",
        "semantic_developer_intent": "disabled",
        "semantic_quality_policy": "disabled",
    }


async def test_late_provider_binding_cannot_claim_a_complete_semantic_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A graph built without credentials keeps semantic nodes for a later provider."""
    _write_skill(tmp_path)
    graph_module = importlib.import_module("skillspector.graph")
    monkeypatch.setattr(
        graph_module,
        "is_llm_available",
        lambda: (False, "not configured"),
        raising=False,
    )
    late_bound_graph = graph_module.create_graph()

    def transport_failure(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated late-bound provider failure")

    captured: dict[str, object] = {}

    async def invoke_real_graph(
        state: dict[str, object], config: dict[str, object]
    ) -> dict[str, object]:
        captured["result"] = await late_bound_graph.ainvoke(state, config=config)
        return captured["result"]  # type: ignore[return-value]

    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    monkeypatch.setattr("skillspector.llm_analyzer_base.get_chat_model", transport_failure)
    monkeypatch.setattr(mcp_server, "graph", SimpleNamespace(ainvoke=invoke_real_graph))

    verdict = await run_scan(str(tmp_path), use_llm=True, output_format="json")
    graph_result = captured["result"]

    statuses = {
        status["analyzer_id"]: status["status"]
        for status in graph_result["analyzer_status_events"]  # type: ignore[index]
        if status["analyzer_id"].startswith("semantic_")  # type: ignore[index]
    }
    assert statuses == {
        "semantic_security_discovery": "unavailable",
        "semantic_developer_intent": "unavailable",
        "semantic_quality_policy": "unavailable",
    }
    assert verdict["scan_mode"] == "static-only"
    assert verdict["safe_to_install"] is False
    assert verdict["recommendation"] != "SAFE"


async def test_requested_unavailable_llm_blocks_safe_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmet requested analysis pass cannot produce an install-safe verdict."""
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "not configured"))
    monkeypatch.setattr(
        mcp_server.graph,
        "ainvoke",
        AsyncMock(return_value=_complete_zero_risk_graph_result()),
    )

    verdict = await run_scan("fixture", use_llm=True, output_format="json")

    assert verdict["llm_requested"] is True
    assert verdict["llm_available"] is False
    assert verdict["llm_used"] is False
    assert verdict["scan_mode"] == "static-only"
    assert verdict["risk_score"] == 0
    assert verdict["safe_to_install"] is False
    assert verdict["recommendation"] == "CAUTION"


async def test_unrequested_llm_keeps_static_scan_eligible_for_safe_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit static-only scans retain the normal complete low-risk safety predicate."""
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "not configured"))
    monkeypatch.setattr(
        mcp_server.graph,
        "ainvoke",
        AsyncMock(return_value=_complete_zero_risk_graph_result()),
    )

    verdict = await run_scan("fixture", use_llm=False, output_format="json")

    assert verdict["llm_requested"] is False
    assert verdict["llm_available"] is False
    assert verdict["llm_used"] is False
    assert verdict["scan_mode"] == "static-only"
    assert verdict["risk_score"] == 0
    assert verdict["safe_to_install"] is True
    assert verdict["recommendation"] == "SAFE"


async def test_all_failed_runtime_llm_calls_block_safe_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful preflight cannot hide a requested LLM pass that failed at runtime."""
    graph_result = _complete_zero_risk_graph_result()
    graph_result["llm_call_log"] = [
        {"node": "semantic_security_discovery", "ok": False, "error": "transport error"},
        {"node": "semantic_quality_policy", "ok": False, "error": "invalid response"},
    ]
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    monkeypatch.setattr(
        mcp_server.graph,
        "ainvoke",
        AsyncMock(return_value=graph_result),
    )

    verdict = await run_scan("fixture", use_llm=True, output_format="json")

    assert verdict["llm_requested"] is True
    assert verdict["llm_available"] is True
    assert verdict["llm_used"] is False
    assert verdict["scan_mode"] == "static-only"
    assert verdict["risk_score"] == 0
    assert verdict["safe_to_install"] is False
    assert verdict["recommendation"] == "CAUTION"


async def test_partial_runtime_llm_failure_blocks_safe_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partially completed requested pass remains degraded and cannot be install-safe."""
    graph_result = _complete_zero_risk_graph_result()
    graph_result["llm_call_log"] = [
        {"node": "semantic_security_discovery", "ok": True, "error": None},
        {"node": "semantic_quality_policy", "ok": False, "error": "rate limited"},
    ]
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    monkeypatch.setattr(
        mcp_server.graph,
        "ainvoke",
        AsyncMock(return_value=graph_result),
    )

    verdict = await run_scan("fixture", use_llm=True, output_format="json")

    assert verdict["llm_requested"] is True
    assert verdict["llm_available"] is True
    assert verdict["llm_used"] is True
    assert verdict["scan_mode"] == "static+llm"
    assert verdict["risk_score"] == 0
    assert verdict["safe_to_install"] is False
    assert verdict["recommendation"] == "CAUTION"


async def test_successful_runtime_llm_calls_keep_complete_scan_install_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete low-risk pass remains eligible for an install-safe verdict."""
    graph_result = _complete_zero_risk_graph_result()
    graph_result["effective_finding_ids"] = ["finding-1"]
    graph_result["analyzer_status_events"] = [
        {"analyzer_id": "semantic_security_discovery", "status": "completed"},
        {"analyzer_id": "semantic_developer_intent", "status": "completed"},
        {"analyzer_id": "semantic_quality_policy", "status": "completed"},
    ]
    graph_result["llm_call_log"] = [
        {"node": "semantic_security_discovery", "ok": True, "error": None},
        {"node": "semantic_developer_intent", "ok": True, "error": None},
        {"node": "semantic_quality_policy", "ok": True, "error": None},
        {"node": "meta_analyzer", "ok": True, "error": None},
    ]
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    monkeypatch.setattr(
        mcp_server.graph,
        "ainvoke",
        AsyncMock(return_value=graph_result),
    )

    verdict = await run_scan("fixture", use_llm=True, output_format="json")

    assert verdict["llm_available"] is True
    assert verdict["llm_used"] is True
    assert verdict["scan_mode"] == "static+llm"
    assert verdict["safe_to_install"] is True
    assert verdict["recommendation"] == "SAFE"


async def _render_complete_zero_risk_result(
    state: dict[str, object], config: dict[str, object]
) -> dict[str, object]:
    """Render a canonical complete report from the wrapper-provided graph state."""
    del config
    return report(
        {
            **state,
            "filtered_findings": [],
            "component_metadata": [],
            "has_executable_scripts": False,
            "manifest": {"name": "mcp-test"},
            "analysis_completeness": {
                "is_complete": True,
                "status": "complete",
                "execution_successful": True,
                "entirely_uninspected_files": 0,
            },
            "execution_successful": True,
        }
    )


async def test_unavailable_requested_llm_aligns_embedded_json_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The embedded JSON report reflects requested-but-unavailable LLM analysis."""
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "not configured"))
    monkeypatch.setattr(
        "skillspector.nodes.report.is_llm_available",
        lambda: (False, "not configured"),
    )
    monkeypatch.setattr(mcp_server.graph, "ainvoke", _render_complete_zero_risk_result)

    verdict = await run_scan("fixture", use_llm=True, output_format="json")
    payload = json.loads(verdict["report"])

    assert verdict["risk_score"] == payload["risk_assessment"]["score"] == 0
    assert verdict["recommendation"] == payload["risk_assessment"]["recommendation"] == "CAUTION"
    assert payload["metadata"]["llm_requested"] is True
    assert payload["metadata"]["llm_available"] is False
    assert payload["metadata"]["meta_analysis_applied"] is False
    assert payload["metadata"]["filtering_mode"] == "heuristic"


async def test_empty_runtime_telemetry_aligns_mcp_and_embedded_json_caution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful preflight cannot leave the embedded report fail-open."""
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    monkeypatch.setattr(
        "skillspector.nodes.report.is_llm_available",
        lambda: (True, None),
    )
    monkeypatch.setattr(mcp_server.graph, "ainvoke", _render_complete_zero_risk_result)

    verdict = await run_scan("fixture", use_llm=True, output_format="json")
    payload = json.loads(verdict["report"])

    assert verdict["recommendation"] == "CAUTION"
    assert verdict["safe_to_install"] is False
    assert payload["risk_assessment"]["recommendation"] == "CAUTION"
    assert payload["metadata"]["llm_degraded"] is True
    assert "runtime telemetry was incomplete" in payload["metadata"]["llm_error"]


async def test_malformed_runtime_telemetry_preserves_failed_meta_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed siblings cannot erase valid failed meta-analysis evidence."""
    llm_call_log: list[object] = [
        {"node": "meta_analyzer", "ok": False, "error": "runtime failure"},
        "malformed-record",
    ]

    async def render_mixed_telemetry(
        state: dict[str, object], config: dict[str, object]
    ) -> dict[str, object]:
        del config
        completeness = {
            "is_complete": True,
            "status": "complete",
            "execution_successful": True,
            "entirely_uninspected_files": 0,
        }
        return {
            **report(
                {
                    **state,
                    "filtered_findings": [],
                    "component_metadata": [],
                    "has_executable_scripts": False,
                    "manifest": {"name": "mcp-test"},
                    "llm_call_log": llm_call_log,  # type: ignore[typeddict-item]
                    "analysis_completeness": completeness,
                    "execution_successful": True,
                }
            ),
            "llm_call_log": llm_call_log,
            "analysis_completeness": completeness,
        }

    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    monkeypatch.setattr(
        "skillspector.nodes.report.is_llm_available",
        lambda: (True, None),
    )
    monkeypatch.setattr(mcp_server.graph, "ainvoke", render_mixed_telemetry)

    verdict = await run_scan("fixture", use_llm=True, output_format="json")
    payload = json.loads(verdict["report"])

    assert verdict["llm_available"] is False
    assert payload["metadata"]["llm_available"] is False
    assert verdict["recommendation"] == "CAUTION"
    assert payload["risk_assessment"]["recommendation"] == "CAUTION"
    assert verdict["safe_to_install"] is False


async def test_truthy_malformed_ok_is_not_counted_as_runtime_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the literal boolean True can prove a successful LLM call."""
    llm_call_log = [
        {"node": "meta_analyzer", "ok": "false", "error": None},
    ]

    async def render_truthy_malformed_telemetry(
        state: dict[str, object], config: dict[str, object]
    ) -> dict[str, object]:
        del config
        completeness = {
            "is_complete": True,
            "status": "complete",
            "execution_successful": True,
            "entirely_uninspected_files": 0,
        }
        return {
            **report(
                {
                    **state,
                    "filtered_findings": [],
                    "component_metadata": [],
                    "has_executable_scripts": False,
                    "manifest": {"name": "mcp-test"},
                    "llm_call_log": llm_call_log,  # type: ignore[typeddict-item]
                    "analysis_completeness": completeness,
                    "execution_successful": True,
                }
            ),
            "llm_call_log": llm_call_log,
            "analysis_completeness": completeness,
        }

    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    monkeypatch.setattr(
        "skillspector.nodes.report.is_llm_available",
        lambda: (True, None),
    )
    monkeypatch.setattr(mcp_server.graph, "ainvoke", render_truthy_malformed_telemetry)

    verdict = await run_scan("fixture", use_llm=True, output_format="json")
    payload = json.loads(verdict["report"])
    metadata = payload["metadata"]

    assert verdict["recommendation"] == "CAUTION"
    assert verdict["safe_to_install"] is False
    assert verdict["llm_available"] is False
    assert payload["risk_assessment"]["recommendation"] == "CAUTION"
    assert metadata["llm_available"] is False
    assert metadata["meta_analysis_applied"] is False
    assert metadata["llm_calls_succeeded"] == 0
    assert metadata["llm_degraded"] is True


async def test_failed_meta_analysis_aligns_mcp_and_embedded_json_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP and its embedded report expose the same runtime availability."""
    llm_call_log = [
        {"node": "meta_analyzer", "ok": False, "error": "runtime failure"},
    ]

    async def render_failed_meta_analysis(
        state: dict[str, object], config: dict[str, object]
    ) -> dict[str, object]:
        del config
        completeness = {
            "is_complete": True,
            "status": "complete",
            "execution_successful": True,
            "entirely_uninspected_files": 0,
        }
        return {
            **report(
                {
                    **state,
                    "filtered_findings": [],
                    "component_metadata": [],
                    "has_executable_scripts": False,
                    "manifest": {"name": "mcp-test"},
                    "llm_call_log": llm_call_log,
                    "analysis_completeness": completeness,
                    "execution_successful": True,
                }
            ),
            "llm_call_log": llm_call_log,
            "analysis_completeness": completeness,
        }

    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    monkeypatch.setattr(
        "skillspector.nodes.report.is_llm_available",
        lambda: (True, None),
    )
    monkeypatch.setattr(mcp_server.graph, "ainvoke", render_failed_meta_analysis)

    verdict = await run_scan("fixture", use_llm=True, output_format="json")
    payload = json.loads(verdict["report"])

    assert verdict["llm_available"] is False
    assert verdict["llm_available"] == payload["metadata"]["llm_available"]
    assert verdict["llm_used"] is False
    assert verdict["safe_to_install"] is False
    assert verdict["recommendation"] == "CAUTION"


async def test_explicit_static_only_keeps_embedded_json_report_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit static-only mode keeps its existing SAFE report and request metadata."""
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "not configured"))
    monkeypatch.setattr(
        "skillspector.nodes.report.is_llm_available",
        lambda: (False, "not configured"),
    )
    monkeypatch.setattr(mcp_server.graph, "ainvoke", _render_complete_zero_risk_result)

    verdict = await run_scan("fixture", use_llm=False, output_format="json")
    payload = json.loads(verdict["report"])

    assert verdict["risk_score"] == payload["risk_assessment"]["score"] == 0
    assert verdict["recommendation"] == payload["risk_assessment"]["recommendation"] == "SAFE"
    assert payload["metadata"]["llm_requested"] is False
    assert payload["metadata"]["meta_analysis_applied"] is False


async def test_run_scan_reports_llm_available_with_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credentials present but use_llm=False: available, but honestly not used."""
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    _write_skill(tmp_path)

    result = await run_scan(str(tmp_path), use_llm=False, output_format="json")

    assert result["llm_available"] is True
    assert result["llm_requested"] is False
    assert result["llm_used"] is False
    assert result["scan_mode"] == "static-only"


async def test_run_scan_reports_missing_telemetry_for_bound_provider_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound provider alone cannot make an empty semantic pass look used."""

    class _InjectedProvider:
        DEFAULT_MODEL = "injected-default"
        SLOT_DEFAULTS = {"meta_analyzer": "injected-default"}

        def get_context_length(self, model: str) -> int | None:
            return 4096 if model == "injected-default" else None

        def get_max_output_tokens(self, model: str) -> int | None:
            return 128 if model == "injected-default" else None

        def resolve_model(self, slot: str = "default") -> str:
            return "injected-default"

        def resolve_credentials(self) -> tuple[str, str | None] | None:
            return None

        def create_chat_model(
            self,
            model: str,
            *,
            max_tokens: int,
            timeout: float | None = 120,
        ) -> object:
            return object()

    class _Graph:
        async def ainvoke(self, state, config):
            assert state["use_llm"] is True
            return {
                "filtered_findings": [],
                "risk_score": 0,
                "risk_severity": "LOW",
                "risk_recommendation": "OK",
                "report_body": "report",
                "llm_call_log": [],
            }

    token = use_provider(_InjectedProvider())
    monkeypatch.setattr(mcp_server, "graph", _Graph())
    _write_skill(tmp_path)

    try:
        result = await run_scan(str(tmp_path), use_llm=True, output_format="json")
    finally:
        reset_provider(token)

    assert result["llm_available"] is True
    assert result["llm_requested"] is True
    assert result["llm_used"] is False
    assert result["scan_mode"] == "static-only"
    assert result["safe_to_install"] is False


async def test_run_scan_disables_llm_for_unavailable_bound_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound provider that cannot build a chat model must stay static-only."""

    class _UnavailableInjectedProvider:
        DEFAULT_MODEL = "injected-default"
        SLOT_DEFAULTS = {"meta_analyzer": "injected-default"}

        def get_context_length(self, model: str) -> int | None:
            return 4096 if model == "injected-default" else None

        def get_max_output_tokens(self, model: str) -> int | None:
            return 128 if model == "injected-default" else None

        def resolve_model(self, slot: str = "default") -> str:
            return "injected-default"

        def resolve_credentials(self) -> tuple[str, str | None] | None:
            return None

        def create_chat_model(
            self,
            model: str,
            *,
            max_tokens: int,
            timeout: float | None = 120,
        ) -> object | None:
            return None

    class _Graph:
        async def ainvoke(self, state, config):
            assert state["use_llm"] is False
            return {
                "filtered_findings": [],
                "risk_score": 0,
                "risk_severity": "LOW",
                "risk_recommendation": "OK",
                "report_body": "report",
            }

    token = use_provider(_UnavailableInjectedProvider())
    monkeypatch.setattr(mcp_server, "graph", _Graph())
    _write_skill(tmp_path)

    try:
        result = await run_scan(str(tmp_path), use_llm=True, output_format="json")
    finally:
        reset_provider(token)

    assert result["llm_available"] is False
    assert result["llm_requested"] is True
    assert result["llm_used"] is False
    assert result["scan_mode"] == "static-only"


async def test_run_scan_rejects_invalid_format(tmp_path: Path) -> None:
    """An unsupported output_format is rejected before any scan runs."""
    with pytest.raises(ValueError):
        await run_scan(str(tmp_path), output_format="xml")


async def test_mcp_blocks_install_when_execution_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A low risk score cannot override failed inspection execution."""

    async def failed_execution_result(state: dict, config: dict) -> dict:
        return {
            "risk_score": 0,
            "risk_severity": "LOW",
            "risk_recommendation": "CAUTION",
            "execution_successful": False,
            "analysis_completeness": {
                "entirely_uninspected_files": 1,
                "ledger_exceptions": [],
            },
            "filtered_findings": [],
            "report_body": "{}",
        }

    monkeypatch.setattr(mcp_server.graph, "ainvoke", failed_execution_result)
    verdict = await mcp_server.run_scan("fixture", use_llm=False)

    assert verdict["safe_to_install"] is False
    assert verdict["execution_successful"] is False


async def test_mcp_blocks_install_when_analysis_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful, low-risk but partial scan is never safe to install."""

    async def incomplete_result(state: dict, config: dict) -> dict:
        return {
            "risk_score": 0,
            "risk_severity": "LOW",
            "risk_recommendation": "CAUTION",
            "execution_successful": True,
            "analysis_completeness": {
                "is_complete": False,
                "status": "partial",
                "entirely_uninspected_files": 0,
                "ledger_exceptions": [],
            },
            "filtered_findings": [],
            "report_body": "{}",
        }

    monkeypatch.setattr(mcp_server.graph, "ainvoke", incomplete_result)
    verdict = await mcp_server.run_scan("fixture", use_llm=False)

    assert verdict["safe_to_install"] is False
    assert verdict["execution_successful"] is True
    assert verdict["analysis_completeness"]["status"] == "partial"


async def test_run_scan_rejects_local_target_when_disallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP-style scans reject local targets before the graph is invoked."""
    graph_ainvoke = AsyncMock()
    monkeypatch.setattr(mcp_server.graph, "ainvoke", graph_ainvoke)
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))

    with pytest.raises(ValueError, match="local targets are disabled"):
        await run_scan(str(tmp_path), allow_local_targets=False)

    assert graph_ainvoke.await_count == 0


async def test_run_scan_rejects_file_url_when_local_targets_disallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same HTTP guard rejects file:// targets before any scan runs."""
    graph_ainvoke = AsyncMock()
    monkeypatch.setattr(mcp_server.graph, "ainvoke", graph_ainvoke)
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))

    with pytest.raises(ValueError, match="local targets are disabled"):
        await run_scan(tmp_path.as_uri(), allow_local_targets=False)

    assert graph_ainvoke.await_count == 0


async def test_run_scan_rejects_local_yara_rules_when_targets_are_disallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTTP policy covers local YARA configuration as well as scan targets."""
    graph_ainvoke = AsyncMock()
    monkeypatch.setattr(mcp_server.graph, "ainvoke", graph_ainvoke)
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))

    with pytest.raises(ValueError, match="local targets are disabled"):
        await run_scan(
            "https://example.com/skills/safe.git",
            allow_local_targets=False,
            yara_rules_dir=str(tmp_path),
        )

    assert graph_ainvoke.await_count == 0


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (r"\\server\share\skill", True),
        ("//server/share/skill", True),
        ("git@github.com:NVIDIA/SkillSpector.git", False),
        ("ssh://git@github.com/NVIDIA/SkillSpector.git", False),
        ("git+ssh://git@github.com/NVIDIA/SkillSpector.git", False),
        ("custom://example/skill", False),
    ],
)
def test_is_local_target_classifies_protocol_edges(target: str, expected: bool) -> None:
    """Classifier treats UNC-style paths as local and known remote schemes as remote."""
    assert mcp_server._is_local_target(target) is expected


def test_is_local_target_checks_relative_paths_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing relative paths are local; missing relative paths stay unresolved."""
    (tmp_path / "skill").mkdir()
    monkeypatch.chdir(tmp_path)

    assert mcp_server._is_local_target("skill") is True
    assert mcp_server._is_local_target("missing-skill") is False


def test_is_local_target_fails_closed_when_home_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unresolvable tilde paths remain local instead of leaking a runtime error."""

    def fail_to_expanduser(self: Path) -> Path:
        raise RuntimeError("Could not determine home directory")

    monkeypatch.setattr(Path, "expanduser", fail_to_expanduser)

    assert mcp_server._is_local_target("~nosuchuser/skill") is True


async def test_run_scan_allows_remote_target_when_local_targets_disallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote HTTP targets still reach the resolver path when local targets are blocked."""
    graph_ainvoke = AsyncMock(
        return_value={
            "risk_score": 0,
            "risk_severity": "low",
            "risk_recommendation": "safe",
            "filtered_findings": [],
            "report_body": "ok",
        }
    )
    monkeypatch.setattr(mcp_server.graph, "ainvoke", graph_ainvoke)
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))

    target = "https://example.com/skills/safe.git"
    result = await run_scan(target, allow_local_targets=False)

    assert result["target"] == target
    assert graph_ainvoke.await_count == 1
    assert graph_ainvoke.await_args.args[0]["input_path"] == target


async def test_run_scan_keeps_default_local_target_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default run_scan path still accepts local targets."""
    graph_ainvoke = AsyncMock(
        return_value={
            "risk_score": 0,
            "risk_severity": "low",
            "risk_recommendation": "safe",
            "filtered_findings": [],
            "report_body": "ok",
        }
    )
    monkeypatch.setattr(mcp_server.graph, "ainvoke", graph_ainvoke)
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))

    result = await run_scan(str(tmp_path))

    assert result["target"] == str(tmp_path)
    assert graph_ainvoke.await_count == 1
    assert graph_ainvoke.await_args.args[0]["input_path"] == str(tmp_path)


@pytest.mark.parametrize(
    ("transport", "expected_allow_local_targets"),
    [("stdio", True), ("http", False)],
)
def test_run_passes_transport_local_target_policy(
    transport: str,
    expected_allow_local_targets: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run() keeps stdio local scans available and disables them for HTTP."""
    captured: dict[str, bool] = {}
    server = SimpleNamespace(
        settings=SimpleNamespace(host=None, port=None),
        run=MagicMock(),
    )

    def fake_build_server(*, allow_local_targets: bool = False):
        captured["allow_local_targets"] = allow_local_targets
        return server

    monkeypatch.setattr(mcp_server, "build_server", fake_build_server)

    mcp_server.run(transport=transport, host="0.0.0.0", port=9000)

    assert captured["allow_local_targets"] is expected_allow_local_targets
    if transport == "http":
        assert server.settings.host == "0.0.0.0"
        assert server.settings.port == 9000
        server.run.assert_called_once_with(transport="streamable-http")
    else:
        server.run.assert_called_once_with(transport="stdio")


def test_run_rejects_unknown_transport_without_allowing_local_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown transports fail closed before a server can start."""
    captured: dict[str, bool] = {}
    server = SimpleNamespace(
        settings=SimpleNamespace(host=None, port=None),
        run=MagicMock(),
    )

    def fake_build_server(*, allow_local_targets: bool = False):
        captured["allow_local_targets"] = allow_local_targets
        return server

    monkeypatch.setattr(mcp_server, "build_server", fake_build_server)

    with pytest.raises(ValueError, match="transport must be"):
        mcp_server.run(transport="sse")

    assert captured["allow_local_targets"] is False
    server.run.assert_not_called()


async def test_build_server_registers_scan_skill() -> None:
    """build_server wires up the scan_skill tool (requires the mcp extra)."""
    pytest.importorskip("mcp")

    server = mcp_server.build_server()
    tools = await server.list_tools()
    assert "scan_skill" in {tool.name for tool in tools}


async def test_build_server_disables_local_targets_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct server construction remains fail-closed before transport selection."""
    pytest.importorskip("mcp")
    from mcp.server.fastmcp.exceptions import ToolError

    graph_ainvoke = AsyncMock()
    monkeypatch.setattr(mcp_server.graph, "ainvoke", graph_ainvoke)
    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))

    server = mcp_server.build_server()

    with pytest.raises(ToolError, match="local targets are disabled"):
        await server.call_tool("scan_skill", {"target": str(tmp_path)})

    assert graph_ainvoke.await_count == 0


def test_build_server_reports_incompatible_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """An installed package without FastMCP must not be reported as missing."""
    import builtins

    original_import = builtins.__import__

    def import_without_fastmcp(name: str, *args: object, **kwargs: object) -> object:
        if name == "mcp.server.fastmcp":
            raise ModuleNotFoundError("No module named 'mcp.server.fastmcp'", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_fastmcp)

    with pytest.raises(ModuleNotFoundError, match="installed 'mcp' package is incompatible"):
        mcp_server.build_server()


async def test_mcp_stdio_initialize_registers_scan_skill() -> None:
    """The real stdio CLI must initialize and expose the scan_skill tool."""
    pytest.importorskip("mcp")

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    repo_root = Path(__file__).resolve().parents[2]
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "skillspector.cli", "mcp"],
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=15)
            tools = await asyncio.wait_for(session.list_tools(), timeout=15)

    assert "scan_skill" in {tool.name for tool in tools.tools}


async def test_run_scan_findings_exclude_the_suppressed_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MCP verdict lists the findings that drove the score, not kept+suppressed.

    `run_scan` serialises this list straight to the calling agent, so a
    baseline-suppressed finding leaking in tells the agent a skill is dirtier
    than the risk score it is gating on.
    """
    kept = Finding(rule_id="SQP-1", message="kept")
    dropped = Finding(rule_id="SQP-2", message="suppressed")
    result = {
        "findings": [kept, dropped],
        "filtered_findings": [kept, dropped],
        "suppressed_findings": [SuppressedFinding(finding=dropped, reason="baselined")],
        "risk_score": 10,
        "risk_severity": "LOW",
        "report_body": "# report",
    }
    monkeypatch.setattr(mcp_server.graph, "ainvoke", AsyncMock(return_value=result))

    verdict = await run_scan(str(_write_skill(tmp_path)), use_llm=False, output_format="json")

    assert [finding["id"] for finding in verdict["findings"]] == ["SQP-1"]


async def test_run_scan_respects_an_empty_filtered_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every-finding-filtered reports no findings, not the raw pre-filter list."""
    result = {
        "findings": [Finding(rule_id="SQP-1", message="one")],
        "filtered_findings": [],
        "suppressed_findings": [],
        "risk_score": 0,
        "risk_severity": "LOW",
        "report_body": "# report",
    }
    monkeypatch.setattr(mcp_server.graph, "ainvoke", AsyncMock(return_value=result))

    verdict = await run_scan(str(_write_skill(tmp_path)), use_llm=False, output_format="json")

    assert verdict["findings"] == []
