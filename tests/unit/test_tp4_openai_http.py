# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real SDK HTTP transport, parser, retry, graph and public-report contracts."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from skillspector import llm_analyzer_base
from skillspector.inspection_ledger import LedgerReason
from skillspector.nodes.analyzers import mcp_tool_poisoning as tp
from skillspector.providers import reset_provider, use_provider
from skillspector.providers.openai import OpenAIProvider
from skillspector.sarif_models import validate_sarif_report

_MODEL = "azure/anthropic/claude-opus-5"


@pytest.fixture
def openai_http_endpoint(monkeypatch):
    requests = []
    behavior = {"mode": "clean", "tp4_calls": 0, "tool_name": "_TP4AnalysisResult"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            tools = body.get("tools", [])
            tool_name = behavior["tool_name"]
            is_tp4 = bool(tools and tools[0]["function"]["name"] == tool_name)
            if not is_tp4:
                response_format = body.get("response_format", {})
                is_tp4 = response_format.get("json_schema", {}).get("name") == tool_name
            message = {"role": "assistant", "content": json.dumps({"findings": []})}
            finish_reason = "stop"
            if is_tp4:
                behavior["tp4_calls"] += 1
                mode = behavior["mode"]
                if mode in {"recover", "recover-multiple"} and behavior["tp4_calls"] > 1:
                    mode = "clean"
                if mode == "recover-multiple":
                    mode = "multiple-mismatch"
                if mode in {"refusal", "recover"}:
                    message = {"role": "assistant", "content": "Cannot provide this assessment."}
                else:
                    arguments = {
                        "is_mismatch": mode == "mismatch",
                        "confidence": 7.0 if mode == "out-of-range" else 0.95,
                        "declared_purpose_summary": "Format values",
                        "actual_behavior_summary": "Changes persistent state",
                        "mismatched_capabilities": ["persistence"] if mode == "mismatch" else [],
                        "explanation": "The returned assessment is intentionally controlled.",
                    }
                    encoded = "not valid JSON" if mode == "malformed" else json.dumps(arguments)
                    if tools:
                        tool_calls = [
                            {
                                "id": "call_tp4",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": encoded,
                                },
                            }
                        ]
                        if mode.startswith("multiple-"):
                            second_arguments = dict(arguments)
                            if mode == "multiple-mismatch":
                                second_arguments.update(
                                    is_mismatch=True, mismatched_capabilities=["persistence"]
                                )
                            tool_calls.append(
                                {
                                    "id": "call_tp4_second",
                                    "type": "function",
                                    "function": {
                                        "name": (
                                            "UnexpectedAssessment"
                                            if mode == "multiple-unknown"
                                            else tool_name
                                        ),
                                        "arguments": (
                                            "not valid JSON"
                                            if mode == "multiple-malformed"
                                            else json.dumps(second_arguments)
                                        ),
                                    },
                                }
                            )
                        message = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        }
                        finish_reason = "tool_calls"
                    else:
                        message = {"role": "assistant", "content": encoded}
            payload = {
                "id": "chatcmpl-loopback",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            }
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-loopback-test")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("SKILLSPECTOR_MODEL", _MODEL)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("SKILLSPECTOR_STRUCTURED_OUTPUT_METHOD", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_REASONING_EFFORT", raising=False)
    token = use_provider(OpenAIProvider())
    try:
        yield requests, behavior
    finally:
        reset_provider(token)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "mode,attempts,complete",
    [
        ("clean", 1, True),
        ("mismatch", 1, True),
        ("recover", 2, True),
        ("refusal", 4, False),
        ("malformed", 4, False),
        ("out-of-range", 4, False),
        ("multiple-clean", 4, False),
        ("multiple-mismatch", 4, False),
        ("multiple-unknown", 4, False),
        ("multiple-malformed", 4, False),
        ("recover-multiple", 2, True),
    ],
)
@pytest.mark.parametrize("refresh", [False, True])
def test_tp4_real_http_retries_and_deadline_refresh(
    monkeypatch, openai_http_endpoint, mode, attempts, complete, refresh
):
    requests, behavior = openai_http_endpoint
    behavior["mode"] = mode
    if refresh:
        monkeypatch.setattr(llm_analyzer_base, "_retarget_request_timeout", lambda *_args: False)
    analyzer = tp._TP4Analyzer(model=_MODEL, timeout=(lambda: 30.0) if refresh else 30.0)
    outcome = analyzer.run_batches_detailed(
        [tp.Batch(file_path="format.py", content="Assess the declared formatting behavior.")]
    )
    assert behavior["tp4_calls"] == attempts
    assert bool(outcome.successful) is complete
    assert bool(outcome.failures) is not complete
    for request in requests:
        assert request["tool_choice"] == {
            "type": "function",
            "function": {"name": "_TP4AnalysisResult"},
        }
        assert request["parallel_tool_calls"] is False
        assert "response_format" not in request
        assert "Report your result by calling" not in request["messages"][0]["content"]
    if not complete:
        assert outcome.failures[0].reason is LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID
    assert analyzer.response_received
    assert sum(record["total_tokens"] for record in analyzer.inference_usage) == 20 * attempts


def test_reasoning_effort_uses_existing_json_schema_route_over_real_http(
    monkeypatch, openai_http_endpoint
):
    requests, _behavior = openai_http_endpoint
    monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", "high")
    monkeypatch.setattr(llm_analyzer_base, "_retarget_request_timeout", lambda *_args: False)
    analyzer = tp._TP4Analyzer(model=_MODEL, timeout=lambda: 30.0)
    outcome = analyzer.run_batches_detailed([tp.Batch(file_path="format.py", content="Assess")])
    assert outcome.successful and not outcome.failures
    assert requests[0]["reasoning_effort"] == "high"
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert "tool_choice" not in requests[0]


def test_real_http_validation_uses_schema_title_as_tool_name(openai_http_endpoint):
    requests, behavior = openai_http_endpoint
    behavior["tool_name"] = "GatewayTP4Assessment"

    class TitledTP4Assessment(tp._TP4AnalysisResult):
        model_config = {"title": "GatewayTP4Assessment"}

    class TitledTP4Analyzer(tp._TP4Analyzer):
        response_schema = TitledTP4Assessment

    analyzer = TitledTP4Analyzer(model=_MODEL, timeout=30.0)
    outcome = analyzer.run_batches_detailed([tp.Batch(file_path="format.py", content="Assess")])

    assert behavior["tp4_calls"] == 1
    assert outcome.successful and not outcome.failures
    assert isinstance(outcome.successful[0][1][0], TitledTP4Assessment)
    assert requests[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "GatewayTP4Assessment"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,attempts,complete",
    [
        ("clean", 1, True),
        ("mismatch", 1, True),
        ("refusal", 4, False),
        ("malformed", 4, False),
        ("multiple-clean", 4, False),
        ("multiple-mismatch", 4, False),
        ("multiple-unknown", 4, False),
        ("multiple-malformed", 4, False),
        ("recover-multiple", 2, True),
    ],
)
async def test_async_real_http_tool_results_use_same_validation(
    openai_http_endpoint, mode, attempts, complete
):
    requests, behavior = openai_http_endpoint
    behavior["mode"] = mode
    analyzer = tp._TP4Analyzer(model=_MODEL, timeout=30.0)
    outcome = await analyzer.arun_batches_detailed(
        [tp.Batch(file_path="format.py", content="Assess the declared formatting behavior.")]
    )
    assert behavior["tp4_calls"] == attempts
    assert bool(outcome.successful) is complete
    assert bool(outcome.failures) is not complete
    if not complete:
        assert outcome.failures[0].reason is LedgerReason.LLM_STRUCTURED_RESPONSE_INVALID
    assert analyzer.response_received
    assert all("response_format" not in request for request in requests)


@pytest.mark.parametrize("output_format", ["json", "sarif"])
@pytest.mark.parametrize(
    "mode,attempts,complete",
    [
        ("clean", 1, True),
        ("mismatch", 1, True),
        ("refusal", 4, False),
        ("malformed", 4, False),
        ("multiple-clean", 4, False),
        ("multiple-mismatch", 4, False),
        ("multiple-unknown", 4, False),
        ("multiple-malformed", 4, False),
        ("recover-multiple", 2, True),
    ],
)
def test_full_scanner_graph_reports_tp4_http_results(
    tmp_path: Path, openai_http_endpoint, output_format, mode, attempts, complete
):
    from skillspector.graph import create_graph

    requests, behavior = openai_http_endpoint
    behavior["mode"] = mode
    (tmp_path / "SKILL.md").write_text(
        "---\nname: formatting-helper\ndescription: Format supplied values.\n---\n"
        "# Formatting Helper\n",
        encoding="utf-8",
    )
    (tmp_path / "format.py").write_text(
        "def format_value(value):\n    return str(value)\n", encoding="utf-8"
    )
    result = create_graph().invoke(
        {
            "input_path": str(tmp_path),
            "output_format": output_format,
            "use_llm": True,
            "llm_requested": True,
            "model_config": {"default": _MODEL},
        }
    )
    tp4_status = next(
        event
        for event in result["analyzer_status_events"]
        if event["analyzer_id"] == tp.ANALYZER_ID
    )
    assert tp4_status["status"] == ("completed" if complete else "degraded")
    assert behavior["tp4_calls"] == attempts
    findings = [f for f in result["findings"] if f.rule_id == "TP4"]
    assert bool(findings) is (mode == "mismatch")
    if findings:
        assert findings[0].file == "SKILL.md"
        assert findings[0].evidence["code_path"] == "format.py"
        assert findings[0].evidence["code_start_line"] == 1
        assert findings[0].evidence["code_end_line"] == 2
        assert any(
            findings[0].finding_id in event.get("emitted_finding_ids", [])
            for event in result["inspection_ledger"]
            if event["analyzer_id"] == tp.ANALYZER_ID and event["path"] == "format.py"
        )
    report = json.loads(result["report_body"])
    if output_format == "json":
        assert report["execution_successful"] is True
        assert report["analysis_completeness"]["is_complete"] is complete
        assert bool([issue for issue in report["issues"] if issue["id"] == "TP4"]) is (
            mode == "mismatch"
        )
    else:
        validate_sarif_report(report)
        run = report["runs"][0]
        assert run["invocations"][0]["properties"]["analysisCompleteness"]["isComplete"] is complete
        assert bool([item for item in run["results"] if item["ruleId"] == "TP4"]) is (
            mode == "mismatch"
        )
    tp4_requests = [
        request
        for request in requests
        if request.get("tools") and request["tools"][0]["function"]["name"] == "_TP4AnalysisResult"
    ]
    assert len(tp4_requests) == attempts
    assert all(request["model"] == _MODEL for request in tp4_requests)
    assert all(
        request["tool_choice"] == {"type": "function", "function": {"name": "_TP4AnalysisResult"}}
        for request in tp4_requests
    )
