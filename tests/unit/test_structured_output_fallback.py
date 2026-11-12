# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured-output rejection fallback for DeepSeek-style gateways (issue #1089).

A gateway without JSON-schema ``response_format`` answers every analyzer
request with HTTP 400. The analyzers must retry the batch with an unforced
tool call instead of failing the whole LLM stage, and must fail loudly when
the gateway rejects that shape too.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from openai import BadRequestError, RateLimitError
from pydantic import BaseModel

from skillspector.inspection_ledger import LedgerReason
from skillspector.llm_analyzer_base import (
    BatchExecutionResult,
    BatchFailure,
    llm_call_record_for_outcome,
)
from skillspector.llm_utils import (
    bind_structured_output_fallback,
    is_structured_output_rejection,
)
from skillspector.nodes.analyzers import mcp_tool_poisoning as tp
from skillspector.providers import reset_provider, use_provider
from skillspector.providers.openai_compatible import OpenAICompatibleProvider
from skillspector.semantic_runtime import (
    STRUCTURED_OUTPUT_UNAVAILABLE_ERROR,
    all_llm_calls_failed_with_structured_output_rejection,
)

_MODEL = "deepseek-flash"
_TP4_TOOL = "_TP4AnalysisResult"


class _Result(BaseModel):
    answer: str
    score: int


def _provider_error(status: int, message: str) -> BadRequestError | RateLimitError:
    body = {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": "invalid_request_error",
        }
    }
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "https://gateway.example/v1/chat/completions"),
        json=body,
    )
    if status == 429:
        return RateLimitError(message, response=response, body=body)
    return BadRequestError(message, response=response, body=body)


@pytest.mark.parametrize(
    "error,expected",
    [
        (
            _provider_error(400, "This response_format type is unavailable now"),
            True,
        ),
        (
            _provider_error(400, "tool_choice with a specific function is not supported"),
            True,
        ),
        (_provider_error(400, "The model `deepseek-flash` does not exist"), False),
        (_provider_error(400, "Input tokens exceed the configured limit"), False),
        (_provider_error(429, "Rate limit exceeded for this model"), False),
        (ValueError("boom"), False),
    ],
)
def test_is_structured_output_rejection(error, expected):
    assert is_structured_output_rejection(error) is expected


def test_is_structured_output_rejection_follows_wrapped_cause():
    inner = _provider_error(400, "This response_format type is unavailable now")
    wrapped = RuntimeError("invoke failed")
    wrapped.__cause__ = inner
    assert is_structured_output_rejection(wrapped) is True


def test_fallback_binding_returns_none_for_non_openai_models():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    fake = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
    assert bind_structured_output_fallback(fake, _Result, "fake-model") is None


@pytest.fixture
def deepseek_like_endpoint(monkeypatch):
    requests: list[dict] = []
    behavior = {"mode": "fallback"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            fmt = body.get("response_format") or {}
            tools = body.get("tools") or []
            choice = body.get("tool_choice")
            mode = behavior["mode"]
            if fmt.get("type") == "json_schema":
                if behavior.get("expire_deadline_on_rejection"):
                    behavior["remaining"] = 0.0
                self._send_error(400, "This response_format type is unavailable now")
                return
            if isinstance(choice, dict):
                # DeepSeek accepts tools with tool_choice auto, not a forced
                # named function call.
                self._send_error(400, "tool_choice with a specific function is not supported")
                return
            if mode == "reject-all":
                self._send_error(400, "This response_format type is unavailable now")
                return
            tool_name = tools[0]["function"]["name"] if tools else None
            if tool_name == _TP4_TOOL:
                arguments = {
                    "is_mismatch": False,
                    "confidence": 0.95,
                    "declared_purpose_summary": "Format values",
                    "actual_behavior_summary": "Changes persistent state",
                    "mismatched_capabilities": [],
                    "explanation": "The returned assessment is intentionally controlled.",
                }
            else:
                arguments = {"findings": []}
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
            payload = {
                "id": "chatcmpl-loopback",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            }
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_error(self, status: int, message: str) -> None:
            payload = {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "invalid_request_error",
                }
            }
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SKILLSPECTOR_COMPAT_API_KEY", "sk-loopback-test")
    monkeypatch.setenv("SKILLSPECTOR_COMPAT_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("SKILLSPECTOR_MODEL", _MODEL)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("SKILLSPECTOR_STRUCTURED_OUTPUT_METHOD", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_REASONING_EFFORT", raising=False)
    token = use_provider(OpenAICompatibleProvider())
    try:
        yield requests, behavior
    finally:
        reset_provider(token)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_rejected_json_schema_retries_with_unforced_tool_call(deepseek_like_endpoint):
    requests, _behavior = deepseek_like_endpoint
    analyzer = tp._TP4Analyzer(model=_MODEL, timeout=30.0)
    outcome = analyzer.run_batches_detailed(
        [tp.Batch(file_path="format.py", content="Assess the declared formatting behavior.")]
    )
    assert outcome.successful and not outcome.failures
    assert len(requests) == 2
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert "response_format" not in requests[1]
    assert requests[1]["tools"][0]["function"]["name"] == _TP4_TOOL
    assert "tool_choice" not in requests[1]
    assert analyzer._use_structured_output_fallback is True


def test_second_batch_binds_fallback_directly(deepseek_like_endpoint):
    requests, _behavior = deepseek_like_endpoint
    analyzer = tp._TP4Analyzer(model=_MODEL, timeout=30.0)
    batch = tp.Batch(file_path="format.py", content="Assess the declared formatting behavior.")
    first = analyzer.run_batches_detailed([batch])
    second = analyzer.run_batches_detailed([batch])
    assert first.successful and second.successful
    assert len(requests) == 3
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert all("response_format" not in request for request in requests[1:])
    assert all("tool_choice" not in request for request in requests[1:])


@pytest.mark.asyncio
async def test_async_rejected_json_schema_retries_with_unforced_tool_call(
    deepseek_like_endpoint,
):
    requests, _behavior = deepseek_like_endpoint
    analyzer = tp._TP4Analyzer(model=_MODEL, timeout=30.0)
    outcome = await analyzer.arun_batches_detailed(
        [tp.Batch(file_path="format.py", content="Assess the declared formatting behavior.")]
    )
    assert outcome.successful and not outcome.failures
    assert len(requests) == 2
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert "response_format" not in requests[1]
    assert "tool_choice" not in requests[1]


def test_exhausted_fallback_records_distinct_reason(deepseek_like_endpoint):
    requests, behavior = deepseek_like_endpoint
    behavior["mode"] = "reject-all"
    analyzer = tp._TP4Analyzer(model=_MODEL, timeout=30.0)
    outcome = analyzer.run_batches_detailed(
        [tp.Batch(file_path="format.py", content="Assess the declared formatting behavior.")]
    )
    assert not outcome.successful
    assert len(outcome.failures) == 1
    assert outcome.failures[0].reason is LedgerReason.LLM_STRUCTURED_OUTPUT_REJECTED
    assert outcome.failures[0].error_class == "LLMStructuredOutputUnavailableError"
    assert len(requests) == 2
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert "response_format" not in requests[1]
    record = llm_call_record_for_outcome(tp.ANALYZER_ID, outcome)
    assert record["ok"] is False
    assert record["error"] == STRUCTURED_OUTPUT_UNAVAILABLE_ERROR
    assert all_llm_calls_failed_with_structured_output_rejection([record]) is True


def test_call_record_predicate_ignores_partial_and_other_failures():
    ok_record = {"node": "x", "ok": True, "error": None}
    rejected_record = {
        "node": "x",
        "ok": False,
        "error": STRUCTURED_OUTPUT_UNAVAILABLE_ERROR,
    }
    other_record = {"node": "x", "ok": False, "error": "boom"}
    assert all_llm_calls_failed_with_structured_output_rejection([]) is False
    assert all_llm_calls_failed_with_structured_output_rejection([rejected_record]) is True
    assert (
        all_llm_calls_failed_with_structured_output_rejection([rejected_record, ok_record]) is False
    )
    assert (
        all_llm_calls_failed_with_structured_output_rejection([rejected_record, other_record])
        is False
    )


def test_full_scanner_graph_reports_llm_unavailable_when_fallback_exhausted(
    tmp_path: Path, deepseek_like_endpoint
):
    from skillspector.graph import create_graph

    requests, behavior = deepseek_like_endpoint
    behavior["mode"] = "reject-all"
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
            "output_format": "json",
            "use_llm": True,
            "llm_requested": True,
            "model_config": {"default": _MODEL},
        }
    )
    assert requests, "expected the analyzers to call the stub gateway"
    assert all(
        "response_format" not in request or request["response_format"]["type"] == "json_schema"
        for request in requests
    )
    report = json.loads(result["report_body"])
    metadata = report["metadata"]
    assert metadata["llm_requested"] is True
    assert metadata["llm_available"] is False
    assert metadata["llm_calls_attempted"] > 0
    assert metadata["llm_calls_succeeded"] == 0
    assert metadata["llm_degraded"] is True
    assert "structured output rejected by provider" in metadata["llm_error"]
    assert report["risk_assessment"]["recommendation"] == "CAUTION"


@pytest.mark.asyncio
async def test_async_two_simultaneous_rejections_both_recover(deepseek_like_endpoint):
    """Two batches rejected at the same time must each retry with tool calling.

    The shared fallback flag only guides future bindings. A batch whose
    initial request is still in flight when a sibling batch flips the flag
    must not be classified as already having tried the fallback.
    """
    requests, _behavior = deepseek_like_endpoint
    analyzer = tp._TP4Analyzer(model=_MODEL, timeout=30.0)
    batches = [
        tp.Batch(file_path="one.py", content="Assess the declared formatting behavior."),
        tp.Batch(file_path="two.py", content="Assess the declared formatting behavior."),
    ]
    outcome = await analyzer.arun_batches_detailed(batches, max_concurrency=2)
    assert len(outcome.successful) == 2
    assert not outcome.failures
    assert len(requests) == 4
    initial = [request for request in requests if request.get("response_format")]
    assert len(initial) == 2
    assert all(request["response_format"]["type"] == "json_schema" for request in initial)
    retries = [request for request in requests if "response_format" not in request]
    assert len(retries) == 2
    assert all("tool_choice" not in request for request in retries)


def test_mixed_outcome_record_keeps_coverage_failure(deepseek_like_endpoint):
    """A successful or recovered batch keeps the unavailable error away."""
    _requests, _behavior = deepseek_like_endpoint
    ok_batch = tp.Batch(file_path="ok.py", content="Assess the declared formatting behavior.")
    bad_batch = tp.Batch(file_path="bad.py", content="Assess the declared formatting behavior.")
    rejection = BatchFailure(
        bad_batch,
        "LLMStructuredOutputUnavailableError",
        LedgerReason.LLM_STRUCTURED_OUTPUT_REJECTED,
    )
    mixed = BatchExecutionResult(
        successful=[(ok_batch, [])],
        failures=[rejection],
    )
    record = llm_call_record_for_outcome(tp.ANALYZER_ID, mixed)
    assert record["ok"] is False
    assert record["error"] is None
    recovered_outcome = BatchExecutionResult(
        recovered=[(ok_batch, [])],
        failures=[rejection],
    )
    record = llm_call_record_for_outcome(tp.ANALYZER_ID, recovered_outcome)
    assert record["ok"] is False
    assert record["error"] is None
    total = BatchExecutionResult(failures=[rejection])
    record = llm_call_record_for_outcome(tp.ANALYZER_ID, total)
    assert record["ok"] is False
    assert record["error"] == STRUCTURED_OUTPUT_UNAVAILABLE_ERROR


def _tp4_arguments() -> dict:
    return {
        "is_mismatch": False,
        "confidence": 0.95,
        "declared_purpose_summary": "Format values",
        "actual_behavior_summary": "Changes persistent state",
        "mismatched_capabilities": [],
        "explanation": "The returned assessment is intentionally controlled.",
    }


def _tp4_state() -> dict:
    return {
        "manifest": {"description": "Documentation only."},
        "file_cache": {
            "one.py": "def one():\n    return 1\n",
            "two.py": "def two():\n    return 2\n",
        },
        "component_metadata": [
            {"path": "one.py", "type": "python"},
            {"path": "two.py", "type": "python"},
        ],
        "model_config": {"default": "test-model"},
    }


def test_tp4_mixed_outcome_keeps_coverage_failure(
    deepseek_like_endpoint, monkeypatch: pytest.MonkeyPatch
):
    """TP4 reserves the unavailable error for total rejection, like the record helper."""
    _requests, _behavior = deepseek_like_endpoint

    async def fake_arun(self, batches, **kwargs):
        return BatchExecutionResult(
            successful=[(batches[0], [tp._TP4AnalysisResult(**_tp4_arguments())])],
            failures=[
                BatchFailure(
                    batches[1],
                    "LLMStructuredOutputUnavailableError",
                    LedgerReason.LLM_STRUCTURED_OUTPUT_REJECTED,
                )
            ],
        )

    monkeypatch.setattr(tp._TP4Analyzer, "arun_batches_detailed", fake_arun)
    result = tp._check_tp4(_tp4_state())
    assert result.record["ok"] is False
    assert result.record["error"] != STRUCTURED_OUTPUT_UNAVAILABLE_ERROR
    assert "TP4 LLM batch failed" in str(result.record["error"])


def test_tp4_total_rejection_keeps_unavailable_error(
    deepseek_like_endpoint, monkeypatch: pytest.MonkeyPatch
):
    """TP4 still reports structured output unavailable when every batch is rejected."""
    _requests, _behavior = deepseek_like_endpoint

    async def fake_arun(self, batches, **kwargs):
        return BatchExecutionResult(
            failures=[
                BatchFailure(
                    batch,
                    "LLMStructuredOutputUnavailableError",
                    LedgerReason.LLM_STRUCTURED_OUTPUT_REJECTED,
                )
                for batch in batches
            ]
        )

    monkeypatch.setattr(tp._TP4Analyzer, "arun_batches_detailed", fake_arun)
    result = tp._check_tp4(_tp4_state())
    assert result.record["ok"] is False
    assert result.record["error"] == STRUCTURED_OUTPUT_UNAVAILABLE_ERROR


def test_expired_deadline_during_first_rejection_sends_no_fallback(deepseek_like_endpoint):
    """A deadline that expires during the first rejection must stop the retry."""
    requests, behavior = deepseek_like_endpoint
    behavior["remaining"] = 30.0
    behavior["expire_deadline_on_rejection"] = True
    analyzer = tp._TP4Analyzer(model=_MODEL, timeout=lambda: behavior["remaining"])
    outcome = analyzer.run_batches_detailed(
        [tp.Batch(file_path="format.py", content="Assess the declared formatting behavior.")]
    )
    assert not outcome.successful
    assert len(outcome.failures) == 1
    assert outcome.failures[0].reason is LedgerReason.RUNTIME_LIMIT
    assert len(requests) == 1
    assert requests[0]["response_format"]["type"] == "json_schema"
