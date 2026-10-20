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

"""Tests for the MiniMax provider."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import skillspector.providers as providers_module
from skillspector.inference_usage import provider_name
from skillspector.llm_provenance import capture_llm_provenance, sanitize_llm_provenance
from skillspector.llm_utils import bind_structured_output, get_chat_model
from skillspector.providers import get_metadata_provider, registry, resolve_provider_credentials
from skillspector.providers._agent_cli import _scrub_env
from skillspector.providers.chat_models import UnsupportedControlError
from skillspector.providers.minimax import (
    MINIMAX_CN_BASE_URL,
    MINIMAX_GLOBAL_BASE_URL,
    REGISTRY_PATH,
    MiniMaxProvider,
)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch):
    for key in (
        "MINIMAX_API_KEY",
        "MINIMAX_BASE_URL",
        "MINIMAX_REGION",
        "SKILLSPECTOR_MODEL",
        "SKILLSPECTOR_MODEL_REGISTRY",
        "SKILLSPECTOR_PROVIDER",
        "SKILLSPECTOR_REASONING_EFFORT",
        "SKILLSPECTOR_TEMPERATURE",
        "SKILLSPECTOR_SEED",
        "SKILLSPECTOR_STRUCTURED_OUTPUT_METHOD",
    ):
        monkeypatch.delenv(key, raising=False)
    providers_module._INJECTED_PROVIDER.set(None)
    registry._load.cache_clear()
    yield
    providers_module._INJECTED_PROVIDER.set(None)
    registry._load.cache_clear()


class TestMiniMaxProvider:
    def test_returns_none_without_api_key(self) -> None:
        assert MiniMaxProvider().resolve_credentials() is None

    def test_uses_global_endpoint_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        assert MiniMaxProvider().resolve_credentials() == (
            "test-key",
            MINIMAX_GLOBAL_BASE_URL,
        )

    def test_selects_china_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        monkeypatch.setenv("MINIMAX_REGION", "cn_zh")
        assert MiniMaxProvider().resolve_credentials() == (
            "test-key",
            MINIMAX_CN_BASE_URL,
        )

    def test_honors_base_url_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        monkeypatch.setenv("MINIMAX_BASE_URL", "https://minimax.example.com/v1")
        assert MiniMaxProvider().resolve_credentials() == (
            "test-key",
            "https://minimax.example.com/v1",
        )

    def test_rejects_unknown_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        monkeypatch.setenv("MINIMAX_REGION", "unknown")
        with pytest.raises(ValueError, match="global_en.*cn_zh"):
            MiniMaxProvider().resolve_credentials()

    def test_creates_chat_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        llm = MiniMaxProvider().create_chat_model("MiniMax-M2.7", max_tokens=123)
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "MiniMax-M2.7"
        assert llm.max_tokens == 123
        assert str(llm.openai_api_base).rstrip("/") == MINIMAX_GLOBAL_BASE_URL

    def test_bundled_models_and_context_windows(self) -> None:
        provider = MiniMaxProvider()
        assert provider.resolve_model() == "MiniMax-M3.1-Flash"
        assert list(yaml.safe_load(Path(REGISTRY_PATH).read_text())["models"]) == [
            "MiniMax-M3.1-Flash",
            "MiniMax-M3",
        ]
        assert provider.get_context_length("MiniMax-M3.1-Flash") == 1_000_000
        assert provider.get_max_output_tokens("MiniMax-M3.1-Flash") == 524_288
        assert provider.get_context_length("MiniMax-M3") == 1_000_000
        assert provider.get_context_length("MiniMax-M2.7") == 204_800
        assert provider.get_max_output_tokens("MiniMax-M3") is None

    def test_selector_uses_minimax_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "minimax")
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        assert resolve_provider_credentials() == ("test-key", MINIMAX_GLOBAL_BASE_URL)
        assert isinstance(get_metadata_provider(), MiniMaxProvider)

    def test_api_key_is_removed_from_cli_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        assert "MINIMAX_API_KEY" not in _scrub_env()


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Capture real SDK HTTP requests without opening a network connection."""
    captured: list[httpx.Request] = []

    def send(_client: httpx.Client, request: httpx.Request, **kwargs: object) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        message: dict[str, object] = {
            "role": "assistant",
            "content": "safe",
            "reasoning_content": "Check permissions.",
        }
        if body.get("tools"):
            message["tool_calls"] = [
                {
                    "id": "call_scan",
                    "type": "function",
                    "function": {
                        "name": body["tools"][0]["function"]["name"],
                        "arguments": '{"safe": true}',
                    },
                }
            ]
        if body.get("stream"):
            tool_calls = message.get("tool_calls", [])
            for index, tool_call in enumerate(tool_calls):
                tool_call["index"] = index
            chunk = {
                "id": "test-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "delta": message, "finish_reason": None}],
            }
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            },
        )

    async def async_send(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: object
    ) -> httpx.Response:
        return send(client, request, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    return captured


@pytest.mark.parametrize(
    ("region", "override", "base_url"),
    [
        ("global_en", None, MINIMAX_GLOBAL_BASE_URL),
        ("cn_zh", None, MINIMAX_CN_BASE_URL),
        ("cn_zh", "https://proxy.example/v1", "https://proxy.example/v1"),
    ],
)
@pytest.mark.parametrize("model", [None, "MiniMax-M3", "MiniMax-M2.7", "custom-model"])
def test_scanner_serializes_default_and_explicit_models(
    monkeypatch: pytest.MonkeyPatch,
    requests: list[httpx.Request],
    region: str,
    override: str | None,
    base_url: str,
    model: str | None,
) -> None:
    monkeypatch.setenv("MINIMAX_REGION", region)
    if override:
        monkeypatch.setenv("MINIMAX_BASE_URL", override)
    if model:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", model)
    llm = get_chat_model()
    assert llm.invoke("Inspect this skill").content == "safe"
    request = requests[0]
    body = json.loads(request.content)
    assert str(request.url) == base_url + "/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["model"] == (model or "MiniMax-M3.1-Flash")
    assert body["max_completion_tokens"] > 0
    if model == "MiniMax-M2.7":
        assert body["max_completion_tokens"] == 51_200
    assert not {"service_tier", "reasoning_split", "thinking", "reasoning_effort"} & body.keys()


def test_saved_registry_budget_and_schema_remain_effective(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, requests: list[httpx.Request]
) -> None:
    saved_registry = tmp_path / "models.yaml"
    saved_registry.write_text(
        "models:\n  MiniMax-M2.7:\n    context_length: 80000\n"
        "    max_output_tokens: 8192\n    structured_output: json_mode\n"
    )
    monkeypatch.setenv("SKILLSPECTOR_MODEL_REGISTRY", str(saved_registry))
    monkeypatch.setenv("SKILLSPECTOR_MODEL", "MiniMax-M2.7")
    get_chat_model().invoke("Inspect this skill")
    assert json.loads(requests[0].content)["max_completion_tokens"] == 8192
    assert MiniMaxProvider().structured_output_method("MiniMax-M2.7") == "json_mode"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_flash_serializes_supported_effort(
    monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request], effort: str
) -> None:
    monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", effort)
    get_chat_model().invoke("Inspect this skill")
    assert json.loads(requests[0].content)["reasoning_effort"] == effort


@pytest.mark.parametrize("effort", ["none", "minimal", "invalid"])
def test_flash_rejects_invalid_effort_before_http(
    monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request], effort: str
) -> None:
    monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", effort)
    with pytest.raises(UnsupportedControlError, match="MiniMax-M3.1-Flash"):
        get_chat_model()
    assert requests == []


@pytest.mark.parametrize("model", ["MiniMax-M3", "MiniMax-M2.7", "custom-model"])
def test_explicit_models_keep_existing_effort_passthrough(
    monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request], model: str
) -> None:
    monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", "none")
    get_chat_model(model).invoke("Inspect this skill")
    assert json.loads(requests[0].content)["reasoning_effort"] == "none"


def test_scanner_uses_tool_schema_and_preserves_multimodal_input(
    requests: list[httpx.Request],
) -> None:
    class ScanResult(BaseModel):
        """Result of inspecting a skill."""

        safe: bool

    content = [
        {"type": "text", "text": "Inspect this skill"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}},
    ]
    llm = get_chat_model()
    structured = bind_structured_output(llm, ScanResult, "MiniMax-M3.1-Flash")
    assert structured.invoke([{"role": "user", "content": content}]).safe is True
    body = json.loads(requests[0].content)
    assert "response_format" not in body
    assert body["tools"][0]["function"]["name"] == "ScanResult"
    assert body["messages"][0]["content"] == content


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_reasoning_and_tool_history_round_trip(
    requests: list[httpx.Request], stream: bool, asynchronous: bool
) -> None:
    class ScanResult(BaseModel):
        """Result of inspecting a skill."""

        safe: bool

    llm = get_chat_model().bind_tools([ScanResult])
    if asynchronous:
        if stream:
            chunks = [chunk async for chunk in llm.astream("Inspect this skill")]
            reply = sum(chunks[1:], chunks[0])
        else:
            reply = await llm.ainvoke("Inspect this skill")
    elif stream:
        chunks = list(llm.stream("Inspect this skill"))
        reply = sum(chunks[1:], chunks[0])
    else:
        reply = llm.invoke("Inspect this skill")
    assert reply.additional_kwargs["reasoning_content"] == "Check permissions."
    assert reply.tool_calls[0]["id"] == "call_scan"
    llm.invoke([reply, ToolMessage(content="checked", tool_call_id="call_scan")])
    messages = json.loads(requests[-1].content)["messages"]
    assert messages[0]["reasoning_content"] == "Check permissions."
    assert messages[0]["tool_calls"][0]["id"] == "call_scan"
    assert messages[1]["tool_call_id"] == "call_scan"


def test_minimax_provenance_tracks_forwarded_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    controls = {"temperature": 0.5, "seed": 7, "reasoning_effort": "high"}
    for name, value in controls.items():
        monkeypatch.setenv(f"SKILLSPECTOR_{name.upper()}", str(value))
    assert provider_name(MiniMaxProvider()) == "minimax"
    result = sanitize_llm_provenance(
        capture_llm_provenance({}),
        use_llm=True,
        inference_usage=[
            {
                "provider": "minimax",
                "usage_source": "provider_response",
                "requested_controls": controls,
                "forwarded_controls": controls,
            }
        ],
    )
    assert result["provider"]["resolved_adapter"] == "minimax"
    for name, value in controls.items():
        assert result["sampling"][name]["adapter_support"] is True
        assert result["sampling"][name]["forwarded_to_client"] == value
