# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess as sp
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from skillspector.providers import _select_active_provider, create_chat_model
from skillspector.providers.subprocess.provider import (
    SubprocessChatModel,
    SubprocessProvider,
    _augment_messages_with_json_instruction,
    _strip_fences,
)


def _model(command: str = "echo") -> SubprocessChatModel:
    return SubprocessChatModel(command=command)


class TestSubprocessChatModelGenerate:
    def test_formats_system_and_human_messages(self):
        model = _model()
        captured: list[str] = []

        def fake_call(prompt: str) -> str:
            captured.append(prompt)
            return "response"

        with patch.object(model, "_call_subprocess", side_effect=fake_call):
            messages = [
                SystemMessage(content="You are a security analyst."),
                HumanMessage(content="Review this file."),
            ]
            result = model.invoke(messages)

        assert len(captured) == 1
        assert "You are a security analyst." in captured[0]
        assert "Review this file." in captured[0]

    def test_returns_ai_message_with_subprocess_output(self):
        model = _model()
        with patch.object(model, "_call_subprocess", return_value="hello world"):
            result = model.invoke([HumanMessage(content="hi")])

        assert isinstance(result, AIMessage)
        assert result.content == "hello world"

    def test_raises_on_nonzero_exit(self):
        model = _model(command="false")  # always exits 1
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stderr = "command failed"

        with patch("subprocess.run", return_value=fake_result):
            with pytest.raises(RuntimeError, match="LLM subprocess failed"):
                model.invoke([HumanMessage(content="hi")])

    def test_passes_full_prompt_to_stdin(self):
        model = _model(command="cat -")  # echoes stdin
        prompt_seen: list[str] = []

        def fake_run(args, *, input, capture_output, text, timeout):
            prompt_seen.append(input)
            result = MagicMock()
            result.returncode = 0
            result.stdout = "ok"
            return result

        with patch("subprocess.run", side_effect=fake_run):
            model.invoke([HumanMessage(content="test prompt")])

        assert "test prompt" in prompt_seen[0]

    def test_raises_on_timeout(self):
        model = _model()
        with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="echo", timeout=120)):
            with pytest.raises(RuntimeError, match="timed out"):
                model.invoke([HumanMessage(content="hi")])


class TestSubprocessProvider:
    def test_resolve_credentials_returns_command_when_env_set(self, monkeypatch):
        monkeypatch.setenv("SKILLSPECTOR_LLM_COMMAND", "claude -p")
        p = SubprocessProvider()
        creds = p.resolve_credentials()
        assert creds == ("subprocess", None)

    def test_resolve_credentials_returns_none_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("SKILLSPECTOR_LLM_COMMAND", raising=False)
        p = SubprocessProvider()
        assert p.resolve_credentials() is None

    def test_create_chat_model_returns_subprocess_model(self, monkeypatch):
        monkeypatch.setenv("SKILLSPECTOR_LLM_COMMAND", "cat -")
        p = SubprocessProvider()
        model = p.create_chat_model("subprocess", max_tokens=512, timeout=30.0)
        assert isinstance(model, SubprocessChatModel)
        assert model.command == "cat -"

    def test_create_chat_model_raises_when_no_command(self, monkeypatch):
        monkeypatch.delenv("SKILLSPECTOR_LLM_COMMAND", raising=False)
        p = SubprocessProvider()
        with pytest.raises(ValueError, match="SKILLSPECTOR_LLM_COMMAND"):
            p.create_chat_model("subprocess", max_tokens=512)

    def test_resolve_model_returns_skillspector_model_env(self, monkeypatch):
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "my-local-model")
        p = SubprocessProvider()
        assert p.resolve_model() == "my-local-model"

    def test_resolve_model_falls_back_to_sentinel(self, monkeypatch):
        monkeypatch.delenv("SKILLSPECTOR_MODEL", raising=False)
        p = SubprocessProvider()
        assert p.resolve_model() == "subprocess"

    def test_get_context_length_returns_default(self):
        p = SubprocessProvider()
        length = p.get_context_length("subprocess")
        assert length == 200_000

    def test_get_max_output_tokens_returns_default(self):
        p = SubprocessProvider()
        tokens = p.get_max_output_tokens("subprocess")
        assert tokens == 8_192


class TestSubprocessProviderSelection:
    def test_select_active_provider_returns_subprocess(self, monkeypatch):
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "subprocess")
        monkeypatch.setenv("SKILLSPECTOR_LLM_COMMAND", "echo hi")
        provider = _select_active_provider()
        assert isinstance(provider, SubprocessProvider)

    def test_create_chat_model_uses_subprocess_command(self, monkeypatch):
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "subprocess")
        monkeypatch.setenv("SKILLSPECTOR_LLM_COMMAND", "echo hi")
        model = create_chat_model("subprocess", max_tokens=512)
        assert isinstance(model, SubprocessChatModel)


class TestHelperFunctions:
    def test_strip_fences_removes_markdown(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        assert _strip_fences(text) == '{"key": "value"}'

    def test_strip_fences_passthrough_plain(self):
        text = '{"key": "value"}'
        assert _strip_fences(text) == '{"key": "value"}'

    def test_augment_messages_appends_to_last_human(self):
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="ask"),
        ]
        augmented = _augment_messages_with_json_instruction(msgs, '{"type": "object"}')
        assert isinstance(augmented[-1], HumanMessage)
        assert "JSON Schema" in augmented[-1].content
        assert augmented[0].content == "sys"
