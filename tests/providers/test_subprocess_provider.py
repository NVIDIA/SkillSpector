# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from skillspector.providers.subprocess.provider import SubprocessChatModel


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
        import subprocess

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


import os
from unittest.mock import patch

from skillspector.providers.subprocess.provider import SubprocessProvider


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

    def test_create_chat_model_returns_none_when_no_command(self, monkeypatch):
        monkeypatch.delenv("SKILLSPECTOR_LLM_COMMAND", raising=False)
        p = SubprocessProvider()
        assert p.create_chat_model("subprocess", max_tokens=512) is None

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
