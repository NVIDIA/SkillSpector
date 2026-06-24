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
        with patch.object(model, "_call_subprocess", return_value="  hello world  "):
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
        import subprocess as sp

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
