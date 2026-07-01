# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
            model.invoke(messages)

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
        """Test that markdown code fences are stripped from response text."""
        text = "```json\n{\"key\": \"value\"}\n```"
        assert _strip_fences(text) == '{"key": "value"}'

    def test_strip_fences_passthrough_plain(self):
        """Test that plain JSON passes through unchanged."""
        text = '{"key": "value"}'
        assert _strip_fences(text) == '{"key": "value"}'

    def test_augment_messages_appends_to_last_human(self):
        """Test that JSON schema instruction is appended to the last HumanMessage."""
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="ask"),
        ]
        augmented = _augment_messages_with_json_instruction(msgs, '{"type": "object"}')
        assert isinstance(augmented[-1], HumanMessage)
        assert "JSON Schema" in augmented[-1].content
        assert augmented[0].content == "sys"


class TestFormatMessages:
    """Tests for _format_messages covering all message type branches."""

    def test_ai_message_renders_as_assistant_tag(self):
        """Test that AIMessage content is wrapped in assistant tags."""
        from skillspector.providers.subprocess.provider import _format_messages

        msgs = [AIMessage(content="I am the assistant.")]
        result = _format_messages(msgs)
        assert "<assistant>" in result
        assert "I am the assistant." in result

    def test_fallback_string_content_renders_as_str(self):
        """Test that unknown message types with string content are rendered."""
        from langchain_core.messages import ChatMessage

        from skillspector.providers.subprocess.provider import _format_messages

        msgs = [ChatMessage(content="raw text", role="custom")]
        result = _format_messages(msgs)
        assert "raw text" in result

    def test_fallback_list_content_extracts_str_items(self):
        """Test that list content with string items is joined correctly."""
        from langchain_core.messages import ChatMessage

        from skillspector.providers.subprocess.provider import _format_messages

        msgs = [ChatMessage(content=["part one", "part two"], role="custom")]
        result = _format_messages(msgs)
        assert "part one" in result
        assert "part two" in result

    def test_fallback_list_content_extracts_dict_text_key(self):
        """Test that list content with dict items extracts the 'text' key."""
        from langchain_core.messages import ChatMessage

        from skillspector.providers.subprocess.provider import _format_messages

        msgs = [ChatMessage(content=[{"type": "text", "text": "hello"}], role="custom")]
        result = _format_messages(msgs)
        assert "hello" in result


class TestWithStructuredOutput:
    """Tests for SubprocessChatModel.with_structured_output paths."""

    def test_pydantic_schema_path_parses_json_response(self):
        """Test that a Pydantic BaseModel schema returns a validated model instance."""
        from pydantic import BaseModel as PydanticModel

        class MySchema(PydanticModel):
            value: str

        model = _model()
        runnable = model.with_structured_output(MySchema)

        with patch.object(model, "_call_subprocess", return_value='{"value": "ok"}'):
            result = runnable.invoke([HumanMessage(content="test")])

        assert isinstance(result, MySchema)
        assert result.value == "ok"

    def test_dict_schema_path_returns_parsed_dict(self):
        """Test that a dict JSON Schema returns a parsed Python dict."""
        model = _model()
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        runnable = model.with_structured_output(schema)

        with patch.object(model, "_call_subprocess", return_value='{"x": 42}'):
            result = runnable.invoke([HumanMessage(content="test")])

        assert result == {"x": 42}

    def test_invalid_schema_type_raises_type_error(self):
        """Test that an unsupported schema type raises TypeError."""
        model = _model()
        with pytest.raises(TypeError, match="requires a Pydantic BaseModel"):
            model.with_structured_output("not-a-schema")  # type: ignore[arg-type]

    def test_pydantic_path_strips_markdown_fences(self):
        """Test that markdown fences in the response are stripped before parsing."""
        from pydantic import BaseModel as PydanticModel

        class MySchema(PydanticModel):
            value: str

        model = _model()
        runnable = model.with_structured_output(MySchema)
        fenced = '```json\n{"value": "fenced"}\n```'

        with patch.object(model, "_call_subprocess", return_value=fenced):
            result = runnable.invoke([HumanMessage(content="test")])

        assert result.value == "fenced"

    def test_pydantic_schema_path_accepts_plain_string_prompt(self):
        """A bare string prompt (as LLMAnalyzerBase passes) must still get the
        JSON-schema instruction appended, not be iterated character-by-character.
        """
        from pydantic import BaseModel as PydanticModel

        class MySchema(PydanticModel):
            value: str

        model = _model()
        runnable = model.with_structured_output(MySchema)
        captured: list[str] = []

        def fake_call(prompt: str) -> str:
            captured.append(prompt)
            return '{"value": "ok"}'

        with patch.object(model, "_call_subprocess", side_effect=fake_call):
            result = runnable.invoke("plain string prompt")

        assert isinstance(result, MySchema)
        assert result.value == "ok"
        assert len(captured) == 1
        assert "plain string prompt" in captured[0]
        assert "JSON Schema" in captured[0]

    def test_dict_schema_path_accepts_plain_string_prompt(self):
        """A bare string prompt must work for the dict-schema path too."""
        model = _model()
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        runnable = model.with_structured_output(schema)
        captured: list[str] = []

        def fake_call(prompt: str) -> str:
            captured.append(prompt)
            return '{"x": 42}'

        with patch.object(model, "_call_subprocess", side_effect=fake_call):
            result = runnable.invoke("plain string prompt")

        assert result == {"x": 42}
        assert len(captured) == 1
        assert "plain string prompt" in captured[0]
        assert "JSON Schema" in captured[0]


class TestExitCode1Diagnostic:
    """exit code 1 diagnostic hint for headless claude sessions."""

    def test_exit_code_1_no_stdout_gives_enterprise_hint(self):
        """exit code 1 with no stdout and 'claude' in command should raise with enterprise hint."""
        model = SubprocessChatModel(command="claude -p", timeout=10.0)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="enterprise session credentials"):
                model._call_subprocess("test prompt")

    def test_exit_code_1_with_stdout_gives_generic_error(self):
        """exit code 1 with stdout present should give the generic error (not enterprise hint)."""
        model = SubprocessChatModel(command="some-other-tool", timeout=10.0)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "some output"
        mock_result.stderr = "error detail"
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError) as exc_info:
                model._call_subprocess("test prompt")
        assert "enterprise session credentials" not in str(exc_info.value)
        assert "exit 1" in str(exc_info.value)


class TestLLMAnalyzerBaseIntegration:
    """End-to-end regression test: LLMAnalyzerBase.run_batches through the
    subprocess provider's with_structured_output() RunnableLambda.

    This is the exact call path that motivated the fix: LLMAnalyzerBase
    invokes the structured runnable with a plain string prompt (not a
    message list), and the runnable must coerce that string before
    appending the JSON-schema instruction.
    """

    def test_run_batches_end_to_end_with_subprocess_provider(self, monkeypatch):
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "subprocess")
        monkeypatch.setenv("SKILLSPECTOR_LLM_COMMAND", "claude -p")

        from skillspector.llm_analyzer_base import Batch, LLMAnalyzerBase

        canned_json = (
            '{"findings": [{"rule_id": "TEST001", "message": "found it", '
            '"severity": "HIGH", "start_line": 1}]}'
        )
        captured: list[str] = []

        def fake_call(prompt: str) -> str:
            captured.append(prompt)
            return canned_json

        with patch.object(SubprocessChatModel, "_call_subprocess", side_effect=fake_call):
            analyzer = LLMAnalyzerBase(base_prompt="Look for issues.", model="subprocess")
            batch = Batch(file_path="foo.py", content="print('hi')")
            results = analyzer.run_batches([batch])

        # The prompt built by LLMAnalyzerBase must reach _call_subprocess intact
        # (not iterated character-by-character) and carry the JSON-schema
        # instruction appended by with_structured_output().
        assert len(captured) == 1
        assert "foo.py" in captured[0]
        assert "JSON Schema" in captured[0]

        assert len(results) == 1
        result_batch, findings = results[0]
        assert result_batch is batch
        assert len(findings) == 1
        assert findings[0].rule_id == "TEST001"
        assert findings[0].message == "found it"
        assert findings[0].severity == "HIGH"
