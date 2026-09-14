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

"""Unit tests for the OpenCode CLI argv builder + auth check.

Security invariants verified:
  - argv is a plain ``list[str]`` (``shell=False`` downstream) — the prompt
    travels via stdin, never in argv, so Windows-hostile content (spaces,
    embedded quotes, unicode, backslashes, trailing backslash) needs no
    quoting and survives byte-exact.
  - ``--model`` is omitted when no model is set and validated otherwise.
  - ``--pure`` disables external plugins; ``--auto`` (auto-approve) is NEVER
    in argv.
  - The auth probe (``opencode auth list``) is cheap, non-inference, bounded,
    uses the same scrubbed environment as inference, and fail-closed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from skillspector.inference_usage import provider_name
from skillspector.providers import (
    _agent_cli,
    get_metadata_provider,
    has_cli_capability,
    resolve_provider_credentials,
)
from skillspector.providers._agent_cli import (
    AgentCLIError,
    _build_opencode_argv,
    _opencode_auth_check,
    _parse_opencode_output,
    _run_bounded,
)
from skillspector.providers.opencode_cli import OpencodeCLIProvider

OPENCODE_BINARY = "/usr/bin/opencode"
MODEL = "anthropic/claude-sonnet-4-6"

_AUTH_LIST_OK = (
    "\x1b[90m\u2514\x1b[39m  4 credentials\n\x1b[90m\u2514\x1b[39m  3 environment variables\n"
).encode()
_AUTH_LIST_EMPTY = b"0 credentials\n0 environment variables\n"
_AUTH_LIST_SINGULAR = b"1 credential\n1 environment variable\n"
_AUTH_LIST_UNPARSEABLE = b"authentication status unknown\n"


def _ok_result(stdout: bytes = _AUTH_LIST_OK) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")


# ---------------------------------------------------------------------------
# _build_opencode_argv
# ---------------------------------------------------------------------------


class TestBuildOpencodeArgv:
    def test_argv_is_plain_list_of_str(self) -> None:
        argv = _build_opencode_argv(OPENCODE_BINARY, "", 0)
        assert isinstance(argv, list), "argv must be a list (ensures shell=False)"
        assert argv and all(isinstance(a, str) for a in argv)

    def test_argv_exact_shape_without_model(self) -> None:
        assert _build_opencode_argv(OPENCODE_BINARY, "", 0) == [
            OPENCODE_BINARY,
            "run",
            "--pure",
            "--format",
            "json",
        ]

    def test_argv_disables_external_plugins(self) -> None:
        assert "--pure" in _build_opencode_argv(OPENCODE_BINARY, "", 0)

    def test_argv_format_json_pair(self) -> None:
        argv = _build_opencode_argv(OPENCODE_BINARY, "", 0)
        assert "--format" in argv
        assert argv[argv.index("--format") + 1] == "json"

    def test_argv_model_forwarded(self) -> None:
        argv = _build_opencode_argv(OPENCODE_BINARY, MODEL, 0)
        assert "--model" in argv
        assert argv[argv.index("--model") + 1] == MODEL

    def test_argv_model_omitted_when_empty(self) -> None:
        # No SKILLSPECTOR_MODEL -> opencode runs with the CLI default model.
        assert "--model" not in _build_opencode_argv(OPENCODE_BINARY, "", 0)

    def test_argv_model_label_validated_against_injection(self) -> None:
        with pytest.raises(AgentCLIError):
            _build_opencode_argv(OPENCODE_BINARY, "--auto", 0)
        with pytest.raises(AgentCLIError):
            _build_opencode_argv(OPENCODE_BINARY, "model;rm -rf /", 0)

    def test_argv_never_auto_approve(self) -> None:
        # --auto auto-approves permissions (dangerous); never use it.
        for model in ("", MODEL):
            argv = _build_opencode_argv(OPENCODE_BINARY, model, 0)
            assert "--auto" not in argv
            assert "dangerously" not in " ".join(argv).lower()

    def test_argv_max_output_tokens_accepted_but_not_forwarded(self) -> None:
        # CliSpec uniformity: the parameter exists but opencode run has no
        # token flag, so it must not change argv.
        assert _build_opencode_argv(OPENCODE_BINARY, MODEL, 0) == _build_opencode_argv(
            OPENCODE_BINARY, MODEL, 8192
        )


# ---------------------------------------------------------------------------
# Windows quoting: hostile prompt bytes survive stdin delivery byte-exact
# (real subprocesses, not mocks — this is the transport argv shape relies on)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "spaces in the prompt stay whole",
        "embedded \"double\" and 'single' quotes",
        "unicode h\xe9llo \u4e16\u754c \U0001f389",
        "backslashes C:\\path\\to\\skill",
        "trailing backslash endswith\\",
    ],
)
def test_hostile_prompt_roundtrips_byte_exact(prompt: str) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    raw = prompt.encode("utf-8")
    rc, out, _err, overflow = _run_bounded(proc, raw, timeout=30)
    assert rc == 0
    assert overflow is False
    assert out == raw


# ---------------------------------------------------------------------------
# _opencode_auth_check
# ---------------------------------------------------------------------------


class TestOpencodeAuthCheck:
    @pytest.fixture(autouse=True)
    def _binary_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Probe tests focus on probe behavior; the lookup itself is covered
        # by test_lookup_receives_passed_binary below.
        monkeypatch.setattr(_agent_cli, "find_binary", lambda name: name)

    def test_assumes_caller_resolved_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail_on_lookup(_name: str) -> str:
            raise AssertionError("auth check must not re-resolve the binary")

        monkeypatch.setattr(_agent_cli, "find_binary", fail_on_lookup)
        with patch("skillspector.providers._agent_cli.subprocess.run") as mock_run:
            mock_run.return_value = _ok_result()
            ok, _reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is True
        mock_run.assert_called_once()

    def test_missing_binary_is_fail_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: None)
        with patch("skillspector.providers._agent_cli.subprocess.run") as mock_run:
            ok, reason = _agent_cli.is_available("opencode")
            mock_run.assert_not_called()
        assert ok is False
        assert reason

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ok_result()
        assert _opencode_auth_check(OPENCODE_BINARY) == (True, None)

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_uses_auth_list(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ok_result()
        _opencode_auth_check(OPENCODE_BINARY)
        assert mock_run.call_args[0][0][:3] == [OPENCODE_BINARY, "auth", "list"]

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_uses_scrubbed_environment(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ok_result()
        _opencode_auth_check(OPENCODE_BINARY)
        assert mock_run.call_args[1].get("env") == _agent_cli._scrub_env()

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_shell_is_false_and_bounded(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ok_result()
        _opencode_auth_check(OPENCODE_BINARY)
        kwargs = mock_run.call_args[1]
        assert kwargs.get("shell") is False
        assert isinstance(kwargs.get("timeout"), int | float)
        assert kwargs["timeout"] <= 15

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_nonzero_exit_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.return_value = SimpleNamespace(returncode=1, stdout=b"", stderr=b"boom")
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert reason

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_timeout_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="opencode", timeout=15)
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert reason

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_no_credentials_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ok_result(_AUTH_LIST_EMPTY)
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert "auth login" in (reason or "")

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_accepts_singular_counts(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ok_result(_AUTH_LIST_SINGULAR)
        assert _opencode_auth_check(OPENCODE_BINARY) == (True, None)

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_unparseable_output_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ok_result(_AUTH_LIST_UNPARSEABLE)
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert "auth login" in (reason or "")


# ---------------------------------------------------------------------------
# _parse_opencode_output
# ---------------------------------------------------------------------------


# Real opencode 1.18.30 envelope shapes (Step-0 probe:
# `opencode run "say hi" --format json` in an empty dir; verbatim raw saved
# to Temp scratch only). Top-level `type` is one of step_start / text /
# step_finish; the reply text lives at part.text of `text` events.
_STEP_START = (
    '{"type":"step_start","timestamp":1789228628310,"sessionID":"ses_abc",'
    '"part":{"id":"prt_1","messageID":"msg_1","sessionID":"ses_abc","type":"step-start"}}'
)
_TEXT_EVENT = (
    '{"type":"text","timestamp":1789228631145,"sessionID":"ses_abc",'
    '"part":{"id":"prt_2","messageID":"msg_1","sessionID":"ses_abc",'
    '"type":"text","text":"hi","time":{"start":1789228631133,"end":1789228631140}}}'
)
_STEP_FINISH = (
    '{"type":"step_finish","timestamp":1789228631194,"sessionID":"ses_abc",'
    '"part":{"id":"prt_3","reason":"stop","messageID":"msg_1",'
    '"sessionID":"ses_abc","type":"step-finish"}}'
)


def _text_event(text: str) -> str:
    return (
        '{"type":"text","timestamp":1789228631145,"sessionID":"ses_abc",'
        '"part":{"id":"prt_9","messageID":"msg_1","sessionID":"ses_abc",'
        f'"type":"text","text":{json.dumps(text)}'
        ',"time":{"start":1789228631133,"end":1789228631140}}}'
    )


class TestParseOpencodeOutput:
    def test_extracts_text_from_full_session(self) -> None:
        # All three observed envelope shapes; only the text event carries reply.
        raw = f"{_STEP_START}\n{_TEXT_EVENT}\n{_STEP_FINISH}\n"
        assert _parse_opencode_output(raw) == "hi"

    def test_concatenates_multiple_text_events_in_order(self) -> None:
        raw = f"{_text_event('Hello, ')}\n{_text_event('world!')}\n"
        assert _parse_opencode_output(raw) == "Hello, world!"

    def test_skips_non_json_noise_lines(self) -> None:
        # Banner/TUI noise must never leak into the extracted output.
        raw = f"opencode v1.18.30\n{_TEXT_EVENT}\nattaching session...\n"
        assert _parse_opencode_output(raw) == "hi"

    def test_empty_stdout_raises(self) -> None:
        with pytest.raises(AgentCLIError, match="no assistant text"):
            _parse_opencode_output("")

    def test_whitespace_only_stdout_raises(self) -> None:
        with pytest.raises(AgentCLIError, match="no assistant text"):
            _parse_opencode_output("   \n  \n")

    def test_no_text_events_raises(self) -> None:
        # Non-empty output but no assistant text (step boundaries only).
        with pytest.raises(AgentCLIError, match="no assistant text"):
            _parse_opencode_output(f"{_STEP_START}\n{_STEP_FINISH}\n")

    def test_ignores_missing_and_malformed_parts(self) -> None:
        raw = "\n".join(
            [
                '{"type":"text","part":null}',
                '{"type":"text","part":[]}',
                '{"type":"text","part":{"type":"text","text":123}}',
                '{"type":"text","part":{"type":"text","text":"   "}}',
                '["not", "an", "object"]',
                _TEXT_EVENT,
            ]
        )
        assert _parse_opencode_output(raw) == "hi"

    def test_colon_model_labels_pass_validation(self) -> None:
        argv = _build_opencode_argv(OPENCODE_BINARY, "openrouter/poolside/laguna-s-2.1:free", 0)
        assert argv[argv.index("--model") + 1] == ("openrouter/poolside/laguna-s-2.1:free")


# ---------------------------------------------------------------------------
# Provider wiring: SKILLSPECTOR_PROVIDER=opencode_cli selects the
# provider end to end. No subprocess calls here — selection, model
# resolution, and labeling are pure env/class lookups.
# ---------------------------------------------------------------------------


class TestOpencodeCLIProviderWiring:
    def test_provider_selected_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "opencode_cli")
        provider = get_metadata_provider()
        assert isinstance(provider, OpencodeCLIProvider)
        # CLI provider returns no HTTP credentials
        assert resolve_provider_credentials() is None

    def test_resolve_model_empty_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No model is pinned: with SKILLSPECTOR_MODEL unset, resolve_model is ""
        # so opencode receives no explicit --model override.
        monkeypatch.delenv("SKILLSPECTOR_MODEL", raising=False)
        assert OpencodeCLIProvider().resolve_model() == ""
        assert OpencodeCLIProvider.DEFAULT_MODEL == ""

    def test_resolve_model_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "anthropic/claude-sonnet-4-6")
        assert OpencodeCLIProvider().resolve_model() == "anthropic/claude-sonnet-4-6"

    def test_provider_name_label(self) -> None:
        assert provider_name(OpencodeCLIProvider()) == "opencode_cli"

    def test_has_cli_capability(self) -> None:
        assert has_cli_capability(OpencodeCLIProvider())
