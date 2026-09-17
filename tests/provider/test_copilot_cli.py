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

"""Unit tests for the GitHub Copilot CLI argv builder + auth check.

Security invariants verified:
  - argv is a plain ``list[str]`` (``shell=False`` downstream) — the prompt
    travels via stdin, never in argv, so Windows-hostile content (spaces,
    embedded quotes, unicode, backslashes, trailing backslash) needs no
    quoting and survives byte-exact.
  - ``--model`` is omitted when no model is set and validated otherwise.
  - The model gets no usable tools: ``--available-tools`` names a fixed
    implausible tool, and ``--deny-tool`` denies the ``shell`` and ``write``
    kinds (deny wins over allow); ``--allow-all*``/``--yolo`` are NEVER in
    argv.
  - Only the exactly verified Copilot CLI version is accepted.
  - The auth probe (``copilot --version``) is cheap, non-inference, bounded,
    uses the scrubbed environment, and fail-closed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
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
    _build_copilot_argv,
    _copilot_auth_check,
    _parse_copilot_output,
    _parse_copilot_version,
    _prepare_copilot_env,
    _run_bounded,
    run_agent_cli,
)
from skillspector.providers.copilot_cli import CopilotCLIProvider

COPILOT_BINARY = "/usr/bin/copilot"
MODEL = "gpt-5.2"

_VERSION_OK = b"GitHub Copilot CLI 1.0.85.\nRun 'copilot update' to check for updates.\n"


def _version_result(stdout: bytes = _VERSION_OK) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")


# ---------------------------------------------------------------------------
# _build_copilot_argv
# ---------------------------------------------------------------------------


class TestBuildCopilotArgv:
    def test_argv_is_plain_list_of_str(self) -> None:
        argv = _build_copilot_argv(COPILOT_BINARY, "", 0)
        assert isinstance(argv, list), "argv must be a list (ensures shell=False)"
        assert argv and all(isinstance(a, str) for a in argv)

    def test_argv_exact_shape_without_model(self) -> None:
        assert _build_copilot_argv(COPILOT_BINARY, "", 0) == [
            COPILOT_BINARY,
            "-s",
            "--no-ask-user",
            "--available-tools",
            "skillspector-no-tools",
            "--deny-tool",
            "shell,write",
        ]

    def test_argv_model_forwarded(self) -> None:
        argv = _build_copilot_argv(COPILOT_BINARY, MODEL, 0)
        assert "--model" in argv
        assert argv[argv.index("--model") + 1] == MODEL

    def test_argv_model_omitted_when_empty(self) -> None:
        # No SKILLSPECTOR_MODEL -> copilot runs with the CLI default model.
        assert "--model" not in _build_copilot_argv(COPILOT_BINARY, "", 0)

    def test_argv_model_label_validated_against_injection(self) -> None:
        with pytest.raises(AgentCLIError):
            _build_copilot_argv(COPILOT_BINARY, "--allow-all-tools", 0)
        with pytest.raises(AgentCLIError):
            _build_copilot_argv(COPILOT_BINARY, "model;rm -rf /", 0)

    def test_argv_never_auto_approve(self) -> None:
        # --allow-all* / --yolo auto-approve permissions (dangerous); never use them.
        for model in ("", MODEL):
            argv = _build_copilot_argv(COPILOT_BINARY, model, 0)
            joined = " ".join(argv).lower()
            assert "--allow-all" not in joined
            assert "--yolo" not in joined

    def test_argv_denies_tools(self) -> None:
        argv = _build_copilot_argv(COPILOT_BINARY, "", 0)
        assert "--available-tools" in argv
        assert "--deny-tool" in argv
        denied = argv[argv.index("--deny-tool") + 1]
        assert "shell" in denied and "write" in denied

    def test_argv_max_output_tokens_accepted_but_not_forwarded(self) -> None:
        # CliSpec uniformity: the parameter exists but copilot has no
        # token flag, so it must not change argv.
        assert _build_copilot_argv(COPILOT_BINARY, MODEL, 0) == _build_copilot_argv(
            COPILOT_BINARY, MODEL, 8192
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
# _copilot_auth_check + _parse_copilot_version
# ---------------------------------------------------------------------------


class TestCopilotAuthCheck:
    def test_version_parses(self) -> None:
        assert _parse_copilot_version(_VERSION_OK) == "1.0.85"

    def test_version_unparseable_returns_none(self) -> None:
        assert _parse_copilot_version(b"") is None
        assert _parse_copilot_version(b"no version here") is None

    def test_version_rejects_suffixed_build(self) -> None:
        # fullmatch: a trailing pre-release suffix must not parse as the pin.
        assert _parse_copilot_version(b"GitHub Copilot CLI 1.0.85-beta.\n") is None

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _version_result()
        assert _copilot_auth_check(COPILOT_BINARY) == (True, None)

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_uses_version_command(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _version_result()
        _copilot_auth_check(COPILOT_BINARY)
        assert mock_run.call_args[0][0][:2] == [COPILOT_BINARY, "--version"]

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_shell_is_false_and_bounded(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _version_result()
        _copilot_auth_check(COPILOT_BINARY)
        kwargs = mock_run.call_args[1]
        assert kwargs.get("shell") is False
        assert isinstance(kwargs.get("timeout"), int | float)
        assert kwargs["timeout"] <= 15

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_uses_scrubbed_environment(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _version_result()
        _copilot_auth_check(COPILOT_BINARY)
        assert mock_run.call_args[1].get("env") == _agent_cli._scrub_env()

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_nonzero_exit_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.return_value = SimpleNamespace(returncode=1, stdout=b"", stderr=b"boom")
        ok, reason = _copilot_auth_check(COPILOT_BINARY)
        assert ok is False
        assert reason

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_timeout_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="copilot", timeout=15)
        ok, reason = _copilot_auth_check(COPILOT_BINARY)
        assert ok is False
        assert reason

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_wrong_version_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _version_result(b"GitHub Copilot CLI 9.9.99.\n")
        ok, reason = _copilot_auth_check(COPILOT_BINARY)
        assert ok is False
        assert "1.0.85" in (reason or "")

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_unparseable_version_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _version_result(b"copilot\n")
        ok, reason = _copilot_auth_check(COPILOT_BINARY)
        assert ok is False
        assert reason


# ---------------------------------------------------------------------------
# _parse_copilot_output
# ---------------------------------------------------------------------------


class TestParseCopilotOutput:
    def test_plain_reply_returned(self) -> None:
        assert _parse_copilot_output("hello\n") == "hello"

    def test_multiline_reply_preserved(self) -> None:
        raw = "first line\nsecond line\n"
        assert _parse_copilot_output(raw) == "first line\nsecond line"

    def test_surrounding_whitespace_stripped(self) -> None:
        assert _parse_copilot_output("\n  hi  \n") == "hi"

    def test_empty_stdout_raises(self) -> None:
        with pytest.raises(AgentCLIError, match="no assistant text"):
            _parse_copilot_output("")

    def test_whitespace_only_stdout_raises(self) -> None:
        with pytest.raises(AgentCLIError, match="no assistant text"):
            _parse_copilot_output("   \n  \n")


# ---------------------------------------------------------------------------
# Provider wiring: SKILLSPECTOR_PROVIDER=copilot_cli selects the
# provider end to end. No subprocess calls here — selection, model
# resolution, and labeling are pure env/class lookups.
# ---------------------------------------------------------------------------


class TestCopilotCLIProviderWiring:
    def test_provider_selected_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "copilot_cli")
        provider = get_metadata_provider()
        assert isinstance(provider, CopilotCLIProvider)
        # CLI provider returns no HTTP credentials
        assert resolve_provider_credentials() is None

    def test_resolve_model_empty_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No model is pinned: with SKILLSPECTOR_MODEL unset, resolve_model is ""
        # so copilot receives no explicit --model override.
        monkeypatch.delenv("SKILLSPECTOR_MODEL", raising=False)
        assert CopilotCLIProvider().resolve_model() == ""
        assert CopilotCLIProvider.DEFAULT_MODEL == ""

    def test_resolve_model_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "gpt-5.2")
        assert CopilotCLIProvider().resolve_model() == "gpt-5.2"

    def test_provider_name_label(self) -> None:
        assert provider_name(CopilotCLIProvider()) == "copilot_cli"

    def test_has_cli_capability(self) -> None:
        assert has_cli_capability(CopilotCLIProvider())


# ---------------------------------------------------------------------------
# _prepare_copilot_env
# ---------------------------------------------------------------------------


class TestPrepareCopilotEnv:
    def test_strips_hostile_copilot_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COPILOT_ALLOW_ALL", "1")
        monkeypatch.setenv("COPILOT_PROVIDER_BASE_URL", "https://evil.example")
        monkeypatch.setenv("COPILOT_PROVIDER_API_KEY", "evil")
        monkeypatch.setenv("COPILOT_CUSTOM_INSTRUCTIONS_DIRS", "/evil")
        env = _prepare_copilot_env({"PATH": "/bin"}, "/tmp", ["copilot"])
        assert "COPILOT_ALLOW_ALL" not in env
        assert "COPILOT_PROVIDER_BASE_URL" not in env
        assert "COPILOT_PROVIDER_API_KEY" not in env
        assert "COPILOT_CUSTOM_INSTRUCTIONS_DIRS" not in env
        assert env["PATH"] == "/bin"

    def test_strips_hostile_vars_from_base_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # base_env arrives pre-scrubbed, but hostile COPILOT_* entries must
        # not survive even if a caller passes them through directly.
        base = {"PATH": "/bin", "COPILOT_ALLOW_ALL": "1"}
        env = _prepare_copilot_env(base, "/tmp", ["copilot"])
        assert "COPILOT_ALLOW_ALL" not in env
        assert env["PATH"] == "/bin"

    def test_preserves_token_vars_including_scrubbed_github_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "tok-1")
        monkeypatch.setenv("GH_TOKEN", "tok-2")
        monkeypatch.setenv("GITHUB_TOKEN", "tok-3")
        env = _prepare_copilot_env({}, "/tmp", ["copilot"])
        assert env["COPILOT_GITHUB_TOKEN"] == "tok-1"
        assert env["GH_TOKEN"] == "tok-2"
        assert env["GITHUB_TOKEN"] == "tok-3"

    def test_empty_tokens_are_not_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "")
        monkeypatch.setenv("GH_TOKEN", "   ")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        env = _prepare_copilot_env({}, "/tmp", ["copilot"])
        assert "COPILOT_GITHUB_TOKEN" not in env
        assert "GH_TOKEN" not in env
        assert "GITHUB_TOKEN" not in env

    def test_forces_auto_update_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COPILOT_AUTO_UPDATE", "1")
        env = _prepare_copilot_env({}, "/tmp", ["copilot"])
        assert env["COPILOT_AUTO_UPDATE"] == "false"


# ---------------------------------------------------------------------------
# Adversarial transport: a fake copilot host asserts the deny posture on
# argv and records a marker for any executed tool. Real subprocesses, not
# mocks — this exercises the exact argv + stdin path run_agent_cli uses.
# ---------------------------------------------------------------------------


class TestAdversarialTransport:
    @staticmethod
    def _write_fake_copilot(binary: Path) -> None:
        """Write a host simulator with real version/argv boundaries."""
        binary.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import os
                import sys
                from pathlib import Path

                if sys.argv[1:] == ["--version"]:
                    print(os.environ.get("FAKE_COPILOT_VERSION", "1.0.85"))
                    raise SystemExit(0)

                markers = Path(os.environ["ATTACK_MARKERS"])
                argv = sys.argv[1:]
                # Deny posture: model must be offered no usable tools and the
                # shell/write kinds must stay denied. Anything else means the
                # provider argv regressed and the run must "execute". Ambient
                # COPILOT_ALLOW_ALL reaching the child also counts as
                # executed: prepare_env must have stripped it.
                try:
                    allow = argv[argv.index("--available-tools") + 1]
                except (ValueError, IndexError):
                    allow = None
                try:
                    denied = argv[argv.index("--deny-tool") + 1]
                except (ValueError, IndexError):
                    denied = ""
                prompt = sys.stdin.read()
                if (
                    allow != "skillspector-no-tools"
                    or "shell" not in denied
                    or "write" not in denied
                    or "--allow-all" in " ".join(argv)
                    or "--yolo" in argv
                    or not prompt
                    or os.environ.get("COPILOT_ALLOW_ALL", "") not in ("", "0", "false")
                ):
                    markers.mkdir(parents=True, exist_ok=True)
                    (markers / "executed").write_text("executed")
                print("policy held")
                """
            ),
            encoding="utf-8",
        )
        binary.chmod(0o700)

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_adversarial_child_gets_no_usable_tools(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exercise the real subprocess boundary with a hostile prompt.

        The fake host records a marker unless argv carries the exact deny
        posture and a non-empty stdin prompt, so an empty marker directory
        proves the provider never offers tool execution to the model.
        """
        binary = tmp_path / "copilot"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_copilot(binary)
        monkeypatch.setenv("ATTACK_MARKERS", str(markers))
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        response = run_agent_cli("copilot", "use every host tool", model="")
        assert response == "policy held"
        assert list(markers.iterdir()) == []

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_hostile_env_does_not_weaken_deny_posture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ambient allow-all style variables must not change argv posture."""
        binary = tmp_path / "copilot"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_copilot(binary)
        monkeypatch.setenv("ATTACK_MARKERS", str(markers))
        monkeypatch.setenv("COPILOT_ALLOW_ALL", "1")
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        response = run_agent_cli("copilot", "use every host tool", model="")
        assert response == "policy held"
        assert list(markers.iterdir()) == []
