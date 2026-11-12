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
  - ``--pure`` disables external plugins and an unguessable per-invocation
    agent carries a wildcard deny for all current and future OpenCode tools;
    ``--auto`` is NEVER in argv.
  - Only the exactly verified OpenCode version is accepted, and resolved
    managed config cannot reopen tools, ambient input, or sharing.
  - The auth probe (``opencode auth list``) is cheap, non-inference, bounded,
    uses the same scrubbed environment as inference, and fail-closed.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import replace
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
    _OPENCODE_AGENT_PREFIX,
    _OPENCODE_DENY_ALL,
    _OPENCODE_SUPPORTED_VERSIONS,
    _OPENCODE_SUPPORTED_VERSIONS_V2,
    AgentCLIError,
    _build_opencode_argv,
    _is_supported_opencode_version,
    _is_supported_opencode_version_v2,
    _opencode_auth_check,
    _opencode_config_2x,
    _parse_opencode_debug_agents,
    _parse_opencode_output,
    _parse_opencode_version,
    _prepare_opencode_env,
    _run_bounded,
    get_spec,
    run_agent_cli,
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
_VERSION_OK = b"1.18.33\n"


def _ok_result(stdout: bytes = _AUTH_LIST_OK) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")


def _probe_results(auth_stdout: bytes = _AUTH_LIST_OK) -> list[SimpleNamespace]:
    return [_ok_result(_VERSION_OK), _ok_result(auth_stdout)]


# ---------------------------------------------------------------------------
# _build_opencode_argv
# ---------------------------------------------------------------------------


class TestBuildOpencodeArgv:
    @pytest.fixture(autouse=True)
    def _pin_probe_version_1x(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The builder probes the real binary for its version. Pin the probe
        # to 1.18.33 so the 1.x expectations below do not depend on which
        # binary (if any) a developer has installed: with a supported 2.x
        # binary on PATH the unmocked builder omits --pure and the exact
        # shape assertions fail even though the builder is correct.
        monkeypatch.setattr(_agent_cli, "_probe_opencode_version", lambda _binary: (1, 18, 33))

    def test_argv_is_plain_list_of_str(self) -> None:
        argv = _build_opencode_argv(OPENCODE_BINARY, "", 0)
        assert isinstance(argv, list), "argv must be a list (ensures shell=False)"
        assert argv and all(isinstance(a, str) for a in argv)

    def test_argv_exact_shape_without_model(self) -> None:
        argv = _build_opencode_argv(OPENCODE_BINARY, "", 0)
        agent_name = argv[argv.index("--agent") + 1]
        assert argv == [
            OPENCODE_BINARY,
            "run",
            "--pure",
            "--agent",
            agent_name,
            "--format",
            "json",
        ]

    def test_argv_disables_external_plugins(self) -> None:
        assert "--pure" in _build_opencode_argv(OPENCODE_BINARY, "", 0)

    @pytest.mark.parametrize("version", [(2, 2, 6), (2, 2, 7)])
    def test_argv_2x_omits_pure_and_uses_standalone(
        self, monkeypatch: pytest.MonkeyPatch, version: tuple[int, int, int]
    ) -> None:
        # The 2.x branch is pinned explicitly: --pure is unknown on 2.x,
        # and --standalone keeps the run on a private server instead of
        # the background service on the fixed channel port.
        monkeypatch.setattr(_agent_cli, "_probe_opencode_version", lambda _binary: version)
        argv = _build_opencode_argv(OPENCODE_BINARY, "", 0)
        assert "--pure" not in argv
        assert "--standalone" in argv
        assert argv[argv.index("--agent") + 1].startswith(f"{_OPENCODE_AGENT_PREFIX}-")
        assert "--format" in argv and argv[argv.index("--format") + 1] == "json"

    def test_argv_unknown_version_keeps_1x_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An unknowable version keeps the 1.x shape; the preflight then
        # fails closed on version before any inference runs.
        monkeypatch.setattr(_agent_cli, "_probe_opencode_version", lambda _binary: None)
        argv = _build_opencode_argv(OPENCODE_BINARY, "", 0)
        assert "--pure" in argv
        assert "--standalone" not in argv

    def test_argv_selects_unguessable_per_invocation_agent(self) -> None:
        first = _build_opencode_argv(OPENCODE_BINARY, "", 0)
        second = _build_opencode_argv(OPENCODE_BINARY, "", 0)
        first_agent = first[first.index("--agent") + 1]
        second_agent = second[second.index("--agent") + 1]
        assert first_agent.startswith(f"{_OPENCODE_AGENT_PREFIX}-")
        assert second_agent.startswith(f"{_OPENCODE_AGENT_PREFIX}-")
        assert first_agent != second_agent

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
        with patch("skillspector.providers._agent_cli.secrets.token_hex", return_value="a" * 32):
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
            mock_run.side_effect = _probe_results()
            ok, _reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is True
        assert mock_run.call_count == 2

    def test_missing_binary_is_fail_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: None)
        with patch("skillspector.providers._agent_cli.subprocess.run") as mock_run:
            ok, reason = _agent_cli.is_available("opencode")
            mock_run.assert_not_called()
        assert ok is False
        assert reason

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_success(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = _probe_results()
        assert _opencode_auth_check(OPENCODE_BINARY) == (True, None)

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_uses_auth_list(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = _probe_results()
        _opencode_auth_check(OPENCODE_BINARY)
        assert mock_run.call_args[0][0][:3] == [OPENCODE_BINARY, "auth", "list"]

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_uses_scrubbed_environment(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = _probe_results()
        _opencode_auth_check(OPENCODE_BINARY)
        env = mock_run.call_args[1].get("env")
        assert env["OPENCODE_PERMISSION"] == _OPENCODE_DENY_ALL
        assert env["OPENCODE_PURE"] == "1"
        assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["share"] == "disabled"
        # The 1.x probe keeps OPENCODE_DISABLE_PROJECT_CONFIG: the mocked
        # binary is not installed, so the version probe cannot tell the
        # version and the environment keeps the 1.x shape.
        assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_shell_is_false_and_bounded(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = _probe_results()
        _opencode_auth_check(OPENCODE_BINARY)
        kwargs = mock_run.call_args[1]
        assert kwargs.get("shell") is False
        assert isinstance(kwargs.get("timeout"), int | float)
        assert kwargs["timeout"] <= 15

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_nonzero_exit_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _ok_result(_VERSION_OK),
            SimpleNamespace(returncode=1, stdout=b"", stderr=b"boom"),
        ]
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert reason

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_rejects_unverified_version(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ok_result(b"1.18.99\n")
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert "1.18.31" in (reason or "") and "1.18.34" in (reason or "")
        mock_run.assert_called_once()

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_timeout_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="opencode", timeout=15)
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert reason

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_no_credentials_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = _probe_results(_AUTH_LIST_EMPTY)
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert "auth login" in (reason or "")

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_accepts_singular_counts(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = _probe_results(_AUTH_LIST_SINGULAR)
        assert _opencode_auth_check(OPENCODE_BINARY) == (True, None)

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_unparseable_output_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = _probe_results(_AUTH_LIST_UNPARSEABLE)
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert "auth login" in (reason or "")


class TestOpencodeVersionRange:
    def test_supported_versions_are_an_explicit_verified_set(self) -> None:
        assert _OPENCODE_SUPPORTED_VERSIONS == frozenset(
            {
                (1, 18, 31),
                (1, 18, 32),
                (1, 18, 33),
                (1, 18, 34),
                (1, 18, 35),
                (1, 19, 0),
                (1, 19, 1),
                (1, 19, 2),
                (1, 19, 3),
                (1, 19, 4),
            }
        )

    @pytest.mark.parametrize(
        "version",
        [
            "1.18.31",
            "1.18.32",
            "1.18.33",
            "1.18.34",
            "1.18.35",
            "1.19.0",
            "1.19.1",
            "1.19.2",
            "1.19.3",
            "1.19.4",
        ],
    )
    def test_parse_supported_versions_compare_in_range(self, version: str) -> None:
        parsed = _parse_opencode_version(f"{version}\n".encode())
        assert parsed is not None
        assert _is_supported_opencode_version(parsed)

    @pytest.mark.parametrize(
        "raw", [b"1.18.30\n", b"1.18.36\n", b"1.19.5\n", b"1.18.34-1\n", b"garbage\n", b"\n"]
    )
    def test_parse_unsupported_versions_rejected(self, raw: bytes) -> None:
        parsed = _parse_opencode_version(raw)
        assert not _is_supported_opencode_version(parsed)

    @pytest.mark.parametrize(
        "version",
        [
            "1.18.31",
            "1.18.32",
            "1.18.34",
            "1.18.35",
            "1.19.0",
            "1.19.1",
            "1.19.2",
            "1.19.3",
            "1.19.4",
        ],
    )
    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_accepts_range_versions(self, mock_run: MagicMock, version: str) -> None:
        mock_run.side_effect = [_ok_result(f"{version}\n".encode()), _ok_result()]
        assert _opencode_auth_check(OPENCODE_BINARY) == (True, None)

    @pytest.mark.parametrize("version", ["1.18.30", "1.18.36", "1.19.5"])
    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_rejects_out_of_range_versions(self, mock_run: MagicMock, version: str) -> None:
        mock_run.return_value = _ok_result(f"{version}\n".encode())
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert "1.18.31" in (reason or "") and "1.19.4" in (reason or "")
        mock_run.assert_called_once()


class TestOpencodeVersionRangeV2:
    def test_supported_v2_versions_are_an_explicit_verified_set(self) -> None:
        assert _OPENCODE_SUPPORTED_VERSIONS_V2 == frozenset({(2, 2, 6), (2, 2, 7)})

    @pytest.mark.parametrize("version", ["2.2.6", "2.2.7"])
    def test_parse_supported_v2_versions(self, version: str) -> None:
        parsed = _parse_opencode_version(f"{version}\n".encode())
        assert parsed is not None
        assert _is_supported_opencode_version_v2(parsed)
        assert not _is_supported_opencode_version(parsed)

    @pytest.mark.parametrize("raw", [b"2.2.5\n", b"2.2.8\n", b"2.3.0\n", b"2.2.6-1\n"])
    def test_parse_unsupported_v2_versions_rejected(self, raw: bytes) -> None:
        parsed = _parse_opencode_version(raw)
        assert not _is_supported_opencode_version_v2(parsed)

    def _v2_probe_results(self, auth_stdout: bytes) -> list[SimpleNamespace]:
        # The --version probe in _build_opencode_argv uses Popen directly,
        # so the mocked subprocess.run calls are the version check and
        # auth list --format json only.
        return [
            _ok_result(b"2.2.6\n"),
            _ok_result(auth_stdout),
        ]

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_accepts_v2_with_connections(self, mock_run: MagicMock) -> None:
        auth_json = (
            b'[{"id": "anthropic", "name": "Anthropic", "connections": '
            b'[{"type": "credential", "label": "API key"}]}]'
        )
        mock_run.side_effect = self._v2_probe_results(auth_json)
        assert _opencode_auth_check(OPENCODE_BINARY) == (True, None)

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_uses_auth_list_format_json_on_v2(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = self._v2_probe_results(b"[]")
        _opencode_auth_check(OPENCODE_BINARY)
        assert mock_run.call_args[0][0] == [
            OPENCODE_BINARY,
            "auth",
            "list",
            "--standalone",
            "--format",
            "json",
        ]

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_empty_v2_auth_list_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = self._v2_probe_results(b"[]")
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert "auth login" in (reason or "")

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_unparseable_v2_auth_list_is_fail_closed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = self._v2_probe_results(b"not json")
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert "auth login" in (reason or "")

    @patch("skillspector.providers._agent_cli.subprocess.run")
    def test_probe_rejects_unverified_v2_version(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ok_result(b"2.2.8\n")
        ok, reason = _opencode_auth_check(OPENCODE_BINARY)
        assert ok is False
        assert "2.2.6" in (reason or "") and "2.2.7" in (reason or "")
        mock_run.assert_called_once()


class TestParseOpencodeDebugAgents:
    # Resolved permissions for the probe agent measured live on 2.2.6/2.2.7
    # (#1084): defaults, then the agent-local deny, then the top-level deny,
    # then the remaining default denies.
    MEASURED_PERMS = [
        {"action": "*", "resource": "*", "effect": "allow"},
        {"action": "read", "resource": "*.env", "effect": "ask"},
        {"action": "read", "resource": "*.env.*", "effect": "ask"},
        {"action": "read", "resource": "*.env.example", "effect": "allow"},
        {"action": "*", "resource": "*", "effect": "deny"},
        {"action": "*", "resource": "*", "effect": "deny"},
        {"action": "browser", "resource": "*", "effect": "deny"},
        {"action": "skill", "resource": "browser", "effect": "deny"},
    ]

    def _agents_json(self, agent_name: str, perms: list[dict] | None) -> bytes:
        agents = [
            {"id": "build", "name": "Build", "mode": "primary", "permissions": []},
            {"id": agent_name, "name": agent_name, "mode": "primary", "permissions": perms},
        ]
        return json.dumps(agents).encode()

    def test_accepts_measured_deny_ending_list(self) -> None:
        raw = self._agents_json("ssprobe", self.MEASURED_PERMS)
        assert _parse_opencode_debug_agents(raw, "ssprobe") is None

    def test_accepts_single_wildcard_deny(self) -> None:
        raw = self._agents_json("ssprobe", [{"action": "*", "resource": "*", "effect": "deny"}])
        assert _parse_opencode_debug_agents(raw, "ssprobe") is None

    def test_rejects_missing_agent(self) -> None:
        raw = self._agents_json("ssprobe", self.MEASURED_PERMS)
        with pytest.raises(AgentCLIError, match="could not find the selected agent"):
            _parse_opencode_debug_agents(raw, "other-agent")

    def test_rejects_trailing_allow(self) -> None:
        perms = self.MEASURED_PERMS + [{"action": "read", "resource": "*.env", "effect": "allow"}]
        with pytest.raises(AgentCLIError, match="non-deny rule after the final wildcard deny"):
            _parse_opencode_debug_agents(self._agents_json("ssprobe", perms), "ssprobe")

    def test_rejects_trailing_ask(self) -> None:
        perms = self.MEASURED_PERMS + [{"action": "read", "resource": "*.env", "effect": "ask"}]
        with pytest.raises(AgentCLIError, match="non-deny rule after the final wildcard deny"):
            _parse_opencode_debug_agents(self._agents_json("ssprobe", perms), "ssprobe")

    def test_rejects_last_wildcard_allow(self) -> None:
        perms = [dict(rule) for rule in self.MEASURED_PERMS]
        perms[5] = {"action": "*", "resource": "*", "effect": "allow"}
        with pytest.raises(AgentCLIError, match="last wildcard rule is not a deny"):
            _parse_opencode_debug_agents(self._agents_json("ssprobe", perms), "ssprobe")

    def test_rejects_malformed_json(self) -> None:
        with pytest.raises(AgentCLIError, match="malformed JSON"):
            _parse_opencode_debug_agents(b"not json", "ssprobe")

    def test_rejects_non_list_output(self) -> None:
        with pytest.raises(AgentCLIError, match="non-list agent set"):
            _parse_opencode_debug_agents(b'{"id": "ssprobe"}', "ssprobe")

    def test_rejects_missing_permissions(self) -> None:
        raw = self._agents_json("ssprobe", None)
        with pytest.raises(AgentCLIError, match="no resolved permissions"):
            _parse_opencode_debug_agents(raw, "ssprobe")

    def test_rejects_no_wildcard_rule(self) -> None:
        perms = [{"action": "read", "resource": "*.env", "effect": "deny"}]
        with pytest.raises(AgentCLIError, match="no wildcard rule"):
            _parse_opencode_debug_agents(self._agents_json("ssprobe", perms), "ssprobe")

    def test_rejects_malformed_rule(self) -> None:
        perms = [{"action": "*", "resource": "*"}]
        with pytest.raises(AgentCLIError, match="malformed permission rule"):
            _parse_opencode_debug_agents(self._agents_json("ssprobe", perms), "ssprobe")


class TestOpencodeConfig2x:
    def test_project_policy_uses_agents_map_and_deny_lists(self) -> None:
        config = json.loads(_opencode_config_2x("skillspector-deny-all-abc"))
        assert "agent" not in config
        agent = config["agents"]["skillspector-deny-all-abc"]
        assert agent["mode"] == "primary"
        assert agent["permissions"] == [{"action": "*", "resource": "*", "effect": "deny"}]
        assert config["permissions"] == [{"action": "*", "resource": "*", "effect": "deny"}]


class TestOpencodeDenyAllPolicy:
    def test_policy_overrides_hostile_ambient_configuration(self, tmp_path: Path) -> None:
        base = {
            "PATH": os.environ.get("PATH", ""),
            "OPENCODE_AUTO_SHARE": "1",
            "OPENCODE_CONFIG_CONTENT": '{"permission":"allow","share":"auto"}',
            "OPENCODE_PERMISSION": '{"*":"allow"}',
            "OPENCODE_EXPERIMENTAL": "1",
            # An ambient value must not switch the OAuth login plugins back off (#809).
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
        }

        argv = _build_opencode_argv(OPENCODE_BINARY, "", 0)
        agent_name = argv[argv.index("--agent") + 1]
        env = _prepare_opencode_env(base, str(tmp_path), argv)
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])

        assert json.loads(env["OPENCODE_PERMISSION"]) == {"*": "deny"}
        assert config["permission"] == {"*": "deny"}
        assert config["agent"][agent_name]["permission"] == {"*": "deny"}
        assert config["default_agent"] == agent_name
        assert config["share"] == "disabled"
        assert config["autoshare"] is False
        assert env["OPENCODE_AUTO_SHARE"] == "0"
        # The version probe cannot see a binary here (OPENCODE_BINARY is
        # not installed), so the environment keeps the 1.x shape and the
        # flag stays set: on 1.x it is the only thing stopping the
        # ancestor configuration walk (verified live on real 1.19.4). On
        # 2.x it must stay unset so the project opencode.jsonc policy
        # is delivered (see TestOpencodeProjectConfigFlagVersionSpecific).
        assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
        # Internal plugins carry the OAuth logins (ChatGPT, Copilot, ...) and
        # register no tools; disabling them breaks the user's login (#809).
        # External plugins stay off through OPENCODE_PURE below.
        assert "OPENCODE_DISABLE_DEFAULT_PLUGINS" not in env
        assert env["OPENCODE_DISABLE_EXTERNAL_SKILLS"] == "1"
        assert env["OPENCODE_DISABLE_CLAUDE_CODE"] == "1"
        assert env["OPENCODE_PURE"] == "1"
        assert Path(env["OPENCODE_CONFIG_DIR"]).is_relative_to(tmp_path)
        assert Path(env["OPENCODE_DB"]).is_relative_to(tmp_path)
        assert Path(env["OPENCODE_TEST_MANAGED_CONFIG_DIR"]).is_relative_to(tmp_path)

    @staticmethod
    def _write_fake_opencode(
        binary: Path,
        markers: Path,
        *,
        managed: dict | None = None,
        version: str = "1.18.34",
    ) -> None:
        """Write a host simulator with real version/config/run boundaries."""
        binary.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import os
                import sys
                from pathlib import Path

                if sys.argv[1:] == ["--version"]:
                    print({version!r})
                    raise SystemExit(0)

                config = json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])
                managed = {managed or {}!r}
                config.setdefault("agent", {{}}).update(managed.get("agent", {{}}))
                for key, value in managed.items():
                    if key != "agent":
                        config[key] = value

                if sys.argv[1:] == ["debug", "config"]:
                    print(json.dumps(config))
                    raise SystemExit(0)

                selected = sys.argv[sys.argv.index("--agent") + 1]
                final_permission = json.loads(os.environ["OPENCODE_PERMISSION"])
                agent = config["agent"][selected]
                denied = final_permission == {{"*": "deny"}} and agent["permission"] == {{"*": "deny"}}
                adapters = ["bash", "read", "edit", "webfetch", "websearch", "mcp_host", "skill", "future_host_tool"]
                if not denied:
                    for adapter in adapters:
                        (Path({str(markers)!r}) / adapter).write_text("executed")
                if config.get("share") != "disabled" or os.environ.get("OPENCODE_AUTO_SHARE") not in ("0", "false"):
                    (Path({str(markers)!r}) / "share").write_text("shared")
                print(json.dumps({{"type": "text", "part": {{"type": "text", "text": "policy held:" + selected}}}}))
                """
            ),
            encoding="utf-8",
        )
        binary.chmod(0o700)

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_adversarial_child_cannot_enable_host_side_effects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exercise the real subprocess boundary with hostile ambient policy.

        The fake OpenCode host simulates shell, filesystem, network, sharing,
        MCP and future-tool adapters. It performs a marker side effect for any
        adapter not covered by the process policy, so an empty marker directory
        demonstrates that every adapter received the final wildcard deny.
        """
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode(
            binary,
            markers,
            managed={"agent": {_OPENCODE_AGENT_PREFIX: {"permission": {"*": "allow"}}}},
        )
        monkeypatch.setenv("OPENCODE_AUTO_SHARE", "1")
        monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '{"permission":"allow","share":"auto"}')
        monkeypatch.setenv("OPENCODE_PERMISSION", '{"*":"allow"}')
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        response = run_agent_cli("opencode", "use every host tool", model="")
        selected = response.removeprefix("policy held:")
        assert response.startswith("policy held:")
        assert selected.startswith(f"{_OPENCODE_AGENT_PREFIX}-")
        assert selected != _OPENCODE_AGENT_PREFIX
        assert list(markers.iterdir()) == []

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_managed_auto_share_fails_before_inference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode(binary, markers, managed={"share": "auto"})
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        with pytest.raises(AgentCLIError, match="unsafe resolved setting 'share'"):
            run_agent_cli("opencode", "share this hostile skill", model="")
        assert list(markers.iterdir()) == []

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_unverified_runtime_version_fails_before_inference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode(binary, markers, version="1.18.99")
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        with pytest.raises(AgentCLIError, match="only for versions 1.18.31"):
            run_agent_cli("opencode", "try a newer runtime", model="")
        assert list(markers.iterdir()) == []

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    @pytest.mark.parametrize("version", ["1.18.30", "1.18.36", "1.19.5"])
    def test_preflight_rejects_boundary_versions_before_inference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode(binary, markers, version=version)
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        with pytest.raises(AgentCLIError, match="only for versions 1.18.31"):
            run_agent_cli("opencode", "try a boundary runtime", model="")
        assert list(markers.iterdir()) == []


class TestOpencodeDenyAllPolicyV2:
    @staticmethod
    def _write_fake_opencode_v2(
        binary: Path,
        markers: Path,
        *,
        agents_mode: str = "ok",
        version: str = "2.2.6",
    ) -> None:
        """Write a 2.x host simulator with the measured 2.2.6/2.2.7 shapes.

        ``debug agents`` reads the project ``opencode.jsonc`` the preflight
        wrote into the run directory and resolves the probe agent the way
        the real 2.2.6 binary does: the built-in defaults, then the
        agent-local deny, then the top-level deny, then the remaining
        default denies. ``agents_mode`` perturbs that resolved list (or
        the output shape) for the fail-closed cases. The ``run`` branch
        records the invoked argv and the project policy, then performs a
        marker side effect for any adapter not covered by the project
        policy, mirroring the 1.x fake.
        """
        binary.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import os
                import sys
                from pathlib import Path

                if sys.argv[1:] == ["--version"]:
                    print({version!r})
                    raise SystemExit(0)

                markers = Path({str(markers)!r})
                defaults = [
                    {{"action": "*", "resource": "*", "effect": "allow"}},
                    {{"action": "read", "resource": "*.env", "effect": "ask"}},
                    {{"action": "read", "resource": "*.env.*", "effect": "ask"}},
                    {{"action": "read", "resource": "*.env.example", "effect": "allow"}},
                    {{"action": "question", "resource": "*", "effect": "allow"}},
                    {{"action": "browser", "resource": "*", "effect": "deny"}},
                    {{"action": "skill", "resource": "browser", "effect": "deny"}},
                ]

                def probe_agent():
                    project = Path.cwd() / "opencode.jsonc"
                    if not project.exists():
                        return None
                    config = json.loads(project.read_text())
                    agents = config.get("agents", {{}})
                    if not agents:
                        return None
                    name = next(iter(agents))
                    agent_perms = agents[name].get("permissions", [])
                    top_perms = config.get("permissions", [])
                    resolved = defaults[:4] + agent_perms + top_perms + defaults[5:]
                    return {{"id": name, "name": name, "mode": "primary", "permissions": resolved}}

                def probe_plugins():
                    # The real 2.2.7 loads a configured system plugin (a
                    # /etc/opencode/plugins directory plugin) when the
                    # project policy does not deny it: `plugins: ["-*"]`
                    # alone stops activation but not the import, so the
                    # module's top-level code still runs in the server.
                    # A `plugin.use` deny-all policy stops the load
                    # entirely. Fail the probe when it is missing, the
                    # way the preflight must fail closed.
                    project = Path.cwd() / "opencode.jsonc"
                    if not project.exists():
                        return False
                    config = json.loads(project.read_text())
                    return any(
                        policy.get("action") == "plugin.use"
                        and policy.get("resource") == "*"
                        and policy.get("effect") == "deny"
                        for policy in config.get("policies", [])
                    )

                if sys.argv[1:] == ["debug", "config"]:
                    print(json.dumps([{{"type": "directory", "path": str(Path.cwd())}}]))
                    raise SystemExit(0)

                if sys.argv[1:3] == ["debug", "agents"]:
                    mode = {agents_mode!r}
                    if "--standalone" in sys.argv:
                        (markers / "agents-standalone").write_text("standalone")
                    if not probe_plugins():
                        print("a system plugin would load: no plugin.use deny-all policy", file=sys.stderr)
                        raise SystemExit(1)
                    probe = probe_agent()
                    if mode == "garbage":
                        print("not json")
                        raise SystemExit(0)
                    if mode == "non-list":
                        print(json.dumps({{"id": "build"}}))
                        raise SystemExit(0)
                    agents = [
                        {{"id": "build", "name": "Build", "mode": "primary", "permissions": defaults}},
                    ]
                    if probe is not None and mode != "missing":
                        resolved = list(probe["permissions"])
                        if mode == "trailing-allow":
                            resolved.append({{"action": "read", "resource": "*.env", "effect": "allow"}})
                        elif mode == "trailing-ask":
                            resolved.append({{"action": "read", "resource": "*.env", "effect": "ask"}})
                        elif mode == "last-wildcard-allow":
                            for rule in reversed(resolved):
                                if rule["action"] == "*" and rule["resource"] == "*":
                                    rule["effect"] = "allow"
                                    break
                        elif mode == "no-wildcard":
                            resolved = [
                                rule
                                for rule in resolved
                                if not (rule["action"] == "*" and rule["resource"] == "*")
                            ]
                        probe["permissions"] = resolved
                        agents.append(probe)
                    print(json.dumps(agents))
                    raise SystemExit(0)

                selected = sys.argv[sys.argv.index("--agent") + 1]
                (markers / "argv").write_text(json.dumps(sys.argv[1:]))
                (markers / "env-flag").write_text(
                    os.environ.get("OPENCODE_DISABLE_PROJECT_CONFIG", "absent")
                )
                project = Path.cwd() / "opencode.jsonc"
                (markers / "jsonc").write_text(project.read_text() if project.exists() else "")
                config = json.loads(project.read_text()) if project.exists() else {{}}
                agent = config.get("agents", {{}}).get(selected, {{}})
                denied = any(
                    perm.get("action") == "*"
                    and perm.get("resource") == "*"
                    and perm.get("effect") == "deny"
                    for perm in agent.get("permissions", [])
                )
                adapters = ["bash", "read", "edit", "webfetch", "websearch", "mcp_host", "skill", "future_host_tool"]
                if not denied:
                    for adapter in adapters:
                        (markers / adapter).write_text("executed")
                print(json.dumps({{"type": "text", "part": {{"type": "text", "text": "policy held:" + selected}}}}))
                """
            ),
            encoding="utf-8",
        )
        binary.chmod(0o700)

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_2x_preflight_accepts_measured_permissions_and_run_holds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode_v2(binary, markers)
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        response = run_agent_cli("opencode", "use every host tool", model="")
        selected = response.removeprefix("policy held:")
        assert response.startswith("policy held:")
        assert selected.startswith(f"{_OPENCODE_AGENT_PREFIX}-")
        assert sorted(path.name for path in markers.iterdir()) == [
            "agents-standalone",
            "argv",
            "env-flag",
            "jsonc",
        ]
        argv = json.loads((markers / "argv").read_text())
        assert "--pure" not in argv
        assert "--standalone" in argv
        assert argv[argv.index("--agent") + 1] == selected
        config = json.loads((markers / "jsonc").read_text())
        assert config["agents"][selected]["permissions"] == [
            {"action": "*", "resource": "*", "effect": "deny"}
        ]
        assert config["permissions"] == [{"action": "*", "resource": "*", "effect": "deny"}]

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_2x_project_config_denies_all_plugins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Finding 1 (#1084): a configured system plugin (/etc/opencode/plugins)
        # loads into the 2.x server unless the project policy denies it. The
        # preflight writes that deny into the project opencode.jsonc, and the
        # fake refuses the probe without it, so removing the policy fails
        # this test the way the real binary would load the plugin.
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode_v2(binary, markers)
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        response = run_agent_cli("opencode", "use every host tool", model="")
        assert response.startswith("policy held:")
        config = json.loads((markers / "jsonc").read_text())
        assert {"action": "plugin.use", "resource": "*", "effect": "deny"} in config.get(
            "policies", []
        )

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_2x_preflight_accepts_2_2_7(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode_v2(binary, markers, version="2.2.7")
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        response = run_agent_cli("opencode", "use every host tool", model="")
        assert response.startswith("policy held:")
        assert sorted(path.name for path in markers.iterdir()) == [
            "agents-standalone",
            "argv",
            "env-flag",
            "jsonc",
        ]

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_2x_argv_omits_pure_for_2x_binary(self, tmp_path: Path) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode_v2(binary, markers)
        argv = _build_opencode_argv(str(binary), "", 0)
        assert "--pure" not in argv
        assert "--standalone" in argv
        assert "--agent" in argv

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    @pytest.mark.parametrize(
        "agents_mode,match",
        [
            ("missing", "could not find the selected agent"),
            ("trailing-allow", "non-deny rule after the final wildcard deny"),
            ("trailing-ask", "non-deny rule after the final wildcard deny"),
            ("last-wildcard-allow", "last wildcard rule is not a deny"),
            ("garbage", "malformed JSON"),
            ("non-list", "non-list agent set"),
            ("no-wildcard", "no wildcard rule"),
        ],
    )
    def test_2x_preflight_rejects_bad_resolved_policy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        agents_mode: str,
        match: str,
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode_v2(binary, markers, agents_mode=agents_mode)
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        with pytest.raises(AgentCLIError, match=match):
            run_agent_cli("opencode", "try a hostile runtime", model="")
        # The probe ran (with --standalone) and failed closed; the run
        # never started, so no run markers exist.
        assert sorted(path.name for path in markers.iterdir()) == ["agents-standalone"]

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_2x_rejects_unverified_version_before_inference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode_v2(binary, markers, version="2.2.8")
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        with pytest.raises(AgentCLIError, match="only for versions 1.18.31") as excinfo:
            run_agent_cli("opencode", "try a newer runtime", model="")
        assert "2.2.7" in str(excinfo.value)
        assert list(markers.iterdir()) == []


# ---------------------------------------------------------------------------
# Review round 2 (#1086): version-specific OPENCODE_DISABLE_PROJECT_CONFIG
# and an isolated 2.x auth probe. Both findings were verified live:
# real 1.19.4 writes a .gitignore into an ancestor .opencode project when
# the flag is absent, and real 2.2.7 accepts `auth list --standalone`.
# ---------------------------------------------------------------------------


class TestOpencodeProjectConfigFlagVersionSpecific:
    @pytest.mark.parametrize(
        "version,flag_expected",
        [
            ((1, 18, 34), "1"),
            ((1, 19, 4), "1"),
            ((2, 2, 6), None),
            ((2, 2, 7), None),
            (None, "1"),
        ],
    )
    def test_prepare_env_flag_is_version_specific(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        version: tuple[int, int, int] | None,
        flag_expected: str | None,
    ) -> None:
        # The 2.x policy-delivery change must not leak into 1.x: the flag
        # stays set for every 1.x runtime (and for an unknowable version,
        # which keeps the 1.x shape) and is omitted only on 2.x.
        monkeypatch.setattr(_agent_cli, "_probe_opencode_version", lambda _binary: version)
        argv = _build_opencode_argv(OPENCODE_BINARY, "", 0)
        env = _prepare_opencode_env({"PATH": os.environ.get("PATH", "")}, str(tmp_path), argv)
        assert env.get("OPENCODE_DISABLE_PROJECT_CONFIG") == flag_expected


class TestOpencodeAuthProbeIsolation:
    @staticmethod
    def _write_fake_auth_2x(binary: Path, markers: Path) -> None:
        """Fake modelling the 2.x auth probe boundary.

        ``auth list`` fails when ``--standalone`` is absent (an ordinary
        service occupies the fixed port) and when the working directory
        holds no project policy denying ``plugin.use`` (a configured
        project/system plugin would load). With both present it prints
        the 2.x JSON shape with one connection.
        """
        binary.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import os
                import sys
                from pathlib import Path

                markers = Path({str(markers)!r})

                if sys.argv[1:] == ["--version"]:
                    print("2.2.7")
                    raise SystemExit(0)

                if sys.argv[1:3] == ["auth", "list"]:
                    (markers / "argv").write_text(json.dumps(sys.argv[1:]))
                    (markers / "cwd").write_text(os.getcwd())
                    (markers / "env-flag").write_text(
                        os.environ.get("OPENCODE_DISABLE_PROJECT_CONFIG", "absent")
                    )
                    project = Path.cwd() / "opencode.jsonc"
                    (markers / "policy").write_text(
                        project.read_text() if project.exists() else ""
                    )
                    if "--standalone" not in sys.argv:
                        print("background service already occupies the fixed port", file=sys.stderr)
                        raise SystemExit(1)
                    if not project.exists():
                        print("no project policy: a configured plugin would load", file=sys.stderr)
                        raise SystemExit(1)
                    config = json.loads(project.read_text())
                    denied = any(
                        policy.get("action") == "plugin.use"
                        and policy.get("resource") == "*"
                        and policy.get("effect") == "deny"
                        for policy in config.get("policies", [])
                    )
                    if not denied:
                        print("no plugin.use deny: a configured plugin would load", file=sys.stderr)
                        raise SystemExit(1)
                    print(json.dumps([{{"id": "anthropic", "connections": [{{"type": "credential"}}]}}]))
                    raise SystemExit(0)

                raise SystemExit(2)
                """
            ),
            encoding="utf-8",
        )
        binary.chmod(0o700)

    @staticmethod
    def _write_fake_auth_1x(binary: Path, markers: Path) -> None:
        """Fake modelling the 1.x auth probe boundary (counts output)."""
        binary.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import os
                import sys
                from pathlib import Path

                markers = Path({str(markers)!r})

                if sys.argv[1:] == ["--version"]:
                    print("1.18.34")
                    raise SystemExit(0)

                if sys.argv[1:] == ["auth", "list"]:
                    (markers / "argv").write_text(json.dumps(sys.argv[1:]))
                    (markers / "cwd").write_text(os.getcwd())
                    (markers / "env-flag").write_text(
                        os.environ.get("OPENCODE_DISABLE_PROJECT_CONFIG", "absent")
                    )
                    print("1 credential\\n1 environment variable")
                    raise SystemExit(0)

                raise SystemExit(2)
                """
            ),
            encoding="utf-8",
        )
        binary.chmod(0o700)

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_2x_auth_probe_uses_standalone_isolated_dir_policy_and_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_auth_2x(binary, markers)
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        created: list[str] = []
        real_mkdtemp = tempfile.mkdtemp

        def recording_mkdtemp(*args: object, **kwargs: object) -> str:
            path = real_mkdtemp(*args, **kwargs)  # type: ignore[arg-type]
            created.append(path)
            return path

        monkeypatch.setattr(tempfile, "mkdtemp", recording_mkdtemp)

        assert _opencode_auth_check(str(binary)) == (True, None)
        assert json.loads((markers / "argv").read_text()) == [
            "auth",
            "list",
            "--standalone",
            "--format",
            "json",
        ]
        cwd = (markers / "cwd").read_text()
        assert "skillspector_opencode_auth_" in cwd
        assert cwd != os.getcwd()
        policy = json.loads((markers / "policy").read_text())
        assert {"action": "plugin.use", "resource": "*", "effect": "deny"} in policy.get(
            "policies", []
        )
        assert (markers / "env-flag").read_text() == "absent"
        assert created, "auth probe must create its isolated temp dir"
        assert all(not os.path.exists(path) for path in created), "auth temp dir leaked"

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_2x_auth_probe_old_shape_fails_on_busy_port(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pins the regression the fix addresses: without --standalone the
        # probe resolves the default service, and an ordinary service
        # already occupying the fixed port makes availability fail even
        # with valid credentials.
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_auth_2x(binary, markers)
        (tmp_path / "opencode.jsonc").write_text(
            _opencode_config_2x("skillspector-deny-all-test"), encoding="utf-8"
        )

        old_argv = subprocess.run(
            [str(binary), "auth", "list", "--format", "json"],
            capture_output=True,
            cwd=tmp_path,
            timeout=30,
        )
        assert old_argv.returncode == 1
        assert b"fixed port" in old_argv.stderr

        new_argv = subprocess.run(
            [str(binary), "auth", "list", "--standalone", "--format", "json"],
            capture_output=True,
            cwd=tmp_path,
            timeout=30,
        )
        assert new_argv.returncode == 0
        assert len(json.loads(new_argv.stdout.decode())) == 1

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_1x_auth_probe_keeps_flag_and_isolated_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_auth_1x(binary, markers)
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        assert _opencode_auth_check(str(binary)) == (True, None)
        assert json.loads((markers / "argv").read_text()) == ["auth", "list"]
        assert "--standalone" not in json.loads((markers / "argv").read_text())
        cwd = (markers / "cwd").read_text()
        assert "skillspector_opencode_auth_" in cwd
        assert cwd != os.getcwd()
        assert (markers / "env-flag").read_text() == "1"


class TestOpencodeAncestorProject:
    @pytest.fixture()
    def project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        # A repository containing .opencode, with the run/auth temp dirs
        # placed beneath it: tempfile.tempdir is honoured by mkdtemp, so
        # pointing it at the nested dir reproduces TMPDIR beneath a repo.
        project = tmp_path / "project"
        (project / ".opencode").mkdir(parents=True)
        (project / ".opencode" / "opencode.json").write_text(
            json.dumps({"plugin": ["some-system-plugin"]}), encoding="utf-8"
        )
        nested = project / "nested"
        nested.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(nested))
        monkeypatch.setenv("TMPDIR", str(nested))
        return project

    @staticmethod
    def _write_fake_opencode_ancestor(binary: Path, markers: Path) -> None:
        """1.x fake simulating config initialisation in an ancestor project.

        With OPENCODE_DISABLE_PROJECT_CONFIG unset, a `debug config` or
        `auth list` invocation walks up from its cwd, finds the ancestor
        `.opencode` project, and writes a missing `.gitignore` there,
        the way real 1.19.4 does before the preflight finishes.
        """
        binary.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import os
                import sys
                from pathlib import Path

                def ancestor_opencode():
                    here = Path.cwd()
                    for parent in [here, *here.parents]:
                        if (parent / ".opencode").is_dir():
                            return parent / ".opencode"
                    return None

                def maybe_touch_ancestor():
                    if os.environ.get("OPENCODE_DISABLE_PROJECT_CONFIG") == "1":
                        return
                    found = ancestor_opencode()
                    if found is not None:
                        (found / ".gitignore").write_text("node_modules\\n")

                if sys.argv[1:] == ["--version"]:
                    print("1.18.34")
                    raise SystemExit(0)

                if sys.argv[1:] == ["debug", "config"]:
                    maybe_touch_ancestor()
                    print(json.dumps(json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])))
                    raise SystemExit(0)

                if sys.argv[1:] == ["auth", "list"]:
                    maybe_touch_ancestor()
                    print("1 credential\\n1 environment variable")
                    raise SystemExit(0)

                selected = sys.argv[sys.argv.index("--agent") + 1]
                print(json.dumps({{"type": "text", "part": {{"type": "text", "text": "policy held:" + selected}}}}))
                """
            ),
            encoding="utf-8",
        )
        binary.chmod(0o700)

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_1x_run_beneath_project_leaves_ancestor_untouched(
        self, tmp_path: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode_ancestor(binary, markers)
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        response = run_agent_cli("opencode", "use every host tool", model="")
        assert response.startswith("policy held:")
        assert not (project / ".opencode" / ".gitignore").exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_1x_auth_probe_beneath_project_leaves_ancestor_untouched(
        self, tmp_path: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        self._write_fake_opencode_ancestor(binary, markers)
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        assert _opencode_auth_check(str(binary)) == (True, None)
        assert not (project / ".opencode" / ".gitignore").exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="test helper uses a POSIX shebang")
    def test_2x_run_beneath_project_delivers_policy_and_omits_flag(
        self, tmp_path: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binary = tmp_path / "opencode"
        markers = tmp_path / "outside"
        markers.mkdir()
        TestOpencodeDenyAllPolicyV2._write_fake_opencode_v2(binary, markers)
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: str(binary))

        response = run_agent_cli("opencode", "use every host tool", model="")
        assert response.startswith("policy held:")
        assert (markers / "env-flag").read_text() == "absent"
        config = json.loads((markers / "jsonc").read_text())
        assert {"action": "plugin.use", "resource": "*", "effect": "deny"} in config.get(
            "policies", []
        )
        assert not (project / ".opencode" / ".gitignore").exists()


# ---------------------------------------------------------------------------
# _parse_opencode_output
# ---------------------------------------------------------------------------


# Real opencode 1.18.33 envelope shapes (Step-0 probe:
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
        raw = f"opencode v1.18.33\n{_TEXT_EVENT}\nattaching session...\n"
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


# ---------------------------------------------------------------------------
# Zen free-tier refusal mapping
# ---------------------------------------------------------------------------

# Verbatim envelope shape from a Zen 403 (free tier refused
# under the deny-all isolation; any permission deny trips it).
_ZEN_REFUSAL_ENVELOPE = (
    '{"type":"error","timestamp":1789784969134,"sessionID":"ses_probe",'
    '"error":{"name":"APIError","data":{"message":"Error from provider '
    "(Console): OpenCode's free tier can only be used from within OpenCode\","
    '"statusCode":403,"responseBody":"{\\"type\\":\\"error\\",\\"error\\":'
    '{\\"type\\":\\"FreeTierError\\"}"}}}}}'
)


class _FakePopen:
    """Minimal Popen stand-in for _run_bounded (cross-platform, no subprocess)."""

    def __init__(self, stdout: bytes, returncode: int) -> None:
        self.stdin: io.BytesIO | None = io.BytesIO()
        self.stdout: io.BytesIO | None = io.BytesIO(stdout)
        self.stderr: io.BytesIO | None = io.BytesIO(b"")
        self._returncode = returncode

    def wait(self, timeout: float | None = None) -> int:
        return self._returncode

    def kill(self) -> None:
        pass


class TestZenFreeTierRefusal:
    def _run_with_fake_host(
        self, monkeypatch: pytest.MonkeyPatch, stdout: bytes, returncode: int
    ) -> str:
        spec = replace(get_spec("opencode"), preflight=None)
        monkeypatch.setattr(_agent_cli, "get_spec", lambda _name: spec)
        monkeypatch.setattr(_agent_cli, "find_binary", lambda _name: "/usr/bin/opencode")
        monkeypatch.setattr(
            subprocess,
            "Popen",
            lambda *args, **kwargs: _FakePopen(stdout, returncode),
        )
        return run_agent_cli("opencode", "probe", model="opencode/nemotron-3-ultra-free")

    def test_free_tier_refusal_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Zen FreeTierError envelope must fail closed with guidance — never
        the opaque exit-code message, and never a silent sandbox weakening."""
        with pytest.raises(AgentCLIError, match="free tier can only be used"):
            self._run_with_fake_host(monkeypatch, _ZEN_REFUSAL_ENVELOPE.encode(), 1)

    def test_plain_nonzero_exit_keeps_generic_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(AgentCLIError, match="exited with code 1"):
            self._run_with_fake_host(monkeypatch, b"boom", 1)
