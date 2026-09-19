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

"""Hardened subprocess helper for agent CLI providers (claude, codex, gemini,
opencode, and copilot).

This is the single security chokepoint for all agent-CLI calls. Per-CLI
knowledge (argv, output parsing, auth check) lives in a small ``CliSpec``
registry (see ``_REGISTRY`` / "HOW TO ADD A NEW AGENT CLI" below); the
security core is CLI-agnostic. Every call goes through :func:`run_agent_cli`
which enforces:

- **No shell**: ``shell=False`` with an explicit argv list.
- **Untrusted content via stdin only**: the prompt (which may contain
  adversarial skill content) is written to the process stdin, never
  injected into argv.
- **Capability stripping** (per-binary): tools disabled, MCP disabled,
  no extra directories, deny permission mode (claude); read-only sandbox
  (codex).  ``--dangerously-skip-permissions`` is NEVER used.
- **Environment scrubbing**: API keys, SSH keys, cloud credentials, and
  other secrets are stripped from the child environment.
- **Timeout enforcement**: the call raises ``TimeoutError`` rather than
  hanging indefinitely.
- **Input / output caps**: prompt exceeding ``MAX_INPUT_BYTES`` is
  rejected; stdout is capped at ``MAX_OUTPUT_BYTES``.
- **Fail-closed**: non-zero exit, timeout, missing binary, or bad
  output all raise ``AgentCLIError``.
- **Prompt-layer hardening**: the caller wraps untrusted content in
  clear DATA delimiters before passing it here (defense-in-depth on top
  of capability removal).

The JSON output envelope (``claude -p --output-format json``) is parsed
and the assistant text is returned.  ``codex exec --json`` produces
JSONL events; the last assistant message is extracted.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from skillspector.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Static analyzers stop at a one-million-character decoded text limit.
# The CLI prompt path separately caps encoded UTF-8 bytes.
MAX_INPUT_BYTES = 1_000_000  # 1 MB encoded prompt cap
MAX_OUTPUT_BYTES = 10_000_000  # 10 MB safety cap on stdout
MAX_STDERR_BYTES = 64_000  # stderr is only used for error snippets
CLI_TIMEOUT_SECONDS = 300  # 5-minute per-call hard limit

# Environment variables that must NOT be forwarded to child processes.
# Includes API keys, cloud creds, SSH agent, and SkillSpector's own keys.
_SECRET_ENV_PREFIXES: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_PROXY_API_KEY",
    "OPENAI_API_KEY",
    "SKILLSPECTOR_API_KEYS",
    "SKILLSPECTOR_COMPAT_API_KEY",
    "NVIDIA_INFERENCE_KEY",
    "NVIDIA_INFERENCE_METADATA_KEY",
    "AWS_",
    "AZURE_",
    "GOOGLE_",
    "GCLOUD_",
    "GCP_",
    "SSH_",
    "GPG_",
    "GITHUB_TOKEN",
    "GITLAB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_TOKEN",
    "COHERE_API_KEY",
    "REPLICATE_API_TOKEN",
    "MISTRAL_API_KEY",
    "TOGETHER_API_KEY",
    "GROQ_API_KEY",
    "FIREWORKS_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
)


class AgentCLIError(RuntimeError):
    """Raised when an agent CLI call fails for any reason (fail-closed)."""


# ---------------------------------------------------------------------------
# Environment scrubbing
# ---------------------------------------------------------------------------


def _scrub_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with secret variables removed.

    Any variable whose name starts with a prefix in ``_SECRET_ENV_PREFIXES``
    is stripped.  The resulting environment is passed to the subprocess.
    """
    clean: dict[str, str] = {}
    for key, val in os.environ.items():
        upper = key.upper()
        if any(upper.startswith(p.upper()) for p in _SECRET_ENV_PREFIXES):
            continue
        clean[key] = val
    return clean


# ---------------------------------------------------------------------------
# Binary lookup
# ---------------------------------------------------------------------------


def find_binary(name: str) -> str | None:
    """Return the absolute path of *name* on PATH, or ``None`` if absent."""
    return shutil.which(name)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _validate_model_label(model: str) -> str:
    """Ensure *model* cannot be used as an argument injection vector.

    Model labels come from ``SKILLSPECTOR_MODEL`` (user-controlled) or the
    provider's defaults.  We verify the label does not start with ``-``
    (which would look like a flag to the CLI) and contains only safe
    characters.

    Raises:
        AgentCLIError: when the label fails validation.
    """
    if not model:
        raise AgentCLIError("model label must be a non-empty string")
    if model.startswith("-"):
        raise AgentCLIError(
            f"model label {model!r} starts with '-'; this looks like an argument injection attempt"
        )
    # Allow alphanumeric, dash, dot, slash, colon, underscore (covers all
    # known claude/codex model identifiers).
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-./: _")
    bad = [c for c in model if c not in allowed]
    if bad:
        raise AgentCLIError(f"model label {model!r} contains disallowed characters: {bad!r}")
    return model


# ---------------------------------------------------------------------------
# Claude CLI invocation
# ---------------------------------------------------------------------------


def _build_claude_argv(binary: str, model: str, max_output_tokens: int) -> list[str]:
    """Build the argv list for a capability-stripped ``claude -p`` call.

    ``-p`` / ``--print``
        Non-interactive single-shot mode. The prompt is read from stdin;
        the response is written to stdout and the process exits.

    ``--output-format text``
        Emit the assistant's response as plain text — nothing else.  This is
        the most stable format the claude CLI offers: it has been the canonical
        headless contract since ``-p`` was introduced, predates the JSON
        envelope formats, and is unaffected by changes to the event-stream
        schema.  The envelope formats (``json`` / ``stream-json``) have changed
        shape across builds (single dict → JSON array → JSONL); ``text`` never
        has.  Because we need only the response text and not the metadata
        (session ID, stop reason, etc.) that the envelope carries, ``text`` is
        the right choice here: the format we request defines exactly what we
        parse, with no version detection and no fallbacks.

    ``--model <label>``
        Use the requested model. ``--model`` is a known flag, so the label
        cannot be placed after ``--``; we validate it instead.

    ``--allowed-tools ""``
        Allow-list with NO entries = deny by default. This is the primary
        capability removal. An allow-list (not a deny-list) is used on
        purpose: any tool not explicitly allowed — including tools added in
        future Claude versions — is blocked. The value is our own fixed
        string; untrusted content never reaches argv.

    ``--permission-mode dontAsk``
        Backstop: any action the model attempts anyway is denied without
        prompting (a prompt would hang in non-interactive mode). ``dontAsk``
        is a valid mode (``claude`` rejects unknown modes).

    ``--strict-mcp-config``
        Use only MCP servers from ``--mcp-config`` — which we never pass — so
        zero MCP servers load. (Note: ``--no-mcp-config`` is NOT a real flag.)

    ``--setting-sources=``
        Load no user, project, or local settings, preventing their hooks from
        running in the spawned Claude CLI process. This also means filesystem
        model settings do not flow into the child unless SkillSpector pins a
        model explicitly.

    ``--disable-slash-commands``
        Prevents skill/plugin invocations from within the sandboxed call.

    Deliberately NOT included:
    - ``--dangerously-skip-permissions`` / ``--allow-dangerously-skip-permissions``
      — explicitly forbidden.
    - ``--bare`` — it skips keychain reads, which breaks authentication
      ("Not logged in"); security comes from the allow-list + permission mode,
      not from ``--bare``.
    - ``--add-dir`` — no extra directory access needed.
    """
    # Forward --model ONLY when SKILLSPECTOR_MODEL is explicitly set; otherwise
    # omit it and let Claude fall back without loading user/project/local
    # settings from disk.
    model_arg = ["--model", _validate_model_label(model)] if model else []
    return [
        binary,
        "-p",
        "--output-format",
        "text",
        *model_arg,
        "--allowed-tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--strict-mcp-config",
        "--setting-sources=",
        "--disable-slash-commands",
    ]


def _parse_claude_output(raw: str) -> str:
    """Return the assistant text from ``claude -p --output-format text`` stdout.

    With ``--output-format text`` the claude CLI writes only the response to
    stdout and nothing else, so no parsing is required: the contract is the
    format flag itself.  The only failure case is an empty response (which
    indicates an auth failure, rate-limit, or other non-zero-exit scenario
    that the caller's fail-closed checks should have already caught).

    Raises:
        AgentCLIError: when stdout is empty.
    """
    text = raw.strip()
    if not text:
        raise AgentCLIError("claude returned empty stdout; cannot extract assistant response")
    return text


# ---------------------------------------------------------------------------
# Codex CLI invocation
# ---------------------------------------------------------------------------


def _build_codex_argv(binary: str, model: str, max_output_tokens: int = 0) -> list[str]:
    """Build the argv list for a capability-stripped ``codex exec`` call.

    Flags chosen (verified end-to-end against codex 0.139.0):

    ``exec``
        Non-interactive subcommand. With NO positional prompt, codex reads the
        instructions from stdin — which is exactly where the runner pipes the
        prompt. (Passing ``-`` makes the prompt literally ``"-"`` and demotes
        the real content to a ``<stdin>`` block, so we do not pass it.)

    ``--json``
        Emit JSONL events to stdout, enabling structured parsing.

    ``--sandbox read-only``
        Most restrictive sandbox mode. Model-generated shell commands are
        restricted to read-only filesystem access; no code execution. Unlike
        claude/gemini (which block model tool use entirely), codex's strictest
        mode still permits read-only filesystem *reads* by model-generated
        commands. This is informational, not an exfil channel: the call runs in
        an isolated empty temp CWD, output returns only to the operator's own
        report, and there is no network egress path.

    ``--ephemeral``
        Do not persist session files to disk (no residue from the scan).

    ``--ignore-user-config``
        Ignore ``$CODEX_HOME/config.toml``; use only our explicit flags.

    ``--ignore-rules``
        Do not load user/project ``.rules`` files.

    ``--model <label>``
        Use the requested model.

    ``-m`` / ``--model`` label is validated via ``_validate_model_label``.
    """
    return [
        binary,
        "exec",
        "--json",
        "--sandbox",
        "read-only",
        # We run in an isolated empty temp dir (not a git repo); codex refuses
        # an "untrusted" dir without this. Safe: --sandbox read-only still bars
        # code execution, and the temp dir holds no project files.
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        # --model omitted by default -> codex uses the account's default model
        # (forwarded only when SKILLSPECTOR_MODEL is set).
        *(["--model", _validate_model_label(model)] if model else []),
    ]


def _parse_codex_output(raw: str) -> str:
    """Extract assistant text from ``codex exec --json`` JSONL output.

    Verified against codex 0.139.0, whose final message arrives nested::

        {"type": "item.completed", "item": {"type": "agent_message", "text": "..."}}

    The older flat ``{"type": "agent_message", ...}`` shape is also accepted for
    resilience across versions. Non-JSON lines (e.g. "Reading prompt from
    stdin...") are skipped.

    Raises:
        AgentCLIError: when no assistant message is found.
    """
    last_text: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        event_type = str(obj.get("type", "")).lower()
        # Current shape: the assistant message is nested under "item".
        if event_type in ("item.completed", "item.updated"):
            item = obj.get("item")
            if isinstance(item, dict) and str(item.get("type", "")).lower() in (
                "agent_message",
                "assistant",
                "message",
            ):
                text = item.get("text") or item.get("content") or item.get("message")
                if isinstance(text, str) and text.strip():
                    last_text = text.strip()
            continue
        # Older/flat shapes (defensive).
        if event_type in ("message", "agent_message", "assistant", "output"):
            content = obj.get("content") or obj.get("text") or obj.get("message")
            if isinstance(content, str) and content.strip():
                last_text = content.strip()

    if last_text is None:
        raise AgentCLIError(
            f"codex returned no assistant message in JSONL output; raw={raw[:400]!r}"
        )
    return last_text


# ---------------------------------------------------------------------------
# Gemini CLI invocation  (verified against gemini 0.46.0)
# ---------------------------------------------------------------------------


def _build_gemini_argv(binary: str, model: str, max_output_tokens: int = 0) -> list[str]:
    """Build a capability-stripped, non-interactive Gemini CLI argv.

    Flags chosen (verified end-to-end against ``gemini`` 0.46.0):

    ``-p ""``
        Headless (non-interactive) mode. The ``-p`` value is appended to stdin
        input, so with an empty value the effective prompt is exactly what the
        runner pipes to stdin — untrusted content never reaches argv.

    ``-m <label>`` / ``-o json``
        Model (validated) and structured JSON output we can parse.

    ``--approval-mode plan``
        Read-only mode: the model cannot execute tools — the primary capability
        removal. ``-y`` / ``--yolo`` (auto-approve) and ``--raw-output`` (which
        disables output sanitisation) are deliberately NEVER used.
    """
    # -m omitted by default -> gemini uses the user's own configured model
    # (forwarded only when SKILLSPECTOR_MODEL is set).
    model_arg = ["-m", _validate_model_label(model)] if model else []
    return [
        binary,
        "-p",
        "",  # headless; the real prompt is piped to stdin by run_agent_cli
        *model_arg,
        "-o",
        "json",
        "--approval-mode",
        "plan",  # read-only: no tool execution
        # We run in an isolated empty temp dir; without trust gemini silently
        # downgrades --approval-mode to "default". Safe: the temp dir is empty,
        # and "plan" keeps the session read-only (no tool execution).
        "--skip-trust",
    ]


def _parse_gemini_output(raw: str) -> str:
    """Extract assistant text from ``gemini -o json`` output.

    ``gemini -o json`` returns a JSON object with a ``response`` key
    (alongside ``session_id`` / ``stats``).  Other common keys are accepted
    for resilience across minor gemini CLI versions.  When JSON parsing fails
    entirely, the raw stdout is returned as-is (gemini may fall back to plain
    text in some error states, and returning it is better than raising and
    dropping the whole analysis).

    Raises:
        AgentCLIError: on empty stdout only.
    """
    text = raw.strip()
    if not text:
        raise AgentCLIError("gemini returned empty stdout")
    try:
        obj: Any = json.loads(text)
    except json.JSONDecodeError:
        return text  # plain-text fallback (non-JSON gemini output)
    if isinstance(obj, dict):
        for key in ("response", "text", "content", "result", "output"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return text


# ---------------------------------------------------------------------------
# OpenCode CLI invocation  (verified against opencode 1.18.31)
# ---------------------------------------------------------------------------


_OPENCODE_AGENT_PREFIX = "skillspector-deny-all"
_OPENCODE_DENY_ALL = json.dumps({"*": "deny"}, separators=(",", ":"))
_OPENCODE_SUPPORTED_VERSION = "1.18.31"


def _opencode_agent_name(argv: list[str]) -> str:
    """Return the unpredictable agent selected by an OpenCode argv."""
    try:
        name = argv[argv.index("--agent") + 1]
    except (ValueError, IndexError) as exc:
        raise AgentCLIError("OpenCode invocation is missing its fixed deny-all agent") from exc
    if not name.startswith(f"{_OPENCODE_AGENT_PREFIX}-"):
        raise AgentCLIError("OpenCode invocation selected an unexpected agent")
    return name


def _opencode_config(agent_name: str) -> str:
    """Build the inline policy for one unguessable OpenCode agent identity."""
    deny_all = {"*": "deny"}
    return json.dumps(
        {
            "$schema": "https://opencode.ai/config.json",
            "agent": {
                agent_name: {
                    "description": "SkillSpector text-only semantic analysis",
                    "mode": "primary",
                    "permission": deny_all,
                }
            },
            "autoshare": False,
            "autoupdate": False,
            "default_agent": agent_name,
            "instructions": [],
            "mcp": {},
            "permission": deny_all,
            "plugin": [],
            "share": "disabled",
            "skills": {"paths": [], "urls": []},
            "snapshot": False,
        },
        separators=(",", ":"),
    )


def _parse_opencode_version(raw: bytes) -> str | None:
    """Parse an exact stable semantic version from ``opencode --version``."""
    text = raw.decode("utf-8", errors="replace").strip()
    match = re.fullmatch(
        r"(?:opencode(?:\s+version)?\s+)?v?(\d+\.\d+\.\d+)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match is not None else None


def _prepare_opencode_env(
    base_env: dict[str, str], temp_root: str, argv: list[str]
) -> dict[str, str]:
    """Return an isolated, non-overridable deny-all OpenCode environment.

    OpenCode merges several ambient configuration layers and lets agent-local
    permissions override top-level permissions. The inline config therefore
    defines an unpredictable per-invocation agent selected in argv *and* denies
    that agent every current or future tool. A later managed config therefore
    cannot name the agent in advance and append a more-specific allow rule.
    ``OPENCODE_PERMISSION`` repeats the wildcard at OpenCode's final top-level
    permission-merge stage. Project/global extension loading, auto-sharing,
    updates, external skills, and optional tool surfaces are disabled
    independently as defense in depth.

    The real OpenCode authentication store remains available so this CLI
    provider can use the user's existing login. All mutable config, cache,
    state, database, and temporary paths are redirected below the invocation's
    already-isolated temporary root.
    """
    env = {key: value for key, value in base_env.items() if not key.upper().startswith("OPENCODE_")}
    agent_name = _opencode_agent_name(argv)
    config_home = os.path.join(temp_root, "xdg-config")
    config_dir = os.path.join(config_home, "opencode")
    cache_home = os.path.join(temp_root, "xdg-cache")
    state_home = os.path.join(temp_root, "xdg-state")
    isolated_home = os.path.join(temp_root, "home")
    managed_config = os.path.join(temp_root, "managed-config")
    child_tmp = os.path.join(temp_root, "tmp")
    for directory in (
        config_dir,
        cache_home,
        state_home,
        isolated_home,
        managed_config,
        child_tmp,
    ):
        os.makedirs(directory, mode=0o700, exist_ok=True)

    env.update(
        {
            "OPENCODE_AUTO_SHARE": "0",
            "OPENCODE_CLIENT": "skillspector",
            "OPENCODE_CONFIG_CONTENT": _opencode_config(agent_name),
            "OPENCODE_CONFIG_DIR": config_dir,
            "OPENCODE_DB": os.path.join(temp_root, "opencode.db"),
            "OPENCODE_DISABLE_AUTOCOMPACT": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_PRUNE": "1",
            "OPENCODE_DISABLE_TERMINAL_TITLE": "1",
            "OPENCODE_ENABLE_EXA": "0",
            "OPENCODE_ENABLE_PARALLEL": "0",
            "OPENCODE_ENABLE_QUESTION_TOOL": "0",
            "OPENCODE_EXPERIMENTAL": "0",
            "OPENCODE_EXPERIMENTAL_CODE_MODE": "0",
            "OPENCODE_EXPERIMENTAL_LSP_TOOL": "0",
            "OPENCODE_EXPERIMENTAL_PLAN_MODE": "0",
            "OPENCODE_PERMISSION": _OPENCODE_DENY_ALL,
            "OPENCODE_PURE": "1",
            # OpenCode 1.18.31 uses this internal override for the home paths
            # searched for AGENTS.md, .opencode, .claude, and .agents content.
            "OPENCODE_TEST_HOME": isolated_home,
            # The same pinned version uses this override instead of the
            # platform's /etc, ProgramData, or /Library managed-config path.
            "OPENCODE_TEST_MANAGED_CONFIG_DIR": managed_config,
            "TEMP": child_tmp,
            "TMP": child_tmp,
            "TMPDIR": child_tmp,
            "XDG_CACHE_HOME": cache_home,
            "XDG_CONFIG_HOME": config_home,
            "XDG_STATE_HOME": state_home,
        }
    )
    return env


def _preflight_opencode_policy(
    binary: str, argv: list[str], child_env: dict[str, str], temp_root: str
) -> None:
    """Fail closed unless OpenCode's fully resolved sensitive config is exact.

    macOS MDM preferences are loaded after ``OPENCODE_CONFIG_CONTENT`` and do
    not have a supported disable flag in OpenCode 1.18.31. Querying the final
    config with the exact environment and directory prevents a managed
    ``share: auto`` (or another late source) from silently reopening ambient
    capabilities before the inference process is allowed to start.
    """

    def run_probe(command: list[str], label: str) -> bytes:
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                cwd=temp_root,
                env=child_env,
            )
        except (FileNotFoundError, OSError) as exc:
            raise AgentCLIError(f"OpenCode {label} preflight could not start: {exc}") from exc

        returncode, stdout_raw, _stderr_raw, overflow = _run_bounded(proc, b"", 15)
        if overflow:
            raise AgentCLIError(f"OpenCode {label} preflight exceeded its output limit")
        if returncode is None:
            raise AgentCLIError(f"OpenCode {label} preflight timed out")
        if returncode != 0:
            raise AgentCLIError(f"OpenCode {label} preflight exited with code {returncode}")
        return stdout_raw

    version_raw = run_probe([binary, "--version"], "version")
    version = _parse_opencode_version(version_raw)
    if version != _OPENCODE_SUPPORTED_VERSION:
        version_text = version_raw.decode("utf-8", errors="replace").strip()
        raise AgentCLIError(
            "OpenCode security policy is verified only for version "
            f"{_OPENCODE_SUPPORTED_VERSION}; found {version_text[:80]!r}"
        )

    agent_name = _opencode_agent_name(argv)
    stdout_raw = run_probe([binary, "debug", "config"], "policy")
    try:
        config = json.loads(stdout_raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentCLIError("OpenCode policy preflight returned malformed JSON") from exc
    if not isinstance(config, dict):
        raise AgentCLIError("OpenCode policy preflight returned a non-object config")

    expected_sensitive = {
        "autoshare": False,
        "autoupdate": False,
        "default_agent": agent_name,
        "instructions": [],
        "mcp": {},
        "permission": {"*": "deny"},
        "plugin": [],
        "share": "disabled",
        "skills": {"paths": [], "urls": []},
        "snapshot": False,
        "tools": None,
    }
    for key, expected in expected_sensitive.items():
        if config.get(key) != expected:
            raise AgentCLIError(f"OpenCode policy preflight found unsafe resolved setting {key!r}")

    agents = config.get("agent")
    selected = agents.get(agent_name) if isinstance(agents, dict) else None
    if not isinstance(selected, dict) or selected.get("permission") != {"*": "deny"}:
        raise AgentCLIError("OpenCode policy preflight found an overridable agent permission")


def _build_opencode_argv(binary: str, model: str, max_output_tokens: int = 0) -> list[str]:
    """Build argv for a text-only, deny-all ``opencode run`` call.

    Flags chosen (verified against ``opencode`` 1.18.31 ``run --help``):

    ``run``
        Non-interactive single-shot mode. With no positional message, the
        prompt is piped to stdin by run_agent_cli — untrusted content never
        reaches argv.

    ``--pure``
        Run without external plugins.

    ``--agent skillspector-deny-all-<nonce>``
        Select the unguessable per-invocation agent installed through
        ``OPENCODE_CONFIG_CONTENT``. Its agent-local permission is a wildcard
        deny; the child environment also supplies final-merge
        ``OPENCODE_PERMISSION={"*":"deny"}`` so built-in, custom, MCP, and
        future tool names all remain unavailable.

    ``--format json``
        Emit raw JSON events to stdout for structured parsing.

    ``--model <label>``
        Model in ``provider/model`` form (validated). Omitted by default so
        opencode uses the CLI default model (forwarded only when
        SKILLSPECTOR_MODEL is set).

    The shared runner also isolates every OpenCode config/state path, disables
    ambient instructions, skills, plugins, auto-sharing, snapshots and updates,
    scrubs secret-bearing environment variables, delivers untrusted content
    via stdin only, and never passes ``--auto``.

    Deliberately NOT included:
    - ``--auto`` — auto-approves permissions (dangerous); never use it.
    - ``max_output_tokens`` — ``run`` has no token flag (accepted for
      CliSpec uniformity and ignored, like codex/gemini).
    """
    # --model omitted by default -> opencode uses the CLI default model
    # (forwarded only when SKILLSPECTOR_MODEL is set).
    model_arg = ["--model", _validate_model_label(model)] if model else []
    agent_name = f"{_OPENCODE_AGENT_PREFIX}-{secrets.token_hex(16)}"
    return [
        binary,
        "run",
        "--pure",
        "--agent",
        agent_name,
        "--format",
        "json",
        *model_arg,
    ]


def _parse_opencode_output(raw: str) -> str:
    """Extract assistant text from ``opencode run --format json`` JSONL events.

    Verified against opencode 1.18.31, whose events look like::

        {"type": "step_start", ..., "part": {"type": "step-start", ...}}
        {"type": "text", ..., "part": {"type": "text", "text": "hi", ...}}
        {"type": "step_finish", ..., "part": {"type": "step-finish", ...}}

    The assistant text arrives in ``part.text`` of ``text`` events; every
    other event (step boundaries and the like) carries no reply text and is
    skipped. Non-JSON lines (banner/TUI noise) are skipped, mirroring the
    codex parser's tolerance. Multiple ``text`` events are concatenated in
    order (streamed chunks).

    Raises:
        AgentCLIError: when stdout is empty or holds no ``text`` event.
    """
    chunks: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        part = obj.get("part")
        if not isinstance(part, dict):
            continue
        if str(part.get("type", "")).lower() != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            chunks.append(text)
    reply = "".join(chunks).strip()
    if not reply:
        raise AgentCLIError(
            f"opencode returned no assistant text in JSON output; raw={raw[:400]!r}"
        )
    return reply


# ---------------------------------------------------------------------------
# Per-CLI authentication probes (cheap, local — run once per scan)
# ---------------------------------------------------------------------------


def _claude_auth_check(binary: str) -> tuple[bool, str | None]:
    """Check claude is authenticated via ``claude auth status`` (no inference)."""
    try:
        result = subprocess.run(
            [binary, "auth", "status"], capture_output=True, shell=False, timeout=15
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return False, f"claude auth status check failed: {exc}"
    out = (result.stdout or b"").decode("utf-8", errors="replace").strip()
    try:
        logged_in = bool(json.loads(out).get("loggedIn"))
    except (json.JSONDecodeError, AttributeError):
        logged_in = result.returncode == 0 and "not logged in" not in out.lower()
    if result.returncode != 0 or not logged_in:
        return False, "claude is not authenticated (run `claude auth login`)"
    return True, None


def _codex_auth_check(binary: str) -> tuple[bool, str | None]:
    """Check codex is authenticated via ``codex login status`` (no inference)."""
    try:
        result = subprocess.run(
            [binary, "login", "status"], capture_output=True, shell=False, timeout=15
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return False, f"codex login status check failed: {exc}"
    out = (result.stdout or b"").decode("utf-8", errors="replace").lower()
    if result.returncode != 0 or "not logged in" in out:
        return False, "codex is not authenticated (run `codex login`)"
    return True, None


def _gemini_auth_check(binary: str) -> tuple[bool, str | None]:
    """Gemini availability probe.

    The Gemini CLI (0.46.0) has no cheap non-interactive auth-status command, so
    we treat binary-on-PATH as available and let the first real call fail closed
    if auth is missing.
    """
    return True, None


def _opencode_auth_check(binary: str) -> tuple[bool, str | None]:
    """Check opencode is authenticated via ``opencode auth list`` (no inference).

    The caller must already have resolved ``binary``. The probe uses the same
    scrubbed environment as inference, performs no inference, and completes
    well under 15s. Fail-closed: probe error/timeout, non-zero exit, zero
    parsed credentials everywhere, or unparseable output all return
    ``(False, reason)``. ``True`` is returned only when at least one parsed
    credential/environment-key count is positive.
    """
    tmp_root = tempfile.mkdtemp(prefix="skillspector_opencode_auth_")
    try:
        try:
            policy_argv = _build_opencode_argv(binary, "", 0)
            child_env = _prepare_opencode_env(_scrub_env(), tmp_root, policy_argv)
            version_result = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                shell=False,
                timeout=15,
                env=child_env,
            )
            if (
                version_result.returncode != 0
                or _parse_opencode_version(version_result.stdout or b"")
                != _OPENCODE_SUPPORTED_VERSION
            ):
                return (
                    False,
                    "opencode_cli requires exactly OpenCode "
                    f"{_OPENCODE_SUPPORTED_VERSION} for its verified deny-all policy",
                )
            result = subprocess.run(
                [binary, "auth", "list"],
                capture_output=True,
                shell=False,
                timeout=15,
                env=child_env,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            return False, f"opencode auth list check failed: {exc}"
    finally:
        _cleanup_temp_dir(tmp_root)
    if result.returncode != 0:
        return False, "opencode is not authenticated (run `opencode auth login`)"
    out = re.compile(r"\x1b\[[0-9;]*m").sub(
        "", (result.stdout or b"").decode("utf-8", errors="replace")
    )
    creds_match = re.compile(r"(\d+)\s+credentials?").search(out)
    env_match = re.compile(r"(\d+)\s+environment variables?").search(out)
    num_creds = int(creds_match.group(1)) if creds_match else 0
    num_env_keys = int(env_match.group(1)) if env_match else 0
    if num_creds <= 0 and num_env_keys <= 0:
        return False, "opencode is not authenticated (run `opencode auth login`)"
    return True, None


# ---------------------------------------------------------------------------
# GitHub Copilot CLI invocation  (verified against copilot 1.0.86)
# ---------------------------------------------------------------------------


def _parse_copilot_version(raw: bytes) -> str | None:
    """Parse an exact stable semantic version from ``copilot --version``."""
    lines = raw.decode("utf-8", errors="replace").strip().splitlines() or [""]
    match = re.fullmatch(
        r"(?:github\s+copilot\s+cli\s+)?v?(\d+\.\d+\.\d+)\.?",
        lines[0].strip(),
        flags=re.IGNORECASE,
    )
    return match.group(1) if match is not None else None


def _prepare_copilot_env(
    base_env: dict[str, str], temp_root: str, argv: list[str]
) -> dict[str, str]:
    """Return the child environment for a copilot invocation.

    ``temp_root``/``argv`` are unused (CliSpec signature uniformity).
    Starts from the already-scrubbed base and applies an explicit allowlist
    to ``COPILOT_*``: every such variable is dropped EXCEPT the three
    documented token variables (``COPILOT_GITHUB_TOKEN``, ``GH_TOKEN``,
    ``GITHUB_TOKEN`` — re-read from the operator environment because the
    shared scrub strips ``GITHUB_TOKEN``) and ``COPILOT_HOME`` (a path, not
    a policy control — argv-level deny rules take precedence over anything
    a config file could add, so hiding it would only break login-file auth
    without hardening anything). The tokens are the CLI's supported
    headless auth path and therefore work at inference time. In particular
    ``COPILOT_ALLOW_ALL`` never reaches the child, so ambient shell config
    cannot re-enable tools; ``COPILOT_PROVIDER_*`` cannot redirect inference
    to an arbitrary endpoint; and ``COPILOT_CUSTOM_INSTRUCTIONS_DIRS``
    cannot inject instructions. ``COPILOT_AUTO_UPDATE`` is forced off so the
    version gate cannot be invalidated mid-scan.
    """
    env = {key: value for key, value in base_env.items() if not key.upper().startswith("COPILOT_")}
    for name in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "COPILOT_HOME"):
        value = os.environ.get(name, "").strip()
        if value:
            env[name] = value
    env["COPILOT_AUTO_UPDATE"] = "false"
    return env


def _build_copilot_argv(binary: str, model: str, max_output_tokens: int = 0) -> list[str]:
    """Build the argv list for a non-interactive ``copilot`` call.

    Flags chosen (verified against Copilot CLI 1.0.86 ``--help``):

    (no ``-p``)
        With no prompt flag, the prompt is piped to stdin by run_agent_cli —
        untrusted content never reaches argv (verified by nonce round-trip).

    ``-s``
        Suppress stats and decoration, emitting only the agent's response.

    ``--no-ask-user``
        Disable the ask_user tool so the agent cannot pause for input.

    ``--no-custom-instructions``
        Disable loading of custom instructions from AGENTS.md and related
        files, so ambient instruction files in the operator's home or
        project cannot steer the semantic verdict. (`COPILOT_HOME` is
        deliberately retained for login, which is why flag-level disabling
        is required rather than home isolation.)

    ``--disable-builtin-mcps``
        Disable all built-in MCP servers as defense in depth alongside the
        tool allowlist below.

    ``--no-auto-update``
        Disable CLI auto-updates so the pinned-version gate cannot be
        invalidated mid-scan.

    ``--available-tools skillspector-no-tools``
        Allowlist holding a fixed implausible name, so the model is offered
        no usable tools (verified: a file-creation request was refused with
        no side effects). A fictitious name fails closed if a future CLI
        ever rejects unknown tool names.

    ``--deny-tool shell,write``
        Belt-and-braces deny of the shell and file-writing tool kinds in the
        documented ``Kind(argument)`` form; deny rules take precedence over
        allow rules. (No wildcard deny exists; single-tool deny alone does
        not stop reads through other tools, hence the allowlist above.)

    ``--model <label>``
        Model in plain form (validated). Omitted by default so copilot uses
        the CLI default model (forwarded only when SKILLSPECTOR_MODEL is set).

    Deliberately NOT included:
    - ``--allow-all*`` / ``--yolo`` — auto-approve permissions (dangerous); never use them.
    - ``max_output_tokens`` — copilot has no token flag (accepted for
      CliSpec uniformity and ignored, like codex/gemini).
    """
    # --model omitted by default -> copilot uses the CLI default model
    # (forwarded only when SKILLSPECTOR_MODEL is set).
    model_arg = ["--model", _validate_model_label(model)] if model else []
    return [
        binary,
        "-s",
        "--no-ask-user",
        "--no-custom-instructions",
        "--disable-builtin-mcps",
        "--no-auto-update",
        "--available-tools",
        "skillspector-no-tools",
        "--deny-tool",
        "shell,write",
        *model_arg,
    ]


def _parse_copilot_output(raw: str) -> str:
    """Extract the assistant reply from ``copilot -s`` plain-text output.

    Verified against Copilot CLI 1.0.86: ``-s`` emits only the response text.
    The whole stripped output is the reply; empty output raises fail-closed
    (an empty response must never be mistaken for a clean analysis).
    """
    reply = raw.strip()
    if not reply:
        raise AgentCLIError(f"copilot returned no assistant text in output; raw={raw[:400]!r}")
    return reply


def _copilot_auth_check(binary: str) -> tuple[bool, str | None]:
    """Check copilot is usable via ``copilot --version`` (no inference).

    The caller must already have resolved ``binary``. The probe uses the
    shared scrubbed environment (token re-injection is inference-only and
    version output does not depend on it), performs no inference, and
    completes well under 15s. Fail-closed: probe error/timeout, non-zero
    exit, or a version other than the verified 1.0.86 all return
    ``(False, reason)``.

    There is no status subcommand, so a passing probe means the binary runs
    the verified version — not proof of login. Authentication works two
    ways: the persistent login session (``copilot login``), validated by the
    first inference call failing closed with the CLI's real error; or one of
    ``COPILOT_GITHUB_TOKEN`` / ``GH_TOKEN`` / ``GITHUB_TOKEN``, which
    ``_prepare_copilot_env`` deliberately preserves through the scrub while
    dropping every other ``COPILOT_*`` variable.
    """
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            shell=False,
            timeout=15,
            env=_scrub_env(),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return False, f"copilot --version check failed: {exc}"
    if result.returncode != 0:
        return False, (
            "copilot --version probe failed "
            f"(exit {result.returncode}); check the binary, then `copilot login`"
        )
    version = _parse_copilot_version(result.stdout or b"")
    if version != "1.0.86":
        version_text = (result.stdout or b"").decode("utf-8", errors="replace").strip()
        return False, (
            "copilot_cli requires exactly GitHub Copilot CLI 1.0.86 "
            f"for its verified tool-deny policy; found {version_text[:80]!r}"
        )
    return True, None


# ---------------------------------------------------------------------------
# Antigravity CLI  (registered but DISABLED — verified incompatible)
#
# The Antigravity CLI (binary: ``agy``) was tested end-to-end against the real
# binary (agy 0.x and re-verified on agy 1.0.10, logged in). It CANNOT be driven
# programmatically and so is kept fail-closed:
#   * Its ``--print`` / ``--prompt`` mode renders the response to the TTY only.
#     With stdout captured via a pipe (exactly how run_agent_cli must invoke it),
#     it HANGS and returns EMPTY stdout (and empty stderr) — the response never
#     reaches us. (On 1.0.10, `agy --print` produced 0 bytes of stdout and did
#     not honour even `--print-timeout 30s`, requiring an external kill; its
#     `--help` still exposes only TTY-oriented --print/--prompt with no headless
#     JSON-stdout mode. agy remains a TTY/language-server app, not a
#     stdin->stdout filter like claude/codex/gemini ``-p``.)
#   * It also takes the prompt as an argv VALUE, not stdin — at odds with our
#     "untrusted content via stdin, never argv" rule and bounded by OS argv size.
#   * Its backend is Gemini (Google Cloud Code Assist), so it adds no capability
#     over the working ``gemini_cli`` provider, which returns JSON over stdin.
# It stays in the registry (and fails closed) so the limitation is documented in
# one place. To enable later: if agy gains a headless/structured stdout mode
# (e.g. ``--output-format json`` written to a pipe), wire _build_agy_argv to it
# and replace _agy_auth_check with a real probe — exactly as was done for gemini.
# ---------------------------------------------------------------------------


def _build_agy_argv(binary: str, model: str, max_output_tokens: int = 0) -> list[str]:
    """Antigravity CLI argv — disabled: agy can't be captured from a pipe.

    Fails closed: raising here guarantees ``agy`` is never invoked, since its
    print mode emits to a TTY only and would silently return nothing (an empty
    response must never be mistaken for a clean analysis). See the note above.
    """
    raise AgentCLIError(
        "antigravity_cli (agy) cannot be driven programmatically: its print mode "
        "renders to a TTY and returns empty stdout on a pipe; refusing to run. "
        "Its backend is Gemini — use SKILLSPECTOR_PROVIDER=gemini_cli instead."
    )


def _agy_auth_check(binary: str) -> tuple[bool, str | None]:
    """Report antigravity as unavailable: verified incompatible (fail-closed)."""
    return (
        False,
        "antigravity_cli (agy) is registered but disabled: its print mode renders "
        "to a TTY and emits nothing on a pipe, so it cannot be captured "
        "programmatically. Its backend is Gemini — use gemini_cli instead.",
    )


# ---------------------------------------------------------------------------
# CLI registry
#
# HOW TO ADD A NEW AGENT CLI (no changes to run_agent_cli or the security core):
#   1. Write three small functions above:
#        _build_<name>_argv(binary, model, max_output_tokens) -> argv
#        _parse_<name>_output(raw) -> str
#        _<name>_auth_check(binary) -> (available, reason)
#      If the CLI needs extra process isolation, also provide a fixed
#      ``prepare_env(clean_env, temp_root, argv)`` callback and optional
#      ``preflight(binary, argv, child_env, temp_root)`` in its ``CliSpec``.
#      Keep the security posture: no shell, NO tool execution, NO auto-approve,
#      prompt via stdin (run_agent_cli handles stdin), fail-closed on any error.
#   2. Add a CliSpec entry to _REGISTRY below.
#   3. Add a ~5-line provider subclass of AgentCLIProviderBase under
#      providers/<name>_cli/ (just BINARY_NAME — no model_registry.yaml; CLI
#      providers pin no model and use package-wide default token budgets).
#   4. Register it in providers/__init__.py:_select_active_provider.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliSpec:
    """Everything provider-specific about one agent CLI, behind one lookup."""

    binary: str
    build_argv: Callable[[str, str, int], list[str]]
    parse_output: Callable[[str], str]
    auth_check: Callable[[str], tuple[bool, str | None]]
    prepare_env: Callable[[dict[str, str], str, list[str]], dict[str, str]] | None = None
    preflight: Callable[[str, list[str], dict[str, str], str], None] | None = None


_REGISTRY: dict[str, CliSpec] = {
    "claude": CliSpec("claude", _build_claude_argv, _parse_claude_output, _claude_auth_check),
    "codex": CliSpec("codex", _build_codex_argv, _parse_codex_output, _codex_auth_check),
    "gemini": CliSpec("gemini", _build_gemini_argv, _parse_gemini_output, _gemini_auth_check),
    "opencode": CliSpec(
        "opencode",
        _build_opencode_argv,
        _parse_opencode_output,
        _opencode_auth_check,
        _prepare_opencode_env,
        _preflight_opencode_policy,
    ),
    "copilot": CliSpec(
        "copilot",
        _build_copilot_argv,
        _parse_copilot_output,
        _copilot_auth_check,
        _prepare_copilot_env,
    ),
    # Disabled (fails closed via _build_agy_argv). agy's backend is Gemini, so it
    # reuses _parse_gemini_output rather than duplicating it — though parse is
    # never reached while _build_agy_argv raises. See the antigravity note above.
    "agy": CliSpec("agy", _build_agy_argv, _parse_gemini_output, _agy_auth_check),
}


def get_spec(name: str) -> CliSpec:
    """Return the :class:`CliSpec` for *name*, or raise for an unknown CLI."""
    spec = _REGISTRY.get(name)
    if spec is None:
        raise AgentCLIError(
            f"unsupported agent CLI {name!r}; known: {', '.join(sorted(_REGISTRY))}"
        )
    return spec


def is_available(binary_name: str) -> tuple[bool, str | None]:
    """Return ``(available, reason)``: the binary is on PATH AND authenticated."""
    spec = get_spec(binary_name)
    binary = find_binary(spec.binary)
    if binary is None:
        return False, f"{spec.binary!r} binary not found on PATH"
    return spec.auth_check(binary)


# ---------------------------------------------------------------------------
# Bounded process execution
# ---------------------------------------------------------------------------


def _drain_stream(stream: Any, buf: bytearray, cap: int, on_overflow: Any) -> None:
    """Read *stream* into *buf* up to *cap* bytes, then stop reading.

    Calls *on_overflow* once if the cap is reached so the caller can react
    (e.g. kill a runaway process). Never raises.
    """
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            remaining = cap - len(buf)
            if remaining > 0:
                buf.extend(chunk[:remaining])
            if len(buf) >= cap:
                on_overflow()
                break
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


_CLEANUP_RETRIES = 10
_CLEANUP_RETRY_DELAY_S = 0.2


def _cleanup_temp_dir(path: str) -> None:
    """Best-effort removal of the per-invocation temp cwd.

    On Windows a directory cannot be deleted while any process has it as its
    working directory or holds a handle inside it. The agent CLI's process
    tree can outlive the direct child by a beat (a ``.cmd`` shim's node child,
    or a killed-but-not-reaped child on the timeout path), so the first
    attempts may fail transiently with ``WinError 32``. Retry briefly, then
    leak the directory with a warning rather than raise: by the time cleanup
    runs the model's response is already in hand, and a temp-dir leak must
    never fail the batch (#315).
    """
    for _ in range(_CLEANUP_RETRIES):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            time.sleep(_CLEANUP_RETRY_DELAY_S)
    shutil.rmtree(path, ignore_errors=True)
    if os.path.isdir(path):
        logger.warning(
            "Could not remove temp dir %s (handle still held by the CLI "
            "process tree?); leaking it rather than failing the batch",
            path,
        )


def _run_bounded(
    proc: subprocess.Popen, prompt_bytes: bytes, timeout: float
) -> tuple[int | None, bytes, bytes, bool]:
    """Drive *proc* to completion with memory and time bounds.

    Feeds *prompt_bytes* to stdin and drains stdout/stderr concurrently (so a
    large prompt cannot deadlock against a chatty child). stdout is capped at
    ``MAX_OUTPUT_BYTES`` and stderr at ``MAX_STDERR_BYTES``; if stdout exceeds
    its cap the process is killed immediately rather than buffered to memory.

    Returns ``(returncode, stdout, stderr, overflow)``. ``returncode`` is
    ``None`` when the call timed out; ``overflow`` is True when stdout hit the
    cap (the process was then killed).
    """
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    overflow = threading.Event()

    def _kill_on_overflow() -> None:
        overflow.set()
        proc.kill()

    def _feed_stdin() -> None:
        try:
            if proc.stdin is not None:
                proc.stdin.write(prompt_bytes)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass

    threads = [
        threading.Thread(target=_feed_stdin, daemon=True),
        threading.Thread(
            target=_drain_stream,
            args=(proc.stdout, stdout_buf, MAX_OUTPUT_BYTES, _kill_on_overflow),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(proc.stderr, stderr_buf, MAX_STDERR_BYTES, lambda: None),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    try:
        returncode: int | None = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        returncode = None

    for t in threads:
        t.join(timeout=5)

    return returncode, bytes(stdout_buf), bytes(stderr_buf), overflow.is_set()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_agent_cli(
    binary_name: str,
    prompt: str,
    *,
    model: str,
    max_output_tokens: int = 8192,
    timeout: float = CLI_TIMEOUT_SECONDS,
) -> str:
    """Run an agent CLI and return the assistant response text.

    This is the single security-hardened entry point.  All security
    invariants are enforced here:

    - Binary is located via ``shutil.which``; missing binary raises.
    - Untrusted ``prompt`` is delivered via stdin, **never** in argv.
    - ``shell=False`` throughout — no shell interpolation.
    - Environment is scrubbed of secrets before the child is spawned.
    - Process runs in a fresh temporary directory with no access to the
      caller's CWD.
    - Hard timeout; ``subprocess.TimeoutExpired`` is re-raised as
      :class:`AgentCLIError`.
    - Non-zero exit code raises :class:`AgentCLIError` (fail-closed).
    - stdout is streamed with a hard ``MAX_OUTPUT_BYTES`` cap; the process is
      killed if it exceeds the cap (no unbounded buffering).

    Args:
        binary_name: A registered agent CLI name (see ``_REGISTRY``), e.g.
                     ``"claude"``, ``"codex"``, or ``"gemini"``.
        prompt:       The complete prompt string. Delivered to the CLI via
                      stdin only — never placed in argv.
        model:        Model label (e.g. ``"claude-sonnet-4-6"``).
        max_output_tokens: Hint for claude; not forwarded for codex.
        timeout:      Seconds before the subprocess is killed.

    Returns:
        The assistant's text response as a plain string.

    Raises:
        AgentCLIError: on any failure (missing binary, non-zero exit,
            timeout, empty / malformed output).
    """
    spec = get_spec(binary_name)
    binary = find_binary(spec.binary)
    if binary is None:
        raise AgentCLIError(
            f"{spec.binary!r} binary not found on PATH; "
            "install it or use a different SKILLSPECTOR_PROVIDER"
        )

    # -- Input size guard -----------------------------------------------------
    prompt_bytes = prompt.encode("utf-8", errors="replace")
    if len(prompt_bytes) > MAX_INPUT_BYTES:
        raise AgentCLIError(
            f"prompt exceeds MAX_INPUT_BYTES ({MAX_INPUT_BYTES}); got {len(prompt_bytes)} bytes"
        )

    # -- Build argv via the registry (no untrusted content here) ---------------
    argv = spec.build_argv(binary, model, max_output_tokens)

    # -- Run in a temporary directory (no CWD access) -------------------------
    # mkdtemp + explicit best-effort cleanup instead of TemporaryDirectory:
    # the context manager's rmtree-at-__exit__ raises on Windows while the
    # CLI's process tree still holds the directory as its cwd, turning an
    # already-successful call into a batch failure (#315).
    tmp_cwd = tempfile.mkdtemp(prefix="skillspector_cli_")
    try:
        # -- Scrub and apply any CLI-specific isolation policy ----------------
        child_env = _scrub_env()
        if spec.prepare_env is not None:
            try:
                child_env = spec.prepare_env(child_env, tmp_cwd, argv)
            except OSError as exc:
                raise AgentCLIError(
                    f"{binary_name} environment isolation could not be established: {exc}"
                ) from exc

        # A CLI whose security depends on resolved runtime configuration must
        # prove that configuration safe before receiving untrusted input.
        if spec.preflight is not None:
            spec.preflight(binary, argv, child_env, tmp_cwd)

        logger.debug(
            "Running %s argv=%r cwd=%s timeout=%ss",
            binary_name,
            argv,
            tmp_cwd,
            timeout,
        )
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                cwd=tmp_cwd,
                env=child_env,
            )
        except FileNotFoundError as exc:
            raise AgentCLIError(f"{binary_name} binary disappeared after lookup: {exc}") from exc

        # Stream stdout/stderr with hard memory caps so a runaway or compromised
        # CLI cannot exhaust memory before the cap is enforced (a chatty child
        # could otherwise buffer unbounded output until the timeout).
        returncode, stdout_raw, stderr_raw, overflow = _run_bounded(proc, prompt_bytes, timeout)
    finally:
        _cleanup_temp_dir(tmp_cwd)

    # -- Fail-closed checks ---------------------------------------------------
    if overflow:
        raise AgentCLIError(
            f"{binary_name} produced more than MAX_OUTPUT_BYTES ({MAX_OUTPUT_BYTES}); killed"
        )
    if returncode is None:
        raise AgentCLIError(f"{binary_name} timed out after {timeout}s")
    if returncode != 0:
        stderr_snippet = stderr_raw[:500].decode("utf-8", errors="replace")
        raise AgentCLIError(
            f"{binary_name} exited with code {returncode}; stderr={stderr_snippet!r}"
        )

    raw_text = stdout_raw.decode("utf-8", errors="replace")

    # -- Parse envelope via the registry --------------------------------------
    return spec.parse_output(raw_text)
