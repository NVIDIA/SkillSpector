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

"""Hardened subprocess helper for agent CLI providers (claude, codex, gemini).

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
import shutil
import subprocess
import tempfile
import threading
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
    "OPENAI_API_KEY",
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


_REGISTRY: dict[str, CliSpec] = {
    "claude": CliSpec("claude", _build_claude_argv, _parse_claude_output, _claude_auth_check),
    "codex": CliSpec("codex", _build_codex_argv, _parse_codex_output, _codex_auth_check),
    "gemini": CliSpec("gemini", _build_gemini_argv, _parse_gemini_output, _gemini_auth_check),
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

    # -- Scrub environment ----------------------------------------------------
    child_env = _scrub_env()

    # -- Run in a temporary directory (no CWD access) -------------------------
    with tempfile.TemporaryDirectory(prefix="skillspector_cli_") as tmp_cwd:
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
