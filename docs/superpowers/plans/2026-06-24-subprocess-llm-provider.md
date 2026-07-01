# Subprocess LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `subprocess` LLM provider that pipes prompts through any configurable CLI command, enabling SkillSpector's LLM analysis to work inside Claude Code, OpenClaw, Antigravity, or any AI-tool session without a separate API key.

**Architecture:** A new `SubprocessChatModel` (extends LangChain `BaseChatModel`) serializes each LangChain message list into plain text, pipes it to a user-configured shell command via stdin, and returns the stdout as an `AIMessage`. Structured output is handled by appending JSON-schema instructions to the prompt and parsing the response with a Pydantic parser — no native tool-calling required. The new `SubprocessProvider` fits into the existing `providers/` protocol and is selected via `SKILLSPECTOR_PROVIDER=subprocess`.

**Tech Stack:** Python 3.11+, LangChain Core (`BaseChatModel`, `RunnableLambda`), Pydantic v2, `subprocess` stdlib, `pytest`.

## Global Constraints

- No new third-party dependencies beyond what is already in `pyproject.toml`; use only stdlib `subprocess`, LangChain Core, and Pydantic (already present).
- All new code lives under `src/skillspector/providers/subprocess/` and follows the same Apache-2.0 license header used everywhere else in the repo.
- Provider must satisfy the `LLMProvider` Protocol defined in `src/skillspector/providers/base.py` without modifying that file.
- Follow the existing `ruff` + `mypy` style; no `type: ignore` comments unless strictly unavoidable.
- Tests must pass with `make test` (no live LLM calls in default run; subprocess calls must be mockable).

---

## File Map

| Action   | Path                                                                 | Responsibility                                           |
|----------|----------------------------------------------------------------------|----------------------------------------------------------|
| Create   | `src/skillspector/providers/subprocess/__init__.py`                  | Exports `SubprocessProvider`                             |
| Create   | `src/skillspector/providers/subprocess/provider.py`                  | `SubprocessChatModel` + `SubprocessProvider`             |
| Create   | `src/skillspector/providers/subprocess/model_registry.yaml`          | Default token-budget metadata for subprocess model       |
| Modify   | `src/skillspector/providers/__init__.py`                             | Register `subprocess` in `_select_active_provider()`     |
| Modify   | `.env.example`                                                       | Document `SKILLSPECTOR_LLM_COMMAND` env var              |
| Create   | `tests/providers/test_subprocess_provider.py`                        | Unit tests for SubprocessProvider + SubprocessChatModel  |

---

### Task 1: SubprocessChatModel — core invoke loop

**Files:**
- Create: `src/skillspector/providers/subprocess/__init__.py`
- Create: `src/skillspector/providers/subprocess/provider.py`
- Create: `tests/providers/test_subprocess_provider.py`

**Interfaces:**
- Produces: `SubprocessChatModel` — a `BaseChatModel` subclass with `_generate()` and `_call_subprocess()` methods that other tasks extend.

- [ ] **Step 1: Write the failing test**

```python
# tests/providers/test_subprocess_provider.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```
cd C:\zz\SkillSpector
pytest tests/providers/test_subprocess_provider.py -v
```
Expected: `ImportError: cannot import name 'SubprocessChatModel'`

- [ ] **Step 3: Create the `__init__.py`**

```python
# src/skillspector/providers/subprocess/__init__.py
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

"""Subprocess LLM provider — routes prompts through a configured shell command."""

from .provider import SubprocessProvider

__all__ = ["SubprocessProvider"]
```

- [ ] **Step 4: Implement `SubprocessChatModel` in `provider.py`**

```python
# src/skillspector/providers/subprocess/provider.py
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

"""Subprocess LLM provider.

Routes every LLM call through an external CLI command configured by the user.
The full prompt is written to the command's stdin; the response is read from
stdout.  This lets SkillSpector run inside Claude Code, OpenClaw, Antigravity,
or any other AI-tool session without a separate API key.

Configuration
-------------
SKILLSPECTOR_PROVIDER=subprocess
SKILLSPECTOR_LLM_COMMAND=claude -p
    # or: antigravity ask
    # or: openclaw chat
    # The command is split on whitespace; prompt is piped via stdin.

SKILLSPECTOR_MODEL is used only for display/logging (no semantic meaning for
subprocess calls).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterator

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from skillspector.providers import registry

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))

_DEFAULT_CONTEXT_LENGTH = 200_000
_DEFAULT_MAX_OUTPUT_TOKENS = 8_192
_SENTINEL_MODEL = "subprocess"


def _format_messages(messages: list[BaseMessage]) -> str:
    """Render a LangChain message list as a plain-text prompt."""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(f"<system>\n{msg.content}\n</system>")
        elif isinstance(msg, HumanMessage):
            parts.append(f"<human>\n{msg.content}\n</human>")
        elif isinstance(msg, AIMessage):
            parts.append(f"<assistant>\n{msg.content}\n</assistant>")
        else:
            # Fallback for ToolMessage / FunctionMessage etc.
            parts.append(str(msg.content))
    return "\n\n".join(parts)


class SubprocessChatModel(BaseChatModel):
    """A LangChain chat model that routes calls through a shell command.

    The full prompt is written to the subprocess stdin; stdout is the response.
    """

    command: str = Field(description="Shell command to invoke (split on whitespace)")
    timeout: float = Field(default=120.0, description="Seconds before subprocess times out")

    @property
    def _llm_type(self) -> str:
        return "subprocess"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _format_messages(messages)
        text = self._call_subprocess(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _call_subprocess(self, prompt: str) -> str:
        args = shlex.split(self.command)
        result = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LLM subprocess failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout.strip()

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Return a Runnable that appends JSON-schema instructions and parses output.

        Because subprocess models cannot use native tool-calling, structured
        output is implemented by:
        1. Appending JSON schema + instructions to the last human message.
        2. Calling _generate() normally.
        3. Parsing the JSON from the response with Pydantic.
        """
        json_schema = schema.model_json_schema()
        schema_str = json.dumps(json_schema, indent=2)
        instruction = (
            "\n\n---\nRespond with a single valid JSON object that conforms to "
            "this JSON Schema (no markdown fences, no explanation, only JSON):\n"
            f"{schema_str}"
        )

        def inject_and_parse(messages: list[BaseMessage]) -> BaseModel:
            # Append instruction to the last human message (copy to avoid mutation)
            augmented: list[BaseMessage] = []
            for i, msg in enumerate(messages):
                if i == len(messages) - 1 and isinstance(msg, HumanMessage):
                    augmented.append(HumanMessage(content=msg.content + instruction))
                else:
                    augmented.append(msg)
            raw_text = self.invoke(augmented).content
            # Strip markdown code fences if the model emitted them anyway
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return schema.model_validate_json(clean)

        return RunnableLambda(inject_and_parse)
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/providers/test_subprocess_provider.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 6: Commit**

```
git add src/skillspector/providers/subprocess/ tests/providers/test_subprocess_provider.py
git commit -m "feat: add SubprocessChatModel that routes prompts via shell command"
```

---

### Task 2: SubprocessProvider — LLMProvider protocol compliance

**Files:**
- Modify: `src/skillspector/providers/subprocess/provider.py` (append `SubprocessProvider` class at end)
- Create: `src/skillspector/providers/subprocess/model_registry.yaml`
- Modify: `tests/providers/test_subprocess_provider.py` (append provider tests)

**Interfaces:**
- Consumes: `SubprocessChatModel` from Task 1 at `src/skillspector/providers/subprocess/provider.py`
- Produces: `SubprocessProvider` — satisfies `LLMProvider` protocol; used by `_select_active_provider()` in Task 3.

- [ ] **Step 1: Write the failing tests**

Append to `tests/providers/test_subprocess_provider.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/providers/test_subprocess_provider.py::TestSubprocessProvider -v
```
Expected: `ImportError` or `AttributeError` for `SubprocessProvider`

- [ ] **Step 3: Create `model_registry.yaml`**

```yaml
# src/skillspector/providers/subprocess/model_registry.yaml
# Conservative defaults; the actual limits depend on the configured command.
models:
  "subprocess":
    context_length: 200000
    max_output_tokens: 8192
```

- [ ] **Step 4: Append `SubprocessProvider` to `provider.py`**

Add after the `SubprocessChatModel` class (before the end of the file):

```python
class SubprocessProvider:
    """LLM provider that routes calls through a configurable shell command.

    Required environment variables
    --------------------------------
    SKILLSPECTOR_PROVIDER=subprocess
    SKILLSPECTOR_LLM_COMMAND=<shell command>
        e.g.  claude -p
              antigravity ask
              openclaw chat
        The prompt is written to the command's stdin.
    """

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Return a sentinel tuple when SKILLSPECTOR_LLM_COMMAND is set, else None."""
        command = os.environ.get("SKILLSPECTOR_LLM_COMMAND", "").strip()
        if not command:
            return None
        return ("subprocess", None)

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> SubprocessChatModel | None:
        """Return a SubprocessChatModel using the configured command, or None."""
        command = os.environ.get("SKILLSPECTOR_LLM_COMMAND", "").strip()
        if not command:
            return None
        return SubprocessChatModel(command=command, timeout=timeout or 120.0)

    def get_context_length(self, model: str) -> int | None:
        stored = registry.lookup_context_length(REGISTRY_PATH, model)
        return stored if stored is not None else _DEFAULT_CONTEXT_LENGTH

    def get_max_output_tokens(self, model: str) -> int | None:
        stored = registry.lookup_max_output_tokens(REGISTRY_PATH, model)
        return stored if stored is not None else _DEFAULT_MAX_OUTPUT_TOKENS

    def resolve_model(self, slot: str = "default") -> str:
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        return user_input or _SENTINEL_MODEL
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/providers/test_subprocess_provider.py -v
```
Expected: all 12 tests PASS

- [ ] **Step 6: Commit**

```
git add src/skillspector/providers/subprocess/ tests/providers/test_subprocess_provider.py
git commit -m "feat: add SubprocessProvider implementing LLMProvider protocol"
```

---

### Task 3: Register subprocess in provider selector

**Files:**
- Modify: `src/skillspector/providers/__init__.py` (lines 56–87 and the module docstring)
- Modify: `tests/providers/test_subprocess_provider.py` (append selector tests)

**Interfaces:**
- Consumes: `SubprocessProvider` from Task 2
- Produces: `_select_active_provider()` now returns `SubprocessProvider` when `SKILLSPECTOR_PROVIDER=subprocess`

- [ ] **Step 1: Write the failing tests**

Append to `tests/providers/test_subprocess_provider.py`:

```python
from skillspector.providers import _select_active_provider, create_chat_model


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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/providers/test_subprocess_provider.py::TestSubprocessProviderSelection -v
```
Expected: FAIL — `subprocess` not yet in selector

- [ ] **Step 3: Add `subprocess` to `_select_active_provider()` in `providers/__init__.py`**

Find the block starting at line 56 and update it. The change adds one `if` block and updates the docstring:

In the module docstring block (lines 26–31), add one line:

```python
#     subprocess       → SubprocessProvider      (configured shell command)
```

In `_select_active_provider()`, add after the `anthropic_proxy` block (after line 71) and before the `nv_build` block:

```python
    if name == "subprocess":
        from .subprocess import SubprocessProvider

        return SubprocessProvider()
```

Also update the `ValueError` message at the end of the function to include `subprocess`:

```python
    raise ValueError(
        f"Unknown SKILLSPECTOR_PROVIDER: {name!r}. "
        "Expected one of: openai, anthropic, anthropic_proxy, nv_build, subprocess (or unset)."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/providers/test_subprocess_provider.py -v
```
Expected: all 14 tests PASS

- [ ] **Step 5: Run the full unit test suite to check for regressions**

```
make test
```
Expected: all existing tests still PASS

- [ ] **Step 6: Commit**

```
git add src/skillspector/providers/__init__.py tests/providers/test_subprocess_provider.py
git commit -m "feat: register subprocess provider in provider selector"
```

---

### Task 4: Document the new provider in `.env.example`

**Files:**
- Modify: `.env.example`

**Interfaces:**
- Consumes: nothing from code; purely documentation.
- Produces: users know how to configure `SKILLSPECTOR_LLM_COMMAND`.

- [ ] **Step 1: Read the current `.env.example`**

Open `.env.example` and find the section that lists provider-specific credentials.

- [ ] **Step 2: Add the subprocess provider section**

After the existing provider blocks (NVIDIA, OpenAI, Anthropic), add:

```dotenv
# ---------------------------------------------------------------------------
# subprocess provider  (SKILLSPECTOR_PROVIDER=subprocess)
# ---------------------------------------------------------------------------
# Routes every LLM prompt through a shell command via stdin.
# Use this when running SkillSpector inside Claude Code, OpenClaw, Antigravity,
# or any other AI-tool session where the AI is the session itself.
#
# Examples:
#   SKILLSPECTOR_LLM_COMMAND=claude -p          # Claude Code
#   SKILLSPECTOR_LLM_COMMAND=antigravity ask    # Antigravity
#   SKILLSPECTOR_LLM_COMMAND=openclaw chat      # OpenClaw
#
# The prompt is written to the command's stdin; the response is read from stdout.
# No API key is required — the session AI handles the call.
SKILLSPECTOR_LLM_COMMAND=
```

- [ ] **Step 3: Verify the file is valid (no syntax errors in shell)**

```
python -c "
with open('.env.example') as f:
    content = f.read()
print('OK:', len(content), 'chars')
"
```
Expected: prints `OK:` with character count

- [ ] **Step 4: Commit**

```
git add .env.example
git commit -m "docs: document subprocess provider and SKILLSPECTOR_LLM_COMMAND in .env.example"
```

---

### Task 5: Smoke-test end-to-end inside Claude Code

This task has no code to commit — it verifies the full chain works when running from inside a Claude Code session.

- [ ] **Step 1: Set environment variables in your shell**

```powershell
$env:SKILLSPECTOR_PROVIDER = "subprocess"
$env:SKILLSPECTOR_LLM_COMMAND = "claude -p"
```

- [ ] **Step 2: Run a scan against the test fixtures**

```
skillspector scan tests/fixtures/malicious_skill --format terminal
```
Expected: SkillSpector runs to completion; findings are printed; no error about missing API key.

- [ ] **Step 3: Run with `--no-llm` to confirm static-only path still works**

```
skillspector scan tests/fixtures/malicious_skill --no-llm --format terminal
```
Expected: runs successfully; LLM meta_analyzer is skipped.

- [ ] **Step 4: Run with an invalid command to confirm error surfaces cleanly**

```powershell
$env:SKILLSPECTOR_LLM_COMMAND = "nonexistent-command-xyz"
skillspector scan tests/fixtures/malicious_skill --format terminal
```
Expected: a readable `RuntimeError` or `FileNotFoundError` (not a traceback about missing API key).

---

## Self-Review Checklist

- **Spec coverage:** All four requirements covered — (1) no API key needed, (2) runs from Claude Code session, (3) works with OpenClaw/Antigravity via configurable command, (4) model-agnostic.
- **Placeholder scan:** No TBDs. All code blocks are complete.
- **Type consistency:** `SubprocessChatModel.command` (str) → `SubprocessProvider.create_chat_model()` reads `SKILLSPECTOR_LLM_COMMAND` and passes it as `command=` — consistent across tasks.
- **Protocol compliance:** `SubprocessProvider` implements `get_context_length`, `get_max_output_tokens`, `resolve_model`, `resolve_credentials`, `create_chat_model` — all five methods required by `LLMProvider`.
- **No new dependencies:** Uses only stdlib `subprocess`, `shlex`, `json`, existing LangChain Core, and existing Pydantic — all already in `pyproject.toml`.
