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
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from skillspector.providers import registry

_DEFAULT_TIMEOUT = 120.0
_DEFAULT_CONTEXT_LENGTH = 200_000
_DEFAULT_MAX_OUTPUT_TOKENS = 8_192
_SENTINEL_MODEL = "subprocess"
REGISTRY_PATH = str(Path(__file__).parent / "model_registry.yaml")


def _augment_messages_with_json_instruction(
    messages: list[BaseMessage], schema_str: str
) -> list[BaseMessage]:
    """Append JSON schema instruction to the last HumanMessage."""
    instruction = (
        "\n\n---\nRespond with a single valid JSON object that conforms to "
        "this JSON Schema (no markdown fences, no explanation, only JSON):\n"
        f"{schema_str}"
    )
    augmented: list[BaseMessage] = []
    for i, msg in enumerate(messages):
        if i == len(messages) - 1 and isinstance(msg, HumanMessage):
            augmented.append(HumanMessage(content=msg.content + instruction))
        else:
            augmented.append(msg)
    return augmented


def _strip_fences(text: str) -> str:
    """Strip markdown code fences from a string."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return clean


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
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        text_parts.append(item.get("text", ""))
                parts.append("\n".join(p for p in text_parts if p))
            else:
                parts.append(str(content))
    return "\n\n".join(parts)


class SubprocessChatModel(BaseChatModel):
    """A LangChain chat model that routes calls through a shell command.

    The full prompt is written to the subprocess stdin; stdout is the response.
    """

    command: str = Field(description="Shell command to invoke (split on whitespace)")
    timeout: float = Field(default=_DEFAULT_TIMEOUT, description="Seconds before subprocess times out")

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
        args = shlex.split(self.command, posix=(os.name != "nt"))
        try:
            result = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"LLM subprocess timed out after {self.timeout}s (command: {self.command!r})"
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"LLM subprocess failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout.strip()

    def with_structured_output(
        self,
        schema: type | dict[str, Any],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Return a Runnable that appends JSON-schema instructions and parses output.

        Because subprocess models cannot use native tool-calling, structured
        output is implemented by:
        1. Appending JSON schema + instructions to the last human message.
        2. Calling _generate() normally.
        3. Parsing the JSON from the response with Pydantic (for BaseModel) or
           json.loads (for dict schemas).
        """
        if isinstance(schema, dict):
            schema_str = json.dumps(schema, indent=2)

            def inject_and_parse_dict(messages: list[BaseMessage]) -> Any:
                augmented = _augment_messages_with_json_instruction(messages, schema_str)
                raw_text = self.invoke(augmented).content
                clean = _strip_fences(raw_text)
                return json.loads(clean)

            return RunnableLambda(inject_and_parse_dict)
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_str = json.dumps(schema.model_json_schema(), indent=2)

            def inject_and_parse(messages: list[BaseMessage]) -> BaseModel:
                augmented = _augment_messages_with_json_instruction(messages, schema_str)
                raw_text = self.invoke(augmented).content
                clean = _strip_fences(raw_text)
                return schema.model_validate_json(clean)

            return RunnableLambda(inject_and_parse)
        else:
            raise TypeError(
                f"SubprocessChatModel.with_structured_output requires a Pydantic BaseModel subclass "
                f"or a dict JSON Schema, got {type(schema)!r}."
            )


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

    DEFAULT_MODEL: str = _SENTINEL_MODEL
    SLOT_DEFAULTS: dict[str, str] = {}

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
    ) -> SubprocessChatModel:
        """Return a SubprocessChatModel using the configured command.

        Raises ValueError if SKILLSPECTOR_LLM_COMMAND is not set.
        """
        command = os.environ.get("SKILLSPECTOR_LLM_COMMAND", "").strip()
        if not command:
            raise ValueError(
                "SKILLSPECTOR_PROVIDER=subprocess requires SKILLSPECTOR_LLM_COMMAND to be set. "
                "Example: SKILLSPECTOR_LLM_COMMAND=claude -p"
            )
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
