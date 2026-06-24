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
import shlex
import subprocess
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

_DEFAULT_TIMEOUT = 120.0


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
                text_parts = [item if isinstance(item, str) else "" for item in content]
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
        3. Parsing the JSON from the response with Pydantic.
        """
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(
                "SubprocessChatModel.with_structured_output requires a Pydantic BaseModel subclass."
            )
        json_schema = schema.model_json_schema()
        schema_str = json.dumps(json_schema, indent=2)
        instruction = (
            "\n\n---\nRespond with a single valid JSON object that conforms to "
            "this JSON Schema (no markdown fences, no explanation, only JSON):\n"
            f"{schema_str}"
        )

        def inject_and_parse(messages: list[BaseMessage]) -> BaseModel:
            augmented: list[BaseMessage] = []
            for i, msg in enumerate(messages):
                if i == len(messages) - 1 and isinstance(msg, HumanMessage):
                    augmented.append(HumanMessage(content=msg.content + instruction))
                else:
                    augmented.append(msg)
            raw_text = self.invoke(augmented).content
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return schema.model_validate_json(clean)

        return RunnableLambda(inject_and_parse)
