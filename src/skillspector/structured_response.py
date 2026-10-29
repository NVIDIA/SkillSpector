# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate model answers and admit only lossless formatting recovery."""

from __future__ import annotations

import json
import math
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError

MAX_RESPONSE_CHARS = 1_048_576
MAX_RESPONSE_DEPTH = 64
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


class ResponseIssue(StrEnum):
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"
    MISSING_FIELDS = "missing_fields"
    INVALID_TOOL = "invalid_tool"
    EMPTY = "empty"
    REFUSED = "refused"
    TRUNCATED = "truncated"
    INCOMPLETE = "incomplete"
    LIMIT = "response_limit"
    FORMAT_CHANGED = "format_changed"


class StructuredOutputParseError(ValueError):
    """Content-free diagnostic; response text is retained only for bounded recovery."""

    def __init__(
        self,
        message: str,
        *,
        issue: ResponseIssue = ResponseIssue.INVALID_JSON,
        repair_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.issue = issue
        self.repair_text = repair_text

    @property
    def retryable(self) -> bool:
        return self.issue not in {
            ResponseIssue.REFUSED,
            ResponseIssue.TRUNCATED,
            ResponseIssue.INCOMPLETE,
            ResponseIssue.LIMIT,
            ResponseIssue.FORMAT_CHANGED,
        }


def _json_body(text: str) -> str:
    if len(text) > MAX_RESPONSE_CHARS:
        raise StructuredOutputParseError("response exceeds size limit", issue=ResponseIssue.LIMIT)
    text = text.strip()
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0] in ("```", "```json") and lines[-1] == "```":
        text = "\n".join(lines[1:-1]).strip()
    return text


def formatting_tokens(text: str) -> tuple[str, ...]:
    """Keep every field, value and container; only commas/colons/whitespace may change.

    Requiring balanced containers and complete strings excludes clipped answers.
    Free prose is deliberately ineligible: its meaning cannot be checked losslessly.
    """
    text = _json_body(text)
    if not text.startswith("{") or not text.endswith("}"):
        raise StructuredOutputParseError("response is not a complete JSON object")
    decoder = json.JSONDecoder()
    tokens: list[str] = []
    stack: list[str] = []
    offset = 0
    while offset < len(text):
        char = text[offset]
        if char.isspace() or char in ",:":
            offset += 1
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
            if len(stack) > MAX_RESPONSE_DEPTH:
                raise StructuredOutputParseError(
                    "response exceeds nesting limit", issue=ResponseIssue.LIMIT
                )
            tokens.append(char)
            offset += 1
        elif char in "}]":
            if not stack or stack.pop() != char:
                raise StructuredOutputParseError("response has incomplete containers")
            tokens.append(char)
            offset += 1
            if not stack and offset != len(text):
                raise StructuredOutputParseError("response contains multiple objects")
        elif char == '"':
            try:
                value, end = decoder.raw_decode(text, offset)
            except json.JSONDecodeError as exc:
                raise StructuredOutputParseError("response has an incomplete string") from exc
            tokens.append(json.dumps(value, ensure_ascii=True))
            offset = end
        else:
            number = _NUMBER.match(text, offset)
            literal = next(
                (v for v in ("true", "false", "null") if text.startswith(v, offset)), None
            )
            if number:
                token = number.group()
                if not math.isfinite(float(token)):
                    raise StructuredOutputParseError("response has a non-finite number")
                tokens.append(token)
                offset = number.end()
            elif literal:
                tokens.append(literal)
                offset += len(literal)
            else:
                raise StructuredOutputParseError("response contains non-JSON text")
    if stack:
        raise StructuredOutputParseError("response has incomplete containers")
    return tuple(tokens)


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StructuredOutputParseError(
                "response contains duplicate fields", issue=ResponseIssue.INVALID_SCHEMA
            )
        result[key] = value
    return result


def validate_payload(payload: dict, schema: type[BaseModel]) -> object:
    """Require an explicit findings list even when the internal model has a default."""
    fields = getattr(schema, "model_fields", {})
    if "findings" in fields and "findings" not in payload:
        raise StructuredOutputParseError(
            "response omits findings", issue=ResponseIssue.MISSING_FIELDS
        )
    try:
        return schema.model_validate(payload, extra="forbid")
    except ValidationError as exc:
        missing = any(error["type"] == "missing" for error in exc.errors())
        raise StructuredOutputParseError(
            "response omits required fields"
            if missing
            else "response does not match the required fields",
            issue=ResponseIssue.MISSING_FIELDS if missing else ResponseIssue.INVALID_SCHEMA,
        ) from exc


def parse_response_object(text: str) -> dict:
    """Parse bounded JSON without accepting duplicate fields or non-finite numbers."""
    body = _json_body(text)
    formatting_tokens(body)
    try:
        payload = json.loads(body, object_pairs_hook=_unique_fields)
    except json.JSONDecodeError as exc:
        raise StructuredOutputParseError("model returned malformed JSON") from exc
    assert isinstance(payload, dict)
    return payload


def validate_response_text(
    text: str,
    schema: type[BaseModel],
    *,
    completed: bool = False,
    allow_json_wrapper: bool = False,
    preserve_text: str | None = None,
) -> object:
    body = _json_body(text)
    if not body:
        raise StructuredOutputParseError(
            "model returned an empty response", issue=ResponseIssue.EMPTY
        )
    # Bound nesting before invoking the recursive JSON decoder.
    tokens = formatting_tokens(body)
    try:
        payload = json.loads(body, object_pairs_hook=_unique_fields)
    except json.JSONDecodeError as exc:
        raise StructuredOutputParseError(
            "model returned malformed JSON",
            repair_text=body if completed and preserve_text is None else None,
        ) from exc
    if preserve_text is not None and tokens != formatting_tokens(preserve_text):
        raise StructuredOutputParseError(
            "formatting changed assessment content", issue=ResponseIssue.FORMAT_CHANGED
        )
    if allow_json_wrapper and set(payload) == {"json"} and isinstance(payload["json"], dict):
        payload = payload["json"]
    return validate_payload(payload, schema)


def validate_message(
    message: object,
    schema: type[BaseModel],
    *,
    tool: str,
    require_tool: bool = False,
    allow_json_wrapper: bool = False,
    preserve_text: str | None = None,
) -> object:
    """Read stop/refusal evidence before parsing either text or tool arguments."""
    metadata = getattr(message, "response_metadata", {}) or {}
    additional = getattr(message, "additional_kwargs", {}) or {}
    content = getattr(message, "content", "")
    stop = (
        metadata.get("finish_reason") or metadata.get("stop_reason") or metadata.get("stopReason")
    )
    stop = stop.lower() if isinstance(stop, str) else None
    status = metadata.get("status")
    incomplete = metadata.get("incomplete_details") or {}
    if not stop and isinstance(incomplete, dict):
        stop = incomplete.get("reason")
    if stop in {"length", "max_tokens", "max_output_tokens"}:
        raise StructuredOutputParseError(
            "model answer was cut short", issue=ResponseIssue.TRUNCATED
        )
    if (
        additional.get("refusal")
        or stop in {"refusal", "content_filter", "safety", "guardrail_intervened"}
        or isinstance(content, list)
        and any(isinstance(block, dict) and block.get("type") == "refusal" for block in content)
    ):
        raise StructuredOutputParseError(
            "model declined the assessment", issue=ResponseIssue.REFUSED
        )
    if (
        status in {"incomplete", "in_progress", "queued", "failed", "cancelled"}
        or stop == "pause_turn"
    ):
        raise StructuredOutputParseError(
            "model did not finish the assessment", issue=ResponseIssue.INCOMPLETE
        )
    completed = stop in {"stop", "end_turn", "tool_use", "tool_calls"} or status == "completed"
    if preserve_text is not None and not completed:
        raise StructuredOutputParseError(
            "formatting response has no completion evidence", issue=ResponseIssue.FORMAT_CHANGED
        )
    calls = list(getattr(message, "tool_calls", []) or []) + list(
        getattr(message, "invalid_tool_calls", []) or []
    )
    if require_tool or calls:
        if len(calls) != 1 or calls[0].get("name") != tool:
            raise StructuredOutputParseError(
                f"model must return exactly one {tool} tool call", issue=ResponseIssue.INVALID_TOOL
            )
        args = calls[0].get("args")
        wire_calls = additional.get("tool_calls")
        if wire_calls is None and isinstance(content, list):
            # The Responses API keeps original arguments here, while its parsed
            # tool_calls may already have discarded duplicate fields.
            wire_calls = [
                {"function": block}
                for block in content
                if isinstance(block, dict) and block.get("type") == "function_call"
            ] or None
        if wire_calls is not None:
            if not isinstance(wire_calls, list) or len(wire_calls) != 1:
                raise StructuredOutputParseError(
                    "model returned ambiguous tool calls", issue=ResponseIssue.INVALID_TOOL
                )
            function = wire_calls[0].get("function") if isinstance(wire_calls[0], dict) else None
            if not isinstance(function, dict) or function.get("name") != tool:
                raise StructuredOutputParseError(
                    "model returned an unexpected tool", issue=ResponseIssue.INVALID_TOOL
                )
            args = function.get("arguments")
        text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=True)
    elif isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    else:
        raise StructuredOutputParseError(
            "model returned no usable answer", issue=ResponseIssue.EMPTY
        )
    return validate_response_text(
        text,
        schema,
        completed=completed,
        allow_json_wrapper=allow_json_wrapper,
        preserve_text=preserve_text,
    )
