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
MAX_SCHEMA_ERRORS = 5
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_SCHEMA_ERROR_TYPE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_CODE_FENCE = re.compile(r"```[A-Za-z0-9_+-]*")
# Completion labels that providers return verbatim; anything else is reported as "other".
_FINISH_LABELS = frozenset(
    {
        "stop",
        "end_turn",
        "tool_use",
        "tool_calls",
        "length",
        "max_tokens",
        "max_output_tokens",
        "refusal",
        "content_filter",
        "safety",
        "guardrail_intervened",
        "pause_turn",
        "completed",
        "incomplete",
        "in_progress",
        "queued",
        "failed",
        "cancelled",
    }
)


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


class ResponseCheck(StrEnum):
    """The specific validation check that rejected an answer.

    ``ResponseIssue`` selects the ledger reason; this finer code tells which
    bounded check failed without retaining any of the model's text.
    """

    SIZE_LIMIT = "size_limit"
    NESTING_LIMIT = "nesting_limit"
    NOT_JSON_OBJECT = "not_json_object"
    UNBALANCED_CONTAINERS = "unbalanced_containers"
    TRAILING_CONTENT = "trailing_content"
    INVALID_STRING = "invalid_string"
    NON_FINITE_NUMBER = "non_finite_number"
    NON_JSON_TOKEN = "non_json_token"
    MALFORMED_JSON = "malformed_json"
    DUPLICATE_FIELD = "duplicate_field"
    MISSING_FINDINGS = "missing_findings"
    MISSING_FIELD = "missing_field"
    INVALID_FIELD = "invalid_field"
    EMPTY_RESPONSE = "empty_response"
    NO_RESPONSE = "no_response"
    NO_USABLE_ANSWER = "no_usable_answer"
    NO_STRUCTURED_RESPONSE = "no_structured_response"
    TOOL_CALL_COUNT = "tool_call_count"
    TOOL_CALL_AMBIGUOUS = "tool_call_ambiguous"
    TOOL_CALL_UNEXPECTED = "tool_call_unexpected"
    TOOL_CALL_MISSING = "tool_call_missing"
    TOOL_ARGUMENTS_INVALID = "tool_arguments_invalid"
    TOOL_OUTPUT_INVALID = "tool_output_invalid"
    CLI_JSON_MISSING = "cli_json_missing"
    TRUNCATED = "truncated"
    REFUSED = "refused"
    INCOMPLETE = "incomplete"
    FORMAT_CHANGED = "format_changed"
    FORMAT_UNVERIFIED = "format_unverified"
    FORMAT_UNSUPPORTED = "format_unsupported"
    FORMAT_INPUT_LIMIT = "format_input_limit"


_SHAPED_CHECKS = frozenset(
    {ResponseCheck.NOT_JSON_OBJECT, ResponseCheck.TRAILING_CONTENT, ResponseCheck.NON_JSON_TOKEN}
)


class StructuredOutputParseError(ValueError):
    """Content-free diagnostic; response text is retained only for bounded recovery.

    ``check``, ``channel``, ``finish``, ``shape`` and ``schema_errors`` are fixed
    codes or schema-declared field paths, so they can be reported without the answer.
    """

    def __init__(
        self,
        message: str,
        *,
        issue: ResponseIssue = ResponseIssue.INVALID_JSON,
        repair_text: str | None = None,
        check: ResponseCheck | None = None,
        schema_errors: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.issue = issue
        self.repair_text = repair_text
        self.check = check
        self.schema_errors = schema_errors
        # Set by validate_message: whether the answer arrived as a tool call or
        # as text, and the provider's allowlisted completion label.
        self.channel: str | None = None
        self.finish: str | None = None
        # Set when text was not a bare JSON object: where prose or fences sat.
        self.shape: str | None = None

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
        raise StructuredOutputParseError(
            "response exceeds size limit",
            issue=ResponseIssue.LIMIT,
            check=ResponseCheck.SIZE_LIMIT,
        )
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
        raise StructuredOutputParseError(
            "response is not a complete JSON object", check=ResponseCheck.NOT_JSON_OBJECT
        )
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
                    "response exceeds nesting limit",
                    issue=ResponseIssue.LIMIT,
                    check=ResponseCheck.NESTING_LIMIT,
                )
            tokens.append(char)
            offset += 1
        elif char in "}]":
            if not stack or stack.pop() != char:
                raise StructuredOutputParseError(
                    "response has incomplete containers",
                    check=ResponseCheck.UNBALANCED_CONTAINERS,
                )
            tokens.append(char)
            offset += 1
            if not stack and offset != len(text):
                raise StructuredOutputParseError(
                    "response contains multiple objects", check=ResponseCheck.TRAILING_CONTENT
                )
        elif char == '"':
            try:
                value, end = decoder.raw_decode(text, offset)
            except json.JSONDecodeError as exc:
                raise StructuredOutputParseError(
                    "response has an incomplete string", check=ResponseCheck.INVALID_STRING
                ) from exc
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
                    raise StructuredOutputParseError(
                        "response has a non-finite number",
                        check=ResponseCheck.NON_FINITE_NUMBER,
                    )
                tokens.append(token)
                offset = number.end()
            elif literal:
                tokens.append(literal)
                offset += len(literal)
            else:
                raise StructuredOutputParseError(
                    "response contains non-JSON text", check=ResponseCheck.NON_JSON_TOKEN
                )
    if stack:
        raise StructuredOutputParseError(
            "response has incomplete containers", check=ResponseCheck.UNBALANCED_CONTAINERS
        )
    return tuple(tokens)


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StructuredOutputParseError(
                "response contains duplicate fields",
                issue=ResponseIssue.INVALID_SCHEMA,
                check=ResponseCheck.DUPLICATE_FIELD,
            )
        result[key] = value
    return result


def _declared_field_names(schema: type[BaseModel]) -> frozenset[str]:
    """Property names anywhere in *schema*'s JSON schema, including nested models."""
    names: set[str] = set()
    pending: list[object] = [schema.model_json_schema()]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                names.update(key for key in properties if isinstance(key, str))
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return frozenset(names)


def schema_error_paths(exc: ValidationError, schema: type[BaseModel]) -> tuple[str, ...]:
    """Return ``field.path:error_type`` pairs that never echo model-supplied text.

    Path parts are list indexes or names the schema declares; any other key,
    such as an unexpected field the model invented, is reported as ``*``.
    """
    try:
        names = _declared_field_names(schema)
        paths: list[str] = []
        for error in exc.errors(include_url=False, include_context=False, include_input=False):
            parts = [
                str(part)
                if isinstance(part, int) and not isinstance(part, bool)
                else part
                if isinstance(part, str) and part in names
                else "*"
                for part in error.get("loc", ())
            ]
            error_type = error.get("type")
            if not isinstance(error_type, str) or not _SCHEMA_ERROR_TYPE.fullmatch(error_type):
                error_type = "other"
            path = f"{'.'.join(parts) or '$'}:{error_type}"
            if path not in paths:
                paths.append(path)
            if len(paths) == MAX_SCHEMA_ERRORS:
                break
        return tuple(paths)
    except Exception:
        # A diagnostic must never replace the validation failure it describes.
        return ()


def _text_shape(text: str) -> str:
    """Fixed labels for prose or fences around a JSON-looking object, never the text."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return "no_object"
    before = text[:start]
    after = text[end + 1 :]
    labels = [
        label
        for label, present in (
            ("prefix", bool(_CODE_FENCE.sub("", before).strip())),
            ("fence", "```" in before or "```" in after),
            ("suffix", bool(_CODE_FENCE.sub("", after).strip())),
        )
        if present
    ]
    return "+".join(labels) or "object"


def validate_payload(payload: dict, schema: type[BaseModel]) -> object:
    """Require an explicit findings list even when the internal model has a default."""
    fields = getattr(schema, "model_fields", {})
    if "findings" in fields and "findings" not in payload:
        raise StructuredOutputParseError(
            "response omits findings",
            issue=ResponseIssue.MISSING_FIELDS,
            check=ResponseCheck.MISSING_FINDINGS,
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
            check=ResponseCheck.MISSING_FIELD if missing else ResponseCheck.INVALID_FIELD,
            schema_errors=schema_error_paths(exc, schema),
        ) from exc


def parse_response_object(text: str) -> dict:
    """Parse bounded JSON without accepting duplicate fields or non-finite numbers."""
    body = _json_body(text)
    formatting_tokens(body)
    try:
        payload = json.loads(body, object_pairs_hook=_unique_fields)
    except json.JSONDecodeError as exc:
        raise StructuredOutputParseError(
            "model returned malformed JSON", check=ResponseCheck.MALFORMED_JSON
        ) from exc
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
            "model returned an empty response",
            issue=ResponseIssue.EMPTY,
            check=ResponseCheck.EMPTY_RESPONSE,
        )
    # Bound nesting before invoking the recursive JSON decoder.
    try:
        tokens = formatting_tokens(body)
    except StructuredOutputParseError as exc:
        if exc.check in _SHAPED_CHECKS:
            exc.shape = _text_shape(body)
        raise
    try:
        payload = json.loads(body, object_pairs_hook=_unique_fields)
    except json.JSONDecodeError as exc:
        raise StructuredOutputParseError(
            "model returned malformed JSON",
            repair_text=body if completed and preserve_text is None else None,
            check=ResponseCheck.MALFORMED_JSON,
        ) from exc
    if preserve_text is not None and tokens != formatting_tokens(preserve_text):
        raise StructuredOutputParseError(
            "formatting changed assessment content",
            issue=ResponseIssue.FORMAT_CHANGED,
            check=ResponseCheck.FORMAT_CHANGED,
        )
    if allow_json_wrapper and len(payload) == 1:
        # Exact gateway argument envelopes observed on completed live responses.
        # Only a sole known wrapper may be removed; ambiguous extra data is rejected.
        key = next(iter(payload))
        if key in {"json", "input", "parameters"} and isinstance(payload[key], dict):
            payload = payload[key]
    return validate_payload(payload, schema)


def _finish_label(message: object) -> str:
    metadata = getattr(message, "response_metadata", {}) or {}
    stop = (
        metadata.get("finish_reason") or metadata.get("stop_reason") or metadata.get("stopReason")
    )
    if not isinstance(stop, str):
        stop = metadata.get("status")
    if not isinstance(stop, str):
        return "none"
    stop = stop.lower()
    return stop if stop in _FINISH_LABELS else "other"


def _answer_channel(message: object) -> str:
    if getattr(message, "tool_calls", None) or getattr(message, "invalid_tool_calls", None):
        return "tool_call"
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return "text" if content.strip() else "empty"
    return "content_blocks" if isinstance(content, list) else "none"


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
    try:
        return _validate_message(
            message,
            schema,
            tool=tool,
            require_tool=require_tool,
            allow_json_wrapper=allow_json_wrapper,
            preserve_text=preserve_text,
        )
    except StructuredOutputParseError as exc:
        try:
            exc.channel = _answer_channel(message)
            exc.finish = _finish_label(message)
        except Exception:
            # A diagnostic must never replace the validation failure it describes.
            pass
        raise


def _validate_message(
    message: object,
    schema: type[BaseModel],
    *,
    tool: str,
    require_tool: bool,
    allow_json_wrapper: bool,
    preserve_text: str | None,
) -> object:
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
            "model answer was cut short",
            issue=ResponseIssue.TRUNCATED,
            check=ResponseCheck.TRUNCATED,
        )
    if (
        additional.get("refusal")
        or stop in {"refusal", "content_filter", "safety", "guardrail_intervened"}
        or isinstance(content, list)
        and any(isinstance(block, dict) and block.get("type") == "refusal" for block in content)
    ):
        raise StructuredOutputParseError(
            "model declined the assessment",
            issue=ResponseIssue.REFUSED,
            check=ResponseCheck.REFUSED,
        )
    if (
        status in {"incomplete", "in_progress", "queued", "failed", "cancelled"}
        or stop == "pause_turn"
    ):
        raise StructuredOutputParseError(
            "model did not finish the assessment",
            issue=ResponseIssue.INCOMPLETE,
            check=ResponseCheck.INCOMPLETE,
        )
    completed = stop in {"stop", "end_turn", "tool_use", "tool_calls"} or status == "completed"
    if preserve_text is not None and not completed:
        raise StructuredOutputParseError(
            "formatting response has no completion evidence",
            issue=ResponseIssue.FORMAT_CHANGED,
            check=ResponseCheck.FORMAT_UNVERIFIED,
        )
    calls = list(getattr(message, "tool_calls", []) or []) + list(
        getattr(message, "invalid_tool_calls", []) or []
    )
    if require_tool or calls:
        if len(calls) != 1 or calls[0].get("name") != tool:
            raise StructuredOutputParseError(
                f"model must return exactly one {tool} tool call",
                issue=ResponseIssue.INVALID_TOOL,
                check=ResponseCheck.TOOL_CALL_COUNT,
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
                    "model returned ambiguous tool calls",
                    issue=ResponseIssue.INVALID_TOOL,
                    check=ResponseCheck.TOOL_CALL_AMBIGUOUS,
                )
            function = wire_calls[0].get("function") if isinstance(wire_calls[0], dict) else None
            if not isinstance(function, dict) or function.get("name") != tool:
                raise StructuredOutputParseError(
                    "model returned an unexpected tool",
                    issue=ResponseIssue.INVALID_TOOL,
                    check=ResponseCheck.TOOL_CALL_UNEXPECTED,
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
            "model returned no usable answer",
            issue=ResponseIssue.EMPTY,
            check=ResponseCheck.NO_USABLE_ANSWER,
        )
    return validate_response_text(
        text,
        schema,
        completed=completed,
        allow_json_wrapper=allow_json_wrapper,
        preserve_text=preserve_text,
    )
