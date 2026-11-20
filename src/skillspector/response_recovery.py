# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse one explicitly requested response field without guessing its meaning."""

from __future__ import annotations

import json
import math
import re
from enum import Enum
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from skillspector.structured_response import (
    MAX_RESPONSE_CHARS,
    MAX_RESPONSE_DEPTH,
    ResponseIssue,
    StructuredOutputParseError,
    formatting_tokens,
)


def recover_payload(text: str) -> dict[str, Any] | None:
    """Recover punctuation only, retaining every original key, value and container."""
    try:
        tokens = formatting_tokens(text)
        position = 0

        def read_value(depth: int) -> Any:
            nonlocal position
            if depth > MAX_RESPONSE_DEPTH or position >= len(tokens):
                raise ValueError("missing or over-nested value")
            token = tokens[position]
            position += 1
            if token == "{":
                result: dict[str, Any] = {}
                while position < len(tokens) and tokens[position] != "}":
                    key_token = tokens[position]
                    if not key_token.startswith('"'):
                        raise ValueError("object key is not a string")
                    key = json.loads(key_token)
                    if key in result:
                        raise ValueError("duplicate object key")
                    position += 1
                    result[key] = read_value(depth + 1)
                if position >= len(tokens):
                    raise ValueError("missing object terminator")
                position += 1
                return result
            if token == "[":
                values: list[Any] = []
                while position < len(tokens) and tokens[position] != "]":
                    values.append(read_value(depth + 1))
                if position >= len(tokens):
                    raise ValueError("missing array terminator")
                position += 1
                return values
            if token in {"}", "]"}:
                raise ValueError("missing value")
            return json.loads(token)

        payload = read_value(0)
        return payload if isinstance(payload, dict) and position == len(tokens) else None
    except (ValueError, RecursionError):
        return None


def _recovery_annotation(annotation: Any) -> Any:
    while True:
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        elif get_origin(annotation) in {Union, UnionType}:
            alternatives = [item for item in get_args(annotation) if item is not type(None)]
            if len(alternatives) != 1:
                return annotation
            annotation = alternatives[0]
        else:
            return annotation


def recovery_field_choices(annotation: Any) -> tuple[Any, ...]:
    """Expose the same categorical values accepted by the scalar parser."""
    annotation = _recovery_annotation(annotation)
    if get_origin(annotation) is Literal:
        return get_args(annotation)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple(choice.value for choice in annotation)
    return ()


def parse_recovery_value(text: str, annotation: Any, *, field_name: str | None = None) -> Any:
    """Accept plain strings and unambiguous scalar answers; reject inferred values."""
    if len(text) > MAX_RESPONSE_CHARS:
        raise StructuredOutputParseError(
            "field response exceeds size limit", issue=ResponseIssue.LIMIT
        )
    value = text.strip()
    if not value:
        raise StructuredOutputParseError("field response is empty", issue=ResponseIssue.EMPTY)
    annotation = _recovery_annotation(annotation)
    if annotation is str:
        return value
    if field_name:
        label = re.escape(field_name).replace("_", "[ _]")
        value = re.sub(rf"^{label}\s*:\s*", "", value, count=1, flags=re.IGNORECASE)
    try:
        decoded = json.loads(value)
    except (ValueError, RecursionError):
        decoded = value
    if isinstance(decoded, float) and not math.isfinite(decoded):
        raise StructuredOutputParseError(
            "field response is not finite", issue=ResponseIssue.INVALID_SCHEMA
        )
    choices = recovery_field_choices(annotation)
    if choices:
        matches = []
        for actual in choices:
            if type(decoded) is type(actual) and (
                decoded.casefold() == actual.casefold()
                if isinstance(actual, str)
                else decoded == actual
            ):
                matches.append(actual)
        if len(matches) == 1:
            if isinstance(annotation, type) and issubclass(annotation, Enum):
                return annotation(matches[0])
            return matches[0]
    elif annotation is bool:
        if type(decoded) is bool:
            return decoded
        if isinstance(decoded, str) and decoded.casefold() in {"true", "false", "yes", "no"}:
            return decoded.casefold() in {"true", "yes"}
    elif annotation is int and type(decoded) is int:
        return decoded
    elif annotation is float and type(decoded) in {int, float}:
        try:
            number = float(decoded)
        except OverflowError:
            pass
        else:
            if math.isfinite(number):
                return number
    raise StructuredOutputParseError(
        "field response does not contain one unambiguous value", issue=ResponseIssue.INVALID_SCHEMA
    )
