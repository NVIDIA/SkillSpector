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
    ResponseIssue,
    StructuredOutputParseError,
)


def parse_recovery_value(text: str, annotation: Any, *, field_name: str | None = None) -> Any:
    """Accept plain strings and unambiguous scalar answers; reject inferred values."""
    if len(text) > MAX_RESPONSE_CHARS:
        raise StructuredOutputParseError(
            "field response exceeds size limit", issue=ResponseIssue.LIMIT
        )
    value = text.strip()
    if not value:
        raise StructuredOutputParseError("field response is empty", issue=ResponseIssue.EMPTY)
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) in {Union, UnionType}:
        alternatives = [item for item in get_args(annotation) if item is not type(None)]
        if len(alternatives) == 1:
            annotation = alternatives[0]
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
    if get_origin(annotation) is Literal:
        choices = get_args(annotation)
    elif isinstance(annotation, type) and issubclass(annotation, Enum):
        choices = tuple(annotation)
    else:
        choices = ()
    if choices:
        matches = []
        for choice in choices:
            actual = choice.value if isinstance(choice, Enum) else choice
            if type(decoded) is type(actual) and (
                decoded.casefold() == actual.casefold()
                if isinstance(actual, str)
                else decoded == actual
            ):
                matches.append(choice)
        if len(matches) == 1:
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
