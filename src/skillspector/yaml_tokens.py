# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded scalar and comment ownership from a complete YAML grammar parse."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

import yaml
from yaml.events import CollectionEndEvent, CollectionStartEvent
from yaml.tokens import AliasToken, AnchorToken, ScalarToken, TagToken

_MAX_SOURCE_CHARS = 1_000_000
_MAX_EVENTS = 32_768
_MAX_TOKENS = 16_384
_MAX_DEPTH = 64
_CACHE_SIZE = 8
_CACHE_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class YamlScalar:
    start: int
    end: int
    body_start: int
    body_end: int
    style: str | None
    value: str


@dataclass(frozen=True, slots=True)
class YamlSourceSpans:
    scalars: tuple[YamlScalar, ...]
    comments: tuple[tuple[int, int], ...]


_CACHE: OrderedDict[tuple[int, int], tuple[str, YamlSourceSpans | None]] = OrderedDict()


def yaml_source_spans(content: str, check_runtime: Callable[[], None]) -> YamlSourceSpans | None:
    """Prove scalar and comment bounds without constructing YAML values.

    Tags and aliases are parsed as syntax only. No constructors, alias
    expansion, or skill code execute. A malformed or oversized stream gives
    no ownership; callers keep their existing conservative reading.
    """
    if len(content) > _MAX_SOURCE_CHARS:
        return None
    key = (len(content), hash(content))
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == content:
            _CACHE.move_to_end(key)
        else:
            cached = None
    check_runtime()
    if cached is not None:
        return cached[1]
    result = _parse(content, check_runtime)
    with _CACHE_LOCK:
        _CACHE[key] = (content, result)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return result


def _parse(content: str, check_runtime: Callable[[], None]) -> YamlSourceSpans | None:
    scalars: list[YamlScalar] = []
    comments: list[tuple[int, int]] = []
    try:
        depth = 0
        for index, event in enumerate(yaml.parse(content, Loader=yaml.SafeLoader)):
            check_runtime()
            if index == _MAX_EVENTS:
                return None
            if isinstance(event, CollectionStartEvent):
                depth += 1
                if depth > _MAX_DEPTH:
                    return None
            elif isinstance(event, CollectionEndEvent):
                depth -= 1
        if depth:
            return None
        previous_end = 0
        for index, token in enumerate(yaml.scan(content, Loader=yaml.SafeLoader)):
            check_runtime()
            if index == _MAX_TOKENS:
                return None
            if isinstance(token, (AliasToken, AnchorToken, TagToken)):
                return None
            start, end = token.start_mark.index, token.end_mark.index
            if not 0 <= start <= end <= len(content):
                return None
            cursor = previous_end
            while cursor < start:
                if cursor % 4096 == 0:
                    check_runtime()
                if content[cursor].isspace() or cursor == 0 and content[cursor] == "\ufeff":
                    cursor += 1
                elif content[cursor] == "#":
                    newline = content.find("\n", cursor, start)
                    comment_end = start if newline < 0 else newline
                    comments.append((cursor, comment_end))
                    cursor = comment_end
                else:
                    return None
            previous_end = max(previous_end, end)
            if not isinstance(token, ScalarToken):
                continue
            body_start, body_end = start, end
            if token.style in {"'", '"'}:
                if (
                    content[start : start + 1] != token.style
                    or content[end - 1 : end] != token.style
                ):
                    return None
                body_start, body_end = start + 1, end - 1
            elif token.style in {"|", ">"}:
                newline = content.find("\n", start, end)
                body_start = end if newline < 0 else newline + 1
            scalars.append(YamlScalar(start, end, body_start, body_end, token.style, token.value))
    except (yaml.YAMLError, ValueError, RecursionError):
        return None
    check_runtime()
    return YamlSourceSpans(tuple(scalars), tuple(comments))
