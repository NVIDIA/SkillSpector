# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded JSON syntax ownership for command-parser completeness checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from skillspector.artifacts import SecurityTextView
from skillspector.security_reconstruction import (
    _MAX_JSON_QUOTE_CONTAINER_CHARS,
    MAX_STANDALONE_JSON_SOURCE_CHARS,
    _json_string_spans,
    _validate_json_without_decoding,
    _validated_json_ranges,
)

_PUNCTUATION = 1
_OPEN = 2
_CLOSE = 3
_STRING = 4


@dataclass(frozen=True)
class JSONParserProjection:
    """Length-preserving punctuation boundaries; string contents remain visible."""

    text: str
    openers: set[int]
    closers: set[int]
    strings: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class JSONSourceOwnership:
    """Syntax roles proven by one complete original JSON document."""

    source: str
    roles: bytes

    def project(
        self, view: SecurityTextView, check_runtime: Callable[[], None]
    ) -> JSONParserProjection | None:
        """Carry roles only through ordered, source-bound character mappings.

        A derived value can change string contents, whose security analysis
        stays active. It cannot remove an enclosing quote or manufacture host
        punctuation. Fragments inherit only delimiters actually in that view.
        """
        text = view.text
        if not text or len(text) > MAX_STANDALONE_JSON_SOURCE_CHARS:
            return None
        projected = list(text)
        openers: set[int] = set()
        closers: set[int] = set()
        strings: list[tuple[int, int]] = []
        string_start: int | None = None
        previous = -1
        for index, character in enumerate(text):
            if index % 1024 == 0:
                check_runtime()
            offset = view.source_offset(index)
            if not previous < offset < len(self.source):
                return None
            if (
                previous >= 0
                and offset > previous + 1
                and (
                    self.roles.find(bytes([_OPEN]), previous + 1, offset) >= 0
                    or self.roles.find(bytes([_CLOSE]), previous + 1, offset) >= 0
                )
            ):
                return None
            role = self.roles[offset]
            if role != _STRING and character != self.source[offset]:
                return None
            if role == _PUNCTUATION:
                projected[index] = " "
            elif role == _OPEN:
                openers.add(index)
                string_start = index
            elif role == _CLOSE:
                closers.add(index)
                if string_start is not None:
                    strings.append((string_start, index + 1))
                    string_start = None
            previous = offset
        check_runtime()
        return JSONParserProjection("".join(projected), openers, closers, tuple(strings))


def prepare_json_source(
    source: str, check_runtime: Callable[[], None]
) -> JSONSourceOwnership | None:
    """Prove standalone JSON without decoding large or deeply nested objects.

    Keep the established small-document decoder contract. The iterative grammar
    from quote ownership validates larger documents within an explicit 2 MiB
    source bound; no partial or malformed document grants syntax ownership.
    """
    check_runtime()
    if not source or len(source) > MAX_STANDALONE_JSON_SOURCE_CHARS:
        return None
    if len(source) <= _MAX_JSON_QUOTE_CONTAINER_CHARS:
        if _validated_json_ranges(source, check_runtime) != [(0, len(source))]:
            return None
    elif not _validate_json_without_decoding(
        source, check_runtime, max_characters=MAX_STANDALONE_JSON_SOURCE_CHARS
    ):
        return None
    roles = bytearray(len(source))
    cursor = 0
    for start, end in _json_string_spans(source, check_runtime):
        for offset in range(cursor, start):
            if offset % 1024 == 0:
                check_runtime()
            if source[offset] in "[]{}:,":
                roles[offset] = _PUNCTUATION
        roles[start] = _OPEN
        roles[start + 1 : end - 1] = bytes([_STRING]) * (end - start - 2)
        roles[end - 1] = _CLOSE
        cursor = end
    for offset in range(cursor, len(source)):
        if offset % 1024 == 0:
            check_runtime()
        if source[offset] in "[]{}:,":
            roles[offset] = _PUNCTUATION
    check_runtime()
    return JSONSourceOwnership(source, bytes(roles))
