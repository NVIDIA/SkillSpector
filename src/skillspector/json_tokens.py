# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded JSON syntax ownership for command-parser completeness checks."""

from __future__ import annotations

import json
from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO

from skillspector.artifacts import SecurityTextView, security_text_views
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
class JSONDecodedSource:
    """A decoded document and original spans for every projected character."""

    source: str
    view: SecurityTextView | None
    unsupported: tuple[tuple[int, int], ...] = ()

    def owns(self, view: SecurityTextView, check_runtime: Callable[[], None]) -> bool:
        """Require exact decoded-source provenance, not a caller-supplied name."""
        canonical = self.view
        if canonical is None or not view.name.startswith("json-decoded-") or not view.text:
            return False
        offsets, ends = canonical.source_offsets, canonical.source_end_offsets
        assert offsets is not None and ends is not None
        begin = bisect_left(offsets, view.source_offset(0))
        end = bisect_right(ends, view.source_end_offset(len(view.text) - 1))
        if begin >= end:
            return False
        # Prove the exact trusted transformation over this canonical interval.
        # Marker removal and renamed views cannot borrow ownership, while
        # normalized/compact commands retain their finding-provenance check.
        for expected in security_text_views(canonical.text[begin:end], check_runtime):
            if view.name != "json-decoded-" + expected.name or view.text != expected.text:
                continue
            for index in range(len(view.text)):
                if index % 1024 == 0:
                    check_runtime()
                left = begin + expected.source_offset(index)
                right = begin + expected.source_end_offset(index) - 1
                if view.source_offset(index) != canonical.source_offset(
                    left
                ) or view.source_end_offset(index) != canonical.source_end_offset(right):
                    break
            else:
                return True
        return False

    def window_views(
        self,
        start: int,
        end: int,
        check_runtime: Callable[[], None],
        *,
        right_context: int = 0,
    ) -> tuple[SecurityTextView, ...]:
        """Use ordinary security views and budgets within one original window."""
        canonical = self.view
        if canonical is None:
            return ()
        offsets, ends = canonical.source_offsets, canonical.source_end_offsets
        assert offsets is not None and ends is not None
        begin = bisect_left(offsets, start)
        finish = min(len(offsets), bisect_right(ends, end) + right_context)
        if begin >= finish:
            return ()
        text = canonical.text[begin:finish]
        if text == self.source[start:end]:
            return ()
        views: list[SecurityTextView] = []
        for view in security_text_views(text, check_runtime):
            mapped, mapped_ends = array("I"), array("I")
            for index in range(len(view.text)):
                if index % 1024 == 0:
                    check_runtime()
                left = begin + view.source_offset(index)
                right = begin + view.source_end_offset(index) - 1
                mapped.append(offsets[left] - start)
                mapped_ends.append(ends[right] - start)
            views.append(
                SecurityTextView(
                    "json-decoded-" + view.name,
                    view.text,
                    mapped,
                    right_boundary_is_fixed=finish < len(offsets),
                    source_end_offsets=mapped_ends,
                )
            )
        return tuple(views)


def prepare_decoded_json_source(
    ownership: JSONSourceOwnership, check_runtime: Callable[[], None]
) -> JSONDecodedSource | None:
    """Decode proven JSON string tokens without constructing document objects.

    The established grammar has already bounded the complete source. Only a
    string token is handed to the standard decoder. Mapping walks that decoder's
    result and retains the whole original escape, including surrogate pairs.
    Lone surrogates stay raw and explicitly unsupported.
    """
    source = ownership.source
    if "\\" not in source:
        return None
    output = StringIO()
    offsets, ends = array("I"), array("I")
    unsupported: list[tuple[int, int]] = []
    cursor = 0
    changed = False

    def append_raw(start: int, end: int) -> None:
        output.write(source[start:end])
        offsets.extend(range(start, end))
        ends.extend(range(start + 1, end + 1))

    for start, end in _json_string_spans(source, check_runtime):
        check_runtime()
        if source.find("\\", start, end) < 0:
            continue
        value = json.loads(source[start:end])
        invalid_unicode = False
        for index, character in enumerate(value):
            if index % 1024 == 0:
                check_runtime()
            if 0xD800 <= ord(character) <= 0xDFFF:
                invalid_unicode = True
                break
        if invalid_unicode:
            unsupported.append((start, end))
            continue
        append_raw(cursor, start + 1)
        output.write(value)
        raw_cursor = start + 1
        for index, character in enumerate(value):
            if index % 1024 == 0:
                check_runtime()
            width = 1
            if source[raw_cursor] == "\\":
                width = (
                    (12 if ord(character) > 0xFFFF else 6) if source[raw_cursor + 1] == "u" else 2
                )
            offsets.append(raw_cursor)
            raw_cursor += width
            ends.append(raw_cursor)
        if raw_cursor != end - 1:
            # A source/decoder contract mismatch must never publish a map.
            return JSONDecodedSource(source, None, ((start, end),))
        cursor = end - 1
        changed = True
    check_runtime()
    if not changed:
        return JSONDecodedSource(source, None, tuple(unsupported)) if unsupported else None
    append_raw(cursor, len(source))
    view = SecurityTextView("json-decoded-raw", output.getvalue(), offsets, source_end_offsets=ends)
    return JSONDecodedSource(source, view, tuple(unsupported))


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
    decoded: JSONDecodedSource | None = None

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
        repeated_source_owned: bool | None = None
        for index, character in enumerate(text):
            if index % 1024 == 0:
                check_runtime()
            offset = view.source_offset(index)
            if not 0 <= offset < len(self.source) or offset < previous:
                return None
            role = self.roles[offset]
            if offset == previous:
                if role != _STRING:
                    return None
                if repeated_source_owned is None:
                    repeated_source_owned = self.decoded is not None and self.decoded.owns(
                        view, check_runtime
                    )
                if not repeated_source_owned:
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
