# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, source-mapped CommonMark ownership for container paragraphs."""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

_MAX_SOURCE_CHARS = 131_072
_MAX_TOKENS = 16_384
_LITERAL_CODE = re.compile(r"[A-Za-z0-9_./:<>=@%+,?*~ \t-]+\Z")
_CONTAINER_MARKER = re.compile(r"(?m)^ {0,3}(?:>|[-+*][ \t]|[0-9]{1,9}[.)][ \t])")


@dataclass(frozen=True, slots=True)
class QuotedListSpans:
    code: tuple[tuple[int, int, int, int], ...]
    lines: tuple[tuple[int, int], ...]


@lru_cache(maxsize=8)
def _quoted_list_spans(content: str) -> QuotedListSpans | None:
    # The renderer dependency is optional: absence never grants ownership.
    try:
        from markdown_it import MarkdownIt
        from markdown_it.rules_inline.backticks import backtick
    except ImportError:
        return None

    parser = MarkdownIt("commonmark", {"maxNesting": 32})
    parser.core.ruler.disable(["inline"])
    try:
        blocks = parser.parse(content)
    except (ValueError, RecursionError):
        return None
    if len(blocks) > _MAX_TOKENS:
        return None
    raw_lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n|$)", content)
    if raw_lines and not raw_lines[-1]:
        raw_lines.pop()
    offsets = [0]
    for line in raw_lines:
        offsets.append(offsets[-1] + len(line))
    code: list[tuple[int, int, int, int]] = []
    lines: list[tuple[int, int]] = []
    ancestors: list[str] = []
    for block in blocks:
        if block.nesting == 1:
            ancestors.append(block.type)
            if len(ancestors) >= 32:
                return None
        elif block.nesting == -1:
            ancestors.pop()
        elif (
            block.type == "inline"
            and (
                "blockquote_open" in ancestors
                and "list_item_open" in ancestors
                or "paragraph_open" in ancestors
                and ("blockquote_open" in ancestors or "list_item_open" in ancestors)
            )
            and block.map is not None
        ):
            start, end = block.map
            logical_lines = block.content.split("\n")
            if len(logical_lines) != end - start:
                continue
            mapping: list[tuple[int, int, int]] = []
            position = 0
            for index, logical in enumerate(logical_lines, start):
                raw = raw_lines[index].rstrip("\r\n")
                # Exact logical suffixes preserve source offsets. Quotes still
                # require an explicit marker; normalized tabs grant no proof.
                column = raw.rfind(logical) if logical else -1
                if (
                    column < 0
                    or "\t" in raw[:column]
                    or ("blockquote_open" in ancestors and re.match(r" {0,3}>", raw) is None)
                    or raw[column + len(logical) :].strip()
                ):
                    break
                mapping.append((position, position + len(logical), offsets[index] + column))
                position += len(logical) + 1
            else:
                captured: list[tuple[int, int, int]] = []

                def record_backtick(state, silent, source=block.content, spans=captured):
                    begin = state.pos
                    previous = len(state.tokens)
                    result = backtick(state, silent)
                    if (
                        not silent
                        and state.src == source
                        and len(state.tokens) > previous
                        and state.tokens[-1].type == "code_inline"
                    ):
                        spans.append((begin, state.pos, len(state.tokens[-1].markup)))
                    return result

                # Observe the native rule; no rendering or external plugins run.
                parser.inline.ruler.at("backticks", record_backtick)
                children = []
                try:
                    parser.inline.parse(block.content, parser, {}, children)
                except (ValueError, RecursionError):
                    return None
                if len(children) > _MAX_TOKENS:
                    return None
                starts = [low for low, _, _ in mapping]
                for begin, finish, width in captured:
                    body = block.content[begin + width : finish - width]
                    if _LITERAL_CODE.fullmatch(body) is None:
                        continue
                    index = bisect_right(starts, begin) - 1
                    low, high, absolute = mapping[index]
                    if low <= begin and finish <= high:
                        opening = absolute + begin - low
                        code.append(
                            (
                                opening,
                                opening + width,
                                absolute + finish - width - low,
                                absolute + finish - low,
                            )
                        )
                lines.extend((absolute, absolute + high - low) for low, high, absolute in mapping)
    return QuotedListSpans(tuple(code), tuple(lines))


def quoted_list_spans(content: str, check_runtime: Callable[[], None]) -> QuotedListSpans | None:
    """Return exact ownership only for bounded, complete source artifacts."""
    check_runtime()
    if (
        len(content) > _MAX_SOURCE_CHARS
        or "`" not in content
        or _CONTAINER_MARKER.search(content) is None
    ):
        return None
    result = _quoted_list_spans(content)
    check_runtime()
    return result
