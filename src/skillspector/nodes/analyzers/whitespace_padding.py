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

"""Whitespace-padding detector (rule P9).

Pure-function helper that owns the shared Unicode-whitespace character sets and a
scanner that returns structured "padding run" records. Two consumers build
findings from those records: the prompt-injection analyzer (file bodies) and the
MCP tool-poisoning analyzer (manifest description fields).

This module imports only stdlib (``unicodedata``, ``dataclasses``, ``re``) and
must NOT import sibling analyzer modules so it can serve as the single shared
definition without risking circular imports.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Shared zero-width character set. Must stay character-for-character identical to
# P2's current set and converge mcp_tool_poisoning's _ZERO_WIDTH_RE onto it:
#   U+200B ZERO WIDTH SPACE
#   U+200C ZERO WIDTH NON-JOINER
#   U+200D ZERO WIDTH JOINER
#   U+2060 WORD JOINER
#   U+FEFF ZERO WIDTH NO-BREAK SPACE (BOM)
ZERO_WIDTH_CHARS = frozenset("​‌‍⁠﻿")

# ASCII control characters that count as padding: tab, newline, carriage return,
# vertical tab, form feed.
_ASCII_CONTROL_PADDING = frozenset("\t\n\r\v\f")

# Threshold constants (module-level so tuning is a one-line change).
VERTICAL_BLANK_LINES = 20
VERTICAL_HIGH_SEVERITY_LINES = 40
HORIZONTAL_RUN_CHARS = 80
BLOCK_BYTE_BUDGET = 2048
RATIO_THRESHOLD = 0.90
RATIO_MIN_FILE_BYTES = 4096

# Replacement character emitted by errors="replace" decoding; its presence marks
# binary-ish content, which we bail out of entirely.
_REPLACEMENT_CHAR = "�"

# Markdown fenced-code delimiter (``` or ~~~ with optional leading indentation).
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# How many distinct code points to show in a summary before collapsing.
_SUMMARY_MAX_SEGMENTS = 3


def is_padding_char(ch: str) -> bool:
    """Return True when *ch* is a whitespace/padding character.

    Covers ASCII controls (``\\t \\n \\r \\v \\f``), Unicode whitespace categories
    ``Zs``/``Zl``/``Zp`` (e.g. U+00A0, U+2028, U+2029, U+3000), and the zero-width
    family (which is category ``Cf``/``Bn`` and so must be listed explicitly).
    """
    if ch in _ASCII_CONTROL_PADDING or ch in ZERO_WIDTH_CHARS:
        return True
    return unicodedata.category(ch) in ("Zs", "Zl", "Zp")


def _is_blank_line(line: str) -> bool:
    """Return True when every character of *line* is a padding char (or empty)."""
    return all(is_padding_char(ch) for ch in line)


def _escape_for_summary(ch: str) -> str:
    """Render a single padding char for a human-readable summary segment."""
    escapes = {
        "\n": "\\n",
        "\t": "\\t",
        "\r": "\\r",
        "\v": "\\v",
        "\f": "\\f",
    }
    if ch in escapes:
        return escapes[ch]
    return f"U+{ord(ch):04X}"


def summarize_run(text: str) -> str:
    """Render *text* (a padding run) as ``"<char> xN"`` segments.

    Counts each distinct code point and renders the top few as e.g. ``"U+00A0 x82"``
    or ``"\\n x82"``. Mixed runs collapse to the most frequent code points; the rest
    are summarised as ``"+K more"``. Returns ``""`` for empty input.
    """
    if not text:
        return ""
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    # Sort by descending count, then by code point for determinism.
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], ord(kv[0])))
    segments = [f"{_escape_for_summary(ch)} x{n}" for ch, n in ordered[:_SUMMARY_MAX_SEGMENTS]]
    remaining = len(ordered) - _SUMMARY_MAX_SEGMENTS
    if remaining > 0:
        segments.append(f"+{remaining} more")
    return ", ".join(segments)


@dataclass
class PaddingRun:
    """A contiguous run of whitespace padding detected in content."""

    kind: str  # "vertical" | "horizontal" | "block" | "ratio"
    start_offset: int  # char offset where the run starts
    start_line: int  # 1-based line number
    length: int  # chars (or line count for "vertical")
    followed_by_content: bool
    summary: str  # visible-ized snippet, e.g. "U+00A0 x82" or "\\n x82"


def _fence_line_flags(lines: list[str]) -> list[bool]:
    """Return per-line booleans marking which lines sit inside a Markdown fence.

    The fence delimiter lines themselves are treated as inside the fenced region.
    """
    inside = False
    flags: list[bool] = []
    for line in lines:
        if _FENCE_RE.match(line):
            # The delimiter line is part of the fenced region either way.
            flags.append(True)
            inside = not inside
        else:
            flags.append(inside)
    return flags


def _vertical_char_end(content: str, lines: list[str], run: PaddingRun) -> int:
    """Return the char offset just past a vertical run (its blank lines)."""
    # run.length blank lines starting at 1-based run.start_line.
    end_line_idx = (run.start_line - 1) + run.length  # first line after the run
    if end_line_idx >= len(lines):
        return len(content)
    # Offset of the start of end_line_idx == start_offset + sum of run line lengths.
    offset = run.start_offset
    for line in lines[run.start_line - 1 : end_line_idx]:
        offset += len(line) + 1  # +1 for the newline split removed
    return offset


def _detect_vertical(content: str, lines: list[str]) -> list[PaddingRun]:
    """Detect runs of >= VERTICAL_BLANK_LINES consecutive blank/whitespace-only lines."""
    runs: list[PaddingRun] = []
    blank = [_is_blank_line(line) for line in lines]
    # Precompute char offset of the start of each line.
    line_offsets: list[int] = []
    off = 0
    for line in lines:
        line_offsets.append(off)
        off += len(line) + 1  # +1 for the newline that splitlines stripped

    i = 0
    n = len(lines)
    while i < n:
        if not blank[i]:
            i += 1
            continue
        j = i
        while j < n and blank[j]:
            j += 1
        run_len = j - i
        if run_len >= VERTICAL_BLANK_LINES:
            followed_by_content = j < n and not blank[j]
            start_offset = line_offsets[i]
            # Summary built from the actual run text (the blank lines + newlines).
            end_offset = line_offsets[j] if j < n else len(content)
            summary = summarize_run(content[start_offset:end_offset])
            runs.append(
                PaddingRun(
                    kind="vertical",
                    start_offset=start_offset,
                    start_line=i + 1,
                    length=run_len,
                    followed_by_content=followed_by_content,
                    summary=summary,
                )
            )
        i = j
    return runs


def _detect_horizontal(
    content: str, lines: list[str], file_type: str
) -> list[PaddingRun]:
    """Detect in-line runs of >= HORIZONTAL_RUN_CHARS consecutive padding chars.

    For ``file_type == "markdown"``, runs whose line falls inside a fenced code
    region are skipped (false-positive guard).
    """
    runs: list[PaddingRun] = []
    fence_flags = (
        _fence_line_flags(lines) if file_type == "markdown" else [False] * len(lines)
    )
    line_offset = 0
    for idx, line in enumerate(lines):
        if not fence_flags[idx]:
            k = 0
            line_len = len(line)
            while k < line_len:
                if not is_padding_char(line[k]):
                    k += 1
                    continue
                start = k
                while k < line_len and is_padding_char(line[k]):
                    k += 1
                run_len = k - start
                if run_len >= HORIZONTAL_RUN_CHARS:
                    start_offset = line_offset + start
                    followed_by_content = k < line_len
                    summary = summarize_run(line[start:k])
                    runs.append(
                        PaddingRun(
                            kind="horizontal",
                            start_offset=start_offset,
                            start_line=idx + 1,
                            length=run_len,
                            followed_by_content=followed_by_content,
                            summary=summary,
                        )
                    )
        line_offset += len(line) + 1
    return runs


def _detect_block_and_ratio(content: str) -> list[PaddingRun]:
    """Detect a contiguous block > BLOCK_BYTE_BUDGET and the >90%-of-file ratio.

    Returns at most one "block" run (the largest contiguous padding span exceeding
    the byte budget) and at most one "ratio" run.
    """
    runs: list[PaddingRun] = []
    n = len(content)

    # Largest contiguous padding run (counted in bytes via UTF-8 length).
    best_len = 0
    best_start = -1
    i = 0
    while i < n:
        if not is_padding_char(content[i]):
            i += 1
            continue
        start = i
        while i < n and is_padding_char(content[i]):
            i += 1
        byte_len = len(content[start:i].encode("utf-8"))
        if byte_len > best_len:
            best_len = byte_len
            best_start = start
    if best_len > BLOCK_BYTE_BUDGET and best_start >= 0:
        runs.append(
            PaddingRun(
                kind="block",
                start_offset=best_start,
                start_line=content[:best_start].count("\n") + 1,
                length=best_len,
                followed_by_content=False,
                summary=summarize_run(content[best_start : best_start + 200]),
            )
        )

    # Whitespace-to-file ratio (bytes) for files over the floor.
    file_bytes = len(content.encode("utf-8"))
    if file_bytes > RATIO_MIN_FILE_BYTES:
        padding_bytes = sum(
            len(ch.encode("utf-8")) for ch in content if is_padding_char(ch)
        )
        if file_bytes and padding_bytes / file_bytes > RATIO_THRESHOLD:
            runs.append(
                PaddingRun(
                    kind="ratio",
                    start_offset=0,
                    start_line=1,
                    length=padding_bytes,
                    followed_by_content=False,
                    summary=f"{padding_bytes}/{file_bytes} bytes padding",
                )
            )
    return runs


def detect_whitespace_padding(
    content: str, *, file_type: str = "other"
) -> list[PaddingRun]:
    """Scan *content* for whitespace-padding runs and return structured records.

    Implements the three P9 signals:

    1. Vertical blank-line runs (>= ``VERTICAL_BLANK_LINES`` consecutive
       blank/whitespace-only lines), with ``followed_by_content`` set when
       non-blank content follows the gap.
    2. Horizontal in-line runs (>= ``HORIZONTAL_RUN_CHARS`` consecutive padding
       chars within a single line, including leading indentation).
    3. Oversized contiguous block (> ``BLOCK_BYTE_BUDGET`` bytes) and the
       >``RATIO_THRESHOLD`` whitespace ratio for files over ``RATIO_MIN_FILE_BYTES``.

    Guards:
    - Bails out entirely (returns ``[]``) when *content* contains U+FFFD
      (binary-ish content).
    - For ``file_type == "markdown"``, horizontal runs inside ``` fenced regions
      are skipped.
    - A "block" run whose span equals a "vertical" run's span is suppressed
      (the higher-signal vertical finding wins).
    """
    if not content or _REPLACEMENT_CHAR in content:
        return []

    lines = content.split("\n")

    vertical = _detect_vertical(content, lines)
    horizontal = _detect_horizontal(content, lines, file_type)
    block_ratio = _detect_block_and_ratio(content)

    # Dedup: suppress a "block" run that overlaps a vertical run's span (the
    # higher-signal vertical finding wins). A large vertical gap's contiguous
    # padding naturally extends across the same bytes the block signal would flag.
    def _block_overlaps_vertical(block: PaddingRun) -> bool:
        block_end = block.start_offset + block.length
        for run in vertical:
            vert_start = run.start_offset
            # Vertical length is in lines; recompute its char/byte span via offset.
            vert_end_offset = _vertical_char_end(content, lines, run)
            if block.start_offset < vert_end_offset and vert_start < block_end:
                return True
        return False

    block_ratio = [
        run
        for run in block_ratio
        if not (run.kind == "block" and _block_overlaps_vertical(run))
    ]

    return vertical + horizontal + block_ratio
