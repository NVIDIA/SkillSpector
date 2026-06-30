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

# Non-ASCII characters that render as (or were historically classified as)
# whitespace but fall outside the Zs/Zl/Zp categories, so unicodedata.category
# alone would miss them:
#   U+0085 NEXT LINE (NEL)            — category Cc; splits lines, but a *horizontal*
#                                       or block run built from it must still count.
#   U+180E MONGOLIAN VOWEL SEPARATOR  — category Cf today (was Zs pre-Unicode 6.3).
# Listed explicitly so padding runs built from them are not a P9 bypass (issue #20).
_EXTRA_PADDING_CHARS = frozenset("\x85᠎")

# Threshold constants (module-level so tuning is a one-line change).
VERTICAL_BLANK_LINES = 20
VERTICAL_HIGH_SEVERITY_LINES = 40
HORIZONTAL_RUN_CHARS = 80
BLOCK_BYTE_BUDGET = 2048
RATIO_THRESHOLD = 0.90
RATIO_MIN_FILE_BYTES = 4096

# Replacement character emitted by errors="replace" decoding; a high *density* of
# it marks binary-ish content, which we bail out of entirely. We key on density
# rather than mere presence so a single embedded U+FFFD cannot disable P9 for an
# otherwise-textual file (which would itself be a trivial bypass of this rule).
_REPLACEMENT_CHAR = "�"
_REPLACEMENT_CHAR_DENSITY_THRESHOLD = 0.30

# Markdown fenced-code delimiter (``` or ~~~ with optional leading indentation).
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Line-boundary characters/sequences that count as line separators when splitting
# content into logical lines. Beyond ASCII LF, this includes CR / CRLF and the
# Unicode line/paragraph separators U+2028 / U+2029 / U+0085 (NEL) — all of which
# render as line breaks and are named in issue #20's evasion list. A multi-char
# sequence (CRLF) must precede its single-char members so it is matched whole.
_LINE_BOUNDARY_RE = re.compile("\r\n|\r|\n|\u2028|\u2029|\x85")

# How many distinct code points to show in a summary before collapsing.
_SUMMARY_MAX_SEGMENTS = 3


def is_padding_char(ch: str) -> bool:
    """Return True when *ch* is a whitespace/padding character.

    Covers ASCII controls (``\\t \\n \\r \\v \\f``), Unicode whitespace categories
    ``Zs``/``Zl``/``Zp`` (e.g. U+00A0, U+2028, U+2029, U+3000), the zero-width
    family (category ``Cf``/``Bn``, listed explicitly), and the extra padding
    chars U+0085/U+180E (see ``_EXTRA_PADDING_CHARS``).
    """
    # ASCII fast-path: the only padding code points below U+0080 are the five
    # control chars and the space, so the common case skips unicodedata entirely.
    if ord(ch) < 0x80:
        return ch == " " or ch in _ASCII_CONTROL_PADDING
    if ch in ZERO_WIDTH_CHARS or ch in _EXTRA_PADDING_CHARS:
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
    """A contiguous run of whitespace padding detected in content.

    ``length`` is **overloaded by ``kind``** — read it as:

    * ``"vertical"``  → number of blank/whitespace-only LINES in the run.
    * ``"horizontal"`` → number of padding CHARACTERS in the in-line run.
    * ``"block"``     → number of padding CHARACTERS in the contiguous block
      (NOT bytes — char-based so unit-consistent with ``start_offset``).
    * ``"ratio"``     → number of padding BYTES in the whole file.

    Only ``"vertical"`` exposes a line count, so the analyzer's HIGH-severity
    check (``run.length >= VERTICAL_HIGH_SEVERITY_LINES``) is meaningful for the
    ``"vertical"`` kind alone.

    ``end_offset`` is the char offset just past the run, kept unit-consistent
    with ``start_offset`` so consumers can compute spans without re-deriving them
    from ``length`` (whose unit varies). It defaults to ``start_offset`` and is
    set by the detectors that produce span-based runs.
    """

    kind: str  # "vertical" | "horizontal" | "block" | "ratio"
    start_offset: int  # char offset where the run starts
    start_line: int  # 1-based line number
    length: int  # see class docstring — unit depends on kind
    followed_by_content: bool
    summary: str  # visible-ized snippet, e.g. "U+00A0 x82" or "\\n x82"
    end_offset: int = -1  # char offset just past the run (-1 → unset, == start)

    def __post_init__(self) -> None:
        if self.end_offset < 0:
            self.end_offset = self.start_offset


def _split_lines(content: str) -> tuple[list[str], list[int]]:
    """Split *content* into logical lines on Unicode line boundaries.

    Treats LF, CR, CRLF, U+2028 (LINE SEPARATOR), U+2029 (PARAGRAPH SEPARATOR)
    and U+0085 (NEL) as line separators — so padding built from any of them
    counts toward the vertical blank-line signal (issue #20 evasion list).

    Returns ``(lines, line_offsets)`` where ``lines[k]`` is the separator-free
    text of line *k* and ``line_offsets[k]`` is its char offset in *content*.
    ``line_offsets`` has one extra trailing entry equal to ``len(content)`` so
    ``line_offsets[k + 1]`` always gives the start of the next line (or EOF),
    which keeps offset arithmetic correct regardless of separator width.
    """
    lines: list[str] = []
    line_offsets: list[int] = []
    pos = 0
    for m in _LINE_BOUNDARY_RE.finditer(content):
        line_offsets.append(pos)
        lines.append(content[pos : m.start()])
        pos = m.end()
    # Final line (text after the last separator, possibly empty).
    line_offsets.append(pos)
    lines.append(content[pos:])
    # Trailing sentinel so line_offsets[k + 1] is always valid.
    line_offsets.append(len(content))
    return lines, line_offsets


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


def _detect_vertical(content: str, lines: list[str], line_offsets: list[int]) -> list[PaddingRun]:
    """Detect runs of >= VERTICAL_BLANK_LINES consecutive blank/whitespace-only lines.

    ``line_offsets`` is the offset table from :func:`_split_lines` (one entry per
    line plus a trailing ``len(content)`` sentinel), so ``line_offsets[j]`` is the
    start of line *j* regardless of how wide each line's separator was. This keeps
    char-offset arithmetic correct for CRLF and the Unicode line separators
    (U+2028/U+2029/NEL), not just single-char ``\\n``.
    """
    runs: list[PaddingRun] = []
    blank = [_is_blank_line(line) for line in lines]

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
            # Offset just past the run = start of the first line after it (the
            # sentinel guarantees line_offsets[j] is valid even at EOF).
            end_offset = line_offsets[j]
            summary = summarize_run(content[start_offset:end_offset])
            runs.append(
                PaddingRun(
                    kind="vertical",
                    start_offset=start_offset,
                    start_line=i + 1,
                    length=run_len,
                    followed_by_content=followed_by_content,
                    summary=summary,
                    end_offset=end_offset,
                )
            )
        i = j
    return runs


def _detect_horizontal(
    content: str, lines: list[str], line_offsets: list[int], file_type: str
) -> list[PaddingRun]:
    """Detect in-line runs of >= HORIZONTAL_RUN_CHARS consecutive padding chars.

    For ``file_type == "markdown"``, runs whose line falls inside a fenced code
    region are skipped (false-positive guard). ``line_offsets`` is the offset
    table from :func:`_split_lines`, used so char offsets stay correct under
    variable-width line separators.
    """
    runs: list[PaddingRun] = []
    # Only the markdown path needs fence flags; skip building the list otherwise.
    fence_flags = _fence_line_flags(lines) if file_type == "markdown" else None
    for idx, line in enumerate(lines):
        if fence_flags is not None and fence_flags[idx]:
            continue
        line_offset = line_offsets[idx]
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
                        end_offset=start_offset + run_len,
                    )
                )
    return runs


def _detect_block_and_ratio(content: str) -> list[PaddingRun]:
    """Detect a contiguous block > BLOCK_BYTE_BUDGET and the >90%-of-file ratio.

    Returns at most one "block" run (the largest contiguous padding span exceeding
    the byte budget) and at most one "ratio" run.
    """
    runs: list[PaddingRun] = []
    n = len(content)

    # Largest contiguous padding run. The threshold is a BYTE budget (per the
    # signal table), but the run's ``start_offset``/``end_offset``/``length`` are
    # CHAR-based so they stay unit-consistent for span/overlap arithmetic.
    best_byte_len = 0
    best_start = -1
    best_end = -1
    i = 0
    while i < n:
        if not is_padding_char(content[i]):
            i += 1
            continue
        start = i
        while i < n and is_padding_char(content[i]):
            i += 1
        byte_len = len(content[start:i].encode("utf-8"))
        if byte_len > best_byte_len:
            best_byte_len = byte_len
            best_start = start
            best_end = i
    if best_byte_len > BLOCK_BYTE_BUDGET and best_start >= 0:
        runs.append(
            PaddingRun(
                kind="block",
                start_offset=best_start,
                start_line=content[:best_start].count("\n") + 1,
                length=best_end - best_start,  # char count (unit-consistent)
                followed_by_content=False,
                summary=summarize_run(content[best_start : best_start + 200]),
                end_offset=best_end,
            )
        )

    # Whitespace-to-file ratio (bytes) for files over the floor.
    file_bytes = len(content.encode("utf-8"))
    if file_bytes > RATIO_MIN_FILE_BYTES:
        padding_bytes = sum(len(ch.encode("utf-8")) for ch in content if is_padding_char(ch))
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


def detect_whitespace_padding(content: str, *, file_type: str = "other") -> list[PaddingRun]:
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
    - Bails out entirely (returns ``[]``) when the U+FFFD density of *content*
      exceeds ``_REPLACEMENT_CHAR_DENSITY_THRESHOLD`` (binary-ish content). Keying
      on density rather than mere presence means a single embedded U+FFFD cannot
      suppress detection for an otherwise-textual file.
    - For ``file_type == "markdown"``, horizontal runs inside ``` fenced regions
      are skipped.

    Dedup (at most one finding per overlapping span; higher-signal kind wins):
    - "block" and "ratio" runs whose char span is already covered by a reported
      "vertical" or "horizontal" run are suppressed. A single large whitespace
      span therefore yields ONE finding (the vertical/horizontal one), not three.
    """
    if not content:
        return []
    replacement_count = content.count(_REPLACEMENT_CHAR)
    if replacement_count / len(content) > _REPLACEMENT_CHAR_DENSITY_THRESHOLD:
        return []

    lines, line_offsets = _split_lines(content)

    vertical = _detect_vertical(content, lines, line_offsets)
    horizontal = _detect_horizontal(content, lines, line_offsets, file_type)
    block_ratio = _detect_block_and_ratio(content)

    # Higher-signal runs whose spans suppress overlapping block/ratio runs.
    # All offsets are char-based and unit-consistent (see PaddingRun docstring).
    primary = vertical + horizontal

    def _overlaps_primary(run: PaddingRun) -> bool:
        for p in primary:
            if run.start_offset < p.end_offset and p.start_offset < run.end_offset:
                return True
        return False

    # "ratio" spans the whole file (offset 0..len), so treat it as covered when
    # any primary run exists (a primary run is always a subspan of the file).
    # Signal 3 reports at most ONE finding per file: when both "block" and
    # "ratio" qualify for the same oversized-whitespace condition, prefer the
    # more specific/localized "block" run and drop the redundant "ratio".
    deduped_block_ratio: list[PaddingRun] = []
    block_kept = False
    for run in block_ratio:
        if run.kind == "block" and _overlaps_primary(run):
            continue
        if run.kind == "ratio" and (primary or block_kept):
            continue
        if run.kind == "block":
            block_kept = True
        deduped_block_ratio.append(run)

    return vertical + horizontal + deduped_block_ratio
