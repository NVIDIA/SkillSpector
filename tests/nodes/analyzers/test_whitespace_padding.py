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

"""Tests for the whitespace_padding detector helper (rule P9)."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers.whitespace_padding import (
    BLOCK_BYTE_BUDGET,
    HORIZONTAL_RUN_CHARS,
    RATIO_MIN_FILE_BYTES,
    VERTICAL_BLANK_LINES,
    ZERO_WIDTH_CHARS,
    PaddingRun,
    detect_whitespace_padding,
    is_padding_char,
    summarize_run,
)


def _kinds(runs: list[PaddingRun]) -> set[str]:
    return {r.kind for r in runs}


class TestZeroWidthChars:
    def test_exact_membership(self):
        """ZERO_WIDTH_CHARS contains exactly the five P2 code points."""
        assert ZERO_WIDTH_CHARS == frozenset(
            ["​", "‌", "‍", "⁠", "﻿"]
        )


class TestIsPaddingChar:
    def test_ascii_controls(self):
        for ch in "\t\n\r\v\f":
            assert is_padding_char(ch)

    def test_ascii_space(self):
        assert is_padding_char(" ")

    def test_unicode_zs_zl_zp(self):
        # U+00A0 (Zs), U+2028 (Zl), U+2029 (Zp), U+3000 (Zs)
        for ch in [" ", " ", " ", "　"]:
            assert is_padding_char(ch)

    def test_zero_width_family(self):
        for ch in ZERO_WIDTH_CHARS:
            assert is_padding_char(ch)

    def test_non_padding(self):
        for ch in "aZ9.#":
            assert not is_padding_char(ch)


class TestSummarizeRun:
    def test_single_codepoint(self):
        assert summarize_run(" " * 82) == "U+00A0 x82"

    def test_newline_escape(self):
        assert summarize_run("\n" * 82) == "\\n x82"

    def test_tab_escape(self):
        assert summarize_run("\t" * 5) == "\\t x5"

    def test_empty(self):
        assert summarize_run("") == ""

    def test_mixed_collapses(self):
        text = " " * 10 + " " * 5 + "　" * 2 + " "
        out = summarize_run(text)
        # Top segments by frequency rendered; tail collapsed.
        assert "U+00A0 x10" in out
        assert "+1 more" in out


class TestVerticalSignal:
    def test_below_threshold_no_fire(self):
        content = "header\n" + "\n" * (VERTICAL_BLANK_LINES - 1) + "tail"
        runs = detect_whitespace_padding(content)
        assert "vertical" not in _kinds(runs)

    def test_at_threshold_fires(self):
        content = "header\n" + "\n" * VERTICAL_BLANK_LINES + "tail"
        runs = detect_whitespace_padding(content)
        vert = [r for r in runs if r.kind == "vertical"]
        assert len(vert) == 1
        assert vert[0].followed_by_content is True
        assert vert[0].start_line == 2

    def test_followed_by_content_false_when_trailing(self):
        content = "header\n" + "\n" * (VERTICAL_BLANK_LINES + 5)
        runs = detect_whitespace_padding(content)
        vert = [r for r in runs if r.kind == "vertical"]
        assert len(vert) == 1
        assert vert[0].followed_by_content is False

    def test_unicode_blank_lines(self):
        # Lines made of non-ASCII whitespace still count as blank.
        blank = " 　"
        content = "header\n" + ((blank + "\n") * VERTICAL_BLANK_LINES) + "tail"
        runs = detect_whitespace_padding(content)
        assert "vertical" in _kinds(runs)


class TestHorizontalSignal:
    def test_below_threshold_no_fire(self):
        content = "x" + " " * (HORIZONTAL_RUN_CHARS - 1) + "y"
        runs = detect_whitespace_padding(content)
        assert "horizontal" not in _kinds(runs)

    def test_at_threshold_fires(self):
        content = "x" + " " * HORIZONTAL_RUN_CHARS + "y"
        runs = detect_whitespace_padding(content)
        horiz = [r for r in runs if r.kind == "horizontal"]
        assert len(horiz) == 1
        assert horiz[0].length == HORIZONTAL_RUN_CHARS
        assert horiz[0].followed_by_content is True
        assert horiz[0].start_line == 1

    def test_leading_indentation_counts(self):
        content = " " * HORIZONTAL_RUN_CHARS + "instruction"
        runs = detect_whitespace_padding(content)
        assert "horizontal" in _kinds(runs)

    def test_unicode_nbsp_run(self):
        content = "x" + " " * HORIZONTAL_RUN_CHARS + "y"
        runs = detect_whitespace_padding(content)
        horiz = [r for r in runs if r.kind == "horizontal"]
        assert len(horiz) == 1
        assert horiz[0].summary == f"U+00A0 x{HORIZONTAL_RUN_CHARS}"


class TestBlockAndRatioSignal:
    def test_block_boundary(self):
        # Flank with non-padding chars (no newlines) so the contiguous run is
        # exactly the space block. Exactly at the budget: no fire; one over: fires.
        at_budget = "a" + " " * BLOCK_BYTE_BUDGET + "b"
        assert "block" not in _kinds(detect_whitespace_padding(at_budget))
        over = "a" + " " * (BLOCK_BYTE_BUDGET + 1) + "b"
        assert "block" in _kinds(detect_whitespace_padding(over))

    def test_ratio_fires_for_large_whitespace_file(self):
        content = "x" + " " * (RATIO_MIN_FILE_BYTES + 100)
        runs = detect_whitespace_padding(content)
        assert "ratio" in _kinds(runs)

    def test_ratio_not_for_small_file(self):
        content = " " * 100
        runs = detect_whitespace_padding(content)
        assert "ratio" not in _kinds(runs)

    def test_block_dedup_against_vertical(self):
        # A huge vertical run also exceeds the block budget; block is suppressed
        # because it starts at the same offset as the vertical run.
        content = "header\n" + "\n" * (BLOCK_BYTE_BUDGET + 10) + "tail"
        runs = detect_whitespace_padding(content)
        assert "vertical" in _kinds(runs)
        assert "block" not in _kinds(runs)


class TestGuards:
    def test_replacement_char_bails_out(self):
        content = "x�" + " " * (HORIZONTAL_RUN_CHARS + 10) + "y"
        assert detect_whitespace_padding(content) == []

    def test_markdown_fence_skips_horizontal(self):
        inner = "x" + " " * HORIZONTAL_RUN_CHARS + "y"
        content = "intro\n```\n" + inner + "\n```\noutro"
        runs = detect_whitespace_padding(content, file_type="markdown")
        assert "horizontal" not in _kinds(runs)

    def test_non_markdown_fence_still_fires(self):
        inner = "x" + " " * HORIZONTAL_RUN_CHARS + "y"
        content = "intro\n```\n" + inner + "\n```\noutro"
        runs = detect_whitespace_padding(content, file_type="other")
        assert "horizontal" in _kinds(runs)

    def test_empty_content(self):
        assert detect_whitespace_padding("") == []


class TestUnicodeEvasionEndToEnd:
    def test_each_evasion_char_detected_vertically(self):
        # Each candidate from issue #20's evasion list, as blank-line padding.
        for ch in [
            " ",  # NBSP
            " ",  # line separator
            " ",  # paragraph separator
            "",  # vertical tab
            "",  # form feed
            "　",  # ideographic space
        ] + list(ZERO_WIDTH_CHARS):
            blank = ch
            content = "header\n" + ((blank + "\n") * VERTICAL_BLANK_LINES) + "INJECT"
            runs = detect_whitespace_padding(content)
            assert "vertical" in _kinds(runs), f"failed for U+{ord(ch):04X}"


# Every padding character enumerated in issue #20's evasion list. Each must cross
# a P9 detection threshold so injected instructions hidden behind it are flagged.
#   U+00A0 NBSP, U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR,
#   U+000B VERTICAL TAB, U+000C FORM FEED, U+3000 IDEOGRAPHIC SPACE,
#   and the zero-width family U+200B/U+200C/U+200D/U+2060/U+FEFF.
_ISSUE20_EVASION_CHARS = [
    " ",  # U+00A0 NO-BREAK SPACE (Zs)
    " ",  # U+2028 LINE SEPARATOR (Zl)
    " ",  # U+2029 PARAGRAPH SEPARATOR (Zp)
    "",  # U+000B VERTICAL TAB
    "",  # U+000C FORM FEED
    "　",  # U+3000 IDEOGRAPHIC SPACE (Zs)
    "​",  # U+200B ZERO WIDTH SPACE
    "‌",  # U+200C ZERO WIDTH NON-JOINER
    "‍",  # U+200D ZERO WIDTH JOINER
    "⁠",  # U+2060 WORD JOINER
    "﻿",  # U+FEFF ZERO WIDTH NO-BREAK SPACE / BOM
]


class TestIssue20AdversarialEvasionCoverage:
    """Adversarial self-check: P9 must fire on each issue #20 evasion character.

    Two complementary constructions are exercised for every character:

    * An in-line (horizontal) run of 100 copies of the char before a hidden
      ``INJECT`` instruction — covers U+00A0/U+3000/U+000B/U+000C and the
      zero-width family, which form horizontal/block runs within a line.
    * A vertical run of 25 lines each consisting solely of the char — covers the
      line-separator characters U+2028/U+2029 (Zl/Zp) whose "vertical-ish" runs
      sit between a header and the hidden ``INJECT`` line. (All chars also pass
      this construction since a whitespace-only line is a blank line regardless
      of which padding char fills it.)

    Both constructions cross a detection threshold (100 >= HORIZONTAL_RUN_CHARS,
    25 >= VERTICAL_BLANK_LINES). If any character fails to fire, that is a real
    detector bug per the issue's evasion list.
    """

    @pytest.mark.parametrize("ch", _ISSUE20_EVASION_CHARS, ids=[f"U+{ord(c):04X}" for c in _ISSUE20_EVASION_CHARS])
    def test_inline_run_fires(self, ch: str):
        assert 100 >= HORIZONTAL_RUN_CHARS
        content = "x" + ch * 100 + "INJECT"
        runs = detect_whitespace_padding(content)
        assert runs, f"no P9 run for in-line U+{ord(ch):04X}"
        # An in-line run forms a horizontal (and/or block) signal, never vertical.
        assert "horizontal" in _kinds(runs) or "block" in _kinds(runs), (
            f"in-line U+{ord(ch):04X} did not fire horizontal/block: {_kinds(runs)}"
        )

    @pytest.mark.parametrize("ch", _ISSUE20_EVASION_CHARS, ids=[f"U+{ord(c):04X}" for c in _ISSUE20_EVASION_CHARS])
    def test_vertical_run_fires(self, ch: str):
        assert 25 >= VERTICAL_BLANK_LINES
        content = "header\n" + ((ch + "\n") * 25) + "INJECT"
        runs = detect_whitespace_padding(content)
        vert = [r for r in runs if r.kind == "vertical"]
        assert vert, f"no vertical P9 run for U+{ord(ch):04X}"
        assert vert[0].followed_by_content is True

    @pytest.mark.parametrize("ch", _ISSUE20_EVASION_CHARS, ids=[f"U+{ord(c):04X}" for c in _ISSUE20_EVASION_CHARS])
    def test_p9_analyzer_emits_finding(self, ch: str):
        """End-to-end: the prompt-injection analyzer emits a P9 finding."""
        from skillspector.nodes.analyzers import static_patterns_prompt_injection as spi

        content = "x" + ch * 100 + "INJECT"
        findings = spi.analyze(content, "SKILL.md", "other")
        p9 = [f for f in findings if f.rule_id == "P9"]
        assert p9, f"analyzer emitted no P9 finding for U+{ord(ch):04X}"
        assert p9[0].message == "Whitespace Padding"
        assert p9[0].matched_text, "P9 finding has empty matched_text"
