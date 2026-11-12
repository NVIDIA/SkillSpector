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

"""Tests for MP2 false positives on aligned Markdown tables (issue #870)."""

from __future__ import annotations

from skillspector.nodes.analyzers import static_patterns_memory_poisoning as mp_module

_COLUMNS = 24


def _wide_table() -> str:
    header = "| " + " | ".join(f"Feature {i:<3}" for i in range(_COLUMNS)) + " |"
    delimiter = "|" + "|".join("-" * 12 for _ in range(_COLUMNS)) + "|"
    row = "| " + " | ".join("yes" + " " * 8 for _ in range(_COLUMNS)) + " |"
    return "\n".join([header, delimiter, row, row])


def _mp2(content: str, file_type: str = "markdown") -> list:
    findings = mp_module.analyze(content, "SKILL.md", file_type)
    return [f for f in findings if f.rule_id == "MP2"]


class TestMP2MarkdownTables:
    """Aligned Markdown tables repeat cell text/padding by construction."""

    def test_aligned_table_rows_not_detected(self) -> None:
        """Prettier-aligned table rows (incl. the delimiter row) are not stuffing."""
        content = "# Skill\n\nSome intro prose.\n\n" + _wide_table() + "\n"
        assert _mp2(content) == []

    def test_unproven_pipe_row_still_detected(self) -> None:
        """A wide repeated row without a delimiter row stays fail-closed."""
        row = "| " + " | ".join("n/a" + " " * 5 for _ in range(25)) + " |"
        assert len(_mp2("# Skill\n\n" + row + "\n")) >= 1

    def test_table_shape_in_text_file_still_detected(self) -> None:
        """The exemption is Markdown-only; the same bytes in text still fire."""
        content = "# Skill\n\n" + _wide_table() + "\n"
        assert len(_mp2(content, file_type="text")) >= 1

    def test_stuffing_after_table_still_detected(self) -> None:
        """Suppressing table rows does not hide genuine stuffing elsewhere."""
        content = "# Skill\n\n" + _wide_table() + "\n\n" + ("fill the context window " * 30) + "\n"
        assert len(_mp2(content)) >= 1

    def test_single_cell_stuffing_still_detected(self) -> None:
        """The reviewer's blocker case: repetition inside one cell still fires.

        A one-column table whose single data cell repeats ``abcdef `` 1000
        times is stuffing, not layout: the repeated unit carries content
        and never crosses a cell boundary, so the table exemption must not
        apply even though the row belongs to a proven table.
        """
        content = "| Value |\n| --- |\n| " + "abcdef " * 1000 + "|\n"
        assert len(_mp2(content)) >= 1

    def test_stuffed_cell_in_multi_column_table_still_detected(self) -> None:
        """A stuffed cell inside a wider proven table still fires."""
        content = (
            "| Name | Notes | Status |\n"
            "| --- | --- | --- |\n"
            "| alpha | " + "abcdef " * 100 + "| ok |\n"
        )
        assert len(_mp2(content)) >= 1

    def test_escaped_pipe_cell_stuffing_still_detected(self) -> None:
        """The reviewer's escaped-pipe blocker case: escaped pipes are text.

        Under a 24-column header/delimiter, a body line of outer pipes
        plus ``abcdef\\|`` repeated 23 times plus ``abcdef`` has 24 raw
        pipe segments, so a raw-pipe split wrongly proves it a table row
        and the ``abcdef\\|`` unit looks structural. In Markdown the
        escaped pipes are literal text inside one cell, so the repetition
        is stuffing and must still fire.
        """
        header = "| " + " | ".join(f"Feature {i:<3}" for i in range(_COLUMNS)) + " |"
        delimiter = "|" + "|".join("-" * 12 for _ in range(_COLUMNS)) + "|"
        body = "|" + "abcdef\\|" * 23 + "abcdef" + "|"
        content = "# Skill\n\n" + "\n".join([header, delimiter, body]) + "\n"
        assert len(_mp2(content)) >= 1

    def test_repeated_short_cells_still_suppressed(self) -> None:
        """Repeated cell structure stays exempt: identical short cells."""
        row = "| " + " | ".join("yes" for _ in range(24)) + " |"
        table = "| " + " | ".join(f"Col {i}" for i in range(24)) + " |\n"
        table += "|" + "|".join("---" for _ in range(24)) + "|\n" + row + "\n"
        assert _mp2("# Skill\n\n" + table) == []
