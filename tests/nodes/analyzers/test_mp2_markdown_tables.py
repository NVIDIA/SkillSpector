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
