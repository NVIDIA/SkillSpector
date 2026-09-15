# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared analyzer helpers."""

from skillspector.nodes.analyzers.common import (
    SourceLocationIndex,
    get_context,
    get_context_from_lines,
)


def test_context_helpers_bound_long_lines_around_the_finding() -> None:
    lines = ["a" * 1_500, "MATCH" + "b" * 1_500, "tail"]
    content = "\n".join(lines)

    offset_context = get_context(content, content.index("MATCH"), context_lines=1)
    line_context = get_context_from_lines(lines, lineno=2, window=1)

    for context in (offset_context, line_context):
        assert len(context) <= 1_000
        assert "MATCH" in context


def test_line_context_uses_the_finding_column_on_a_long_line() -> None:
    lines = ["a" * 1_500 + "MATCH" + "b" * 1_500]

    context = get_context_from_lines(lines, lineno=1, window=0, column=1_500)

    assert len(context) <= 1_000
    assert "MATCH" in context


def test_source_location_index_reuses_logical_line_offsets() -> None:
    content = "alpha\r\nβeta\nlast"
    locations = SourceLocationIndex(content, "SKILL.md")
    line_starts = locations.line_starts

    beta_start = content.index("β")
    last_end = len(content)
    first = locations.location(beta_start, beta_start + len("βeta"))
    second = locations.location(content.index("last"), last_end)

    assert locations.line_starts is line_starts
    assert (first.start_line, first.start_column, first.end_line, first.end_column) == (2, 0, 2, 4)
    assert (second.start_line, second.start_column, second.end_line, second.end_column) == (
        3,
        0,
        3,
        4,
    )
