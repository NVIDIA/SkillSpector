# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared analyzer helpers."""

from skillspector.nodes.analyzers.common import get_context, get_context_from_lines


def test_context_helpers_bound_long_lines_around_the_finding() -> None:
    lines = ["a" * 1_500, "MATCH" + "b" * 1_500, "tail"]
    content = "\n".join(lines)

    offset_context = get_context(content, content.index("MATCH"), context_lines=1)
    line_context = get_context_from_lines(lines, lineno=2, window=1)

    for context in (offset_context, line_context):
        assert len(context) <= 1_000
        assert "MATCH" in context
