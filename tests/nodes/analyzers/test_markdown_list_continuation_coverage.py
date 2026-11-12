# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List-relative inline syntax preserves raw shell payloads and fallback boundaries."""

from __future__ import annotations

import re

import pytest

from skillspector.nodes.analyzers import static_patterns_prompt_injection as prompt
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tool


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "source",
    [
        "10. Example:\n    `printf 'x' | cat`.\n",
        "- Commands:\n  `delete`, `star`.\n  `--dry-run` checks syntax, permissions. `done`.\n",
    ],
)
def test_original_list_continuation_examples_complete(source: str, newline: str) -> None:
    source = ("# Résumé — 比較\n\n" + source).replace("\n", newline)
    for analyzer in (tool, prompt):
        report = analyzer.node({"components": ["guide.md"], "file_cache": {"guide.md": source}})
        assert report["inspection_ledger"]
        assert all(event["outcome"] == "completed" for event in report["inspection_ledger"])
        assert report["findings"] == []


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("marker", ["1.", "10.", "100.", "123456789.", "10)", "-"])
@pytest.mark.parametrize("extra", [0, 1, 3])
def test_proven_list_content_margin_owns_inline_delimiters(
    marker: str, extra: int, newline: str
) -> None:
    source = (
        f"# Résumé — 比較\n\n{marker} Example:\n"
        + " " * (len(marker) + 1 + extra)
        + "`printf 'x' | cat`.\n"
    )
    source = source.replace("\n", newline)
    projection = tool._markdown_shell_text(source, lambda: None)
    assert len(projection) == len(source)
    assert projection.count("`") == 0
    assert projection.replace(" ", "") == source.replace("`", "").replace(" ", "")
    assert not tool.has_bounded_parse_exhaustion(source, lambda: None, file_type="markdown")


@pytest.mark.parametrize(
    "source",
    [
        "- Parent:\n  10. Example:\n      `printf 'x' | cat`.\n",
        "1. Parent:\n   10. Example:\n       `printf 'x' | cat`.\n",
        "10. Example:\n   `printf 'x' | cat`.\n",
    ],
)
def test_recognized_nested_and_lazy_list_paragraphs_complete(source: str) -> None:
    assert not tool.has_bounded_parse_exhaustion(source, lambda: None, file_type="markdown")


@pytest.mark.parametrize(
    "source",
    [
        "    `printf 'x' | cat`.\n",
        "10. Example:\n\n        `printf 'x' | cat`.\n",
        "10. Example:\n        `printf 'x' | cat`.\n",
        "1234567890. Example:\n            `printf 'x' | cat`.\n",
        "10. Example:\n    <pre>\n    `printf 'x' | cat`.\n    </pre>\n",
        "10. Example:\n    ```text\n    `printf 'x' | cat`.\n    ```\n",
        "10. Example:\n    > `printf 'x' | cat`.\n",
        "10. Example:\n    - `printf 'x' | cat`.\n",
    ],
)
def test_unproven_or_code_blocks_keep_literal_backticks(source: str) -> None:
    projection = tool._markdown_shell_text(source, lambda: None)
    # Fence delimiters may be masked; single code-body backticks must stay raw.
    for match in re.finditer(r"(?<!`)`(?!`)", source):
        assert projection[match.start()] == "`"


_PAYLOADS = [
    '"$(printf "%s" \\\n  rm)" -rf /',
    '"$(printf "%s%s" r \\\n  m)" -rf /',
    "$CMD -rf /home/",
    "rm -rf /",
]


@pytest.mark.parametrize("payload", _PAYLOADS)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("container", ["numbered", "nested", "unknown-nested", "html", "fence"])
def test_list_boundaries_preserve_destructive_detection_or_partial(
    payload: str, newline: str, container: str
) -> None:
    inline = "`" + payload.replace("\n", "\n    ") + "`"
    sources = {
        "numbered": "10. Example:\n    " + inline + ".\n",
        "nested": "- Parent:\n  10. Example:\n      " + inline.replace("\n", "\n  ") + ".\n",
        "unknown-nested": "10. Parent:\n    - Example:\n      "
        + inline.replace("\n", "\n  ")
        + ".\n",
        "html": "10. Example:\n    <pre>\n    " + inline + "\n    </pre>\n",
        "fence": "10. Example:\n    ```sh\n    " + inline + "\n    ```\n",
    }
    source = sources[container].replace("\n", newline)
    report = tool.node({"components": ["guide.md"], "file_cache": {"guide.md": source}})
    assert any(f.rule_id == "TM1" and f.severity == "HIGH" for f in report["findings"]) or any(
        event["outcome"] != "completed" for event in report["inspection_ledger"]
    )


@pytest.mark.parametrize(
    "boundary", ["", "    # New heading", "    ---", "    > quote", "    - Child:"]
)
def test_unclosed_span_cannot_borrow_list_margin_across_block_boundary(boundary: str) -> None:
    source = "10. Example:\n    `$(resolve_tool).example\n" + boundary + "\n    ` -rf /\n"
    report = tool.node({"components": ["guide.md"], "file_cache": {"guide.md": source}})
    assert any(f.rule_id == "TM1" and f.severity == "HIGH" for f in report["findings"]) or any(
        event["outcome"] != "completed" for event in report["inspection_ledger"]
    )


_LIST_TABLE_BODIES = [
    "Left | Right\n--- | ---\n`$(resolve_tool).example | x\n` -rf / | y\n",
    "Left | Right\n--- | ---\n`$(resolve_tool).example | ` -rf /\n",
    "Left | Right\n--- | ---\n`$(resolve_tool).example\n` -rf /\n",
    "`$(resolve_tool).example | ` -rf /\n--- | ---\nleft | right\n",
]


def _list_table(marker: str, body: str) -> str:
    prefix = "- Parent:\n  " if marker == "nested" else ""
    marker = "1." if marker == "nested" else marker
    margin = len(prefix.rsplit("\n", 1)[-1]) + len(marker) + 1
    return (
        prefix
        + marker
        + " Example:\n"
        + "".join(" " * margin + line for line in body.splitlines(keepends=True))
    )


@pytest.mark.parametrize("body", _LIST_TABLE_BODIES)
@pytest.mark.parametrize("marker", ["10.", "100.", "123456789.", "nested"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_list_relative_tables_keep_separate_cell_and_row_coverage(
    body: str, marker: str, newline: str
) -> None:
    # No standalone payload: the actual cross-cell/row source must retain its gap.
    source = ("# Résumé — 比較\n\n" + _list_table(marker, body)).replace("\n", newline)
    assert tool._markdown_shell_text(source, lambda: None) == source
    report = tool.node({"components": ["guide.md"], "file_cache": {"guide.md": source}})
    assert any(f.rule_id == "TM1" and f.severity == "HIGH" for f in report["findings"]) or any(
        event["outcome"] != "completed" for event in report["inspection_ledger"]
    )


@pytest.mark.parametrize("body", _LIST_TABLE_BODIES)
@pytest.mark.parametrize("width", [2, 3])
def test_wider_unmatched_table_ticks_keep_exact_source(body: str, width: int) -> None:
    source = _list_table("10.", body.replace("`", "`" * width))
    assert tool._markdown_shell_text(source, lambda: None) == source


@pytest.mark.parametrize("body", _LIST_TABLE_BODIES)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_table_header_on_ordered_marker_keeps_following_cells_separate(
    body: str, newline: str
) -> None:
    lines = body.splitlines(keepends=True)
    source = ("10. " + lines[0] + "".join("    " + line for line in lines[1:])).replace(
        "\n", newline
    )
    assert tool._markdown_shell_text(source, lambda: None) == source
    report = tool.node({"components": ["guide.md"], "file_cache": {"guide.md": source}})
    assert any(f.rule_id == "TM1" and f.severity == "HIGH" for f in report["findings"]) or any(
        event["outcome"] != "completed" for event in report["inspection_ledger"]
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "body",
    ["`printf x`\n    --- | ---", "`printf 'x' | cat`.\n    --- | --- | ---"],
)
def test_mismatched_table_columns_keep_list_inline_ownership(body: str, newline: str) -> None:
    source = ("10. Example:\n    " + body + "\n").replace("\n", newline)
    assert tool._markdown_shell_text(source, lambda: None) == source.replace("`", " ")
    assert not tool.has_bounded_parse_exhaustion(source, lambda: None, file_type="markdown")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "header",
    ["`$(resolve_tool).example | ` -rf /", "`$CMD -rf /`", "`printf 'x' | cat`."],
)
def test_table_column_lookahead_preserves_runtime_command_gaps(header: str, newline: str) -> None:
    source = ("10. Example:\n    " + header + "\n    --- | ---\n").replace("\n", newline)
    if "|" in header:
        assert tool._markdown_shell_text(source, lambda: None) == source
    assert tool.has_bounded_parse_exhaustion(source, lambda: None, file_type="markdown")
