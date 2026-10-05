# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for batch-scan report serialization."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from markdown_it import MarkdownIt

from contrib.batch_scan.reports import _format_json, _format_markdown


def test_json_marks_error_entries_as_unsuccessful() -> None:
    entry = {
        "skill": {"name": "crashed-skill", "language": "en"},
        "risk_assessment": {"score": 0, "severity": "ERROR", "recommendation": "ERROR"},
        "components": [],
        "issues": [],
        "error": "scan crashed",
    }

    payload = json.loads(_format_json([entry]))

    assert payload["skills"][0]["error"] == "scan crashed"
    assert payload["skills"][0]["execution_successful"] is False


@pytest.mark.parametrize(
    "payload",
    [
        "safe` | 0/100 | LOW | 0 | en |\n<!--",
        "\r\n## Issues (0)\rNo security issues detected.\n<!--",
        "` `` ``` <script>alert(1)</script> [safe](https://example.invalid)",
        "\\| **safe** &lt;!--",
        " leading and trailing spaces ",
        "~~hidden~~",
        "\x1b[2J\x00\u202eLOW\u202c\x9b\u2066safe\u2069",
    ],
)
@pytest.mark.parametrize(
    "field",
    [
        "name",
        "language",
        "id",
        "message",
        "explanation",
        "remediation",
        "file",
        "reason_code",
        "path",
        "ledger_message",
    ],
)
def test_batch_markdown_treats_scan_content_as_literal_text(payload: str, field: str) -> None:
    entry = {
        "skill": {"name": "sample", "language": "zh"},
        "risk_assessment": {"score": 90, "severity": "CRITICAL"},
        "issues": [
            {
                "id": "P1",
                "message": "finding",
                "remediation": "review",
                "location": {"file": "SKILL.md", "start_line": 1},
            }
        ],
        "analysis_completeness": {
            "ledger_exceptions": [
                {"reason_code": "partial", "path": "run.py", "message": "inspect"}
            ]
        },
    }
    parser = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    original_blocks = [token.type for token in parser.parse(_format_markdown([entry]))]
    if field in {"name", "language"}:
        entry["skill"][field] = payload
    elif field == "file":
        entry["issues"][0]["location"][field] = payload
    elif field in {"reason_code", "path", "ledger_message"}:
        entry["analysis_completeness"]["ledger_exceptions"][0][
            "message" if field == "ledger_message" else field
        ] = payload
    else:
        entry["issues"][0][field] = payload
    original = deepcopy(entry)

    report = _format_markdown([entry])
    tokens = parser.parse(report)

    assert all(character.isprintable() or character in "\n\t" for character in report)
    assert [token.type for token in tokens] == original_blocks
    for token in tokens:
        assert token.type not in {"html_block", "fence", "code_block"}
        assert not any(
            child.type in {"html_inline", "link_open", "image", "s_open"}
            for child in token.children or []
        )
    assert entry == original
    assert json.loads(_format_json([entry]))["skills"][0]["skill"]["name"] == entry["skill"]["name"]


@pytest.mark.parametrize("value", ["a|b", r"a\|b", "`a`", "a``b`", "<script> & value"])
@pytest.mark.parametrize("table_cell", [False, True])
def test_batch_markdown_code_preserves_literal_values(value: str, table_cell: bool) -> None:
    from contrib.batch_scan.reports import _markdown_code

    source = _markdown_code(value, table_cell=table_cell)
    if table_cell:
        source = f"| Path |\n|---|\n| {source} |"
    tokens = MarkdownIt("commonmark").enable("table").parse(source)
    code = [
        child.content
        for token in tokens
        for child in token.children or []
        if child.type == "code_inline"
    ]
    assert code == [value]


def test_batch_markdown_strips_complete_ansi_sequences() -> None:
    from contrib.batch_scan.reports import _markdown_plain_text

    assert _markdown_plain_text("a\x1b[2Jb\x1b[31mc\x1b[0m") == "abc"
