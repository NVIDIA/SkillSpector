# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A wrapped inline code span must not become ``static_parse_limit``.

Issue #1023: when an inline code span crosses a soft line break and the
continuation line starts with a ``<placeholder>``, the Markdown projection
read the line as an HTML block and left every delimiter visible. The shell
tokenizer then read the closing delimiter on the last line as an unmatched
substitution opener, so the referenced file was only partially inspected and
the linking ``SKILL.md`` received AE1 HIGH.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.inspection_ledger import LedgerOutcome
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner

_GUIDE = (
    "Pass the config to both `tool inventory --config\n"
    "<path>` and `tool validate --config <path>` honor its `exclude` list; without\n"
    "`GEMINI.md` and `.github/copilot-instructions.md` files that describe a\n"
)
_JOINED_GUIDE = (
    "Pass the config to both `tool inventory --config <path>` and "
    "`tool validate --config <path>` honor its `exclude` list; without\n"
    "`GEMINI.md` and `.github/copilot-instructions.md` files that describe a\n"
)
_SKILL = (
    "---\n"
    "name: wrapped-span\n"
    "description: Demo skill that reproduces a scanner coverage gap.\n"
    "---\n"
    "\n"
    "See [references/guide.md](references/guide.md).\n"
)


def _blank_backticks(text: str) -> str:
    return "".join(" " if character == "`" else character for character in text)


def test_wrapped_inline_span_masks_every_backtick() -> None:
    projected = tm._markdown_shell_text(_GUIDE, lambda: None)
    assert len(projected) == len(_GUIDE)
    assert projected == _blank_backticks(_GUIDE)


def test_wrapped_inline_span_completes_static_analysis() -> None:
    assert tm.has_bounded_parse_exhaustion(_GUIDE, lambda: None, file_type="markdown") is False


def test_wrapped_inline_span_reaches_completed_ledger_outcome() -> None:
    path = "references/guide.md"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: _GUIDE}}, [tm]
    )
    row = result["inspection_ledger"][0]
    assert row["path"] == path
    assert row["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in row
    assert result["findings"] == []


def test_joined_inline_span_still_completes_static_analysis() -> None:
    assert (
        tm.has_bounded_parse_exhaustion(_JOINED_GUIDE, lambda: None, file_type="markdown") is False
    )


def test_raw_html_block_after_open_inline_span_stays_literal() -> None:
    content = "Pass `open\n<pre>\n`rm -rf /`\n</pre>\n"
    projected = tm._markdown_shell_text(content, lambda: None)
    assert projected.count("`") == content.count("`")


def test_type6_html_block_after_open_inline_span_stays_literal() -> None:
    content = "Pass `open\n<div>\n`rm -rf /`\n</div>\n"
    projected = tm._markdown_shell_text(content, lambda: None)
    assert projected == content


def test_wrapped_span_scan_is_complete_and_safe(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    references = tmp_path / "references"
    references.mkdir()
    (references / "guide.md").write_text(_GUIDE, encoding="utf-8", newline="\n")

    result = CliRunner().invoke(
        app,
        ["scan", str(tmp_path), "--no-llm", "--format", "json", "--fail-on-incomplete"],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    completeness = report["analysis_completeness"]
    assert completeness["coverage_percent"] == 100.0
    assert completeness["is_complete"] is True
    assert completeness["ledger_exceptions"] == []
    assert report["risk_assessment"]["score"] == 0
    assert report["risk_assessment"]["recommendation"] == "SAFE"
    assert not any(issue["id"] == "AE1" for issue in report["issues"])
