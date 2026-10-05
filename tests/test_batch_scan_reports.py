# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for batch-scan report serialization."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

from contrib.batch_scan import batch_scan
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


@pytest.fixture(params=[True, False], ids=["rich", "plain"])
def batch_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request):
    root = tmp_path / "skills"
    skill = root / "example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Example\n", encoding="utf-8")
    entry = {
        "skill": {"name": "example", "language": "en"},
        "risk_assessment": {"score": 0, "severity": "LOW", "recommendation": "SAFE"},
        "components": [],
        "issues": [],
    }
    monkeypatch.setattr(batch_scan, "create_api_key_pool_from_env", lambda: None)
    monkeypatch.setattr(
        batch_scan, "_scan_skill", lambda *args, **kwargs: (entry, entry.get("error"), "example")
    )
    monkeypatch.setattr(batch_scan, "format_terminal", lambda results: "Terminal report")
    if not request.param:
        monkeypatch.setitem(sys.modules, "rich.console", None)
    return root, entry


def _run_batch_cli(monkeypatch: pytest.MonkeyPatch, root: Path, format: str, *args: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_scan", str(root), "--no-llm", "--workers", "1", "--format", format, *args],
    )
    batch_scan._main_impl()


@pytest.mark.parametrize("format", ["json", "markdown"])
def test_batch_cli_stdout_contains_only_the_requested_report(
    batch_cli, monkeypatch: pytest.MonkeyPatch, capsys, format: str
) -> None:
    root, _ = batch_cli
    _run_batch_cli(monkeypatch, root, format)

    captured = capsys.readouterr()
    if format == "json":
        assert json.loads(captured.out)["skills"][0]["skill"]["name"] == "example"
    else:
        assert captured.out.startswith("# SkillSpector Batch Scan Report")
    assert "[1/1]" not in captured.out
    assert "SkillSpector Batch Scan" in captured.err
    assert "[1/1]" in captured.err


def test_batch_cli_terminal_keeps_progress_and_report_on_stdout(
    batch_cli, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root, _ = batch_cli
    _run_batch_cli(monkeypatch, root, "terminal")

    captured = capsys.readouterr()
    assert "[1/1]" in captured.out
    assert "Terminal report" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("format", ["terminal", "json", "markdown"])
def test_batch_cli_language_warning_goes_to_stderr(
    batch_cli, monkeypatch: pytest.MonkeyPatch, capsys, format: str
) -> None:
    root, entry = batch_cli
    entry["skill"]["language"] = "zh"
    _run_batch_cli(monkeypatch, root, format)

    captured = capsys.readouterr()
    assert "WARNING:" in captured.err
    assert "non-English skill" in captured.err
    assert "WARNING:" not in captured.out
    if format == "json":
        assert json.loads(captured.out)["skills"][0]["skill"]["language"] == "zh"


@pytest.mark.parametrize("error,score,exit_code", [("scan failed", 0, 2), (None, 85, 1)])
def test_batch_cli_json_remains_parseable_when_exit_status_is_nonzero(
    batch_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    error: str | None,
    score: int,
    exit_code: int,
) -> None:
    root, entry = batch_cli
    entry["risk_assessment"]["score"] = score
    if error:
        entry["error"] = error
    with pytest.raises(SystemExit) as exc:
        _run_batch_cli(monkeypatch, root, "json")

    assert exc.value.code == exit_code
    captured = capsys.readouterr()
    assert json.loads(captured.out)["skills"][0]["risk_assessment"]["score"] == score
    assert "[1/1]" in captured.err


@pytest.mark.parametrize("format", ["json", "markdown"])
def test_batch_cli_output_file_leaves_stdout_empty(
    batch_cli, monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path, format: str
) -> None:
    root, _ = batch_cli
    output = tmp_path / "report.txt"
    _run_batch_cli(monkeypatch, root, format, "--output", str(output))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Batch report saved to:" in captured.err
    if format == "json":
        assert json.loads(output.read_text(encoding="utf-8"))["batch"]["total_skills"] == 1
    else:
        assert output.read_text(encoding="utf-8").startswith("# SkillSpector Batch Scan Report")


@pytest.mark.parametrize("format", ["terminal", "json"])
def test_batch_cli_validation_errors_go_to_stderr(
    batch_cli, monkeypatch: pytest.MonkeyPatch, capsys, format: str
) -> None:
    root, _ = batch_cli
    with pytest.raises(SystemExit) as exc:
        _run_batch_cli(monkeypatch, root / "missing", format)

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "is not a directory" in " ".join(captured.err.split())
