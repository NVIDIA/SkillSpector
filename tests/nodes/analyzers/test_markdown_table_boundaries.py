# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GFM table cells own inline delimiters without hiding executable source.

Table syntax follows https://github.github.com/gfm/#tables-extension-.
Every shell-looking fixture is inert scanner input, never executed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.mcp_server import run_scan
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner
from tests.nodes.analyzers.test_documentation_reconstruction import _assert_llm_mode
from tests.nodes.analyzers.test_documentation_reconstruction import (
    successful_llm_transport as successful_llm_transport,
)

_TABLE_ROWS = "| Example |\n| --- |\n| `$(resolve_tool).example |\n| ` -rf / |"
_TABLE_CELLS = "| Left | Right |\n| --- | --- |\n| `$(resolve_tool).example | ` -rf / |"
_EXCESS_CELL = "| Good |\n| --- |\n| Fine | `$(resolve_tool).example` -rf / |"
_EXACT_TABLE_CASES = [
    pytest.param(_TABLE_ROWS, id="table-rows"),
    pytest.param(_TABLE_CELLS, id="table-cells"),
]
_SAME_CELL = "| Example |\n| --- |\n| `$(hostname).example` |"
_INVALID_TABLE = (
    "| Left | Right |\n| --- |\n| `$(hostname).example | value |\n| /service` | value |"
)


def _scan(content: str, path: str = "references/table.md") -> dict:
    return static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: content}}, [tm_module]
    )


def _assert_partial(result: dict) -> None:
    event = result["inspection_ledger"][0]
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize("content", _EXACT_TABLE_CASES)
def test_exact_table_reproductions_retain_incomplete_coverage(content: str) -> None:
    result = _scan(content)
    _assert_partial(result)
    assert not any(finding.rule_id == "TM1" for finding in result["findings"])


@pytest.mark.parametrize("content", _EXACT_TABLE_CASES)
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_table_cell_and_row_delimiters_cannot_pair_across_boundaries(
    content: str, width: int, line_ending: str
) -> None:
    # Wider unmatched Markdown runs also retain their exact source spelling;
    # they need not have the same executable shell meaning as single backticks.
    source = content.replace("`", "`" * width).replace("\n", line_ending)
    assert tm_module._markdown_shell_text(source, lambda: None) == source


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(_TABLE_ROWS.replace("---", "-"), id="one-hyphen-delimiter"),
        pytest.param(_TABLE_ROWS.replace("| Example |", "Example"), id="pipe-free-header"),
        pytest.param("Example\n:-:\n" + _TABLE_ROWS.split("\n", 2)[2], id="pipe-free-aligned"),
        pytest.param("Example\n---:\n" + _TABLE_ROWS.split("\n", 2)[2], id="pipe-free-right"),
        pytest.param("Example\n:---:\n" + _TABLE_ROWS.split("\n", 2)[2], id="pipe-free-center"),
        pytest.param(_TABLE_CELLS.replace("| --- | --- |", "| :-: | --: |"), id="alignment"),
        pytest.param(
            "Left | Right\n--- | ---\n`$(resolve_tool).example | ` -rf /",
            id="no-outer-pipes",
        ),
        pytest.param(
            "| Left | Right |\n--- | ---\n`$(resolve_tool).example | ` -rf / |",
            id="inconsistent-outer-pipes",
        ),
        pytest.param(
            "| Left | Right |\n| --- | --- |\n`$(resolve_tool).example\n` -rf /",
            id="body-rows-without-pipes",
        ),
        pytest.param(
            "| Left | Right | Extra |\n| --- | --- | --- |\n| `$(resolve_tool).example | ` -rf / |",
            id="fewer-body-cells",
        ),
        pytest.param(
            "| Left | Right |\n| --- | --- |\n| `$(resolve_tool).example | ` -rf / | extra |",
            id="more-body-cells",
        ),
        pytest.param(
            "| `$(resolve_tool).example | ` -rf / |\n| --- | --- |\n| left | right |",
            id="header-cells",
        ),
        pytest.param(
            "| Left | Right |\n| --- | --- |\n| `$(resolve_tool).example | right |\n| ` -rf / |",
            id="fewer-cells-on-following-row",
        ),
        pytest.param(
            _TABLE_ROWS.replace("| Example |", r"| Left\|Right |"), id="escaped-header-pipe"
        ),
        pytest.param(
            "- | Left | Right |\n  | --- | --- |\n  | `$(resolve_tool).example | ` -rf / |",
            id="single-list-table",
        ),
        pytest.param(
            "Intro `$(resolve_tool).example\n| ` -rf / | Right |\n| --- | --- |\n| left | right |",
            id="preceding-paragraph-cannot-pair-with-header",
        ),
        pytest.param(
            "| A |\n| --- |\n| `$(resolve_tool).example |\n| B |\n| --- |\n| ` -rf / |",
            id="adjacent-header-looking-rows-remain-body",
        ),
    ],
)
def test_valid_gfm_table_shapes_keep_cell_ownership(content: str) -> None:
    assert tm_module._markdown_shell_text(content, lambda: None) == content
    _assert_partial(_scan(content))


@pytest.mark.parametrize(
    "preamble",
    [
        pytest.param("| Left | Right |\n| --- |\n", id="header-has-more-cells"),
        pytest.param("| Left |\n| --- | --- |\n", id="delimiter-has-more-cells"),
        pytest.param("| Left | Right |\n| --- | |\n", id="empty-delimiter-cell"),
        pytest.param("| Left | Right |\n| : | --- |\n", id="colon-without-hyphen"),
        pytest.param("| Left | Right |\n| - - | --- |\n", id="internal-delimiter-space"),
        pytest.param("| Left | Right |\n| --x | --- |\n", id="non-delimiter-character"),
        pytest.param("| Left | Right |\n| === | === |\n", id="equals-are-not-delimiters"),
        pytest.param("| Left | Right |\n\n| --- | --- |\n", id="blank-before-delimiter"),
        pytest.param("| Left\\|Right |\n| --- | --- |\n", id="escaped-pipe-is-not-a-cell"),
        pytest.param("", id="no-header-and-delimiter"),
    ],
)
def test_invalid_tables_keep_ordinary_paragraph_inline_ownership(preamble: str) -> None:
    # GFM example203 explicitly makes a mismatched header/delimiter a paragraph.
    # These two rows therefore belong to one valid multiline inline code span.
    source = preamble + "| `$(hostname).example | value |\n| /service` | value |"
    assert tm_module._markdown_shell_text(source, lambda: None) == source.replace("`", " ")
    assert _scan(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("header", ["Header |", "| Header |"])
def test_setext_precedence_does_not_create_a_single_column_table(header: str) -> None:
    # cmark-gfm parses this first pair as a Setext heading. A pipe only in the
    # header does not override the block parser's precedence for a bare '---'.
    source = header + "\n---\n| `$(hostname).example |\n| /service` |"
    assert tm_module._markdown_shell_text(source, lambda: None) == source.replace("`", " ")
    assert _scan(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_single_list_table_preserves_valid_same_cell_code_span() -> None:
    source = "- | Example |\n  | --- |\n  | `$(hostname).example` |"
    assert tm_module._markdown_shell_text(source, lambda: None) == source.replace("`", " ")
    assert _scan(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("content", [_TABLE_CELLS, _SAME_CELL])
@pytest.mark.parametrize("container", ["nested-list", "blockquote", "quoted-list"])
def test_unproven_table_containers_keep_literal_delimiters(content: str, container: str) -> None:
    lines = content.splitlines()
    if container == "nested-list":
        source = "- - " + lines[0] + "\n" + "\n".join("    " + line for line in lines[1:])
    elif container == "blockquote":
        source = "\n".join("> " + line for line in lines)
    else:
        source = "> - " + lines[0] + "\n" + "\n".join(">   " + line for line in lines[1:])
    # Existing conservative handling of unsupported containers must not grant
    # new inline ownership as an incidental consequence of table recognition.
    assert tm_module._markdown_shell_text(source, lambda: None) == source
    _assert_partial(_scan(source))


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize(
    "body",
    [
        pytest.param("$(hostname).example", id="hostname"),
        pytest.param("$(strip $(DEFAULT))", id="make-expression"),
        pytest.param(r"$(hostname).example\|detail", id="escaped-pipe-in-code"),
    ],
)
def test_same_cell_code_spans_preserve_contents_and_complete_coverage(
    width: int, body: str
) -> None:
    delimiter = "`" * width
    source = f"| Example | Notes |\n| --- | --- |\n| {delimiter}{body}{delimiter} | value |"
    projected = tm_module._markdown_shell_text(source, lambda: None)
    assert projected == source.replace(delimiter, " " * width)
    start = source.index(body)
    assert projected[start : start + len(body)] == body
    assert _scan(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_escaped_pipe_in_prose_does_not_split_a_table_cell() -> None:
    source = "| Example |\n| --- |\n| `$(hostname).example\\|suffix` |"
    assert tm_module._markdown_shell_text(source, lambda: None) == source.replace("`", " ")


@pytest.mark.parametrize("backslashes", [0, 1, 2, 3])
def test_gfm_table_pipe_escaping_is_not_shell_backslash_parity(backslashes: int) -> None:
    # github/cmark-gfm extensions/ext_scanners.re accepts a final backslash-pipe
    # pair even after another backslash; its generated C scanner confirms this.
    body = "$(hostname).example" + "\\" * backslashes + "| detail"
    source = f"| Example |\n| --- |\n| `{body}` |"
    expected = source.replace("`", " ") if backslashes else source
    assert tm_module._markdown_shell_text(source, lambda: None) == expected


def test_table_header_delimiters_cannot_pair_with_body_cells() -> None:
    source = "| `$(resolve_tool).example |\n| --- |\n| ` -rf / |"
    assert tm_module._markdown_shell_text(source, lambda: None) == source
    _assert_partial(_scan(source))


def test_table_excess_cells_do_not_hide_concrete_source_findings() -> None:
    # GFM may omit excess cells from rendering; inspection still retains source.
    source = "| Example |\n| --- |\n| value | rm -rf / |"
    result = _scan(source, "references/extra-cell.md")
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]
    assert findings
    assert all(finding.start_line == 3 for finding in findings)
    assert all(finding.file == "references/extra-cell.md" for finding in findings)
    assert tm_module._markdown_shell_text(source, lambda: None) == source


@pytest.mark.parametrize("width", [1, 2, 3])
def test_dropped_table_cells_cannot_grant_inline_code_ownership(width: int) -> None:
    # GFM example204 and cmark-gfm table.c omit body cells beyond header width.
    # With no rendered inline node, their delimiters remain literal scanner input.
    source = _EXCESS_CELL.replace("`", "`" * width)
    assert tm_module._markdown_shell_text(source, lambda: None) == source
    if width == 1:
        _assert_partial(_scan(source))


@pytest.mark.parametrize("size", [4096, 8192, 16384])
def test_long_invalid_table_delimiter_keeps_paragraph_ownership_with_callback_budget(
    size: int,
) -> None:
    source = "| Example |\n| " + "-" * size + "x |\n| `$(hostname).example |\n| /service` |"
    checks = 0

    def check_runtime() -> None:
        nonlocal checks
        checks += 1

    assert tm_module._markdown_shell_text(source, check_runtime) == source.replace("`", " ")
    assert 1 <= checks <= size // 64 + 64


def test_long_invalid_table_delimiter_checks_deadline_before_rejection() -> None:
    source = "| Example |\n| " + "-" * 32768 + "x |"
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 8:
            raise TimeoutError("inert invalid-table deadline")

    with pytest.raises(TimeoutError, match="inert invalid-table deadline"):
        tm_module._markdown_shell_text(source, cancel)
    assert checks == 8


@pytest.mark.parametrize("width", [2, 3])
def test_shorter_literal_backticks_inside_a_table_cell_are_preserved(width: int) -> None:
    delimiter = "`" * width
    body = "literal ` mark"
    source = f"| Example |\n| --- |\n| {delimiter}{body}{delimiter} |"
    assert tm_module._markdown_shell_text(source, lambda: None) == source.replace(
        delimiter, " " * width
    )


@pytest.mark.parametrize("separator", ["\n\n", "\n# Next\n"])
def test_table_termination_restores_multiline_paragraph_spans(separator: str) -> None:
    table = "| Header |\n| --- |\n| value |"
    paragraph = "Use `$(hostname).example\n/service`."
    source = table + separator + paragraph
    assert tm_module._markdown_shell_text(source, lambda: None) == source.replace("`", " ")
    assert _scan(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("content", [_TABLE_ROWS, _TABLE_CELLS, _SAME_CELL])
def test_unknown_table_fragment_grants_no_inline_delimiter_ownership(content: str) -> None:
    assert tm_module._markdown_shell_text(content, lambda: None, complete_context=False) == content


@pytest.mark.parametrize("content", [_TABLE_CELLS, _SAME_CELL])
@pytest.mark.parametrize("container", ["fenced", "indented", "html"])
def test_table_looking_literal_code_retains_backticks(container: str, content: str) -> None:
    if container == "fenced":
        source = "```sh\n" + content + "\n```"
        projected = tm_module._markdown_shell_text(source, lambda: None)
        assert content in projected
    elif container == "indented":
        source = "\n".join("    " + line for line in content.splitlines())
        assert tm_module._markdown_shell_text(source, lambda: None) == source
    else:
        source = "<pre>\n" + content + "\n</pre>"
        assert tm_module._markdown_shell_text(source, lambda: None) == source


@pytest.mark.parametrize("content", _EXACT_TABLE_CASES)
def test_table_boundary_repair_retains_concrete_findings_and_source_lines(content: str) -> None:
    source = "Preface.\n\n" + content + "\n\n```sh\nrm -rf /\n```\n"
    projected = tm_module._markdown_shell_text(source, lambda: None)
    assert len(projected) == len(source)
    assert [i for i, ch in enumerate(projected) if ch in "\r\n"] == [
        i for i, ch in enumerate(source) if ch in "\r\n"
    ]
    command_start = source.index("rm -rf /")
    assert projected[command_start : command_start + len("rm -rf /")] == "rm -rf /"
    result = _scan(source, "references/evidence.md")
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]
    assert findings
    expected_line = source.count("\n", 0, command_start) + 1
    assert all(finding.start_line == expected_line for finding in findings)
    assert all(finding.file == "references/evidence.md" for finding in findings)
    assert any(
        finding.matched_text == "rm -rf /" and finding.severity == "HIGH" for finding in findings
    )
    _assert_partial(result)


@pytest.mark.parametrize("count", [128, 256, 512])
def test_many_table_cells_preserve_projection_with_callback_budget(count: int) -> None:
    header = "|" + " Header |" * count
    delimiter = "|" + " --- |" * count
    row = "|" + " `value` |" * count
    source = "\n".join((header, delimiter, row))
    checks = 0

    def check_runtime() -> None:
        nonlocal checks
        checks += 1

    assert tm_module._markdown_shell_text(source, check_runtime) == source.replace("`", " ")
    # Bound callback overhead without requiring a checkpoint for every cell.
    # This count alone does not measure character reads or prove linear parsing.
    assert checks <= 24 * count + 64


def test_table_row_scan_checks_cancellation_within_many_cells() -> None:
    source = "|" + " Header |" * 8192 + "\n|" + " --- |" * 8192
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 8:
            raise TimeoutError("inert table deadline")

    with pytest.raises(TimeoutError, match="inert table deadline"):
        tm_module._markdown_shell_text(source, cancel)
    assert checks == 8


@pytest.mark.parametrize("content", _EXACT_TABLE_CASES)
def test_table_projection_propagates_cancellation(content: str) -> None:
    def cancel() -> None:
        raise TimeoutError("inert table cancellation")

    with pytest.raises(TimeoutError, match="inert table cancellation"):
        tm_module._markdown_shell_text(content, cancel)


def _public_results(
    tmp_path: Path, content: str, use_llm: bool, referenced: bool, calls: list[str]
) -> tuple[int, dict, tuple[dict, dict]]:
    if referenced:
        (tmp_path / "SKILL.md").write_text(
            "---\nname: table-reference\ndescription: Inspect table examples.\n---\n\n"
            "Read `references/table.md`.\n",
            encoding="utf-8",
        )
        (tmp_path / "references").mkdir()
        (tmp_path / "references" / "table.md").write_text(content, encoding="utf-8")
    else:
        (tmp_path / "SKILL.md").write_text(content, encoding="utf-8")
    args = ["scan", str(tmp_path), "--format", "json", "--fail-on-incomplete"]
    if not use_llm:
        args.append("--no-llm")
    cli = CliRunner().invoke(app, args)
    cli_calls = list(calls)
    calls.clear()
    mcp = asyncio.run(run_scan(str(tmp_path), use_llm=use_llm, output_format="json"))
    mcp_calls = list(calls)
    # Both interfaces have run before any decoding or report assertions.
    assert cli.exit_code in {0, 1}, cli.output
    assert cli.exception is None or (
        isinstance(cli.exception, SystemExit) and cli.exception.code == cli.exit_code
    )
    reports = (json.loads(cli.output), json.loads(mcp["report"]))
    for report, recorded_calls in zip(reports, (cli_calls, mcp_calls), strict=True):
        _assert_llm_mode(report, use_llm, recorded_calls)
        assert report["execution_successful"] is True
        assert report["analysis_completeness"]["execution_successful"] is True
        statuses = {
            row["analyzer_id"]: row for row in report["analysis_completeness"]["analyzer_statuses"]
        }
        for analyzer in (
            "semantic_developer_intent",
            "semantic_quality_policy",
            "semantic_security_discovery",
        ):
            status = statuses[analyzer]
            assert status["status"] == ("completed" if use_llm else "disabled")
            assert status["completed"] == status["planned_work"]
            assert (status["planned_work"] > 0) is use_llm
            assert all(status[key] == 0 for key in ("partial", "skipped", "failed", "unaccounted"))
        if not use_llm:
            assert report["metadata"].get("llm_calls_attempted", 0) == 0
            assert report["metadata"].get("llm_calls_succeeded", 0) == 0
    return cli.exit_code, mcp, reports


def _assert_public_partial(report: dict, referenced: bool) -> None:
    path = "references/table.md" if referenced else "SKILL.md"
    assert any(
        row["path"] == path
        and row["phase"] == "static"
        and row["outcome"] == "partial"
        and row["reason_code"] == "static_parse_limit"
        and "static_patterns_tool_misuse" in row["analyzers"]
        for row in report["analysis_completeness"]["ledger_exceptions"]
    )
    if referenced:
        assert any(
            issue["id"] == "AE1"
            and issue["location"]["file"] == "SKILL.md"
            and path in issue["finding"]
            for issue in report["issues"]
        )


@pytest.mark.parametrize(
    "content,complete",
    [
        pytest.param(_TABLE_ROWS, False, id="table-rows"),
        pytest.param(_TABLE_CELLS, False, id="table-cells"),
        pytest.param(_EXCESS_CELL, False, id="dropped-excess-cell"),
        pytest.param(_SAME_CELL, True, id="same-cell-code"),
        pytest.param(_INVALID_TABLE, True, id="invalid-table-paragraph"),
    ],
)
@pytest.mark.parametrize("use_llm", [False, True], ids=["no-llm", "llm"])
@pytest.mark.parametrize("referenced", [False, True], ids=["entrypoint", "referenced"])
def test_table_boundaries_reach_strict_cli_and_mcp_in_both_semantic_modes(
    tmp_path: Path,
    content: str,
    complete: bool,
    use_llm: bool,
    referenced: bool,
    successful_llm_transport: list[str],
) -> None:
    exit_code, mcp, reports = _public_results(
        tmp_path, content, use_llm, referenced, successful_llm_transport
    )
    assert exit_code == (0 if complete else 1)
    assert mcp["safe_to_install"] is complete
    assert mcp["llm_used"] is use_llm
    for report in reports:
        assert report["analysis_completeness"]["is_complete"] is complete
        if complete:
            assert report["analysis_completeness"]["coverage_percent"] == 100.0
            assert report["risk_assessment"]["recommendation"] == "SAFE"
            assert not any(issue["id"] == "AE1" for issue in report["issues"])
        else:
            assert report["risk_assessment"]["recommendation"] != "SAFE"
            _assert_public_partial(report, referenced)


@pytest.mark.parametrize("use_llm", [False, True], ids=["no-llm", "llm"])
def test_referenced_table_uncertainty_retains_high_severity_evidence_in_both_public_gates(
    tmp_path: Path, use_llm: bool, successful_llm_transport: list[str]
) -> None:
    source = _TABLE_ROWS + "\n\n```sh\nrm -rf /\n```\n"
    exit_code, mcp, reports = _public_results(
        tmp_path, source, use_llm, True, successful_llm_transport
    )
    assert exit_code == 1
    assert mcp["safe_to_install"] is False
    expected_line = source.count("\n", 0, source.index("rm -rf /")) + 1
    for report in reports:
        assert any(
            issue["id"] == "TM1"
            and issue["severity"] == "HIGH"
            and issue["finding"] == "rm -rf /"
            and issue["location"]["file"] == "references/table.md"
            and issue["location"]["start_line"] == expected_line
            for issue in report["issues"]
        )
        assert report["risk_assessment"]["recommendation"] != "SAFE"
        assert report["analysis_completeness"]["is_complete"] is False
        _assert_public_partial(report, True)
