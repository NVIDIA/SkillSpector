# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make documentation remains inspectable without hiding executable commands."""

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

# These are inert scanner inputs. No Make expression or command is executed.
_MAKE_ERROR_DOCUMENTATION = "`$(error)`. Do not add any of these via `--extra-jflags`:"
_MAKE_EXPRESSIONS = [
    pytest.param("$(CC)", id="variable"),
    pytest.param("${CFLAGS}", id="brace-variable"),
    pytest.param("$(error unsupported option)", id="error-function"),
    pytest.param("$(strip $(EXTRA_FLAGS))", id="nested-strip"),
    pytest.param("$(if $(DEBUG),-g,-O2)", id="conditional-function"),
    pytest.param("$(foreach target,$(TARGETS),$(target))", id="nested-foreach"),
]
_RUNTIME_COMMANDS = [
    pytest.param("$($(resolve_tool)/printf %s rm) -rf /", id="runtime-selected-helper"),
    pytest.param("`$(resolve_tool).example` -rf /", id="literal-shell-backticks"),
]


def test_make_error_extra_flags_documentation_is_complete_and_preserves_source() -> None:
    content = _MAKE_ERROR_DOCUMENTATION
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}, [tm_module]
    )

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert result["findings"] == []
    projected = tm_module._markdown_shell_text(content, lambda: None)
    assert projected == content.replace("`", " ")
    for expression in ("$(error)", "--extra-jflags"):
        start = content.index(expression)
        assert projected[start : start + len(expression)] == expression


@pytest.mark.parametrize("expression", _MAKE_EXPRESSIONS)
@pytest.mark.parametrize("width", [1, 2, 3], ids=["single", "double", "triple"])
@pytest.mark.parametrize("layout", ["paragraph", "list", "table"])
def test_inline_make_expressions_keep_complete_coverage_and_source_offsets(
    expression: str, width: int, layout: str
) -> None:
    delimiter = "`" * width
    adjacent = f"{delimiter}{expression}{delimiter}/{delimiter}$(DEFAULT){delimiter}"
    if layout == "paragraph":
        content = f"The documented values are ({adjacent}); compare their expansions.\n"
    elif layout == "list":
        content = f"- Documented values: {adjacent}.\n"
    else:
        content = f"| Expression | Notes |\n| --- | --- |\n| {adjacent} | Documented values. |\n"
    path = "references/make-options.md"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: content}}, [tm_module]
    )

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert result["findings"] == []
    projected = tm_module._markdown_shell_text(content, lambda: None)
    assert projected == content.replace("`", " ")
    start = content.index(expression)
    assert projected[start : start + len(expression)] == expression


@pytest.mark.parametrize("command", _RUNTIME_COMMANDS)
@pytest.mark.parametrize("container", ["shell-fence", "long-inline-span"])
def test_make_documentation_does_not_hide_runtime_selected_shell_commands(
    command: str, container: str
) -> None:
    represented_command = (
        f"```sh\n{command}\n```" if container == "shell-fence" else f"Run ``{command}``."
    )
    content = _MAKE_ERROR_DOCUMENTATION + "\n\n" + represented_command + "\n"
    path = "references/make-options.md"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: content}}, [tm_module]
    )

    entry = result["inspection_ledger"][0]
    assert entry["outcome"] is LedgerOutcome.PARTIAL
    assert entry["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
    projected = tm_module._markdown_shell_text(content, lambda: None)
    start = content.index(command)
    assert projected[start : start + len(command)] == command


@pytest.mark.parametrize("width", [1, 2, 3])
def test_dangerous_command_after_make_documentation_keeps_finding_and_source_line(
    width: int,
) -> None:
    delimiter = "`" * width
    command = "rm -rf /"
    content = _MAKE_ERROR_DOCUMENTATION + f"\n\nRun {delimiter}{command}{delimiter}.\n"
    path = "references/make-options.md"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: content}}, [tm_module]
    )

    destructive = [finding for finding in result["findings"] if finding.rule_id == "TM1"]
    assert destructive
    for finding in destructive:
        assert finding.severity == "HIGH"
        assert finding.file == path
        assert finding.start_line == 3
        assert command in (finding.matched_text or "")
        assert command in (finding.code_snippet or "")
    projected = tm_module._markdown_shell_text(content, lambda: None)
    start = content.index(command)
    assert projected[start : start + len(command)] == command


@pytest.mark.parametrize("use_llm", [False, True], ids=["no-llm", "llm"])
@pytest.mark.parametrize("complete", [True, False], ids=["make-documentation", "runtime-command"])
def test_repeated_make_documentation_references_keep_honest_public_verdicts(
    tmp_path: Path,
    use_llm: bool,
    complete: bool,
    successful_llm_transport: list[str],
) -> None:
    reference = "references/make-options.md"
    instructions = (
        "---\nname: make-options-guide\ndescription: Explain documented build options.\n---\n\n"
        f"Read `{reference}`.\nReview `{reference}` when comparing options.\n"
        f"Check `{reference}` again before selecting a value.\n"
    )
    assert instructions.count(reference) == 3
    (tmp_path / "SKILL.md").write_text(instructions, encoding="utf-8")
    (tmp_path / "references").mkdir()
    documentation = _MAKE_ERROR_DOCUMENTATION + "\n\nUse `$(strip $(CFLAGS))` for comparison.\n"
    if not complete:
        documentation += "\n```sh\n$($(resolve_tool)/printf %s rm) -rf /\n```\n"
    (tmp_path / reference).write_text(documentation, encoding="utf-8")

    args = ["scan", str(tmp_path), "--format", "json", "--fail-on-incomplete"]
    if not use_llm:
        args.append("--no-llm")
    cli_result = CliRunner().invoke(app, args)
    assert cli_result.exit_code == (0 if complete else 1), cli_result.output
    cli_report = json.loads(cli_result.output)
    _assert_llm_mode(cli_report, use_llm, successful_llm_transport)
    successful_llm_transport.clear()
    mcp_result = asyncio.run(run_scan(str(tmp_path), use_llm=use_llm, output_format="json"))
    mcp_report = json.loads(mcp_result["report"])
    _assert_llm_mode(mcp_report, use_llm, successful_llm_transport)
    assert mcp_result["safe_to_install"] is complete
    assert mcp_result["llm_used"] is use_llm

    for report in (cli_report, mcp_report):
        coverage = report["analysis_completeness"]
        assert coverage["is_complete"] is complete
        assert coverage["entirely_uninspected_files"] == 0
        if complete:
            assert coverage["fully_inspected_files"] == 2
            assert coverage["partially_inspected_files"] == 0
            assert coverage["coverage_percent"] == 100.0
            assert report["risk_assessment"]["recommendation"] == "SAFE"
            assert not any(issue["id"] == "AE1" for issue in report["issues"])
        else:
            assert report["risk_assessment"]["recommendation"] != "SAFE"
            assert any(
                event["path"] == reference
                and event["reason_code"] == LedgerReason.STATIC_PARSE_LIMIT
                for event in coverage["ledger_exceptions"]
            )
        if not use_llm:
            assert report["metadata"].get("llm_calls_attempted", 0) == 0
            assert report["metadata"].get("llm_calls_succeeded", 0) == 0
