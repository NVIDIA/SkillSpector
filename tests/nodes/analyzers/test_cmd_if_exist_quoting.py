# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for Windows batch directory tests."""

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason, ParserCheck
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tool_misuse
from skillspector.nodes.analyzers import static_runner

_DIRECTORY_TEST = r'if exist "%out%\" goto :fail_named'
_COMMENTS = "".join(f"rem Benign fixture comment for step {step:03d}\n" for step in range(120))


def _scan(path: str, content: str):
    return static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: content}}, [tool_misuse]
    )


@pytest.mark.parametrize(
    ("path", "statement", "newline"),
    [
        ("check.bat", _DIRECTORY_TEST, "\r\n"),
        ("check.CMD", r'  @IF NOT EXIST "C:\Output Folder\" GOTO fail_named', "\n"),
    ],
)
def test_cmd_directory_operand_can_end_in_backslash(path: str, statement: str, newline: str):
    source = f"{statement}\nexit /b 0\n:fail_named\nexit /b 1\n{_COMMENTS}"
    result = _scan(path, source.replace("\n", newline))

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert not result["findings"]


@pytest.mark.parametrize("path", ["check.sh", "check.txt"])
def test_cmd_quote_semantics_do_not_apply_to_other_sources(path: str):
    result = _scan(path, f"{_DIRECTORY_TEST}\n{_COMMENTS}")

    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_cmd_directory_operand_keeps_later_command_analysis():
    result = _scan("check.cmd", f'{_DIRECTORY_TEST}\n{_COMMENTS}rm -rf /\nbash -c "$PAYLOAD"\n')

    row = result["inspection_ledger"][0]
    assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
    assert row["parser_diagnostic"]["check"] is ParserCheck.COMMAND_STRING_RUNTIME
    assert row["parser_diagnostic"]["start_line"] == 123
    assert any(
        finding.rule_id == "TM1" and finding.start_line == 122 for finding in result["findings"]
    )


def test_cmd_directory_operand_does_not_close_an_unrelated_quote():
    source = f'echo "unterminated\n{_DIRECTORY_TEST}\n{_COMMENTS}'
    result = _scan("check.bat", source)

    row = result["inspection_ledger"][0]
    assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
    assert row["parser_diagnostic"]["check"] is ParserCheck.COMMAND_WORD_BOUNDARY
    assert row["parser_diagnostic"]["start_line"] == 1


@pytest.mark.parametrize(
    ("statement", "environment"),
    [
        (_DIRECTORY_TEST, "BASH_ENV"),
        (r'ｉｆ ｅｘｉｓｔ "%out%\" goto :fail_named', "ＢＡＳＨ＿ＥＮＶ"),
    ],
)
def test_cmd_looking_line_inside_shell_program_keeps_runtime_diagnostic(
    statement: str, environment: str
):
    source = f"""bash -c 'printf "hello
{statement}
{environment}=/tmp/startup bash -c true
'
"""
    result = _scan("check.cmd", source)

    row = result["inspection_ledger"][0]
    assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
    assert row["parser_diagnostic"]["check"] is ParserCheck.COMMAND_STRING_RUNTIME
    assert row["parser_diagnostic"]["start_line"] == 3


def test_cmd_operand_in_raw_source_is_recognized_in_compacted_view():
    source = f"{_DIRECTORY_TEST}\nrem t e s t e x a m p l e\n{_COMMENTS}"

    result = _scan("check.bat", source)

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_cmd_looking_line_keeps_runtime_selected_command_diagnostic():
    source = '${COMMAND:-"\n' + _DIRECTORY_TEST + "\n} -rf /\n"

    result = _scan("check.bat", source)

    row = result["inspection_ledger"][0]
    assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
    assert row["parser_diagnostic"]["check"] is ParserCheck.COMMAND_IDENTITY
