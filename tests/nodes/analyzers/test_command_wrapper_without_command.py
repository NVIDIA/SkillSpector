# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A command wrapper followed directly by a control operator runs no command (#694)."""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner

_PADDING = "\n# ordinary padding\n" * 400


def _exhausted(content: str, file_type: str, *, complete_context: bool = True) -> bool:
    return tm_module.has_bounded_parse_exhaustion(
        content, lambda: None, file_type=file_type, complete_context=complete_context
    )


@pytest.mark.parametrize("wrapper", ["timeout", "sudo", "nice", "xargs"])
def test_markdown_table_cell_naming_a_wrapper_is_complete(wrapper: str) -> None:
    content = f"| Name | Description |\n|---|---|\n| {wrapper} | Max wait in seconds |\n"

    assert not _exhausted(content, "markdown")
    assert not _exhausted(content + _PADDING, "markdown")


def test_markdown_api_parameter_table_is_complete() -> None:
    content = (
        "| Name | Type | Default | Description |\n"
        "|---|---|---|---|\n"
        "| timeout | float | 150 | Seconds to wait |\n"
        "| retries | int | 3 | Attempts |\n"
    )

    assert not _exhausted(content, "markdown")


@pytest.mark.parametrize(
    ("content", "file_type"),
    [
        ("sudo | cat\n", "shell"),
        ("timeout; echo ok\n", "shell"),
        ("xargs || true\n", "shell"),
        ("nice & wait\n", "shell"),
        ("(nice)\n", "shell"),
        ("sudo -u | cat\n", "shell"),
        ("sudo |& cat\n", "shell"),
        ("timeout && echo ok\n", "shell"),
        ("case $a in x) nice ;; esac\n", "shell"),
        ("signal.alarm(timeout)\nvalue = 1\n", "python"),
        ("if (timeout) { start(); }\n", "javascript"),
    ],
    ids=[
        "pipe",
        "semicolon",
        "or-list",
        "background",
        "subshell",
        "option-without-value",
        "pipe-with-stderr",
        "and-list",
        "case-item-end",
        "python-call",
        "javascript-condition",
    ],
)
def test_wrapper_before_a_clause_end_is_complete(content: str, file_type: str) -> None:
    assert not _exhausted(content + _PADDING, file_type)


@pytest.mark.parametrize(
    "content",
    [
        "sudo >log $CMD -rf /\n",
        "sudo <in $CMD -rf /\n",
        "sudo &>log $CMD -rf /\n",
        "timeout ($CMD) -rf /\n",
        'x | sudo sh -c "$CMD"\n',
        'sudo |& sh -c "$CMD"\n',
        'sudo && eval "$CMD"\n',
        "case $a in sudo) $CMD -rf / ;; esac\n",
        'sudo -s; eval "$CMD"\n',
    ],
    ids=[
        "stdout-redirection",
        "stdin-redirection",
        "bash-and-redirection",
        "parenthesis",
        "command-string",
        "command-string-after-pipe",
        "eval-after-and-list",
        "runtime-command-in-case-item",
        "eval-after-shell-option",
    ],
)
def test_wrapper_with_an_unresolved_command_stays_partial(content: str) -> None:
    assert _exhausted(content, "shell")


def test_wrapper_at_the_end_of_a_fragment_stays_partial() -> None:
    assert _exhausted("x | sudo", "shell")
    assert _exhausted("x | timeout", "shell", complete_context=False)


def test_markdown_cell_with_a_real_command_keeps_its_finding() -> None:
    literal = "| Step | Command |\n|---|---|\n| wipe | `sudo rm -rf /` |\n"
    runtime = "| Step | Command |\n|---|---|\n| wipe | `sudo $CMD -rf /` |\n"

    findings = tm_module.analyze(literal, "SKILL.md", "markdown")

    assert any(finding.rule_id == "TM1" for finding in findings)
    assert _exhausted(runtime, "markdown")


def test_scan_level_parameter_table_is_fully_inspected() -> None:
    content = "# Tool\n\n| Name | Type |\n|---|---|\n| timeout | float |\n" + "\nText.\n" * 400

    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}, [tm_module]
    )

    event = result["inspection_ledger"][0]
    assert event["outcome"] is LedgerOutcome.COMPLETED


def test_scan_level_redirected_wrapper_stays_partial() -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": "sudo >log $CMD -rf /\n"}},
        [tm_module],
    )

    event = result["inspection_ledger"][0]
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
