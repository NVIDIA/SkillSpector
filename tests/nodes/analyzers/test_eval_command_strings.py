# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval operands retain their joined command semantics and parsing bounds."""

import shlex

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner


@pytest.mark.parametrize("setting", ["ENV=prod", "export BASH_ENV=setup.sh"])
@pytest.mark.parametrize("wrapper", ["eval", "builtin eval", "command -p eval"])
@pytest.mark.parametrize("body", ["echo ready", "eval 'echo ready'"])
def test_literal_eval_does_not_load_shell_startup_files(
    setting: str, wrapper: str, body: str
) -> None:
    content = f"{setting}; {wrapper} {shlex.quote(body)}"

    assert not tm_module.has_bounded_parse_exhaustion(content, lambda: None, file_type="shell")


@pytest.mark.parametrize(
    "content",
    [
        "BASH_ENV=setup.sh; eval 'bash -c \"echo ready\"'",
        "ENV=prod; builtin eval 'sh -c \"echo ready\"'",
        "BASH_ENV=setup.sh; eval \"eval 'bash -c true'\"",
        "eval 'export BASH_ENV=setup.sh'; bash -c 'echo ready'",
        "builtin eval 'ENV=prod'; eval 'sh -c \"echo ready\"'",
        "eval \"eval 'export BASH_ENV=setup.sh'\"; bash -c true",
        "ENV=prod; eval 'ENV=; bash -c true'",
        "BASH_ENV=setup.sh; eval 'echo \"$(bash -c true)\"'",
        "BASH_ENV=setup.sh; eval 'echo \"`bash -c true`\"'",
        "eval 'export BASH_ENV=setup.sh; echo \"$(bash -c true)\"'",
        "eval 'export BASH_ENV=setup.sh; echo \"`bash -c true`\"'",
        'ENV=prod; eval "$SCRIPT"',
    ],
)
def test_eval_preserves_startup_exposure_for_actual_child_shells(content: str) -> None:
    assert tm_module.has_bounded_parse_exhaustion(content, lambda: None, file_type="shell")


def test_eval_startup_settings_inside_child_do_not_escape_to_parent() -> None:
    content = "bash -c 'ENV=prod; eval \"echo ready\"'; bash -c 'echo ready'"

    assert not tm_module.has_bounded_parse_exhaustion(content, lambda: None, file_type="shell")


def test_literal_eval_with_startup_setting_keeps_destructive_finding() -> None:
    content = "ENV=prod; eval 'rm -rf /'"

    findings = tm_module.analyze(content, "script.sh", "shell")

    assert any(finding.rule_id == "TM1" for finding in findings)


@pytest.mark.parametrize(
    "content",
    [
        "eval '$CMD -rf /'",
        "eval '$CMD' '-rf' '/'",
        "eval '' '$CMD' '-rf' '/'",
        "eval -- '$CMD' '-rf' '/'",
        "eval '$CMD' 2>/dev/null '-rf' '/'",
        "eval '$CMD' '-rf' '/' >/dev/null",
        "eval '$CMD' '-rf' '/' &>/dev/null",
        "eval '$CMD' \\\n'-rf' '/'",
        "eval '$CMD' \\\r\n'-rf' '/'",
        "eval '$CMD' '-rf' '/' # explanatory comment",
        "eval 'echo' \"$SCRIPT\"",
        "eval 'echo' 'unterminated",
        "eval '$CMD' '-rf' '/' >",
        "eval 'echo' <<END\nhello\nEND",
        "eval 'echo' " + "'' " * 32,
        "eval 'echo' " + " " * 8192,
        "eval 'echo " + "x" * 8192 + "'",
    ],
)
def test_eval_reconstruction_keeps_uncertain_commands_partial(content: str) -> None:
    # All shell fragments are inert scanner input, never executed.
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["script.sh"], "file_cache": {"script.sh": content}}, [tm_module]
    )
    event = result["inspection_ledger"][0]
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "content",
    [
        "eval",
        "eval ''",
        "eval 'echo' 'hello'",
        "eval 'echo' " + "'' " * 31,
        "eval '$CMD' # -rf / is only a comment",
        "eval '$CMD' '-rf' >/dev/null",
        "eval '$CMD' '-rf'; echo '/'",
        "eval '$CMD' '-rf'\necho '/'",
        "eval '$CMD' '-rf' | echo '/'",
        "eval '$CMD' '-rf' && echo '/'",
        "eval '$CMD' '-rf' || echo '/'",
        "eval '$CMD' '-rf' & echo '/'",
        "printf '%s' \"eval '$CMD' '-rf' '/'\"",
    ],
)
def test_eval_operands_do_not_borrow_targets_from_other_clauses(content: str) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["script.sh"], "file_cache": {"script.sh": content}}, [tm_module]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_eval_operand_collection_observes_deadline() -> None:
    class DeadlineExceededError(Exception):
        pass

    calls = 0

    def check_runtime() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise DeadlineExceededError

    with pytest.raises(DeadlineExceededError):
        tm_module._eval_command_string("'echo' 'hello' 'world'", 0, check_runtime)
