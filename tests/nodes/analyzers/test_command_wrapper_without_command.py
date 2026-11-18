# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A command wrapper followed directly by a control operator runs no command (#694)."""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner

_PADDING = "\n# ordinary padding\n" * 400


@pytest.mark.parametrize(
    "source",
    [
        "env",
        "env -i",
        "env --",
        "env -u OTHER",
        "env -C /tmp",
        "env NAME=value",
        "command",
        "command -p",
        "command --",
        "builtin",
        "builtin --",
    ],
)
@pytest.mark.parametrize("ending", ["", "\n", "; echo ready\n", " | cat\n", " # explanation\n"])
def test_optional_command_wrappers_without_executable_are_complete(
    source: str, ending: str
) -> None:
    assert not _exhausted(source + ending, "shell")


@pytest.mark.parametrize("key", ["command", "builtin", "env"])
def test_json_key_naming_optional_command_wrapper_is_complete(key: str) -> None:
    source = '# Output\n\n```json\n{"' + key + '": "echo ready"}\n```\n'

    assert not _exhausted(source, "markdown")


@pytest.mark.parametrize(
    "source",
    [
        "env -u",
        "env -C",
        "env -S",
        "env --unknown",
        'env >log bash -c "$CMD"',
        'command >log bash -c "$CMD"',
        'builtin >log eval "$CMD"',
        'env -- >log bash -c "$CMD"',
        'command -- >log bash -c "$CMD"',
        'builtin -- >log eval "$CMD"',
    ],
)
def test_commandless_wrapper_refinement_keeps_unresolved_operands_partial(source: str) -> None:
    assert _exhausted(source, "shell")


def _exhausted(content: str, file_type: str, *, complete_context: bool = True) -> bool:
    return tm_module.has_bounded_parse_exhaustion(
        content, lambda: None, file_type=file_type, complete_context=complete_context
    )


@pytest.mark.parametrize("wrapper", ["timeout", "sudo", "nice", "xargs"])
def test_markdown_table_cell_naming_a_wrapper_is_complete(wrapper: str) -> None:
    content = f"| Name | Description |\n|---|---|\n| {wrapper} | Max wait in seconds |\n"

    assert not _exhausted(content, "markdown")
    assert not _exhausted(content + _PADDING, "markdown")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_text_diagram_arrow_is_not_a_shell_redirection(newline: str) -> None:
    diagram = "```text\n|          |\n|      |  timeout ---->|\n```\n".replace("\n", newline)

    assert not _exhausted(diagram, "markdown")
    assert not _exhausted(_PADDING + diagram + _PADDING, "markdown")
    assert _exhausted(diagram, "markdown", complete_context=False)
    assert not _exhausted(diagram + "```sh\nsudo rm -rf /\n```\n", "markdown")
    findings = tm_module.analyze(diagram + "```sh\nsudo rm -rf /\n```\n", "SKILL.md", "markdown")
    assert any(finding.rule_id == "TM1" for finding in findings)


@pytest.mark.parametrize(
    "source",
    [
        "```sh\n|      |  timeout ---->|\n```\n",
        "```text\n|      |  timeout ---->|\n",
        '```text\n|      |  timeout ---->|"$DEST"\n```\n',
        '```text\n|      |  timeout ---->|\nsudo >log "$CMD"\n```\n',
    ],
)
def test_diagram_projection_preserves_unproven_command_boundaries(source: str) -> None:
    assert _exhausted(source, "markdown")


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


@pytest.mark.parametrize("wrapper", ["timeout", "sudo", "nice", "xargs"])
@pytest.mark.parametrize("quote", ["'", '"', '"""'])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_python_parenthesized_wrapper_name_is_complete(
    wrapper: str, quote: str, newline: str
) -> None:
    content = (
        '"""Describe a fixture beside its manifest."""\n'
        f"WRAPPER_NAME = (\n    {quote}{wrapper}{quote}\n)\n"
        f"def read_option(options):\n    return options.get({quote}{wrapper}{quote})\n"
    ).replace("\n", newline)

    assert not _exhausted(content, "python")
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["scripts/validate.py"], "file_cache": {"scripts/validate.py": content}},
        [tm_module],
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "source",
    [
        'sudo -n fuser "$device" >/dev/null 2>&1 && continue\n',
        "sudo -n true >/dev/null 2>&1\n",
        "sudo -n true 2>&1 >/dev/null\n",
        "sudo -n true 0<&3 3>&-\n",
        "sudo -n true 3>&1-\n",
        "sudo -n true &>/dev/null\n",
        "sudo -n true &>>/dev/null\n",
        "sudo -n -- true 2>&1\n",
        "sudo -n -u root true 2>&1\n",
        "sudo -u root -n true 2>&1\n",
        "sudo --non-interactive true 2>&1\n",
        "sudo -nS true 2>&1\n",
        "nice -n 5 true 2>&1\n",
        "xargs -n 1 true 2>&1\n",
        "sudo -n sh 2>&1 -c 'echo ready'\n",
        "eval 2>&1 'echo ready'\n",
    ],
)
def test_noninteractive_sudo_and_descriptor_redirections_are_complete(source: str) -> None:
    assert not _exhausted(source, "shell")


@pytest.mark.parametrize(
    "source",
    [
        'sudo -n sh -c "$CMD"\n',
        'sudo -n sh 2>&1 -c "$CMD"\n',
        'sudo -n sh >log -c "$CMD"\n',
        'sudo -n -- sh -c "$CMD"\n',
        'sudo -n -u root sh -c "$CMD"\n',
        'nice -n 5 sh -c "$CMD"\n',
        'xargs -n 1 sh -c "$CMD"\n',
        "sudo -n sh -c '$CMD -rf /'\n",
        'eval 2>&1 "$CMD"\n',
        "sudo -n $CMD -rf /\n",
        "sudo -n >log $CMD -rf /\n",
        "sudo -n &>log $CMD -rf /\n",
        "sudo -n true 2>&\n",
        "sudo -n true 0<&\n",
        "sudo -n true 2>\n",
        "sudo -n true >\n",
        "sudo -n",
        "nice -n",
        "xargs -n",
        "timeout",
        'sudo -n sh -c "unterminated',
    ],
)
def test_wrapper_repairs_preserve_unresolved_command_limits(source: str) -> None:
    assert _exhausted(source, "shell")


def test_noninteractive_sudo_keeps_destructive_finding() -> None:
    source = "sudo -n rm -rf / 2>&1\n"

    findings = tm_module.analyze(source, "scripts/run.sh", "shell")

    assert any(finding.rule_id == "TM1" for finding in findings)


@pytest.mark.parametrize("wrapper", ["sudo -n", "nice -n 1", "xargs -n 1"])
def test_wrapper_chain_budget_stays_partial(wrapper: str) -> None:
    source = " ".join([wrapper] * 40) + " sh -c 'echo ready'\n"

    assert _exhausted(source, "shell")


@pytest.mark.parametrize(
    "content",
    [
        "- Cause -> resolution: Find the problem (timeout -> see 42)\n",
        '| Name | Meaning |\n|---|---|\n| wait | Timeout (e.g. `"2m"`, `"3h"`) |\n',
        'The command uses `ssh host "batch -p\n<queue> -A <account> file"` here.\n',
    ],
)
def test_markdown_documentation_punctuation_keeps_later_executable_content(content: str) -> None:
    dangerous = "\n```sh\nsudo rm -rf /\n```\n"
    unresolved = '\n```sh\nsudo >log "$command"\n```\n'

    assert not _exhausted(content, "markdown")
    assert _exhausted(content + unresolved, "markdown")
    assert any(
        f.rule_id == "TM1" for f in tm_module.analyze(content + dangerous, "SKILL.md", "markdown")
    )


def test_only_plain_declared_list_table_roles_own_rst_punctuation() -> None:
    document = ".. role:: note\n\n.. list-table:: Names\n\n   * - :note:`The child's name.`\n"
    projected = tm_module._rst_table_role_text(document, lambda: None)

    assert projected == document.replace("`", " ").replace("'", " ")
    assert _exhausted(document + '\nsudo >log "$command"\n', "rst")
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["references/guide.rst"], "file_cache": {"references/guide.rst": document}},
        [tm_module],
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "body",
    [
        ".. role:: note(code)\n\n.. list-table:: Names\n\n   * - :note:`The child's name.`\n",
        ".. role:: note\n\n.. list-table:: Names\n\n   * - :note:`$(shell)`\n",
        ".. role:: note\n\n.. code-block:: sh\n\n   * - :note:`The child's name.`\n",
        ".. role:: note\n\n.. list-table:: Names\n\n   * - .. code:: bash\n        - :note:`Code`\n",
        ".. role:: note\n\n.. include:: roles.rst\n\n.. list-table:: Names\n\n   * - :note:`Code`\n",
    ],
)
def test_rst_unproved_role_and_literal_context_remains_visible(body: str) -> None:
    assert tm_module._rst_table_role_text(body, lambda: None) == body


@pytest.mark.parametrize(
    "content",
    [
        "| Command (worked example) | Owner |\n|---|---|\n| `ls` | Reader |\n",
        'The summary reports "two of three\ncomplete" for this record.\n',
        "Use inline `**bold**`, `*italic*`, `` `code` ``, or a link.\n",
        "`` `example` `` → `<code>example</code>`.\n",
        "Make expands its own `$(...)` expressions.\n",
        "The runner rejects shell arithmetic: use a counter instead of `$((n+1))`.\n",
        "The notation changes (csh: no `$()`, `&&`).\n",
    ],
)
def test_documented_examples_keep_adjacent_shell_checks(content: str) -> None:
    assert not _exhausted(content, "markdown")
    assert _exhausted(content + '\n```sh\nbash -c "$payload"\n```\n', "markdown")
    assert any(
        f.rule_id == "TM1"
        for f in tm_module.analyze(content + "\n`sudo rm -rf /`\n", "SKILL.md", "markdown")
    )


@pytest.mark.parametrize(
    "content",
    [
        "Execute `` `name` ``.\n",
        "Run `$((n+1))`.\n",
        "Run `$($tool -rf /)`.\n",
        'Make expands its own `$(eval "$value")` expressions.\n',
    ],
)
def test_unproved_shell_contexts_stay_partial(content: str) -> None:
    assert _exhausted(content, "markdown")


def test_run_instruction_does_not_mark_an_empty_substitution_as_prose() -> None:
    content = "Run `$()`.\n"
    start = content.index("$()")
    assert not tm_module._documented_expansion_placeholder(content, start, start + len("$()"))


def test_typeset_notice_requires_closed_prose_and_preserves_shell_consumers() -> None:
    header = "# Third-party license notices\n\n"
    start = "````text\n"
    end = "````\n"
    words = "Use the terms `Sample Project' and ``AS IS'' in descriptions.\n"
    payload = 'sh -c "echo `Example Term\'"\n'
    source = header + start + words + payload + end
    expected = source.replace("`Sample Project'", " Sample Project ").replace(
        "``AS IS''", "  AS IS  "
    )

    assert tm_module._typeset_notice_text(source, lambda: None) == expected
    for variant in [start + words + end, header + start + words, header + "````sh\n" + words + end]:
        assert tm_module._typeset_notice_text(variant, lambda: None) == variant


def test_shell_syntax_never_becomes_a_plain_wrapped_quotation() -> None:
    content = 'The summary reports "one of three\nsudo $command".\n'
    assert tm_module._markdown_shell_text(content, lambda: None) == content
