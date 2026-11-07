# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct ``shell=True`` detection when earlier arguments contain parentheses."""

from __future__ import annotations

import time

import pytest

from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module


def _tm1(content: str, path: str = "run.py") -> list:
    result = tm_module.node({"components": [path], "file_cache": {path: content}})
    return [finding for finding in result["findings"] if finding.rule_id == "TM1"]


@pytest.mark.parametrize(
    "call",
    [
        pytest.param("subprocess.run(' '.join(sys.argv[1:]), shell=True)", id="join_argv"),
        pytest.param("subprocess.check_output('git log {}'.format(ref), shell=True)", id="format"),
        pytest.param("subprocess.Popen(cmd.strip(), shell=True)", id="qualified_popen"),
        pytest.param("Popen(' '.join(shlex.quote(a) for a in args), shell=True)", id="bare_popen"),
        pytest.param("subprocess.run('echo $(whoami)', shell=True)", id="command_substitution"),
        pytest.param(
            "subprocess.run(\n    'tar {}'.format(shlex.quote(path)),\n    shell=True,\n)",
            id="multiline",
        ),
        pytest.param('subprocess.run("echo :(", shell=True)', id="unbalanced_open_in_string"),
        pytest.param(
            "subprocess.run(\"grep -E '(foo|bar' log\", shell=True)",
            id="unbalanced_regex_in_string",
        ),
        pytest.param("subprocess.run('echo )', shell=True)", id="close_paren_in_string"),
        pytest.param(
            "subprocess.run(cmd,  # don't quote\n    shell=True)", id="apostrophe_in_comment"
        ),
        pytest.param("subprocess.run(cmd,  # (\n    shell=True)", id="open_paren_in_comment"),
        pytest.param(
            'subprocess.run(\n    """\n    echo :(\n    """,\n    shell=True,\n)',
            id="open_paren_in_triple_quoted_string",
        ),
        pytest.param(
            'subprocess.run("""echo \\\n:(""", shell=True)',
            id="open_paren_after_line_continuation_in_triple_quoted_string",
        ),
        pytest.param(
            'subprocess.run("echo \\\n:(", shell=True)',
            id="open_paren_after_line_continuation_in_string",
        ),
        pytest.param(
            'subprocess.run("echo \\\r\n:(", shell=True)',
            id="open_paren_after_crlf_line_continuation_in_string",
        ),
        pytest.param(
            'subprocess.run("""echo \\\r\n:(""", shell=True)',
            id="open_paren_after_crlf_line_continuation_in_triple_quoted_string",
        ),
        pytest.param(
            'subprocess.run(f"{"("}", shell=True)',
            id="open_paren_in_same_quote_f_string_replacement_field",
        ),
        pytest.param(
            "subprocess.run(f\"{'('}\", shell=True)",
            id="open_paren_in_different_quote_f_string_replacement_field",
        ),
    ],
)
def test_shell_true_after_nested_call_argument_is_detected(call: str) -> None:
    findings = _tm1(f"import subprocess, sys\nfrom subprocess import Popen\n{call}\n")

    assert len(findings) == 1
    assert findings[0].severity == "HIGH"
    assert findings[0].start_line == 3


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("subprocess.run(cmd)\nconfigure(shell=True)", id="keyword_after_call"),
        pytest.param("subprocess.run(' '.join(argv), shell=False)", id="shell_false"),
    ],
)
def test_shell_true_outside_the_subprocess_call_is_not_attributed(source: str) -> None:
    assert _tm1(f"import subprocess\n{source}\n") == []


def test_shell_true_after_cr_terminated_comment_in_markdown_is_detected() -> None:
    content = (
        "# Skill\n\n```python\nimport subprocess\nsubprocess.run(cmd, # note\rshell=True)\n```\n"
    )

    findings = _tm1(content, "SKILL.md")

    assert len(findings) == 1
    assert findings[0].severity == "HIGH"


@pytest.mark.parametrize("quote", ['"""', "'''"], ids=["double", "single"])
def test_unterminated_triple_quote_with_crlf_continuations_is_bounded(quote: str) -> None:
    content = "# Skill\n\n```python\nsubprocess.run(" + quote + "echo " + "\\\r\n" * 24 + "\n```\n"

    started_at = time.perf_counter()
    _tm1(content, "SKILL.md")

    assert time.perf_counter() - started_at < 2.0
