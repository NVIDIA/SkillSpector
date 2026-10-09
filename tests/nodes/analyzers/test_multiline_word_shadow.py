# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A word that spans lines never hides a later command line (#694)."""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner

_PADDING = "\n# ordinary padding\n" * 400
_TAIL = '\nE\n$CMD"" -rf / # "\n'


def _exhausted(content: str, file_type: str = "shell") -> bool:
    return tm_module.has_bounded_parse_exhaustion(
        content, lambda: None, file_type=file_type, complete_context=True
    )


@pytest.mark.parametrize(
    ("content", "file_type"),
    [
        ('# x "\n$CMD"" -rf / # "\n', "shell"),
        ('cat <<E\nhe said "hi' + _TAIL, "shell"),
        ("cat <<'E'\nhe said \"hi" + _TAIL, "shell"),
        ('cat <<E\n"${X}' + _TAIL, "shell"),
        ('cat <<E\n"${X}" "' + _TAIL, "shell"),
        ('cat <<E\n"$(date)$()"e"' + _TAIL, "shell"),
        ('cat <<E\n"`date`$()""' + _TAIL, "shell"),
        ('cat <<E\r"hi\rE\r$CMD"" -rf / # "\r', "shell"),
        ('Run "a\n```sh\n$CMD"" -rf / # "\n```\n', "markdown"),
    ],
    ids=[
        "comment",
        "heredoc-prose-quote",
        "quoted-delimiter",
        "heredoc-parameter",
        "heredoc-parameter-space",
        "heredoc-letter-after-quote",
        "heredoc-backtick",
        "heredoc-carriage-return",
        "markdown-prose",
    ],
)
def test_quote_spanning_lines_cannot_hide_a_later_command(content: str, file_type: str) -> None:
    # A quote in a heredoc body, a comment or prose may pair with a quote on
    # a later command line. The word between them must not hide that line.
    assert _exhausted(content, file_type)
    assert _exhausted(content + _PADDING, file_type)


def test_heredoc_quote_cannot_hide_a_runtime_rm_at_scan_level() -> None:
    script = (
        '#!/bin/sh\nCMD="${TOOL:-rm}"\ncat <<EOF\nhe said "hi\nEOF\n'
        '$CMD"" -rf / --no-preserve-root # done "\n'
    )

    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["run.sh"], "file_cache": {"run.sh": script}}, [tm_module]
    )

    event = result["inspection_ledger"][0]
    assert event["outcome"] is not LedgerOutcome.COMPLETED


def test_multiline_string_with_parameter_stays_complete() -> None:
    content = 'echo "line1\nline2 $HOME"\nls -la\n'
    assert not _exhausted(content)
    assert not _exhausted(content + _PADDING)


def test_dynamic_multiline_word_matches_main() -> None:
    # Main never advanced past a dynamic ``$(`` word, so its inner commands
    # were already independent candidates.
    content = 'echo "$(date)\nnext"\nls\n'
    assert not _exhausted(content)
    assert not _exhausted(content + _PADDING)


def test_python_source_keeps_main_frontier() -> None:
    content = 'x = """he said "hi\nmore"""\nprint(x)\n'
    assert not _exhausted(content, "python")
    assert not _exhausted(content + _PADDING, "python")
