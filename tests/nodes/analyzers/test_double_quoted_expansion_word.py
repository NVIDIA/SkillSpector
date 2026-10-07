# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An expansion inside a closed double-quoted span ends at that span's quote (#694)."""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner

_PADDING = "\n# ordinary padding\n" * 400


def _exhausted(content: str, file_type: str = "shell") -> bool:
    return tm_module.has_bounded_parse_exhaustion(
        content, lambda: None, file_type=file_type, complete_context=True
    )


@pytest.mark.parametrize(
    ("content", "file_type"),
    [
        ('echo "x: $(date)"\n', "shell"),
        ('echo "a $(date) b $(id -u)"\n', "shell"),
        ('echo "x: ${A:-$(date)}"\n', "shell"),
        ('echo "x: $(echo "a b")"\n', "shell"),
        ('Write-Warning "Stale path: $($cfg.command)"\n', "powershell"),
        ('Write-Host "Home: $($env:HOME)"\n', "powershell"),
        ('echo "Path: $($HOME)/bin"\n', "shell"),
    ],
    ids=[
        "command-substitution",
        "two-substitutions",
        "parameter-default",
        "nested-quotes",
        "powershell-member",
        "powershell-environment",
        "path-suffix",
    ],
)
def test_expansion_inside_a_closed_string_is_complete(content: str, file_type: str) -> None:
    assert not _exhausted(content + _PADDING, file_type)


@pytest.mark.parametrize(
    "content",
    [
        'echo "x: `date`"\n',
        'echo "x: ${HOME}"\n',
        'echo "$(date)"\n',
        'echo "x: $(date) y"\n',
    ],
    ids=["backtick", "parameter", "adjacent-quote", "trailing-word"],
)
def test_forms_that_were_already_complete_stay_complete(content: str) -> None:
    assert not _exhausted(content + _PADDING)


@pytest.mark.parametrize(
    "content",
    [
        'echo "x: $($TOOL -rf /)"\n',
        'echo "x: $(echo "a b"; $TOOL -rf /)"\n',
        'echo "x: `$TOOL -rf /`"\n',
        'echo "x: $(printf %s r m) -rf /"\n',
        'x; "$CMD $(date)" -rf /\n',
        'sh -c "echo $(date); $CMD -rf /"\n',
        'echo "x: $(date)\n',
        'echo "x: $(date) y\n',
        'echo "Path: $($HOME)\\bin"\n',
    ],
    ids=[
        "runtime-command",
        "nested-runtime-command",
        "backtick-runtime-command",
        "printf-reconstruction",
        "quoted-runtime-command-word",
        "shell-command-string",
        "unclosed-string",
        "unclosed-string-with-word",
        "runtime-output-joined-to-an-escape",
    ],
)
def test_unresolved_commands_stay_partial(content: str) -> None:
    assert _exhausted(content + _PADDING)


@pytest.mark.parametrize(
    "content",
    [
        'echo "x: $(date)"\nrm -rf /tmp/build\n',
        'echo "a $(x)" ; rm -rf ./build\n',
        'echo "a $(x)"\nrm -rf "$HOME"/cache\n',
    ],
    ids=["later-cleanup", "same-line-cleanup", "quoted-cleanup-path"],
)
def test_short_script_with_a_later_cleanup_stays_complete(content: str) -> None:
    assert not _exhausted(content)
    assert not _exhausted(content + _PADDING)


@pytest.mark.parametrize(
    "content",
    [
        '# don\'t "$(x)"\n$CMD"" -rf /\n',
        'echo \'a "$(x)\' "$(printf r)"$X -rf /\n',
        'echo "a $(x)"; $(printf r)"$X" -rf /\n',
    ],
    ids=["comment-quote", "single-quoted-quote", "runtime-command-after-string"],
)
def test_runtime_command_after_a_mispaired_or_closed_quote_stays_partial(content: str) -> None:
    assert _exhausted(content)


def test_destructive_command_in_a_substitution_keeps_its_finding() -> None:
    content = 'echo "x: $(rm -rf /)"\n' + _PADDING

    findings = tm_module.analyze(content, "run.sh", "shell")

    assert any(finding.rule_id == "TM1" for finding in findings)


def test_scan_level_helper_script_is_fully_inspected() -> None:
    script = '#!/bin/sh\necho "Started: $(date)"\n' + _PADDING

    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["run.sh"], "file_cache": {"run.sh": script}}, [tm_module]
    )

    event = result["inspection_ledger"][0]
    assert event["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "content",
    [
        '# "prefix $(date)\n$CMD"" -rf /\n',
        '# "prefix $(date)\n$CMD" x"" -rf /\n',
        '#!/bin/sh\n# see "notes $(date)\n$CMD"" -rf /\n',
    ],
    ids=["failed-comment-parse", "completed-comment-parse", "after-shebang"],
)
def test_quote_from_a_comment_parse_grants_no_ownership(content: str) -> None:
    # A comment ends at the newline, so the real command on the next line owns
    # its quotes. A discarded or comment-confined parse must not mark that
    # command as enclosed by a quote and thereby hide ``-rf /``.
    assert _exhausted(content)
    assert _exhausted(content + _PADDING)


def test_reviewer_script_is_not_fully_inspected() -> None:
    script = '# "prefix $(date)\n$CMD"" -rf /\n'

    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["run.sh"], "file_cache": {"run.sh": script}}, [tm_module]
    )

    event = result["inspection_ledger"][0]
    assert event["outcome"] is not LedgerOutcome.COMPLETED
