# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A quoted ``$(printf ... | ...)`` argument must not look partially inspected.

Issue #1107: a POSIX shell ``printf`` whose argument is a double-quoted
command substitution piping ``printf '%s' "$1"`` into another command made
``static_patterns_tool_misuse`` record ``static_parse_limit``
(``printf_format`` / ``unsupported_syntax``), so the file counted as
partially inspected. The inner substitution's output comes from the
downstream command, not from ``printf``, so it is runtime-dependent data in
an argument position of the outer ``printf``, exactly as for the ``echo``
and assignment variants. The quoted-result branch of
``_static_printf_substitution`` failed whenever a quoted substitution
started with ``printf`` and its arguments were not literal, even when the
substitution was a pipeline whose output another command produces.
"""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner

# Verbatim "quote a value for sh/cron" idiom from the issue report.
_CRONQ = (
    "#!/bin/sh\n"
    "cronq() { printf \"'%s'\" \"$(printf '%s' \"$1\" | sed -e \"s/'/'\\\\''/g\" -e 's/%/\\\\%/g')\"; }\n"
)

# The five line variants from the issue table: the two printf-inner cases
# must complete, and the three controls must stay complete.
_ISSUE_VARIANTS = [
    pytest.param(
        """printf "'%s'" "$(printf '%s' "$1" | sed 's/a/b/')" """,
        id="printf-inner-single-quoted-format",
    ),
    pytest.param(
        """printf '%s\\n' "$(printf '%s' "$1" | sed 's/a/b/')" """,
        id="printf-inner-newline-format",
    ),
    pytest.param(
        """printf "'%s'" "$(echo "$1" | sed 's/a/b/')" """,
        id="echo-inner-control",
    ),
    pytest.param(
        """x="$(printf '%s' "$1" | sed 's/a/b/')" """,
        id="assignment-control",
    ),
    pytest.param(
        """printf "'%s'" "$1" """,
        id="plain-parameter-control",
    ),
]


@pytest.mark.parametrize("line", _ISSUE_VARIANTS)
def test_issue_variants_complete_static_analysis(line: str) -> None:
    source = "#!/bin/sh\n" + line.strip() + "\n"
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is False


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(
            """printf "'%s'" "$(printf '%s' "$1" | sed 's/a/b/')" """,
            id="printf-inner-single-quoted-format",
        ),
        pytest.param(
            """printf '%s\\n' "$(printf '%s' "$1" | sed 's/a/b/')" """,
            id="printf-inner-newline-format",
        ),
    ],
)
def test_issue_reproducer_reaches_completed_ledger_outcome(line: str) -> None:
    path = "scripts/cron.sh"
    source = "#!/bin/sh\n" + line.strip() + "\n"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: source}}, [tm]
    )
    row = result["inspection_ledger"][0]
    assert row["path"] == path
    assert row["analyzer_id"] == "static_patterns_tool_misuse"
    assert row["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in row


def test_cronq_idiom_reaches_completed_ledger_outcome() -> None:
    path = "scripts/cron.sh"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: _CRONQ}}, [tm]
    )
    row = result["inspection_ledger"][0]
    assert row["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in row


def test_literal_elf_probe_behavior_is_unchanged() -> None:
    """Issue #694: a literal ``$(printf '\\177ELF\\001')`` still resolves."""
    probe = "$(printf '\\177ELF\\001')"
    assert tm._static_printf_substitution(probe, 0, len(probe)) == "\x7fELF\x01"
    source = (
        "#!/bin/sh\n"
        'if [ "$_current_exe_head" = "$(printf \'\\177ELF\\001\')" ]; then\n'
        "    echo elf\n"
        "fi\n"
    )
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is False


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(
            """printf "'%s'" "$(printf '%s' "$1")" """,
            id="bare-printf-with-runtime-argument",
        ),
        pytest.param(
            """printf "'%s'" "$(printf '%s' "$1" 'a|b')" """,
            id="quoted-pipe-inside-printf-argument",
        ),
    ],
)
def test_non_pipeline_printf_stays_partial(line: str) -> None:
    """Scope guard: only a real operator after the printf words exempts it.

    A bare ``printf`` with a runtime argument is still fail-closed, and a
    pipe character quoted inside a printf argument is not a pipeline.
    """
    source = "#!/bin/sh\n" + line.strip() + "\n"
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is True
