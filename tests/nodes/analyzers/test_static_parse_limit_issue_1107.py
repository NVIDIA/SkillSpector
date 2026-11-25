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


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(
            """$(printf '%s%s' r 'm -rf /' | cat) """,
            id="executable-printf-pipeline",
        ),
        pytest.param(
            """$(command printf '%s' x | cat) """,
            id="executable-wrapper-printf-pipeline",
        ),
    ],
)
def test_executable_printf_pipeline_keeps_coverage_warning(line: str) -> None:
    """Review #1109: the exemption is for proven quoted data arguments only.

    An unquoted printf pipeline in command position supplies the command
    and its arguments through word splitting, so the reconstruction
    diagnostic must stay: the outer-operand check has nothing left to
    inspect once the substitution is treated as resolved data.
    """
    source = "#!/bin/sh\n" + line.strip() + "\n"
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is True


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(
            """\"$(printf '%s' \"$tool_name\" | cat)\" """,
            id="quoted-command-printf-pipeline",
        ),
        pytest.param(
            """\"$(printf '%s%s' r 'm -rf /' | cat)\" """,
            id="quoted-command-printf-pipeline-rm",
        ),
        pytest.param(
            """\"$(command printf '%s' \"$1\" | sed 's/a/b/')\" """,
            id="quoted-command-wrapper-printf-pipeline",
        ),
        pytest.param(
            """echo hi; \"$(printf '%s' \"$tool_name\" | cat)\" """,
            id="quoted-command-after-semicolon",
        ),
        pytest.param(
            """echo hi | \"$(printf '%s' \"$tool_name\" | cat)\" """,
            id="quoted-command-after-pipe",
        ),
        pytest.param(
            """FOO=bar \"$(printf '%s' \"$tool_name\" | cat)\" """,
            id="quoted-command-after-assignment-prefix",
        ),
        pytest.param(
            """FOO=bar BAZ=qux \"$(printf '%s' \"$tool_name\" | cat)\" """,
            id="quoted-command-after-two-assignment-prefixes",
        ),
        pytest.param(
            """FOO=bar \"$(printf '%s' \"$1\" | sed 's/a/b/')\" """,
            id="quoted-command-after-assignment-bounded-pipeline",
        ),
        pytest.param(
            """FOO=bar \"$(command printf '%s' \"$1\" | sed 's/a/b/')\" """,
            id="quoted-command-wrapper-after-assignment-prefix",
        ),
    ],
)
def test_quoted_command_printf_pipeline_keeps_coverage_warning(line: str) -> None:
    """Review #1109 (rng1995): a quoted substitution can name the command.

    A shell executes a double-quoted substitution in command position as
    the executable, including after assignment prefixes, so quoting alone
    never proves a data argument: the reconstruction diagnostic must
    stay, as for the unquoted case. The outer-printf fixture still
    completes (see the #1107 tests above).
    """
    source = "#!/bin/sh\n" + line.strip() + "\n"
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is True


def test_quoted_wrapper_printf_pipeline_still_completes() -> None:
    """Review #1109: the quoted wrapper case keeps the #1107 exemption."""
    line = """printf "'%s'" "$(command printf '%s' "$1" | sed 's/a/b/')" """
    source = "#!/bin/sh\n" + line.strip() + "\n"
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is False


def test_assignment_prefix_before_printf_still_completes() -> None:
    """Review #1109: an assignment prefix before printf keeps the exemption.

    The prefix sets the printf environment; the substitution remains a
    value argument of a bounded printf consumer, unlike a quoted word
    that itself follows the prefix as the command name.
    """
    line = """FOO=bar printf "'%s'" "$(printf '%s' "$1" | sed 's/a/b/')" """
    source = "#!/bin/sh\n" + line.strip() + "\n"
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is False


def test_operator_scan_honors_runtime_check() -> None:
    """Review #1109: the operator scan must not bypass cancellation.

    A long quoted argument has no _SHELL_COMMAND_WORD_CHARS cap, so the
    added full pass over the substitution must invoke the caller's runtime
    check (cancellation and the analyzer deadline) instead of lambda: None.
    """
    long_argument = "'" + "a" * 9000 + "'"
    inner = "printf '%s' " + long_argument + " | cat"

    def cancelled() -> None:
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        tm._printf_invocation_has_following_operator(inner, check_runtime=cancelled)


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
