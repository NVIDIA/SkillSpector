# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A bounded, literal ``printf`` substitution must not look partially inspected.

Issue #694: the ELF-probe idiom used by ``rustup-init.sh`` made
``static_patterns_tool_misuse`` report ``static_parse_limit``, so the file could
never reach ``safe_to_install``.  The substitution is bounded and its output is
fully determined, so the deterministic evaluator has to resolve it instead of
falling back to the span-bound failure path.
"""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner

# Verbatim POSIX probe from the issue report (rustup-init.sh line 246).
_RUSTUP_ELF_PROBE = (
    "#!/bin/sh\n"
    '_current_exe_head=$(head -c 5 "$1")\n'
    'if [ "$_current_exe_head" = "$(printf \'\\177ELF\\001\')" ]; then\n'
    "    echo elf\n"
    "fi\n"
)

# Verbatim shape of the machine-byte probe from rustup-init.sh line 266.
_RUSTUP_MACHINE_PROBE = (
    '#!/bin/sh\n_m=$(head -c 19 "$1" | tail -c 1)\n[ "$_m" = "$(printf \'\\076\')" ]\n'
)

_LITERAL_SUBSTITUTIONS = [
    pytest.param("$(printf '\\177ELF\\001')", "\x7fELF\x01", id="elf-probe-octal"),
    pytest.param("$(printf '\\074')", "<", id="octal-less-than"),
    pytest.param("$(printf '\\076')", ">", id="octal-greater-than"),
    pytest.param("$(printf '\\x41\\x42')", "AB", id="hex-escapes"),
    pytest.param("$(printf '\\162\\155')", "rm", id="octal-letters"),
    pytest.param("$(printf 'abc')", "abc", id="plain-literal"),
    pytest.param("$(printf '%%s')", "%s", id="escaped-percent"),
]


@pytest.mark.parametrize("substitution,expected", _LITERAL_SUBSTITUTIONS)
def test_bounded_printf_is_resolved_statically(substitution: str, expected: str) -> None:
    assert tm._static_printf_substitution(substitution, 0, len(substitution)) == expected


@pytest.mark.parametrize("substitution,_expected", _LITERAL_SUBSTITUTIONS)
def test_bounded_printf_substitution_completes_static_analysis(
    substitution: str, _expected: str
) -> None:
    source = f'echo "{substitution}"\n'
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is False


def test_issue_reproducer_completes_static_analysis() -> None:
    assert (
        tm.has_bounded_parse_exhaustion(_RUSTUP_ELF_PROBE, lambda: None, file_type="shell") is False
    )


@pytest.mark.parametrize("escape", ["076", "077", "001"])
def test_rustup_machine_probe_completes_static_analysis(escape: str) -> None:
    source = _RUSTUP_MACHINE_PROBE.replace("\\076", f"\\{escape}", 1)
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is False


def test_issue_reproducer_reaches_completed_ledger_outcome() -> None:
    path = "install.sh"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: _RUSTUP_ELF_PROBE}}, [tm]
    )
    row = result["inspection_ledger"][0]
    assert row["path"] == path
    assert row["analyzer_id"] == "static_patterns_tool_misuse"
    assert row["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in row


def test_rustup_machine_probe_reaches_completed_ledger_outcome() -> None:
    path = "install.sh"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: _RUSTUP_MACHINE_PROBE}}, [tm]
    )
    row = result["inspection_ledger"][0]
    assert row["path"] == path
    assert row["analyzer_id"] == "static_patterns_tool_misuse"
    assert row["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in row


@pytest.mark.parametrize(
    "substitution",
    [
        pytest.param("$(printf 'rm -rf /')", id="literal-destructive"),
        pytest.param("$(printf '%s%s' 'rm -rf' ' /')", id="assembled-destructive"),
        pytest.param("$(printf '\\162\\155 -rf /')", id="octal-prefixed-destructive"),
        pytest.param("$(printf '\\040rm')", id="escape-emits-separator"),
        pytest.param("$(printf '\\047rm')", id="escape-emits-quote"),
        pytest.param("$(printf '\\400')", id="octal-overflow-nul"),
        pytest.param("$(printf '\\440')", id="octal-overflow-space"),
        pytest.param("$(printf '\\447')", id="octal-truncates-to-quote"),
        pytest.param("$(printf '\\504')", id="octal-truncates-to-dollar"),
        pytest.param("$(printf '\\133\\135')", id="decoded-brackets"),
        pytest.param("$(printf '\\qrm')", id="unknown-escape"),
    ],
)
def test_destructive_or_undecidable_printf_stays_partial(substitution: str) -> None:
    source = f'echo "{substitution}"\n'
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is True


@pytest.mark.parametrize(
    "substitution",
    [
        "$(printf '\\400')",
        "$(printf '\\440')",
        "$(printf '\\447')",
        "$(printf '\\504')",
    ],
)
def test_out_of_range_octal_is_not_resolved(substitution: str) -> None:
    """Octals above one byte stay fail-closed instead of producing a code point."""
    assert tm._static_printf_substitution(substitution, 0, len(substitution)) is None


def test_decoded_brackets_are_not_treated_as_inert() -> None:
    """Unmodeled glob metacharacters must keep the substitution undecidable."""
    substitution = "$(printf '\\133\\135')"
    assert tm._static_printf_substitution(substitution, 0, len(substitution)) is None


def test_destructive_printf_reaches_partial_ledger_outcome() -> None:
    path = "install.sh"
    source = "echo \"$(printf 'rm -rf /')\"\n"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: source}}, [tm]
    )
    row = result["inspection_ledger"][0]
    assert row["outcome"] is LedgerOutcome.PARTIAL
    assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("$CMD -rf /\n", id="runtime-selected-command"),
        pytest.param("$CMD -rf {/,{1..2}}\n", id="unsupported-brace-expansion"),
        pytest.param('name="value\n' + "text " * 1200, id="unclosed-assignment-quote"),
    ],
)
def test_span_bound_failures_are_unchanged(source: str) -> None:
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is True
