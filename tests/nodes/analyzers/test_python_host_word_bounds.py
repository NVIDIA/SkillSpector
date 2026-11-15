# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed shell word cannot borrow unrelated Python string delimiters."""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason, ParserCause
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner

# Alternating Python delimiters make a raw shell word charge later token bodies
# to the first value. Each native value remains short and independent.
_TAIL = "".join(f'FIELD_{index} = "ordinary_value_{index}"\n' for index in range(600))


def _diagnostic(source: str):
    return tm.bounded_parse_diagnostic(source, lambda: None, file_type="python")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "statement",
    [
        'def read(access: str = "read"):\n    return access\n',
        'RECORD = {"status": "diagnostic_only"}\n',
        'STATES = {"resource_limited", "completed"}\n',
        'PREFIXES = ("run_recomputed_reference_", "run_default_")\n',
        'MESSAGE = "café 🧪 literal"\n',
    ],
)
def test_short_python_values_do_not_consume_later_token_bodies(
    statement: str, newline: str
) -> None:
    source = (statement + _TAIL).replace("\n", newline)

    assert _diagnostic(source) is None


@pytest.mark.parametrize(
    "extension", [" + unknown", " % unknown", ".format(unknown)", " * count", "[part:]"]
)
def test_unproved_value_extensions_keep_the_original_failure(extension: str) -> None:
    source = 'def read(access: str = "read"' + extension + "):\n    return access\n" + _TAIL
    owner = tm._PythonSourceOwnership(source, lambda: None)
    start = source.index('read"')
    assert owner.string_value(start, start + len("read")) is None

    assert _diagnostic(source) is not None


def test_later_runtime_command_remains_independently_partial() -> None:
    source = 'FIRST = "ordinary"\n' + _TAIL + 'COMMAND = "$TOOL -rf /"\n'

    diagnostic = _diagnostic(source)
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RUNTIME_DEPENDENT


@pytest.mark.parametrize("prefix", ["", 'FIRST = "ordinary"\n' + _TAIL], ids=["first", "later"])
def test_a_genuinely_over_budget_python_value_keeps_the_limit(prefix: str) -> None:
    source = prefix + 'VALUE = "' + "r" * 4097 + '"\n' + _TAIL

    diagnostic = _diagnostic(source)
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RESOURCE_LIMIT
    assert diagnostic.measurements["observed_characters"] == 4097
    assert diagnostic.measurements["limit_characters"] == 4096


def test_native_ownership_requires_complete_valid_python() -> None:
    source = 'FENCE = "\\n```\\n"\n' + _TAIL + "def broken(:\n"
    assert tm._python_literal_spans(source, lambda: None) is None

    assert _diagnostic(source) is not None
    assert (
        tm.bounded_parse_diagnostic(
            source, lambda: None, file_type="python", complete_context=False
        )
        is not None
    )


def test_later_raw_high_findings_are_preserved_after_value_reassessment() -> None:
    source = 'FIRST = "ordinary"\n' + _TAIL + 'DANGEROUS = "rm -rf /"\n'
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["script.py"], "file_cache": {"script.py": source}}, [tm]
    )

    assert any(f.rule_id == "TM1" and f.severity == "HIGH" for f in result["findings"])
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_later_runtime_limit_reaches_the_ledger() -> None:
    source = 'FIRST = "ordinary"\n' + _TAIL + 'COMMAND = "$TOOL -rf /"\n'
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["script.py"], "file_cache": {"script.py": source}}, [tm]
    )

    assert any(
        event.get("reason_code") is LedgerReason.STATIC_PARSE_LIMIT
        for event in result["inspection_ledger"]
    )
