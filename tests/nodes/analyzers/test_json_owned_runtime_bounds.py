# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned JSON bodies remain visible without reparsing overlapping suffixes."""

from __future__ import annotations

import json

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner

_RUNTIME_COMMAND = "$($(resolve_tool)/printf %s rm) -rf /"
_PLACEHOLDER = "<omit on first request; reuse the returned identifier later>"


def _quote_fence(values: list[str], prefix: str = "> ", marker: str = "```") -> str:
    body = json.dumps(values, separators=(",", ":"))
    return prefix + marker + "json\n" + prefix + body + "\n" + prefix + marker + "\n"


def _expected_spans(source: str, values: list[str]) -> list[tuple[int, int]]:
    """Locate separately encoded fixture values, independent of the JSON lexer."""
    spans = []
    cursor = 0
    for value in values:
        encoded = json.dumps(value)
        start = source.index(encoded, cursor)
        cursor = start + len(encoded)
        spans.append((start, cursor))
    return spans


@pytest.mark.parametrize("prefix", ["> ", ">\t"], ids=["space-quote", "tab-quote"])
@pytest.mark.parametrize("count", [128, 256, 512])
def test_dense_owned_json_strings_have_bounded_total_parse_spans(
    prefix: str, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = ["ordinary"] * count
    source = _quote_fence(values, prefix)
    spans = _expected_spans(source, values)
    owned_characters = sum(end - start for start, end in spans)
    parsed_characters = 0
    parse_calls = 0
    inside_exhaustion = False
    real_parser = tm_module._parse_shell_command_word
    real_exhaustion = tm_module._has_shell_command_word_exhaustion

    def owned_spans(text, check_runtime):
        # Column recognition has separate tests. Supply exact ownership here so
        # a missing tab-container proof cannot hide overlapping parser work.
        assert text == source
        check_runtime()
        return spans

    def record_parser(text, start, *args, **kwargs):
        nonlocal parsed_characters, parse_calls
        parsed = real_parser(text, start, *args, **kwargs)
        if inside_exhaustion:
            parse_calls += 1
            if parsed is not None:
                parsed_characters += parsed.end - start
            # Stop a repeated-suffix regression as soon as it crosses the
            # budget, rather than spending quadratic time finishing the case.
            assert parse_calls <= count + 1
            assert parsed_characters <= len(source) + owned_characters
        return parsed

    def record_exhaustion(*args, **kwargs):
        nonlocal inside_exhaustion
        previous = inside_exhaustion
        inside_exhaustion = True
        try:
            return real_exhaustion(*args, **kwargs)
        finally:
            inside_exhaustion = previous

    monkeypatch.setattr(tm_module, "validated_json_string_spans", owned_spans)
    monkeypatch.setattr(tm_module, "_parse_shell_command_word", record_parser)
    monkeypatch.setattr(tm_module, "_has_shell_command_word_exhaustion", record_exhaustion)

    assert (
        tm_module.has_bounded_parse_exhaustion(
            source, lambda: None, file_type="markdown", complete_context=True
        )
        is False
    )
    assert parse_calls > 0
    assert parsed_characters > 0


_EARLIER_JSON_VALUES = ["`ordinary ", _RUNTIME_COMMAND, "`"]
_HIDDEN_RUNTIME_CASES = [
    pytest.param(
        _quote_fence(_EARLIER_JSON_VALUES, marker="~~~"),
        _EARLIER_JSON_VALUES,
        id="earlier-json-backticks",
    ),
    pytest.param(
        "`ordinary\n\n" + _quote_fence([_RUNTIME_COMMAND], marker="~~~") + "\n`\n",
        [_RUNTIME_COMMAND],
        id="earlier-unrelated-literal-candidate",
    ),
]


@pytest.mark.parametrize("source,values", _HIDDEN_RUNTIME_CASES)
def test_earlier_literal_parse_cannot_hide_an_owned_runtime_json_string(
    source: str, values: list[str]
) -> None:
    assert tm_module.validated_json_string_spans(source, lambda: None) == _expected_spans(
        source, values
    )
    exhausted = tm_module.has_bounded_parse_exhaustion(
        source, lambda: None, file_type="markdown", complete_context=True
    )
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": source}}, [tm_module]
    )

    assert exhausted is True
    assert any(
        row["outcome"] is LedgerOutcome.PARTIAL
        and row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
        and row["analyzer_id"] == "static_patterns_tool_misuse"
        and row["path"] == "SKILL.md"
        for row in result["inspection_ledger"]
    )


def _assert_complete(source: str) -> None:
    assert (
        tm_module.has_bounded_parse_exhaustion(
            source, lambda: None, file_type="markdown", complete_context=True
        )
        is False
    )
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": source}}, [tm_module]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert not any(finding.rule_id == "TM1" for finding in result["findings"])


@pytest.mark.parametrize("prefix", ["> ", ">\t"], ids=["space-quote", "tab-quote"])
def test_owned_json_padding_remains_complete(prefix: str) -> None:
    _assert_complete(_quote_fence([_PLACEHOLDER, "x" * 8292], prefix))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("$(hostname).example/service", id="ordinary-output-suffix"),
        pytest.param("$(resolve_tool).example/service", id="runtime-output-data-suffix"),
        pytest.param('a "quoted" value', id="raw-escaped-quotes"),
        pytest.param("Ordinary prose `example`", id="ordinary-backticks"),
    ],
)
def test_owned_json_literal_and_data_values_remain_complete(value: str) -> None:
    _assert_complete(_quote_fence([value]))


_INLINE_HOST_DOCUMENTATION = "Use `$(hostname).example` for the host name."


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(json.dumps([_INLINE_HOST_DOCUMENTATION]), id="standalone"),
        pytest.param(_quote_fence([_INLINE_HOST_DOCUMENTATION]), id="space-quote"),
    ],
)
def test_owned_json_inline_host_documentation_remains_complete(source: str) -> None:
    _assert_complete(source)


def _legacy_source(value: str, embedded: bool) -> str:
    body = json.dumps([value]) if embedded else value
    return "\n\n" + body + "\n"


@pytest.mark.parametrize("embedded", [False, True], ids=["direct", "json-embedded"])
def test_resolved_legacy_backticks_preserve_existing_static_evidence(embedded: bool) -> None:
    value = "`printf rm` -rf /"
    source = _legacy_source(value, embedded)
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": source}}, [tm_module]
    )
    if embedded:
        # Baseline is conservatively partial here and emits no TM1. Preserve
        # that incomplete contract without inventing a finding to retain.
        assert any(
            row["outcome"] is LedgerOutcome.PARTIAL
            and row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
            and row["analyzer_id"] == "static_patterns_tool_misuse"
            and row["path"] == "SKILL.md"
            for row in result["inspection_ledger"]
        )
    else:
        # Complete deterministic inspection and concrete risk are independent.
        assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
        assert any(
            finding.rule_id == "TM1"
            and finding.severity == "HIGH"
            and finding.file == "SKILL.md"
            and finding.start_line == 3
            and finding.matched_text == value
            for finding in result["findings"]
        )


def test_unsupported_legacy_command_preserves_raw_shell_uncertainty() -> None:
    value = "`$(resolve_tool).example` -rf /"
    assert tm_module.has_bounded_parse_exhaustion(value, lambda: None, file_type="shell") is True
