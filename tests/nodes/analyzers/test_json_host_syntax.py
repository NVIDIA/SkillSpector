# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host JSON punctuation cannot become one enormous shell command word."""

from __future__ import annotations

import json
import time
from array import array

import pytest

from skillspector.artifacts import SecurityTextView
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.json_tokens import prepare_json_source
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.security_reconstruction import MAX_STANDALONE_JSON_SOURCE_CHARS


def _check() -> None:
    pass


def _dense(size: int = 300_000, instruction: str | None = None) -> str:
    item = '{"point":[1,2,3],"label":"sample"}'
    prefix = '{"rows":[' + ",".join([item] * ((size - 100) // (len(item) + 1))) + "]"
    if instruction is not None:
        prefix += ',"instruction":' + json.dumps(instruction)
    return prefix + "}" + " " * max(0, size - len(prefix) - 1)


def _scan(source: str):
    return static_runner.run_static_patterns_with_ledger(
        {"components": ["data.json"], "file_cache": {"data.json": source}}, [tm]
    )


@pytest.mark.parametrize("size", [20_000, 131_073, MAX_STANDALONE_JSON_SOURCE_CHARS])
def test_dense_standalone_json_owns_punctuation_without_raising_shell_word_limit(size: int) -> None:
    source = _dense(size)
    proof = prepare_json_source(source, _check)
    assert len(source) == size
    assert proof is not None
    projection = proof.project(SecurityTextView("raw", source), _check)
    assert projection is not None
    assert len(projection.text) == len(source)
    assert '"point"' in projection.text
    assert tm.bounded_parse_diagnostic(source, _check, file_type="json") is None


def test_full_scanner_carries_ownership_across_finding_windows() -> None:
    result = _scan(_dense(520_000))
    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("suffix", ["", "x", "\nrm -rf /", ",", " NaN"])
def test_truncation_and_trailing_commands_grant_no_host_syntax(suffix: str) -> None:
    source = _dense().rstrip()
    source = source[:-1] + suffix
    assert prepare_json_source(source, _check) is None
    assert tm.bounded_parse_diagnostic(source, _check, file_type="json") is not None


def test_oversized_documents_keep_their_original_limits() -> None:
    source = _dense(MAX_STANDALONE_JSON_SOURCE_CHARS + 1)
    assert prepare_json_source(source, _check) is None
    assert tm.bounded_parse_diagnostic(source, _check, file_type="json") is not None


def test_quoted_data_does_not_acquire_an_unquoted_command_word_limit() -> None:
    source = json.dumps({"value": "a" * 5000}, separators=(",", ":"))
    assert tm.bounded_parse_diagnostic(source, _check, file_type="json") is None


def test_existing_over_budget_nested_shell_command_remains_partial() -> None:
    source = 'echo "$(' + "r" * 4097 + ' -rf /)"'
    assert tm.bounded_parse_diagnostic(source, _check, file_type="shell") is not None


def test_json_looking_shell_and_markdown_keep_their_existing_contract() -> None:
    source = _dense(20_000)
    assert tm.bounded_parse_diagnostic(source, _check, file_type="shell") is None
    assert tm.bounded_parse_diagnostic(source, _check, file_type="markdown") is not None


@pytest.mark.parametrize(
    "instruction",
    ["Execute rm -rf *", "$(rm -rf /)", "`rm -rf /`", "remove 'xyz' and execute 'rxyzm -rxyzf *'"],
)
def test_security_text_inside_json_strings_is_never_masked(instruction: str) -> None:
    source = _dense(instruction=instruction)
    proof = prepare_json_source(source, _check)
    assert proof is not None
    projection = proof.project(SecurityTextView("raw", source), _check)
    assert projection is not None
    assert json.dumps(instruction) in projection.text
    result = _scan(source)
    if instruction.startswith("remove"):
        assert any(finding.rule_id == "TM1" for finding in result["findings"])
    else:
        assert (
            result["findings"] or result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
        )


def test_known_literal_command_without_a_finding_remains_incomplete() -> None:
    source = json.dumps({"message": "Execute rm -rf *"})
    result = _scan(source)
    assert result["findings"] == []
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_ordered_window_coordinates_inherit_only_original_delimiters() -> None:
    source = _dense()
    proof = prepare_json_source(source, _check)
    assert proof is not None
    begin, end = 90_003, 110_007
    view = SecurityTextView("window", source[begin:end], array("I", range(begin, end)))
    projection = proof.project(view, _check)
    assert projection is not None
    assert (
        tm.bounded_parse_diagnostic(
            view.text,
            _check,
            file_type="json",
            complete_context=False,
            prepared_analysis=proof,
            source_view=view,
        )
        is None
    )


@pytest.mark.parametrize("change", ["remove-quote", "replace-punctuation", "reorder", "duplicate"])
def test_unproven_source_mappings_cannot_borrow_json_ownership(change: str) -> None:
    source = '{"name":"sample","value":[1,2]}'
    proof = prepare_json_source(source, _check)
    assert proof is not None
    text = list(source)
    offsets = list(range(len(source)))
    if change == "remove-quote":
        del text[1]
        del offsets[1]
    elif change == "replace-punctuation":
        text[source.index(":")] = ";"
    elif change == "reorder":
        offsets[2], offsets[3] = offsets[3], offsets[2]
    else:
        offsets[2] = offsets[1]
    assert (
        proof.project(SecurityTextView("derived", "".join(text), array("I", offsets)), _check)
        is None
    )


def test_changed_string_contents_remain_visible_to_the_parser() -> None:
    source = '{"message":"harmless; $(rm -rf /)","values":[1,2]}'
    proof = prepare_json_source(source, _check)
    assert proof is not None
    changed = source.replace("harmless", "danger!!")
    view = SecurityTextView("derived", changed, array("I", range(len(source))))
    projection = proof.project(view, _check)
    assert projection is not None
    assert "danger!!; $(rm -rf /)" in projection.text
    assert projection.text.count("$(rm -rf /)") == 1


def test_large_validation_does_not_allocate_decoded_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_decode(*args, **kwargs):
        raise AssertionError("Large JSON must use the existing iterative grammar")

    monkeypatch.setattr("skillspector.security_reconstruction.json.loads", reject_decode)
    assert prepare_json_source(_dense(), _check) is not None


def test_cancellation_propagates_before_any_ownership_escapes() -> None:
    class CancelledError(Exception):
        pass

    calls = 0

    def check() -> None:
        nonlocal calls
        calls += 1
        if calls == 100:
            raise CancelledError

    with pytest.raises(CancelledError):
        prepare_json_source(_dense(), check)


@pytest.mark.parametrize("source", ["r " * 10_000, "'" + "r" * 20_000 + "'"])
def test_literal_effects_walk_checks_cancellation_without_yielding_a_command(source: str) -> None:
    class CancelledError(Exception):
        pass

    calls = 0

    def check() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise CancelledError

    with pytest.raises(CancelledError):
        tm._has_destructive_command_string(source, check_runtime=check)
    assert calls == 3


@pytest.mark.parametrize(
    "source",
    [
        '{"first":"a rm -rf /","second":"Execute rm -rf *"}',
        '{"steps":"a rm -rf /; Execute rm -rf *"}',
    ],
)
def test_reported_command_cannot_hide_a_later_unreported_string_command(source: str) -> None:
    result = _scan(source)
    assert any(finding.rule_id == "TM1" for finding in result["findings"])
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_raw_provenance_does_not_search_unrelated_large_regex_suffixes() -> None:
    source = json.dumps({"command": "a rm -rf /", "data": "git push " * 75_000})
    deadline = time.monotonic() + 10

    def check_runtime() -> None:
        if time.monotonic() > deadline:
            raise TimeoutError("JSON provenance deadline")

    assert tm.bounded_parse_diagnostic(source, check_runtime, file_type="json") is None
