# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host JSON punctuation cannot become one enormous shell command word."""

from __future__ import annotations

import json
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


def _without_native_consumer_roles(monkeypatch: pytest.MonkeyPatch, source: str):
    original = tm._has_shell_command_word_exhaustion

    def without_roles(*args, **kwargs):
        kwargs.pop("json_key_spans", None)
        kwargs.pop("json_value_spans", None)
        return original(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(tm, "_has_shell_command_word_exhaustion", without_roles)
        return tm.bounded_parse_diagnostic(source, _check, file_type="json")


def test_native_colons_prove_key_roles_without_classifying_value_text() -> None:
    source = '{"timeout" \n : 1, "message": "timeout"}'
    proof = prepare_json_source(source, _check)
    assert proof is not None
    projection = proof.project(SecurityTextView("raw", source), _check)
    assert projection is not None
    assert [source[start:end] for start, end in projection.key_spans] == [
        '"timeout"',
        '"message"',
    ]
    assert tuple(span for span in projection.strings if span not in projection.key_spans) == (
        (source.rindex('"timeout"'), source.rindex('"timeout"') + len('"timeout"')),
    )


@pytest.mark.parametrize("left", [0, 1])
def test_clipped_key_or_missing_native_colon_grants_no_key_role(left: int) -> None:
    source = '{"timeout": 1}'
    proof = prepare_json_source(source, _check)
    assert proof is not None
    start, end = 1 + left, source.index(":")
    view = SecurityTextView("window", source[start:end], array("I", range(start, end)))
    projection = proof.project(view, _check)
    assert projection is not None
    assert projection.key_spans == ()


@pytest.mark.parametrize("indent", [None, 2])
def test_native_key_is_not_a_wrapper_waiting_for_executable_operands(
    monkeypatch: pytest.MonkeyPatch, indent: int | None
) -> None:
    source = json.dumps({"marks": {"timeout": 1}, "value": "ordinary data"}, indent=indent)
    assert _without_native_consumer_roles(monkeypatch, source) is not None
    assert tm.bounded_parse_diagnostic(source, _check, file_type="json") is None


@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_native_value_owns_only_its_proven_backtick_closers(
    monkeypatch: pytest.MonkeyPatch, ensure_ascii: bool
) -> None:
    source = json.dumps(
        {"message": "比較 $HOME reference `label` " + "ordinary data " * 600},
        ensure_ascii=ensure_ascii,
    )
    assert _without_native_consumer_roles(monkeypatch, source) is not None
    assert tm.bounded_parse_diagnostic(source, _check, file_type="json") is None


def test_native_roles_keep_later_unreported_commands_partial() -> None:
    source = json.dumps({"timeout": 1, "message": "Execute rm -rf *"})
    result = _scan(source)
    assert result["findings"] == []
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_native_roles_preserve_raw_high_findings() -> None:
    result = _scan(json.dumps({"timeout": 1, "message": "a rm -rf /"}))
    assert any(
        finding.rule_id == "TM1" and finding.severity == "HIGH" for finding in result["findings"]
    )


@pytest.mark.parametrize(
    "command",
    ['eval "$PAYLOAD -rf /"', 'sh -c "$PAYLOAD -rf /"', "$HOME `$(cat command) -rf /`"],
)
def test_proven_keys_and_closers_cannot_hide_executable_runtime_consumers(command: str) -> None:
    source = json.dumps({"timeout": 1, "command": command})
    assert tm.bounded_parse_diagnostic(source, _check, file_type="json") is not None


def test_one_value_cannot_borrow_another_values_backtick_closer() -> None:
    source = json.dumps({"first": "$HOME `unfinished", "second": "closing` " + "data " * 1200})
    assert tm.bounded_parse_diagnostic(source, _check, file_type="json") is not None


@pytest.mark.parametrize("opener, closer", [("$(printf %s", ")"), ("`printf %s", "`")])
def test_native_value_printf_recovery_keeps_its_same_eof(
    monkeypatch: pytest.MonkeyPatch, opener: str, closer: str
) -> None:
    source = json.dumps({"first": "$HOME `tag` " + opener, "second": "later" + closer})
    first_end = source.index('", "second"') + 1
    proof = prepare_json_source(source, _check)
    assert proof is not None
    projection = proof.project(SecurityTextView("raw", source), _check)
    assert projection is not None
    original = (
        tm._skip_backtick_substitution if opener.startswith("`") else tm._skip_command_substitution
    )
    target = source.index(opener)
    observed = []

    def bounded(content, start, limit, *args, **kwargs):
        if content is projection.text and start == target:
            observed.append(limit)
            assert limit <= first_end
        return original(content, start, limit, *args, **kwargs)

    name = "_skip_backtick_substitution" if opener.startswith("`") else "_skip_command_substitution"
    monkeypatch.setattr(tm, name, bounded)
    assert (
        tm._has_shell_command_word_exhaustion(
            projection.text,
            _check,
            structural_quote_openers=projection.openers,
            structural_quote_closers=projection.closers,
            json_key_spans=projection.key_spans,
            json_value_spans=tuple(
                span for span in projection.strings if span not in projection.key_spans
            ),
        )
        is not None
    )
    assert observed


def test_clipped_value_cannot_claim_a_closing_backtick() -> None:
    source = json.dumps({"message": "$HOME reference `label` " + "ordinary data " * 600})
    proof = prepare_json_source(source, _check)
    assert proof is not None
    start = source.index("`", source.index("`") + 1)
    view = SecurityTextView("window", source[start:], array("I", range(start, len(source))))
    assert (
        tm.bounded_parse_diagnostic(
            view.text,
            _check,
            file_type="json",
            complete_context=False,
            prepared_analysis=proof,
            source_view=view,
        )
        is not None
    )


def test_cancellation_during_native_closer_collection_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = tm._parse_shell_command_word
    entered = False

    def collect(*args, **kwargs):
        nonlocal entered
        if kwargs.get("backtick_closures") is not None:
            entered = True
        return original(*args, **kwargs)

    def cancelled() -> None:
        if entered:
            raise TimeoutError("native JSON deadline")

    monkeypatch.setattr(tm, "_parse_shell_command_word", collect)
    source = json.dumps({"message": "$HOME reference `label` " + "ordinary data " * 600})
    with pytest.raises(TimeoutError, match="native JSON deadline"):
        tm.bounded_parse_diagnostic(source, cancelled, file_type="json")


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


def test_raw_provenance_does_not_search_unrelated_large_regex_suffixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = json.dumps({"command": "a rm -rf /", "data": "git push " * 75_000})
    command_start = source.index("rm -rf /")
    command_end = source.index('"', command_start)
    matches: list[tuple[int, int]] = []
    original_match = tm._has_tm1_code_pattern_at

    def bounded_match(content, start, end, check_runtime):
        assert content is source
        assert (start, end) == (command_start, command_end)
        matches.append((start, end))
        return original_match(content, start, end, check_runtime)

    def unrelated_candidates(*args, **kwargs):
        yield from ()
        pytest.fail("Exact raw evidence must not prime unrelated global candidate scans")

    monkeypatch.setattr(tm, "_has_tm1_code_pattern_at", bounded_match)
    monkeypatch.setattr(tm, "_destructive_command_candidates", unrelated_candidates)

    assert tm.bounded_parse_diagnostic(source, _check, file_type="json") is None
    assert matches == [(command_start, command_end)]
