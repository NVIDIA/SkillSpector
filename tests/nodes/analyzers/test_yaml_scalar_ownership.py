# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from array import array

import pytest

from skillspector import yaml_tokens
from skillspector.artifacts import SecurityTextView
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.yaml_tokens import yaml_source_spans

_TAIL = "".join(f"field_{i}: ordinary descriptive text for this interface\n" for i in range(200))


def _result(source: str) -> dict:
    return static_runner.run_static_patterns_with_ledger(
        {"components": ["rules.yaml"], "file_cache": {"rules.yaml": source}}, [tm]
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "prefix",
    [
        'message: "The reader\'s documented interface is available."\n',
        "# The reader's documented interface is available.\n",
        'message: "Use `value` when reporting the result."\n',
        "message: >\n  Use `value` when reporting the result.\n",
        "pattern: " + json.dumps(r"""["'`]value""") + "\n",
    ],
)
def test_complete_yaml_scalars_and_comments_cannot_borrow_later_source(
    prefix: str, newline: str
) -> None:
    source = ("# Unicode: café\n" + prefix + _TAIL).replace("\n", newline)
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="yaml") is None
    assert _result(source)["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "source",
    [
        'run: "$CMD -rf /"\n',
        "run: 'eval \"$CMD\"'\n",
        'run: >\n  sh -c "$CMD -rf /"\n',
        "run: " + json.dumps('sh -c "$CMD -rf /"') + "\n",
    ],
)
def test_yaml_executable_values_keep_runtime_inspection_incomplete(source: str) -> None:
    result = _result(source + _TAIL)
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_yaml_scalar_ownership_preserves_dangerous_literal_finding() -> None:
    result = _result("run: 'rm -rf /'\n" + _TAIL)
    assert any(f.rule_id == "TM1" for f in result["findings"])


@pytest.mark.parametrize(
    "command, prefix",
    [
        ("$CMD -rf /", "# Run "),
        ("$CMD -rf /", "field: ordinary value # Run "),
        ('sh -c "$CMD -rf /"', "# "),
        ('sh -c "$CMD -rf /"', "field: ordinary value # "),
        ('eval "$CMD"', "# "),
        ('eval "$CMD"', "field: ordinary value # "),
    ],
)
def test_yaml_comment_instructions_keep_runtime_inspection_incomplete(
    command: str, prefix: str
) -> None:
    result = _result(prefix + command + "\n" + _TAIL)

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_yaml_comment_ownership_preserves_dangerous_literal_finding() -> None:
    result = _result("# Run rm -rf /\n" + _TAIL)

    assert any(f.rule_id == "TM1" for f in result["findings"])


@pytest.mark.parametrize("style", ["|", ">", "|-", "|+", ">-", "|2", "|2-", "|+2"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_block_scalar_header_comments_keep_runtime_instructions(style: str, newline: str) -> None:
    source = f"message: {style} # Run $CMD -rf /\n  ordinary display text\n" + _TAIL
    source = source.replace("\n", newline)
    proof = yaml_source_spans(source, lambda: None)

    assert proof is not None
    assert source[slice(*proof.comments[0])].rstrip("\r") == "# Run $CMD -rf /"
    result = _result(source)
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "source",
    [
        "run: >\n  rm -rf /\n",
        'run: "rm\\u0020-rf\\u0020/"\n',
    ],
)
def test_transformed_executable_scalar_reports_its_enclosing_source(source: str) -> None:
    result = _result(source + _TAIL)
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    decoded = [f for f in result["findings"] if "yaml_scalar" in f.evidence]
    assert any(f.rule_id == "TM1" and f.severity == "HIGH" for f in decoded)
    assert all(f.file == "rules.yaml" and f.start_line == 1 for f in decoded)
    assert all(
        f.evidence["yaml_scalar"]["location_precision"] == "enclosing_expression" for f in decoded
    )


@pytest.mark.parametrize("offset", [0, -7, 7])
def test_transformed_scalar_after_window_boundary_keeps_source_identity(offset: int) -> None:
    prefix_size = static_runner._RAW_WINDOW_OWNED_CHARS + offset
    prefix = "padding: " + "x" * (prefix_size - len("padding: ") - 1) + "\n"
    source = prefix + 'run: "rm\\u0020-rf\\u0020/"\n' + _TAIL
    result = _result(source)
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert any(
        f.rule_id == "TM1"
        and f.file == "rules.yaml"
        and f.start_line == 2
        and f.evidence.get("yaml_scalar", {}).get("source_start_offset") == len(prefix) + 5
        for f in result["findings"]
    )


def test_oversized_transformed_scalar_keeps_attribution_incomplete() -> None:
    source = (
        "padding: " + "x" * static_runner.SECURITY_VIEW_WINDOW_CHARS + "\n"
        'run: "rm\\u0020-rf\\u0020/ ' + "x" * static_runner._WINDOW_OVERLAP_CHARS + '"\n'
    )
    result = _result(source)
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


def test_unbound_or_derived_parser_context_keeps_attribution_incomplete() -> None:
    source = 'run: "rm\\u0020-rf\\u0020/"\n'
    prepared = tm.prepare_parser_analysis(source, "yaml", lambda: None)
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="yaml") is not None
    for view in [SecurityTextView("normalized", source), SecurityTextView("raw", source + "\n")]:
        assert (
            tm.bounded_parse_diagnostic(
                source,
                lambda: None,
                file_type="yaml",
                prepared_analysis=prepared,
                source_view=view,
            )
            is not None
        )


def test_nonidentity_interior_mapping_cannot_borrow_raw_scalar_provenance() -> None:
    source = 'name: example\nrun: "rm\\u0020-rf\\u0020/"\n'
    offsets = array("I", range(len(source)))
    offsets[2], offsets[3] = offsets[3], offsets[2]
    view = SecurityTextView("raw", source, offsets)
    parser = tm.prepare_parser_analysis(source, "yaml", lambda: None)
    prepared = tm.prepare_analysis(source, "yaml", lambda: None)
    assert parser is not None and prepared is not None
    assert (
        tm.bounded_parse_diagnostic(
            source,
            lambda: None,
            file_type="yaml",
            prepared_analysis=parser,
            source_view=view,
        )
        is not None
    )
    assert not any(
        "yaml_scalar" in f.evidence for f in prepared.analyze(source, "rules.yaml", "yaml", view)
    )


def test_decoded_line_metadata_is_distinct_from_enclosing_raw_line() -> None:
    source = "a: one\nb: two\nc: three\nrun: |\n  ordinary text\n  rm -rf /\n"
    decoded = [f for f in _result(source)["findings"] if "yaml_scalar" in f.evidence]
    assert any(
        f.rule_id == "TM1"
        and f.start_line == 4
        and f.evidence["yaml_scalar"]["decoded_start_line"] == 2
        for f in decoded
    )


def test_distinct_decoded_findings_are_not_collapsed_by_rule() -> None:
    result = _result('run: "rm\\u0020-rf\\u0020/; rm\\u0020-rf\\u0020/tmp/project"\n')
    decoded = [f for f in result["findings"] if "yaml_scalar" in f.evidence and f.rule_id == "TM1"]
    assert len(decoded) >= 2
    assert len({f.fingerprint() for f in decoded}) >= 2


def test_budget_interruption_preserves_raw_enclosing_locations() -> None:
    source = 'run: "rm\\u0020-rf\\u0020/; rm\\u0020-rf\\u0020/tmp/project"\n'
    findings, reason, _ = static_runner._scan_all_views_detailed(
        "rules.yaml", source, [tm], None, max_findings=1
    )
    assert reason is LedgerReason.OUTPUT_LIMIT
    assert findings
    assert all(f.file == "rules.yaml" and f.start_line == 1 for f in findings)
    assert all(f.evidence.get("yaml_scalar") for f in findings)


@pytest.mark.parametrize("kind", ["json", "python", "other"])
def test_non_yaml_finding_prepare_does_not_repeat_parser_proof(monkeypatch, kind) -> None:
    def reject_preparation(*args, **kwargs):
        pytest.fail("Finding preparation must not repeat another language's parser proof")

    monkeypatch.setattr(tm, "prepare_parser_analysis", reject_preparation)
    assert tm.prepare_analysis("{}", kind, lambda: None) is None


@pytest.mark.parametrize(
    "source",
    [
        'value: "unclosed\n',
        "value: &named hello\ncopy: *named\n",
        "value: !!str hello\n",
        "value: [one, two\n",
    ],
)
def test_unproven_yaml_grammar_or_indirection_gives_no_ownership(source: str) -> None:
    assert yaml_source_spans(source, lambda: None) is None


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_yaml_scalar_offsets_and_styles_are_exact(newline: str) -> None:
    source = (
        "# café\nfirst: \"plain text\"\nsecond: 'single text'\nthird: |\n  block text\n".replace(
            "\n", newline
        )
    )
    parsed = yaml_source_spans(source, lambda: None)
    assert parsed is not None
    quoted = [s for s in parsed.scalars if s.style in {'"', "'"}]
    assert [source[s.body_start : s.body_end] for s in quoted] == ["plain text", "single text"]
    assert [source[s.start : s.end][-1] for s in quoted] == ['"', "'"]
    assert source[slice(*parsed.comments[0])].rstrip("\r") == "# café"


def test_yaml_proof_enforces_size_depth_record_and_cancellation_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(yaml_tokens, "_MAX_SOURCE_CHARS", 16)
    assert yaml_source_spans("value: " + ("x" * 20), lambda: None) is None
    monkeypatch.setattr(yaml_tokens, "_MAX_SOURCE_CHARS", 1_000_000)
    monkeypatch.setattr(yaml_tokens, "_MAX_DEPTH", 2)
    assert yaml_source_spans("value: [[[1]]]", lambda: None) is None
    monkeypatch.setattr(yaml_tokens, "_MAX_TOKENS", 2)
    assert yaml_source_spans("one: unique\ntwo: value\n", lambda: None) is None

    def cancelled() -> None:
        raise TimeoutError("cancelled")

    with pytest.raises(TimeoutError, match="cancelled"):
        yaml_source_spans("value: another unique input\n", cancelled)


def test_yaml_ownership_requires_the_complete_stream() -> None:
    assert tm.has_bounded_parse_exhaustion(
        'message: "Use `value when reporting the result."\n' + _TAIL,
        lambda: None,
        file_type="yaml",
        complete_context=False,
    )
