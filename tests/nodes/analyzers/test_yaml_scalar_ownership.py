# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from skillspector import yaml_tokens
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
def test_transformed_executable_scalar_never_becomes_verified_clean(source: str) -> None:
    result = _result(source + _TAIL)
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


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
