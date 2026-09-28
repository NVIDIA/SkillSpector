# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Iterative JSON validation proves complete syntax without decoding a document."""

from __future__ import annotations

import json
import random
from collections.abc import Callable

import pytest

from skillspector import security_reconstruction as reconstruction


def _valid(source: str, check_runtime: Callable[[], None] | None = None) -> bool:
    return reconstruction._validate_json_without_decoding(source, check_runtime)


def test_larger_json_capacity_does_not_extend_frontmatter_boundary() -> None:
    # This closer lies between the preserved prefix bound and the new JSON
    # bound, so using the latter for prefix discovery would grant ownership.
    prefix = "---\n#" + "x" * 65_530 + "\n---\n"
    assert 65_536 < len(prefix) < 131_072
    assert reconstruction._json_body_after_frontmatter(prefix + '["value"]', None) is None
    assert reconstruction.validated_json_string_spans(prefix + '["value"]', None) == []


def _reference_accepts(source: str) -> bool:
    def reject_constant(_value: str) -> None:
        raise ValueError("Non-JSON constant")

    try:
        # Numeric callbacks compare syntax without depending on Python's
        # integer-conversion length limit or floating-point overflow behavior.
        json.loads(
            source,
            parse_constant=reject_constant,
            parse_int=lambda _value: None,
            parse_float=lambda _value: None,
        )
    except (ValueError, RecursionError):
        return False
    return True


@pytest.mark.parametrize(
    "source",
    [
        "null",
        "true",
        "false",
        "0",
        "-0",
        "1234567890",
        "-1234567890",
        "0.0",
        "-0.01",
        "1e0",
        "1E+09",
        "1e-09",
        "-1.23E+45",
        "1e999999",
        '""',
        '"plain Unicode ☃ 🧭"',
        '"line separator \u2028 and paragraph separator \u2029"',
        r'"\"\\\/\b\f\n\r\t\u0041"',
        r'"\uD83E\uDDED"',
        r'"\ud800"',
        r'"\udfff"',
        "[]",
        "{}",
        '[null,true,false,0,-1,2.5,"text",[],{}]',
        '{"":[],"nested":{"array":[{},[null]]}}',
        '{"duplicate":1,"duplicate":2}',
        ' \t\r\n { "a" : [ 1 , 2 ] } \r\n\t ',
    ],
)
def test_valid_json_grammar_matches_independent_decoder(source: str) -> None:
    assert _reference_accepts(source)
    assert _valid(source) is True


@pytest.mark.parametrize(
    "source",
    [
        "",
        " \t\r\n",
        "NaN",
        "Infinity",
        "-Infinity",
        "True",
        "False",
        "NULL",
        "undefined",
        "+1",
        "-",
        "--1",
        "00",
        "01",
        "-01",
        ".1",
        "1.",
        "1.e1",
        "1e",
        "1e+",
        "1e-",
        "1e+-2",
        "1_000",
        "0x10",
        "١",
        "1٢",
        "[1,]",
        "[,1]",
        "[1,,2]",
        "[1 2]",
        "[}",
        "{]",
        "{,}",
        '{"a":1,}',
        '{"a",1}',
        '{"a" 1}',
        '{"a":}',
        '{"a":1 "b":2}',
        "{true:1}",
        "{:1}",
        '{"a":[1}}',
        "[]{}",
        "truefalse",
        "null 0",
        '"a" "b"',
        '"a"x',
        "// comment\n{}",
        "/* comment */ {}",
        "\ufeff{}",
        "\u00a0[]",
        "[]\u00a0",
        "[]\v",
        "[]\f",
        "[\u2028]",
        '"unterminated',
        '"backslash-at-end\\',
        r'"\q"',
        r'"\x41"',
        r'"\u"',
        r'"\u123"',
        r'"\u12G4"',
        r'"\u１２３４"',
        r'"\U00000041"',
        '"\\\n"',
    ],
)
def test_invalid_json_grammar_matches_independent_decoder(source: str) -> None:
    assert _reference_accepts(source) is False
    assert _valid(source) is False


@pytest.mark.parametrize("codepoint", range(32))
def test_unescaped_ascii_control_characters_are_rejected(codepoint: int) -> None:
    source = '"before' + chr(codepoint) + 'after"'
    assert _reference_accepts(source) is False
    assert _valid(source) is False


def test_generated_documents_and_mutations_match_independent_decoder() -> None:
    rng = random.Random(626)
    alphabet = ['"', "\\", "a", " ", "\n", "\r", "\t", "\x00", "☃", "🧭", "\u2028"]

    def value(depth: int):
        if depth and rng.randrange(3) == 0:
            if rng.randrange(2):
                return [value(depth - 1) for _ in range(rng.randrange(4))]
            return {f"key {index}": value(depth - 1) for index in range(rng.randrange(4))}
        return rng.choice(
            [
                None,
                True,
                False,
                rng.randrange(-10000, 10000),
                0.125,
                "".join(rng.choices(alphabet, k=8)),
            ]
        )

    for _ in range(64):
        document = value(4)
        for ensure_ascii in (False, True):
            for indent in (None, 2):
                source = json.dumps(document, ensure_ascii=ensure_ascii, indent=indent)
                assert _valid(source) is True
                for position in {0, len(source) // 2, len(source) - 1}:
                    for mutated in (
                        source[:position] + source[position + 1 :],
                        source[:position] + "?" + source[position:],
                    ):
                        # Deleting a character may leave valid JSON; determine
                        # the expected grammar result independently each time.
                        assert _valid(mutated) is _reference_accepts(mutated), repr(mutated)


def test_every_truncated_prefix_of_a_structured_value_is_rejected() -> None:
    source = r'{"outer":[{"k":"v\"x\\y\u1234"},true,false,null,1.2e-3],"empty":{}}'
    assert _reference_accepts(source)
    for end in range(len(source)):
        truncated = source[:end]
        assert _reference_accepts(truncated) is False
        assert _valid(truncated) is False


def _extended_array() -> str:
    source = json.dumps([f"record {index:05d}" for index in range(6000)], separators=(",", ":"))
    assert reconstruction._MAX_JSON_QUOTE_DECODE_CHARS < len(source)
    assert len(source) <= reconstruction._MAX_JSON_QUOTE_CONTAINER_CHARS
    return source


@pytest.mark.parametrize("kind", ["standalone", "frontmatter", "fence", "quote-list"])
def test_extended_ownership_never_calls_json_decoder(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _extended_array()
    prefix, suffix = {
        "standalone": ("", ""),
        "frontmatter": ("---\nname: json-example\n---\n", ""),
        "fence": ("~~~json\n", "\n~~~\n"),
        "quote-list": ("> - ~~~json\n>   ", "\n>   ~~~\n"),
    }[kind]
    source = prefix + body + suffix

    def reject_decode(*_args, **_kwargs):
        pytest.fail("Extended JSON ownership called the decoding parser")

    monkeypatch.setattr(reconstruction.json, "loads", reject_decode)
    spans = reconstruction.validated_json_string_spans(source, None)
    assert len(spans) == 6000
    assert source[slice(*spans[0])] == '"record 00000"'
    assert source[slice(*spans[-1])] == '"record 05999"'
    assert spans[0][0] == len(prefix) + 1
    assert spans[-1][1] == len(prefix) + len(body) - 1


@pytest.mark.parametrize("failure", ["truncated", "trailing-comma", "trailing-value", "bad-escape"])
def test_late_invalid_extended_json_grants_no_partial_string_ownership(failure: str) -> None:
    source = _extended_array()
    if failure == "truncated":
        source = source[:-1]
    elif failure == "trailing-comma":
        source = source[:-1] + ",]"
    elif failure == "trailing-value":
        source += "true"
    else:
        source = source[:-1] + ',"\\q"]'
    assert _reference_accepts(source) is False
    assert _valid(source) is False
    assert reconstruction.validated_json_string_spans(source, None) == []


@pytest.mark.parametrize("error_type", [ValueError, RecursionError])
def test_small_decoder_failure_does_not_fall_back_to_iterative_validation(
    error_type: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_decode(*_args, **_kwargs):
        raise error_type("legacy validation failure")

    def reject_fallback(*_args, **_kwargs):
        pytest.fail("Small decoder failure must preserve the existing no-ownership outcome")

    monkeypatch.setattr(reconstruction.json, "loads", reject_decode)
    monkeypatch.setattr(reconstruction, "_validate_json_without_decoding", reject_fallback)
    assert reconstruction.validated_json_string_spans('{"key":"value"}', None) == []


@pytest.mark.parametrize("kind", ["arrays", "objects"])
def test_deep_extended_json_uses_no_python_recursion_or_decoded_tree(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if kind == "arrays":
        depth = 35_000
        source = "[" * depth + '"value"' + "]" * depth
        expected_count = 1
    else:
        depth = 12_000
        source = '{"key":' * depth + '"value"' + "}" * depth
        expected_count = depth + 1
    assert reconstruction._MAX_JSON_QUOTE_DECODE_CHARS < len(source)
    assert len(source) <= reconstruction._MAX_JSON_QUOTE_CONTAINER_CHARS

    def reject_decode(*_args, **_kwargs):
        pytest.fail("Deep extended JSON must not build a decoded document")

    monkeypatch.setattr(reconstruction.json, "loads", reject_decode)
    assert _valid(source) is True
    spans = reconstruction.validated_json_string_spans(source, None)
    assert len(spans) == expected_count
    assert len(spans) <= len(source) // 2
    assert _valid(source[:-1]) is False
    assert reconstruction.validated_json_string_spans(source[:-1], None) == []


def test_dense_extended_json_has_disjoint_raw_quote_spans() -> None:
    count = 43_000
    source = "[" + ",".join(['""'] * count) + "]"
    assert reconstruction._MAX_JSON_QUOTE_DECODE_CHARS < len(source)
    assert len(source) <= reconstruction._MAX_JSON_QUOTE_CONTAINER_CHARS
    assert _valid(source) is True
    spans = reconstruction.validated_json_string_spans(source, None)
    assert len(spans) == count
    assert len(spans) <= len(source) // 2
    assert all(span == (1 + 3 * index, 3 + 3 * index) for index, span in enumerate(spans))


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_normalized_validation_copy_has_a_finite_character_ceiling(delta: int) -> None:
    limit = 4 * reconstruction._MAX_JSON_QUOTE_CONTAINER_CHARS
    source = '"' + "x" * (limit + delta - 2) + '"'
    assert len(source) == limit + delta
    assert _valid(source) is (delta <= 0)


def test_raw_capacity_guard_precedes_both_validation_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    source = '["value"]'
    source += " " * (reconstruction._MAX_JSON_QUOTE_CONTAINER_CHARS + 1 - len(source))

    def reject_validation(*_args, **_kwargs):
        pytest.fail("An oversized raw candidate reached syntax validation")

    monkeypatch.setattr(reconstruction.json, "loads", reject_validation)
    monkeypatch.setattr(reconstruction, "_validate_json_without_decoding", reject_validation)
    assert reconstruction.validated_json_string_spans(source, None) == []


class _CountingText(str):
    def __init__(self, value: str) -> None:
        self.indexed_reads = 0
        self.highest_read = -1

    def __getitem__(self, key: int | slice) -> str:
        result = super().__getitem__(key)
        self.indexed_reads += len(result)
        # A generous constant permits lookahead but aborts repeated-suffix
        # regressions deterministically, without a machine-speed timeout.
        assert self.indexed_reads <= 32 * len(self)
        if isinstance(key, int):
            self.highest_read = max(self.highest_read, key if key >= 0 else len(self) + key)
        elif result:
            self.highest_read = max(self.highest_read, key.indices(len(self))[1] - 1)
        return result


def _work_source(kind: str, count: int) -> str:
    if kind == "string":
        return '"' + "a" * count + '"'
    if kind == "escapes":
        return '"' + r"\u0061" * count + '"'
    if kind == "number":
        return "1" + "0" * count
    if kind == "whitespace":
        return " " * count + "null"
    return "[" * count + '"value"' + "]" * count


@pytest.mark.parametrize("kind", ["string", "escapes", "number", "whitespace", "nesting"])
def test_iterative_validation_has_linear_indexed_work(kind: str) -> None:
    previous_reads = 0
    for count in (256, 512, 1024):
        source = _CountingText(_work_source(kind, count))
        assert _valid(source) is True
        assert source.indexed_reads > 0
        if previous_reads:
            assert source.indexed_reads <= 2 * previous_reads + 256
        previous_reads = source.indexed_reads


@pytest.mark.parametrize("kind", ["string", "escapes", "number", "whitespace", "nesting"])
def test_iterative_validation_yields_to_cancellation_inside_long_tokens(kind: str) -> None:
    source = _CountingText(_work_source(kind, 10_000))
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        _valid(source, cancel)
    assert checks == 3
    assert source.highest_read < 1024


@pytest.mark.parametrize("kind", ["string", "escapes", "number", "whitespace", "nesting"])
def test_runtime_checks_are_never_more_than_256_source_characters_apart(kind: str) -> None:
    source = _CountingText(_work_source(kind, 2048))
    previous_read = -1
    checks = 0

    def check_runtime() -> None:
        nonlocal previous_read, checks
        checks += 1
        assert source.highest_read - previous_read <= 256
        previous_read = source.highest_read

    assert _valid(source, check_runtime) is True
    assert source.highest_read - previous_read <= 256
    assert checks >= len(source) // 256
