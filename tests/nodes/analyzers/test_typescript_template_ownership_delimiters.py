# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for TypeScript template ownership across delimiters and names."""

from __future__ import annotations

import pytest

from skillspector.typescript_tokens import typescript_template_spans


def _owned(source: str) -> bool:
    return typescript_template_spans(source, lambda: None) is not None


def test_an_identity_escape_in_a_module_specifier_is_unowned() -> None:
    source = 'import launch from "ex\\eca";\nlaunch(`${command.name}`);'

    assert _owned(source) is False


def test_an_escaped_module_specifier_still_names_the_module() -> None:
    source = 'import launch from "\\u0065xeca";\nlaunch(`${command.name}`);'

    assert _owned(source) is False


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("const v = `${format(items[0])}`;", id="call-then-computed-member"),
        pytest.param("const v = `${[format(1)]}`;", id="array-literal-with-call"),
        pytest.param("consume(a[0]);\nconst v = `${x}`;", id="bracket-inside-call"),
        pytest.param("const y = f([1].map((x) => x));\nconst v = `${x}`;", id="nested-pairs"),
    ],
)
def test_a_bracket_inside_a_call_stays_owned(source: str) -> None:
    assert _owned(source) is True


def test_a_parenthesised_object_in_an_interpolation_stays_owned() -> None:
    assert _owned("const v = `${({name: item.name}).name}`;") is True


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("const v = `${({})}`;", id="empty-object"),
        pytest.param("const v = `${({a: 1}).a}`;", id="member-of-object"),
    ],
)
def test_an_object_literal_inside_parentheses_stays_owned(source: str) -> None:
    assert _owned(source) is True


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("const x = \\u0065val(payload);", id="unicode-escape"),
        pytest.param("const x = \\x65val(payload);", id="hex-escape"),
        pytest.param("const x = ev\\u0061l(payload);", id="mid-word-escape"),
        pytest.param("const x = \\u{65}val(payload);", id="braced-escape"),
    ],
)
def test_an_identifier_escape_spelling_eval_is_unowned(source: str) -> None:
    # An escaped name is the call it spells, so the template it guards cannot
    # be owned. The escape sits in code: the same bytes inside a template are
    # literal text rather than a name, which is a separate value-flow question.
    assert _owned(source + "\nconst v = `${x}`;") is False


def test_an_identifier_escape_spelling_a_benign_name_stays_owned() -> None:
    assert _owned("const label = \\u0068elp(value);\nconst v = `${x}`;") is True


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "const n = {a: 1} / eval(payload) / 2;\nconst label = `${item.name}`;",
            id="object-with-member",
        ),
        pytest.param(
            "f({}) / eval(payload) / 2;\nconst label = `${item.name}`;",
            id="object-argument",
        ),
        pytest.param(
            "const n = ({}) / eval(payload) / 2;\nconst label = `${item.name}`;",
            id="parenthesised-object",
        ),
    ],
)
def test_division_after_an_object_literal_keeps_the_sink_unowned(source: str) -> None:
    # An empty ``{}`` in statement position is a block, so a slash after it is a
    # regex; these are all operand positions, where the slash divides, which
    # leaves the ``eval`` guard reachable. An execution call here is a
    # reproduced unsafe ownership, not a benign difference between readings.
    assert _owned(source) is False


def test_division_between_object_literals_is_not_read_as_a_regex() -> None:
    source = "const n = {} / 2 / 3;\nconst label = `${item.name}`;"

    assert _owned(source) is True


def test_a_regex_after_a_block_stays_a_regex() -> None:
    source = "if (ok) { render(); }\nconst r = /ab+c/g;\nconst label = `${item.name}`;"

    assert _owned(source) is True


def test_a_nested_block_does_not_end_the_scan() -> None:
    source = "if (a) { if (b) { c(); } }\nconst label = `${item.name}`;"

    assert _owned(source) is True


def test_a_slash_after_an_expression_brace_divides_without_a_call() -> None:
    # No call follows the brace, so only the division reading keeps the
    # template owned; a regex reading runs to the next slash and loses it.
    source = "const n = {a: 1} / 2;\nconst v = `${x}`;"

    assert _owned(source) is True


def test_a_nested_object_before_a_template_stays_owned() -> None:
    source = "const n = 4 / 2;\nconst o = {a: {b: 1}};\nconst v = `${x}`;"

    assert _owned(source) is True


def test_a_block_close_before_a_template_keeps_the_template_owned() -> None:
    source = "if (a) { b(); }\n/ab+c/.test(x);\nconst v = `${x}`;"

    assert _owned(source) is True


def test_a_division_closing_a_block_then_a_regex_stays_parsed() -> None:
    source = (
        "const n = (() => { return {}; })() / 2;\nconst r = /ab/;\nconst label = `${item.name}`;"
    )

    assert _owned(source) is True


def test_a_lowercase_function_declaration_is_not_an_execution_call() -> None:
    # ``function`` is the execution name ``Function`` folded, so the exact
    # lowercase keyword must still read as a declaration rather than a call.
    source = "function helper(${p}) {}\nconst v = `${x}`;"

    assert _owned(source) is True
