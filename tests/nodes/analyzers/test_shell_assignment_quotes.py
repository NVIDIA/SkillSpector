# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from skillspector.nodes.analyzers.static_patterns_tool_misuse import has_bounded_parse_exhaustion


@pytest.mark.parametrize(
    "source",
    [
        'out="$(date)"',
        'v="a (b)"',
        'v="|$a|"',
        'printf "%s" "$(f "$x")"',
        '[ "$(f)" = true ]',
        'cat <<< "$(f)"',
        'v="a \\" (b)"',
        'v="a \\" |$a|"',
    ],
)
def test_complete_shell_quotes_do_not_consume_following_padding(source: str) -> None:
    assert not has_bounded_parse_exhaustion(source + "\n" + "#" * 6000 + "\n", lambda: None)


@pytest.mark.parametrize(
    "source", ['x="$(unterminated', 'x="unclosed', '"$(printf "%s" "$x")" -rf /']
)
def test_unresolved_shell_words_still_exhaust(source: str) -> None:
    assert has_bounded_parse_exhaustion(source + "\n" + "#" * 6000 + "\n", lambda: None)


def test_assignment_ownership_retains_nested_command_budget() -> None:
    nested_word = "r" * 4097
    source = f'out="$({nested_word} -rf /)"'
    assert has_bounded_parse_exhaustion(source, lambda: None)


def test_failed_word_parse_does_not_claim_following_command() -> None:
    source = 'Test-Path "$($_.FullName)\\cli-path"; $($CMD) -rf /'
    assert has_bounded_parse_exhaustion(source, lambda: None)
