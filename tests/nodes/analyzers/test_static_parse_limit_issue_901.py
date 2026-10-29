# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary quotes and backticks must not read as ``static_parse_limit``.

Issue #901: two kinds of everyday text made ``static_patterns_tool_misuse``
report ``static_parse_limit``, so the file counted as partially inspected and
the referencing file picked up an AE1 HIGH finding:

1. A backtick-quoted word inside a shell comment, as in
   ``# Uses the `printf` builtin.``. The shell never executes comments, so the
   backticks are literal text and cannot open a command substitution.
2. An unmatched apostrophe in Markdown prose after inline code, as in
   ```Run `rm -f build.log` to clean the project's output.```
   The prose reading already masks possessives; it must also apply when the
   line carries inline-code delimiters.
"""

from __future__ import annotations

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner

# Verbatim reproducer 1 from the issue: backticks inside a shell comment.
_COMMENT_BACKTICK_SCRIPT = "#!/usr/bin/env bash\n# Uses the `printf` builtin.\necho ok\n"

# Verbatim reproducer 2 from the issue: inline code plus a possessive.
_INLINE_CODE_POSSESSIVE_MD = "Run `rm -f build.log` to clean the project's output.\n"


def test_backtick_in_shell_comment_completes_static_analysis() -> None:
    assert (
        tm.has_bounded_parse_exhaustion(_COMMENT_BACKTICK_SCRIPT, lambda: None, file_type="shell")
        is False
    )


def test_backtick_in_python_comment_completes_static_analysis() -> None:
    source = "# Uses the `printf` builtin.\nprint('ok')\n"
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="python") is False


def test_hash_inside_shell_string_is_not_a_comment() -> None:
    # A ``#`` inside quotes never opens a comment: the backtick substitution
    # that follows is still parsed, keeping the conservative bound.
    source = 'echo "# not a comment `printf`"\n'
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is True


def test_quoted_hash_after_closed_string_is_not_a_comment() -> None:
    # Review follow-up on #902: a whitespace-preceded ``#`` inside an
    # already closed string is not a comment start, so the substitution
    # that follows still runs and its failed printf reconstruction keeps
    # the conservative bound. The earlier negative only covered a ``#``
    # directly after the opening quote, which the comment regex already
    # rejected for lack of a preceding separator.
    source = 'echo " # literal"; $(printf "$FMT" "$CMD") -rf /\n'
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is True


def test_multiline_string_hash_is_not_a_comment() -> None:
    # Review follow-up on #902 (second round): a ``#`` on the second line
    # of a string opened on the first line is still inside that string, so
    # the substitution that follows it still runs and its failed printf
    # reconstruction keeps the conservative bound. The previous fix scanned
    # only the candidate's own line, which wrongly read this hash as a
    # comment start and exempted the substitution.
    source = 'echo "prefix\n # literal"; $(printf "$FMT" "$CMD") -rf /\n'
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is True


def test_quote_inside_comment_does_not_leak_to_next_line() -> None:
    # A quote opened inside a comment ends with that comment's newline: it
    # must not swallow the next line's genuine comment start.
    source = '# it\'s a `printf` note\n# real; $(printf "$FMT" "$CMD") -rf /\n'
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is False


def test_real_comment_before_substitution_exempts_it() -> None:
    # The paired case: the same substitution after a genuine comment start
    # never executes, so the failed reconstruction is exempt and the
    # analysis completes.
    source = '# literal; $(printf "$FMT" "$CMD") -rf /\n'
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is False


def test_hash_inside_python_string_is_not_a_comment() -> None:
    source = 'x = "# not a comment `printf`"\n'
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="python") is True


def test_runtime_command_in_comment_keeps_conservative_result() -> None:
    # A runtime-selected destructive command documented in a comment keeps
    # its partial result: only the printf-reconstruction trigger is
    # resolved for comments, never the operand-undecidability one.
    source = "# $TOOL -rf /\nvalue = 1\n"
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is True


def test_apostrophe_after_inline_code_completes_static_analysis() -> None:
    assert (
        tm.has_bounded_parse_exhaustion(
            _INLINE_CODE_POSSESSIVE_MD, lambda: None, file_type="markdown"
        )
        is False
    )


def test_unpaired_backtick_keeps_prose_check_conservative() -> None:
    # A backtick that cannot pair on its own line may still open a shell
    # substitution, so the prose reading must not mask the apostrophe.
    source = "Run `date and bob's your uncle\n"
    projected = tm._markdown_shell_text(source, lambda: None)
    assert "`" in projected


@pytest.mark.parametrize(
    "line,expected",
    [
        pytest.param(
            "Run `rm -f build.log` to clean.",
            "Run  rm -f build.log  to clean.",
            id="paired-delimiters-blanked",
        ),
        pytest.param(
            "Run `date and bob's your uncle",
            "Run `date and bob's your uncle",
            id="unpaired-tick-visible",
        ),
        pytest.param(
            "Use ``a ` b`` here",
            "Use   a ` b   here",
            id="inner-tick-kept",
        ),
    ],
)
def test_line_local_backtick_pairing(line: str, expected: str) -> None:
    assert tm._blank_line_local_paired_backticks(line) == expected


def test_ansi_c_escaped_quote_hash_is_not_a_comment() -> None:
    # Review follow-up on #902 (third round): in a Bash ANSI-C string the
    # escaped apostrophe does not close the string, so the ``#`` after it
    # is string content, not a comment start. The substitution that follows
    # the string still runs, so its failed printf reconstruction keeps the
    # conservative bound. Before the fix the comment scan popped its quote
    # at the escaped apostrophe, read the hash as a comment start, and
    # wrongly exempted the executable substitution.
    source = "echo $'x\\' # literal'; $(printf \"$FMT\" \"$CMD\") -rf /\n"
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is True


def test_ansi_c_string_then_genuine_comment_exempts() -> None:
    # The paired case: once the ANSI-C string genuinely closes, a later
    # ``#`` on the same line is a real comment start, so the substitution
    # after it never executes and the analysis completes.
    source = "echo $'x\\' # literal' # real; $(printf \"$FMT\" \"$CMD\") -rf /\n"
    assert tm.has_bounded_parse_exhaustion(source, lambda: None, file_type="shell") is False


def test_ansi_c_comment_rule_is_shell_only() -> None:
    # The ANSI-C ownership rule is a Bash rule: ``$'`` is a variable, not a
    # quote, in Perl and Ruby, so the helper only applies it when the
    # caller says shell. The same source therefore reads the hash as a
    # comment start without the flag and as string content with it.
    source = "echo $'x\\' # literal'; $(printf \"$FMT\" \"$CMD\") -rf /\n"
    start = source.index("$(printf")
    assert tm._shell_comment_start_before(source, start, ansi_c=True) is None
    assert tm._shell_comment_start_before(source, start, ansi_c=False) == 11


def test_issue_reproducer_scripts_reach_completed_ledger_outcome() -> None:
    for path, source in (
        ("scripts/run.sh", _COMMENT_BACKTICK_SCRIPT),
        ("references/guide.md", _INLINE_CODE_POSSESSIVE_MD),
    ):
        result = static_runner.run_static_patterns_with_ledger(
            {"components": [path], "file_cache": {path: source}}, [tm]
        )
        row = result["inspection_ledger"][0]
        assert row["path"] == path
        assert row["analyzer_id"] == "static_patterns_tool_misuse"
        assert row["outcome"] is LedgerOutcome.COMPLETED
        assert "reason_code" not in row
        assert row.get("reason") != LedgerReason.STATIC_PARSE_LIMIT
