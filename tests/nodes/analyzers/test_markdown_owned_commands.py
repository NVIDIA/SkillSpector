# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native inline bodies retain their executable child-command coverage."""

from __future__ import annotations

import json

import pytest

from skillspector.inspection_ledger import ParserCause, ParserCheck
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "body",
    [
        'sh -c "$PAYLOAD"',
        'env sh -c "$PAYLOAD"',
        'eval "$PAYLOAD"',
        "$(cat command.txt)",
        "alias launch='eval'; launch \"$PAYLOAD\"",
        'BASH_ENV="$STARTUP"; bash -c "printf done"',
        'env BASH_ENV="$STARTUP" bash -c "printf done"',
        "sh -c 'echo `$PAYLOAD`'",
        "sh -c 'echo `$(cat command.txt)`'",
    ],
)
def test_owned_executable_operands_and_generated_command_heads_are_partial(
    body: str, newline: str
) -> None:
    source = ("# 比較 🧪\n\nRun ``" + body + "`` now.\n").replace("\n", newline)
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert diagnostic.cause in {ParserCause.RUNTIME_DEPENDENT, ParserCause.RESOURCE_LIMIT}
    assert source.index(body) <= diagnostic.start < diagnostic.end <= source.index(body) + len(body)


@pytest.mark.parametrize(
    "source",
    [
        "Use `eval` as a reference.",
        "A command reference `bash -c`.",
        "The reader's reference `label` explains a user's choice.",
        "The reader’s reference `label` explains a user’s choice.",
        "A variable reference `$HOME` is displayed.",
        "Run `sh -c 'printf done'` now.",
        "Run `$(printf echo)` now.",
        "Run `echo 'sh -c \"$PAYLOAD\"'` now.",
        "Run `printf '%s' '$(cat command.txt)'` now.",
        "Run `echo 'alias launch=eval'; printf done` now.",
        "Use `label` without borrowing the later prose's quotes.",
    ],
)
def test_literal_references_data_and_proved_printf_values_stay_complete(source: str) -> None:
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown") is None


@pytest.mark.parametrize(
    "label",
    [
        "can't find the entry",
        "reader's reference label",
        "'unclosed display label",
        '"unclosed display label',
    ],
)
def test_display_quote_alone_does_not_establish_an_executable_head(label: str) -> None:
    source = "Diagnostic text ``" + label + "``."
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown") is None


@pytest.mark.parametrize("label", ["can't find the entry", "'unclosed display label"])
@pytest.mark.parametrize("utility", ["sh -c", "eval"])
def test_the_same_unresolved_quote_in_an_executed_operand_stays_partial(
    label: str, utility: str
) -> None:
    body = utility + " " + json.dumps(label)
    source = "Run ``" + body + "``."
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.BOUNDARY_UNRESOLVED
    assert diagnostic.check is ParserCheck.COMMAND_WORD_BOUNDARY
    assert source.index(body) <= diagnostic.start < diagnostic.end <= source.index(body) + len(body)


@pytest.mark.parametrize(
    "word",
    ['"$CMD', '"$(resolve_tool)', "$(unfinished", '"${unfinished', "${unfinished", "`unfinished"],
)
def test_opaque_or_unclosed_expansion_heads_keep_their_boundary(word: str) -> None:
    source = "An owned fragment `` " + word + " ``."
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.BOUNDARY_UNRESOLVED


@pytest.mark.parametrize("word", ["$(hostname).example", "$(resolve_tool).example/service"])
def test_display_word_fragment_does_not_independently_establish_execution(word: str) -> None:
    source = "A data reference ``" + word + "``."
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown") is None


@pytest.mark.parametrize("word", ["$(hostname).example", "$(resolve_tool).example/service"])
@pytest.mark.parametrize(
    "template,check",
    [
        ("sh -c '{}'", ParserCheck.COMMAND_IDENTITY),
        ("eval '{}'", ParserCheck.COMMAND_IDENTITY),
        # Existing substitution reconstruction runs before the owned worker.
        ("echo $({})", ParserCheck.PRINTF_INVOCATION),
    ],
)
def test_executed_and_nested_dynamic_word_fragments_remain_uncertain(
    word: str, template: str, check: ParserCheck
) -> None:
    body = template.format(word + " -rf /")
    source = "Run ``" + body + "``."
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RUNTIME_DEPENDENT
    assert diagnostic.check is check
    assert source.index(body) <= diagnostic.start < diagnostic.end <= source.index(body) + len(body)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_later_inline_consumer_is_not_hidden_by_an_earlier_reference(newline: str) -> None:
    source = 'Use `label` first.\nThen run `sh -c "$PAYLOAD"`.\n'.replace("\n", newline)
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert diagnostic.start >= source.index("sh -c")


@pytest.mark.parametrize(
    "command, check",
    [
        ("a" * 4097, ParserCheck.COMMAND_WORD_CHARACTERS),
        ('"unfinished', ParserCheck.COMMAND_WORD_BOUNDARY),
        ("$(unfinished", ParserCheck.EXPANSION_BOUNDARY),
        ("echo $(" + "a" * 4097 + ")", ParserCheck.COMMAND_WORD_CHARACTERS),
        ("alias " + "a" * 4097, ParserCheck.COMMAND_WORD_CHARACTERS),
        ('alias "unfinished', ParserCheck.COMMAND_WORD_BOUNDARY),
    ],
)
def test_owned_child_word_limits_and_boundaries_are_not_discarded(command: str, check) -> None:
    source = "# 比較\n\nRun ``sh -c '" + command + "'`` now."
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert diagnostic.check is check
    assert source.index("sh -c") <= diagnostic.start < diagnostic.end <= source.index("'``") + 1


def test_owned_command_inspection_preserves_raw_high_findings() -> None:
    source = 'Run `rm -rf /` and then `sh -c "$PAYLOAD"`.'
    result = tm.node({"components": ["guide.md"], "file_cache": {"guide.md": source}})
    assert any(f.rule_id == "TM1" and f.severity == "HIGH" for f in result["findings"])
    assert any(event["outcome"] == "partial" for event in result["inspection_ledger"])


@pytest.mark.parametrize("wrapper", ["env", "command", "sudo -n", "command -p env"])
def test_executed_wrapper_retains_its_terminal_runtime_command(wrapper: str) -> None:
    body = wrapper + ' "$CMD"'
    source = "Run ``sh -c '" + body + "'``."
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RUNTIME_DEPENDENT
    assert diagnostic.check is ParserCheck.COMMAND_IDENTITY
    assert source.index("sh -c") <= diagnostic.start < diagnostic.end <= source.index("``.")


@pytest.mark.parametrize("body", ['env echo "$CMD"', 'command -v "$CMD"', 'sudo -n echo "$CMD"'])
def test_executed_wrapper_keeps_output_arguments_and_queries_as_data(body: str) -> None:
    source = "Run ``sh -c '" + body + "'``."
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown") is None


def test_effective_command_identity_is_an_optional_executed_context_contract() -> None:
    source = 'env "$CMD"'
    diagnostics = []
    assert tm._command_string_from_clause(source, 0, lambda: None, diagnostics=diagnostics) == (
        False,
        None,
    )
    assert diagnostics == []
    assert tm._command_string_from_clause(
        source, 0, lambda: None, diagnostics=diagnostics, require_command_identity=True
    ) == (True, None)
    assert diagnostics[0].check is ParserCheck.COMMAND_IDENTITY


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize(
    "body",
    [
        "Write-Host -NoNewline \"$($text -replace '%s', 'r m')\"",
        "Write-Output (\"$($text -replace '%s', 'r m')\")",
        "\"$($text -replace '%s', 'r m')\" | Write-Output",
        "$result = \"$($text -replace '%s', 'r m')\"",
    ],
)
def test_existing_powershell_value_owner_is_retained(body: str, newline: str) -> None:
    source = ("# 比較\n\nPowerShell example: ``" + body + "``.\n").replace("\n", newline)
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown") is None


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("characters", [512, 4097])
@pytest.mark.parametrize("pipeline", [False, True])
def test_long_native_replace_value_uses_its_physical_line_owner(
    newline: str, characters: int, pipeline: bool
) -> None:
    replacement = ("data " * ((characters + 4) // 5))[:characters]
    value = "\"$($text -replace '%s', '" + replacement + "')\""
    body = value + " | Write-Output" if pipeline else "Write-Output (" + value + ")"
    source = ("# 比較\n\nPowerShell example: ``" + body + "``.\n").replace("\n", newline)
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown") is None


@pytest.mark.parametrize(
    "body",
    [
        "Write-Host -NoNewline \"$($text -replace '%s', 'r m')\"",
        "Write-Output (\"$($text -replace '%s', 'r m')\")",
        "\"$($text -replace '%s', 'r m')\" | Write-Output",
        "$result = \"$($text -replace '%s', 'r m')\"",
    ],
)
def test_one_value_owner_precedes_whole_word_and_nested_command_reads(
    monkeypatch, body: str
) -> None:
    original = tm._parse_shell_command_word
    parsed_starts = []

    def parse(content, start, *args, **kwargs):
        parsed_starts.append(start)
        return original(content, start, *args, **kwargs)

    monkeypatch.setattr(tm, "_parse_shell_command_word", parse)
    assert tm._owned_inline_command_diagnostic(body, lambda: None, [16 * len(body)]) is None
    opening = body.index("$(")
    assert opening - 1 not in parsed_starts
    assert opening + 2 not in parsed_starts


def test_native_value_proof_obeys_shared_work_allowance() -> None:
    body = "\"$($text -replace '%s', 'r m')\" | Write-Output"
    diagnostic = tm._owned_inline_command_diagnostic(body, lambda: None, [len(body) + 1])
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RESOURCE_LIMIT
    assert diagnostic.check is ParserCheck.MARKDOWN_LITERAL_OWNERSHIP


@pytest.mark.parametrize("separator", ["\r", "\r\n"])
def test_native_replace_value_cannot_own_a_following_inner_statement(separator: str) -> None:
    body = "Write-Output \"$($text -replace '%s', 'r m'" + separator + ' $CMD)"'
    start, end = body.index("$("), body.index(")") + 1
    assert not tm._is_powershell_replace_value_context(body, start, end, body[start + 2 : end - 1])
    diagnostic = tm._owned_inline_command_diagnostic(body, lambda: None, [16 * len(body)])
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RUNTIME_DEPENDENT


@pytest.mark.parametrize("consumer", ["sh -c", "eval"])
def test_explicit_shell_operand_cannot_borrow_native_value_ownership(consumer: str) -> None:
    body = consumer + " 'Write-Output \"$($text -replace x y)\"'"
    source = "Run ``" + body + "``."
    diagnostic = tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RUNTIME_DEPENDENT
    assert diagnostic.check is ParserCheck.COMMAND_IDENTITY
    assert source.index(body) <= diagnostic.start < diagnostic.end <= source.index("``.")


def test_execution_context_directly_controls_the_existing_native_value_owner() -> None:
    body = 'Write-Output "$($text -replace x y)"'
    assert tm._owned_inline_command_diagnostic(body, lambda: None, [16 * len(body)]) is None
    diagnostic = tm._owned_inline_command_diagnostic(
        body, lambda: None, [16 * len(body)], executing=True
    )
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RUNTIME_DEPENDENT
    assert diagnostic.check is ParserCheck.COMMAND_IDENTITY


@pytest.mark.parametrize(
    "body",
    [
        "echo \"$($text -replace '%s', 'r m')\"",
        "Write-Output \"$($text -replace '%s', 'r m'; $CMD)\"",
        "Write-Output \"$($text -replace '%s', $($CMD))\"",
        "Write-Output \"$($text -replace '%s', 'r m')\"; $CMD",
        "Write-Output (\"$($text -replace '%s', 'r m')\"); eval \"$PAYLOAD\"",
        "\"$($text -replace '%s', 'r m')\" | Write-Output; sh -c \"$PAYLOAD\"",
    ],
)
def test_replace_syntax_does_not_hide_executable_or_following_code(body: str) -> None:
    diagnostic = tm.bounded_parse_diagnostic(
        "PowerShell example: ``" + body + "``.", lambda: None, file_type="markdown"
    )
    assert diagnostic is not None


def test_cancellation_inside_owned_command_walk_propagates(monkeypatch) -> None:
    entered = False
    original = tm._owned_inline_command_diagnostic

    def owned(*args, **kwargs):
        nonlocal entered
        entered = True
        return original(*args, **kwargs)

    def expired() -> None:
        if entered:
            raise TimeoutError("owned command deadline")

    monkeypatch.setattr(tm, "_owned_inline_command_diagnostic", owned)
    with pytest.raises(TimeoutError, match="owned command deadline"):
        tm.bounded_parse_diagnostic('Run `sh -c "$PAYLOAD"`.', expired, file_type="markdown")


def test_many_literal_bodies_share_a_finite_work_allowance() -> None:
    source = "Reference `label`. " * 1000
    calls = 0

    def check() -> None:
        nonlocal calls
        calls += 1
        assert calls <= 32 * len(source) + 8192

    assert tm.bounded_parse_diagnostic(source, check, file_type="markdown") is None
    assert calls > 0
