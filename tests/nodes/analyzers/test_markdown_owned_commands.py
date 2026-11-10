# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native inline bodies retain their executable child-command coverage."""

from __future__ import annotations

import json

import pytest

from skillspector.inspection_ledger import LedgerOutcome, ParserCause, ParserCheck
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner


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
    ['"$CMD', '"$(resolve_tool)', "$(unfinished", '"${unfinished', "`unfinished"],
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
        ('"${unfinished', ParserCheck.EXPANSION_BOUNDARY),
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


@pytest.mark.parametrize(
    "value",
    [
        "$(CC)",
        "${CFLAGS}",
        "$(error)",
        "$(error unsupported option)",
        "$(strip $(EXTRA_FLAGS))",
        "$(if $(DEBUG),-g,-O2)",
        "$(foreach target,$(TARGETS),$(target))",
        " \t$(CC) \t",
    ],
)
def test_whole_native_display_values_do_not_independently_prove_execution(value: str) -> None:
    assert tm._is_inline_make_display_value(value, lambda: None, [8 * len(value)])
    assert tm._owned_inline_command_diagnostic(value, lambda: None, [8 * len(value)]) is None
    diagnostic = tm._owned_inline_command_diagnostic(
        value, lambda: None, [16 * len(value)], executing=True
    )
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RUNTIME_DEPENDENT


@pytest.mark.parametrize(
    "value",
    [
        "$(shell)",
        "$(shell echo done)",
        "$(file)",
        "$(eval)",
        "$(call)",
        "$(guile)",
        "$(unknown value)",
        "$($(NAME))",
        "$(if yes,done,$(shell echo done))",
        "$(strip $(call tool))",
        "$(strip $(guile value))",
    ],
)
def test_execution_capable_unknown_or_indirect_expressions_keep_fallback(value: str) -> None:
    assert not tm._is_inline_make_display_value(value, lambda: None, [8 * len(value)])
    diagnostic = tm.bounded_parse_diagnostic(
        "Reference ``" + value + "``.", lambda: None, file_type="markdown"
    )
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RUNTIME_DEPENDENT


@pytest.mark.parametrize(
    "value",
    [
        "$(CC) -rf /",
        "$(CC)suffix",
        "$(strip $(VAR))suffix",
        "$(strip $(VAR); sh -c $PAYLOAD)",
        "$(foreach $(NAME),items,data)",
        "$(if yes,done,other,extra)",
        "$(strip $(VAR)",
        "$(strip 'quoted value')",
    ],
)
def test_display_value_ownership_requires_the_entire_supported_expression(value: str) -> None:
    assert not tm._is_inline_make_display_value(value, lambda: None, [8 * len(value)])


def test_display_value_cannot_bypass_work_depth_or_deadline(monkeypatch) -> None:
    value = "$(strip $(FLAGS))"
    for work, depth in [(len(value), 0), (8 * len(value), 8)]:
        diagnostic = tm._owned_inline_command_diagnostic(value, lambda: None, [work], depth=depth)
        assert diagnostic is not None
        assert diagnostic.cause is ParserCause.RESOURCE_LIMIT
    entered = False
    proof = tm._is_inline_make_display_value

    def display_value(*args, **kwargs):
        nonlocal entered
        entered = True
        return proof(*args, **kwargs)

    def expired() -> None:
        if entered:
            raise TimeoutError("display value deadline")

    monkeypatch.setattr(tm, "_is_inline_make_display_value", display_value)
    with pytest.raises(TimeoutError, match="display value deadline"):
        tm._owned_inline_command_diagnostic(value, expired, [8 * len(value)])


@pytest.mark.parametrize(
    "command",
    ["Test-Path", "Join-Path", "Split-Path", "Resolve-Path", "Convert-Path", "Write-Host"],
)
@pytest.mark.parametrize("member", ["$_.FullName", "$item.Parent.Name"])
@pytest.mark.parametrize("prefix", ["", "prefix/"])
def test_native_quoted_property_arguments_keep_value_ownership(
    command: str, member: str, prefix: str
) -> None:
    body = command + ' "' + prefix + "$(" + member + ')/item"'
    end = len(body)
    if command == "Join-Path":
        body += ' "leaf"'
    assert tm._powershell_property_argument_spans(body, lambda: None, [8 * len(body)]) == []
    spans = tm._powershell_property_argument_spans(
        body, lambda: None, [8 * len(body)], bindings_proven=True
    )
    assert spans == [(body.index('"'), end)]
    assert (
        tm.bounded_parse_diagnostic(
            "Reference ``" + body + "``.", lambda: None, file_type="markdown"
        )
        is None
    )
    diagnostic = tm._owned_inline_command_diagnostic(
        body, lambda: None, [16 * len(body)], executing=True
    )
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RUNTIME_DEPENDENT


@pytest.mark.parametrize(
    "expression",
    [
        "$item.Method()",
        "$item.Items[0]",
        "$item.Name + $CMD",
        "$item.Name; $CMD",
        "$item.Name\r$CMD",
        "$item.Name\n$CMD",
    ],
)
def test_property_value_owner_excludes_calls_operators_and_statements(expression: str) -> None:
    body = 'Test-Path "$(' + expression + ')"'
    assert (
        tm._powershell_property_argument_spans(
            body, lambda: None, [8 * len(body)], bindings_proven=True
        )
        == []
    )
    assert (
        tm.bounded_parse_diagnostic(
            "Reference ``" + body + "``.", lambda: None, file_type="markdown"
        )
        is not None
    )


def test_property_value_does_not_own_following_execution_or_raw_findings() -> None:
    body = 'Test-Path "$($item.Name)"; sh -c "$PAYLOAD"'
    diagnostic = tm.bounded_parse_diagnostic(
        "Reference ``" + body + "``.", lambda: None, file_type="markdown"
    )
    assert diagnostic is not None
    assert diagnostic.start >= len("Reference ``") + body.index("sh -c")
    source = "Reference `$(error rm -rf /)`."
    result = tm.node({"components": ["guide.md"], "file_cache": {"guide.md": source}})
    assert any(
        f.rule_id == "TM1"
        and f.severity == "HIGH"
        and f.file == "guide.md"
        and "rm -rf /" in (f.matched_text or "")
        for f in result["findings"]
    )


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c",
        "powershell -Command",
        "pwsh -Command",
        "python -c",
        "node -e",
        "perl -e",
        "Compare-Item",
        "sh -c",
        "bash -c",
        "eval",
    ],
)
@pytest.mark.parametrize("prefix", ["", "prefix/"])
def test_property_shape_does_not_prove_unknown_or_executed_argument_roles(
    command: str, prefix: str
) -> None:
    body = command + ' "' + prefix + '$($item.Name)/item"'
    assert (
        tm._powershell_property_argument_spans(
            body, lambda: None, [8 * len(body)], bindings_proven=True
        )
        == []
    )
    diagnostic = tm.bounded_parse_diagnostic(
        "Reference ``" + body + "``.", lambda: None, file_type="markdown"
    )
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RUNTIME_DEPENDENT


@pytest.mark.parametrize(
    "body",
    [
        '"$($item.Name)"',
        '$result = Test-Path "$($item.Name)"',
        'Test-Path "$($item.Name)" | Invoke-Expression',
    ],
)
def test_property_arguments_cannot_own_dynamic_heads_or_captured_values(body: str) -> None:
    assert (
        tm._powershell_property_argument_spans(
            body, lambda: None, [8 * len(body)], bindings_proven=True
        )
        == []
    )
    assert (
        tm.bounded_parse_diagnostic(
            "Reference ``" + body + "``.", lambda: None, file_type="markdown"
        )
        is not None
    )


@pytest.mark.parametrize(
    "binding",
    [
        "Set-Alias Test-Path unknown",
        "function Test-Path {}",
        "filter Test-Path {}",
        "workflow Test-Path {}",
        "Import-Module unknown",
        ". helper.ps1",
        "Im`port-`Module unknown",
        "Add-Member -InputObject $item -MemberType ScriptProperty -Name Name "
        "-Value { $getter.Invoke() }",
        "Update-TypeData -TypeName Example -MemberType ScriptProperty -MemberName Name "
        "-Value { $getter.Invoke() }",
        "Remove-TypeData -TypeName Example",
        "Add-Type 'class Example {}'",
        "New-Object PSCodeProperty Name $getter",
        "Import-Clixml item.xml",
    ],
)
def test_property_bindings_must_hold_across_the_whole_original_markdown(
    monkeypatch, binding: str
) -> None:
    body = 'Test-Path "$($item.Name)"'
    observed = []
    original = tm._powershell_property_argument_spans

    def spans(content, *args, **kwargs):
        if content == body:
            observed.append(kwargs.get("bindings_proven", False))
        return original(content, *args, **kwargs)

    monkeypatch.setattr(tm, "_powershell_property_argument_spans", spans)
    source = "Earlier ``" + binding + "``.\n\nReference ``" + body + "``."
    assert tm.bounded_parse_diagnostic(source, lambda: None, file_type="markdown") is not None
    assert observed == [False]


_PATH_VALUE_DOCUMENT = (
    'Path values\n\nReference ``Test-Path "$($_.FullName)/item"``.\n'
    'Reference ``Test-Path "prefix/$($item.Name)/item"``.\n'
    'Reference ``Join-Path "$($item.Parent.Name)/item" "leaf"``.\n'
    'Reference ``Split-Path "$($item.Name)/item"``.\n'
    'Reference ``Resolve-Path "$($item.Name)/item"``.\n'
    'Reference ``Convert-Path "$($item.Name)/item"``.\n'
    'Reference ``Write-Host "$($item.Name)"``.\n'
)


@pytest.mark.parametrize("prefix", ["", "# 比較 🧪\n\n"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_multiple_native_property_values_keep_complete_normal_caller_coverage(
    prefix: str, newline: str
) -> None:
    assert len(_PATH_VALUE_DOCUMENT) == 360
    content = (prefix + _PATH_VALUE_DOCUMENT).replace("\n", newline)
    path = "references/paths.md"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: content}}, [tm]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert result["findings"] == []


@pytest.mark.parametrize("count", [1, 10, 100])
@pytest.mark.parametrize("prefix", ["", "prefix/"])
def test_repeated_property_arguments_fit_the_existing_shared_work(count: int, prefix: str) -> None:
    body = 'Test-Path "' + prefix + "$($item.Name)/" + "p" * 1000 + '"'
    content = "\n\n".join("`" + body + "`" for _ in range(count))
    assert len(content) < static_runner.SECURITY_VIEW_WINDOW_CHARS
    assert tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown") is None
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["paths.md"], "file_cache": {"paths.md": content}}, [tm]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert result["findings"] == []


@pytest.mark.parametrize(
    "body,property_count,replace_probed",
    [
        ('Join-Path "$($item.Parent.Name)" "$($item.Name)"', 2, False),
        ("Write-Output \"$($text -replace '%s', 'r m')\"", 0, True),
    ],
)
def test_proven_property_arguments_skip_only_the_redundant_replace_probe(
    monkeypatch, body: str, property_count: int, replace_probed: bool
) -> None:
    spans = tuple(
        tm._powershell_property_argument_spans(
            body, lambda: None, [8 * len(body) + 8192], bindings_proven=True
        )
    )
    assert len(spans) == property_count
    observed = []
    original = tm._is_powershell_replace_value_context

    def replace_value(*args, **kwargs):
        observed.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(tm, "_is_powershell_replace_value_context", replace_value)
    assert (
        tm._owned_inline_command_diagnostic(
            body, lambda: None, [8 * len(body) + 8192], property_argument_spans=spans
        )
        is None
    )
    assert bool(observed) is replace_probed


def test_repeated_property_arguments_keep_later_runtime_and_raw_high() -> None:
    body = 'Test-Path "$($item.Name)/' + "p" * 1000 + '"'
    content = "\n\n".join("`" + body + "`" for _ in range(10))
    content += '\n\nRun `sh -c "$PAYLOAD"`. Run `rm -rf /`.'
    diagnostic = tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RUNTIME_DEPENDENT
    assert content.index("sh -c") <= diagnostic.start < diagnostic.end <= len(content)
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["paths.md"], "file_cache": {"paths.md": content}}, [tm]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert any(
        finding.rule_id == "TM1"
        and finding.severity == "HIGH"
        and finding.file == "paths.md"
        and finding.matched_text == "rm -rf /"
        for finding in result["findings"]
    )


def test_property_structural_reads_are_exact_bounded_quoted_arguments(monkeypatch) -> None:
    observed = []
    original = tm._parse_shell_command_word

    def parse(content, start, *args, **kwargs):
        if kwargs.get("native_argument", False):
            observed.append((content, start))
        return original(content, start, *args, **kwargs)

    monkeypatch.setattr(tm, "_parse_shell_command_word", parse)
    assert (
        tm.bounded_parse_diagnostic(_PATH_VALUE_DOCUMENT, lambda: None, file_type="markdown")
        is None
    )
    assert observed == [
        ('"$($_.FullName)/item"', 0),
        ('"prefix/$($item.Name)/item"', 0),
        ('"$($item.Parent.Name)/item"', 0),
        ('"$($item.Name)/item"', 0),
        ('"$($item.Name)/item"', 0),
        ('"$($item.Name)/item"', 0),
        ('"$($item.Name)"', 0),
    ]


def test_multiple_property_arguments_in_one_native_body_keep_their_own_ends() -> None:
    content = (
        'Reference ``Join-Path "prefix/$($item.Parent.Name)" "$($item.Name)/child"``.\n'
        + _PATH_VALUE_DOCUMENT
    )
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}, [tm]
    )
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED
    assert result["findings"] == []


@pytest.mark.parametrize("end_delta", [-1, 1])
def test_native_argument_word_must_match_the_entire_proven_endpoint(end_delta: int) -> None:
    content = 'Test-Path "prefix/$($item.Name)/item" later'
    begin = content.index('"')
    end = content.index('"', begin + 1) + 1 + end_delta
    diagnostic = tm._has_shell_command_word_exhaustion(
        content,
        lambda: None,
        native_argument_spans=((begin, end),),
        native_argument_work=[8 * len(content)],
    )
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.BOUNDARY_UNRESOLVED
    assert diagnostic.check is ParserCheck.COMMAND_WORD_BOUNDARY
    assert begin <= diagnostic.start < diagnostic.end <= end


def test_native_argument_word_read_cannot_exceed_shared_work() -> None:
    content = 'Test-Path "prefix/$($item.Name)/item"'
    begin, end = content.index('"'), len(content)
    diagnostic = tm._has_shell_command_word_exhaustion(
        content,
        lambda: None,
        native_argument_spans=((begin, end),),
        native_argument_work=[2 * (end - begin) - 1],
    )
    assert diagnostic is not None
    assert diagnostic.cause is ParserCause.RESOURCE_LIMIT
    assert diagnostic.check is ParserCheck.MARKDOWN_LITERAL_OWNERSHIP


@pytest.mark.parametrize(
    "following",
    [
        'Run ``sh -c "$PAYLOAD"``.',
        'Run ``eval "$PAYLOAD"``.',
        'Run ``sh`` -c "$PAYLOAD".',
        'Run ``cmd /c "$($item.Name)"``.',
        'Run ``pwsh -Command "$($item.Name)"``.',
        'Reference ``Compare-Item "$($item.Name)"``.',
        'Reference ``Test-Path "$($item.Method())"``.',
        'Reference ``Test-Path "$($item.Items[0])"``.',
        '```sh\nsh -c "$PAYLOAD"\n```',
    ],
)
def test_property_argument_transport_keeps_other_executable_scopes(following: str) -> None:
    content = _PATH_VALUE_DOCUMENT + "\n" + following
    diagnostic = tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown")
    assert diagnostic is not None
    assert len(_PATH_VALUE_DOCUMENT) <= diagnostic.start < diagnostic.end <= len(content)


@pytest.mark.parametrize(
    "binding",
    [
        "function Test-Path {}",
        "Set-Alias Test-Path unknown",
        ". helper.ps1",
        "Add-Member -InputObject $item -MemberType ScriptProperty -Name Name "
        "-Value { $getter.Invoke() }",
    ],
)
def test_cross_span_bindings_disable_structural_argument_transport(
    monkeypatch, binding: str
) -> None:
    transported = []
    original = tm._parse_shell_command_word

    def parse(content, start, *args, **kwargs):
        if kwargs.get("native_argument", False):
            transported.append(content)
        return original(content, start, *args, **kwargs)

    monkeypatch.setattr(tm, "_parse_shell_command_word", parse)
    content = "Earlier ``" + binding + "``.\n\n" + _PATH_VALUE_DOCUMENT
    assert tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown") is not None
    assert transported == []


@pytest.mark.parametrize("wrapper", ["sh -c", "eval"])
def test_executed_children_receive_no_native_property_argument_grant(
    monkeypatch, wrapper: str
) -> None:
    child = 'Test-Path "prefix/$($item.Name)/item"'
    observed = []
    original = tm._has_shell_command_word_exhaustion

    def exhaustion(content, *args, **kwargs):
        if content == child:
            observed.append(kwargs.get("native_argument_spans", ()))
        return original(content, *args, **kwargs)

    monkeypatch.setattr(tm, "_has_shell_command_word_exhaustion", exhaustion)
    content = _PATH_VALUE_DOCUMENT + "\nRun ``" + wrapper + " '" + child + "'``."
    assert tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown") is not None
    assert observed and all(spans == () for spans in observed)


@pytest.mark.parametrize("length", [4096, 4097])
def test_native_property_transport_retains_independent_word_limits(length: int) -> None:
    content = _PATH_VALUE_DOCUMENT + "\nReference `r" + "a" * (length - 1) + "`."
    diagnostic = tm.bounded_parse_diagnostic(content, lambda: None, file_type="markdown")
    if length == 4096:
        assert diagnostic is None
    else:
        assert diagnostic is not None
        assert diagnostic.cause is ParserCause.RESOURCE_LIMIT
        assert diagnostic.check is ParserCheck.COMMAND_WORD_CHARACTERS


def test_native_property_transport_preserves_raw_destructive_source() -> None:
    path = "references/paths.md"
    content = _PATH_VALUE_DOCUMENT + "\nRun `rm -rf /`.\n"
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: content}}, [tm]
    )
    findings = [finding for finding in result["findings"] if finding.rule_id == "TM1"]
    assert findings
    assert all(finding.severity == "HIGH" and finding.file == path for finding in findings)
    assert all(finding.start_line == 11 and finding.start_column == 5 for finding in findings)
    assert any(finding.matched_text == "rm -rf /" for finding in findings)


def test_cancellation_during_native_property_argument_read_propagates(monkeypatch) -> None:
    entered = False
    original = tm._parse_shell_command_word

    def parse(content, start, *args, **kwargs):
        nonlocal entered
        entered = entered or kwargs.get("native_argument", False)
        return original(content, start, *args, **kwargs)

    def expired() -> None:
        if entered:
            raise TimeoutError("native argument deadline")

    monkeypatch.setattr(tm, "_parse_shell_command_word", parse)
    with pytest.raises(TimeoutError, match="native argument deadline"):
        tm.bounded_parse_diagnostic(_PATH_VALUE_DOCUMENT, expired, file_type="markdown")
