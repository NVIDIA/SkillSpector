# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PowerShell host quoting must not become an unfinished Bash command."""

from __future__ import annotations

import pytest

from skillspector import powershell_tokens
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.powershell_tokens import powershell_source

_TAIL = "\n# ordinary trailing documentation" * 300


def _ledger(content: str, path: str = "sample.ps1") -> dict:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: content}}, [tm]
    )
    return result["inspection_ledger"][0]


@pytest.mark.parametrize(
    "statement",
    [
        '$uri = "$base`?option=value&cl=$change"',
        'Write-Host "line one`nline two"',
        'Write-Output "quoted `"value`""',
        'Write-Output "escaped `$variable"',
        'Write-Output "tab`tvalue"',
        '($lines -join "`n").Trim()',
        '($_ -split "`t")[0]',
        '$text[$index] -eq "`n"',
        '[IO.File]::WriteAllText($path, "alpha`r`nbeta`n")',
        '$text -eq "C:\\tools\\"',
        '$info.Arguments = "-File `"$runnerPath`""',
        '$manualCommand = "powershell -File `"$setupScript`" -User $User"',
        'Write-Output "line one`\nline two"',
        'Write-Output "line one`\r\nline two"',
        'Write-Output "literal `` backtick"',
        'Write-Output "Unicode `u{1F44D}"',
        "Write-Output 'don''t treat this as a shell quote'",
        '$text = @"\nline one`nline two\n"@',
        "$text = @'\nquotes ' and \" remain text\n'@",
        '<# comment with "quotes and a backtick `\n<# nested #>\n#>\nWrite-Host "done"',
        '# comment with "quotes and a backtick `\nWrite-Host "done"',
        '$path = "C:\\tools\\"\nWrite-Host "done"',
        "Write-Host \"$(Write-Output 'nested value') result\"",
        "Write-Host \"apostrophe ' followed by " + "ordinary text " * 400 + '"',
        'python "script.py" `\n  --input "image.yuv" `\n  --width 3840',
        'python "script.py" `\r\n  --input "image.yuv" `\r\n  --width 3840',
    ],
)
def test_powershell_literal_boundaries_remain_complete(statement: str) -> None:
    event = _ledger(statement + _TAIL)
    assert event["outcome"] is LedgerOutcome.COMPLETED


def test_unrelated_native_commands_do_not_invalidate_uri_ownership() -> None:
    content = (
        'Start-Process -FilePath $archiveTool -ArgumentList @("x", "-y")\n'
        '$uri = "$base`?option=value&cl=$change"\n'
        "& $p4.Source print -q -o $destination $depotPath\n"
    ) + _TAIL
    assert _ledger(content)["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("extension", ["ps1", "psm1", "PSM1", "psd1", "PSD1"])
def test_module_and_script_extensions_use_the_same_ownership(extension: str) -> None:
    content = '$uri = "$base`?option=value"' + _TAIL
    assert _ledger(content, f"sample.{extension}")["outcome"] is LedgerOutcome.COMPLETED
    hostile = '& ("r`m") -rf /' + _TAIL
    assert _ledger(hostile, f"sample.{extension}")["outcome"] is LedgerOutcome.PARTIAL
    subexpression = "@{ Value = \"$(bash -c '$CMD -rf /')\" }" + _TAIL
    assert _ledger(subexpression, f"sample.{extension}")["outcome"] is LedgerOutcome.PARTIAL


@pytest.mark.parametrize(
    "statement",
    [
        "bash -c '$CMD -rf /'",
        "& bash -lc '$CMD -rf /'",
        '& "$($CMD)" -rf /',
        '& "r`m" -rf /',
        '& $CMD "`u{2d}rf" "/"',
        '& $CMD "`u{2d}rf" "/$suffix"',
        'rm "`u{2d}rf" "/"',
        '& <# comment #> "r`m" -rf /',
        '. <# comment #> "r`m" -rf /',
        '& ("r`m") -rf /',
        '& (("r`m")) -rf /',
        '& $("r`m") -rf /',
        '& ($("r`m")) -rf /',
        '& $(("r`m")) -rf /',
        '& (Write-Output "r`m") -rf /',
        '. (Write-Output "r`m") -rf /',
        '& <# command expression #> (Write-Output "r`m") -rf /',
        '. (("r`m")) -rf /',
        '& ("r" + "m") -rf /',
        '& (("r") + "m") -rf /',
        '& ("r`m").ToString() -rf /',
        '& ("r`m".ToString()) -rf /',
        '& ("r`m")[0..1] -rf /',
        'bash -c "r`m -rf /"',
        "$uri = \"$base`?option=value\"; bash -c '$CMD -rf /'",
        "$message = \"$(bash -c '$CMD -rf /')\"",
        '$message = "outer $(Write-Host "inner $(bash -c \'$CMD -rf /\')")"',
        "$message = @\"\n$(bash -c '$CMD -rf /')\n\"@",
        "bash `\n -c '$CMD -rf /'",
        "$CMD `\n -rf /",
        "rm `\r\n -rf /",
        '$payload = "r`m -rf /"; Invoke-Expression $payload',
        '$payload = "r`m -rf /"',
        'Write-Host "ordinary"; $CMD -rf /',
        "<# apparent <# nested opener #>\n$CMD -rf /\n#>",
    ],
)
@pytest.mark.parametrize("tail", ["", _TAIL])
def test_executable_shell_and_decoded_destructive_content_stay_partial(
    statement: str, tail: str
) -> None:
    event = _ledger(statement + tail)
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "content",
    [
        'Write-Host "unfinished',
        "$value = @'\nunfinished",
        "<# unfinished",
        'Write-Host "$(unfinished"',
        'Write-Host "bad `u{XYZ}"',
        'Write-Host "bad `u{110000}"',
        'Write-Host "bad `uX"',
        '$text = @"\ntext\n"@invalid',
        "Write-Host “unproven typographic delimiters”",
        'Write-Host "mixed ”; $CMD -rf /; #"',
        "Write-Host 'mixed ’; $CMD -rf /; #'",
        'Write-Host "mixed “; $CMD -rf /; #"',
        "Write-Host 'mixed ‘; $CMD -rf /; #'",
        "b`ash -c '$CMD -rf /'",
        '"$("' * 34 + ")" * 34,
    ],
)
def test_unsupported_ownership_keeps_conservative_fallback(content: str) -> None:
    assert powershell_source(content, lambda: None) is None


def test_subexpression_code_is_not_owned_as_string_data() -> None:
    content = "Write-Host \"before $(Write-Output 'inner'; $CMD -rf /) after\""
    source = powershell_source(content, lambda: None)
    assert source is not None
    assert source.owns(content.index("before"))
    assert source.owns(content.index("inner"))
    assert not source.owns(content.index("$CMD"))
    assert content.index("Write-Output") in source.clause_starts


def test_partial_context_cannot_claim_powershell_quotes() -> None:
    content = '$uri = "$base`?option=value"' + _TAIL
    assert tm.has_bounded_parse_exhaustion(
        content, lambda: None, file_type="powershell", complete_context=False
    )


def test_runtime_cancellation_propagates() -> None:
    class CancelledError(Exception):
        pass

    def cancel() -> None:
        raise CancelledError

    with pytest.raises(CancelledError):
        powershell_source('Write-Host "done"', cancel)


def test_source_offsets_and_decoded_values_preserve_unicode_and_crlf() -> None:
    content = 'Write-Host "👍 prefix"\r\n$uri = "$base`?x=1"\r\n'
    source = powershell_source(content, lambda: None)
    assert source is not None
    target = next(string for string in source.strings if string.value.startswith("$base"))
    assert content[target.start : target.end] == '"$base`?x=1"'
    assert target.value == "$base?x=1"
    assert all(
        end <= next_start for end, next_start in zip(source.ends, source.starts[1:], strict=False)
    )


def test_source_size_and_span_budgets_preserve_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(powershell_tokens, "MAX_PYTHON_AST_SOURCE_CHARS", 20)
    assert powershell_source('Write-Host "' + "x" * 20 + '"', lambda: None) is None
    monkeypatch.setattr(powershell_tokens, "MAX_PYTHON_AST_SOURCE_CHARS", 1000)
    monkeypatch.setattr(powershell_tokens, "_MAX_SPANS", 2)
    assert powershell_source('"a"; "b"; "c"', lambda: None) is None


def test_here_string_footer_does_not_own_following_code() -> None:
    content = '$text = @"\nliteral text\n"@\n$CMD -rf /'
    source = powershell_source(content, lambda: None)
    assert source is not None
    assert not source.owns(content.index("$CMD"))
    assert source.next_boundary(0, len(content)) == content.index('@"')


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("", ""),
        ("\n", ""),
        ("text\n", "text"),
        ("text\n\n", "text\n"),
        ("text \t\n", "text \t"),
        ("first\nlast\n", "first\nlast"),
    ],
)
def test_here_string_value_excludes_only_its_footer_newline(
    quote: str, newline: str, body: str, expected: str
) -> None:
    content = f"$text = @{quote}\n{body}{quote}@".replace("\n", newline)
    source = powershell_source(content, lambda: None)
    assert source is not None
    assert source.strings[0].value == expected.replace("\n", newline)
    assert content[source.strings[0].start : source.strings[0].end] == content[8:]


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_here_string_footer_keeps_an_escaped_body_newline(newline: str) -> None:
    content = '$text = @"\ntext`n\n"@'.replace("\n", newline)
    source = powershell_source(content, lambda: None)
    assert source is not None
    assert source.strings[0].value == "text\n"


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("following", ["", "continued\n"])
def test_here_string_escaped_physical_newline_cannot_prove_a_footer(
    newline: str, following: str
) -> None:
    content = ('$text = @"\nbody`\n' + following + '"@').replace("\n", newline)
    assert powershell_source(content, lambda: None) is None


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("parenthesized", [0, 1, 2])
@pytest.mark.parametrize("command,partial", [("/bin/rm", True), ("/bin/echo", False)])
@pytest.mark.parametrize("suffix", ["", _TAIL], ids=["short", "padded"])
def test_invoked_here_string_keeps_following_operands(
    quote: str, newline: str, parenthesized: int, command: str, partial: bool, suffix: str
) -> None:
    literal = f"@{quote}\n{command}\n{quote}@".replace("\n", newline)
    expression = "(" * parenthesized + literal + ")" * parenthesized
    content = f"& {expression} -rf /" + suffix
    source = powershell_source(content, lambda: None)
    assert source is not None
    string = source.strings[0]
    assert string.value == command
    assert string.invoked
    assert string.operand_start is not None
    assert content[string.operand_start :].startswith(" -rf /")
    event = _ledger(content)
    assert event["outcome"] is (LedgerOutcome.PARTIAL if partial else LedgerOutcome.COMPLETED)
    if partial:
        assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "expression",
    [
        "(@'\n/bin/echo\n'@; $CMD)",
        "(@'\n/bin/echo\n'@ + $suffix)",
        "(@'\n/bin/echo\n'@)[0]",
        "(@'\n/bin/echo\n'@).ToString()",
        '(@"\n$CMD\n"@)',
        '(@"\n$(Get-Command)\n"@)',
        "$(if ($ready) { @'\n/bin/echo\n'@ } else { $CMD })",
    ],
)
@pytest.mark.parametrize("suffix", ["", _TAIL], ids=["short", "padded"])
def test_parenthesized_here_string_cannot_prove_runtime_selection(
    expression: str, suffix: str
) -> None:
    content = f"& {expression} -rf /" + suffix
    event = _ledger(content)
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize("name", ["text", "script:text"])
def test_large_native_here_string_assignment_is_not_a_shell_command(name: str) -> None:
    content = f"    ${name} = @'\n" + "ordinary documentation\n" * 500 + "'@\n"
    assert _ledger(content)["outcome"] is LedgerOutcome.COMPLETED


def test_assignment_ownership_preserves_nested_runtime_shell_consumer() -> None:
    content = "$text = \"$(bash -c '$CMD -rf /')\"\n"
    assert _ledger(content)["outcome"] is LedgerOutcome.PARTIAL


def test_call_operator_operand_does_not_borrow_assignment_ownership() -> None:
    content = "& $CMD = @'\n" + "ordinary documentation\n" * 500 + "'@\n"
    assert _ledger(content)["outcome"] is LedgerOutcome.PARTIAL


@pytest.mark.parametrize("name", ["x" * 513, "config." + "x" * 513, "naïve"])
def test_incomplete_executable_name_slice_cannot_claim_operand_ownership(name: str) -> None:
    assert powershell_source(f"& ${name} -c '$CMD -rf /'\n", lambda: None) is None


@pytest.mark.parametrize(
    "expression",
    [
        "$runner.GetShell()",
        "$programs[$index]",
        "($runner).GetShell()",
        "($programs)[$index]",
        "${shell-path}",
        '"$program".GetShell()',
    ],
)
@pytest.mark.parametrize("path", ["sample.ps1", "SKILL.md"])
@pytest.mark.parametrize("prefix", ["", "# Invoke-Expression is documented here\n"])
def test_unsupported_executable_selection_stays_partial(
    expression: str, path: str, prefix: str
) -> None:
    content = prefix + f"& {expression} -c '$CMD -rf /'\n"
    if path == "SKILL.md":
        content = f"# Instructions\n\n```powershell\n{content}```\n"
    event = _ledger(content, path)
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize(
    "expression",
    [
        '(Write-Output "bash")',
        "($program)",
        "$(Get-Shell)",
        "$program",
        "${script:program}",
        "$programs[0]",
        "$config.Executable",
        r"$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe",
        "$root/bin/bash",
        '"$program"',
        '"${script:program}"',
        '"$($config.Executable)"',
    ],
)
@pytest.mark.parametrize(
    "option",
    [
        "-c",
        "-lc",
        "-ac",
        "-xc",
        "-C",
        "-CO",
        "-Command",
        "-EncodedCommand",
        "-o emacs -c",
        "-ExecutionPolicy Bypass -Command",
    ],
)
def test_expression_selected_shell_keeps_code_operands_partial(
    expression: str, option: str
) -> None:
    content = f"& {expression} {option} '$CMD -rf /'\n"
    assert _ledger(content)["outcome"] is LedgerOutcome.PARTIAL


@pytest.mark.parametrize(
    "arguments",
    [
        "-Change 123 -OutputPath $outputPath",
        "-NonInteractive",
        "-latest -products '*' -property installationPath",
        "files -e $pattern",
    ],
)
def test_native_options_are_not_misread_as_shell_code_flags(arguments: str) -> None:
    assert _ledger(f"& $program {arguments}\n")["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("path", ["sample.ps1", "SKILL.md"])
def test_native_path_suffix_after_variable_preserves_argument_boundaries(path: str) -> None:
    content = r"& $env:SystemRoot\System32\bcdedit.exe /enum '{current}' 2>&1" + "\n"
    if path == "SKILL.md":
        content = f"# Instructions\n\n```powershell\n{content}```\n"
    assert _ledger(content, path)["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("expression", ['"bash"', "'bash'", '"ba`sh"'])
def test_literal_quoted_shell_command_keeps_inspectable_code(expression: str) -> None:
    content = f"& {expression} -c 'printf safe'\n"
    assert _ledger(content)["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("language", ["powershell", "pwsh", "ps1", "PowerShell"])
@pytest.mark.parametrize("marker", ["```", "~~~~"])
def test_markdown_powershell_fences_preserve_host_quotes(language: str, marker: str) -> None:
    content = (
        f"# Instructions\n\n{marker}{language}\n"
        '$uri = "$base`?option=value"\n'
        'python "script.py" `\n  --input "image.yuv"\n'
        f"{marker}\n\n" + "Ordinary trailing explanation.\n" * 300
    )
    assert _ledger(content, "SKILL.md")["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "command",
    [
        "bash -c '$CMD -rf /'",
        "$message = \"$(bash -c '$CMD -rf /')\"",
        "rm `\n -rf /",
        '& "r`m" -rf /',
        '& (Write-Output "r`m") -rf /',
    ],
)
def test_markdown_powershell_fences_keep_executable_payloads_partial(command: str) -> None:
    content = f"# Instructions\n\n```powershell\n{command}\n```\n"
    assert _ledger(content, "SKILL.md")["outcome"] is LedgerOutcome.PARTIAL


@pytest.mark.parametrize("language", ["powershell", "pwsh", "ps1"])
@pytest.mark.parametrize(
    ("command", "partial"),
    [
        ("Write-Output '$CMD -rf /' | bash", True),
        ("$cmd = Write-Output '$CMD -rf /'", True),
        ("Write-Host '$CMD -rf /' 6>&1 | bash", True),
        ("Import-Module ./helpers.psm1\nWrite-Host '$CMD -rf /'", True),
        ("Write-Output '$CMD -rf /'", False),
        ("Write-Host '$CMD -rf /'", False),
    ],
)
def test_fenced_powershell_values_preserve_capture_and_display_contracts(
    language: str, command: str, partial: bool
) -> None:
    content = f"```{language}\n{command}\n```\n"
    event = _ledger(content, "SKILL.md")

    assert (event["outcome"] is LedgerOutcome.PARTIAL) is partial


@pytest.mark.parametrize(
    ("setup", "partial"),
    [
        ("Import-Module ./helpers.psm1", True),
        ("function Write-Host { bash -c $args[0] }", True),
        ("Write-Output 'ordinary setup'", False),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_powershell_fences_do_not_reset_display_command_identity(
    setup: str, partial: bool, reverse: bool
) -> None:
    blocks = [f"```powershell\n{setup}\n```\n", "```powershell\nWrite-Host '$CMD -rf /'\n```\n"]
    if reverse:
        blocks.reverse()
    event = _ledger("\n".join(blocks), "SKILL.md")

    assert (event["outcome"] is LedgerOutcome.PARTIAL) is partial


@pytest.mark.parametrize("prior_fence", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_nested_powershell_fence_owns_only_its_current_body(
    prior_fence: bool, newline: str
) -> None:
    prior = "```powershell\nWrite-Host 'first'\n```\n\n" if prior_fence else "Intro.\n\n"
    body = "  Write-Host 'second'\n"
    content = (prior + "- Instructions\n  ```powershell\n" + body + "  ```\n").replace(
        "\n", newline
    )
    blocks: list[tuple[int, int]] = []
    tm._markdown_shell_text(content, lambda: None, powershell_blocks=blocks)
    start = content.index(body.replace("\n", newline))

    assert blocks[-1] == (start, start + len(body.replace("\n", newline)))
    assert all(
        left_end <= right_start
        for (_, left_end), (right_start, _) in zip(blocks, blocks[1:], strict=False)
    )
    assert _ledger(content, "SKILL.md")["outcome"] is LedgerOutcome.COMPLETED
    hostile = content + "\n```bash\nbash -c '$CMD -rf /'\n```\n"
    assert _ledger(hostile, "SKILL.md")["outcome"] is LedgerOutcome.PARTIAL


def test_fence_language_and_closure_are_required_for_ownership() -> None:
    statement = '$uri = "$base`?option=value"\n'
    tail = "Ordinary trailing explanation.\n" * 300
    for content in (
        "```bash\n" + statement + "```\n" + tail,
        "```powershell\n" + statement + tail,
    ):
        assert _ledger(content, "SKILL.md")["outcome"] is LedgerOutcome.PARTIAL


@pytest.mark.parametrize("reverse", [False, True])
def test_markdown_fence_offsets_preserve_following_shell_and_raw_findings(reverse: bool) -> None:
    host = '```powershell\r\n$uri = "$base`?option=value"\r\n```\r\n'
    shell = "```bash\r\nbash -c '$CMD -rf /'\r\n```\r\n"
    blocks = [host, shell] if not reverse else [shell, host]
    content = "# 👍 instructions\r\n\r\n" + "\r\n".join(blocks)
    assert _ledger(content, "SKILL.md")["outcome"] is LedgerOutcome.PARTIAL
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content + "\nrm -rf /\n"}},
        [tm],
    )
    assert result["findings"]
