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
