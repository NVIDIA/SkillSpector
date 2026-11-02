# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PowerShell host syntax must not invent shell parse failures (#711)."""

from pathlib import Path

import pytest

from skillspector.graph import graph
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner


def _scan(content: str) -> dict:
    return static_runner.run_static_patterns_with_ledger(
        {"components": ["scripts/helper.ps1"], "file_cache": {"scripts/helper.ps1": content}},
        [tm],
    )


def _limited(content: str) -> bool:
    return tm.has_bounded_parse_exhaustion(content, lambda: None, file_type="powershell")


_HELP_SCRIPT = """<#
.SYNOPSIS
  Cleans the build cache. It won't touch sources; never run rm -rf $( yourself.
#>
param([string]$Path = '.cache')
# Don't delete anything outside $Path - rm -rf isn't needed here.
Write-Host "Cleaning $Path... it's safe."
Write-Verbose 'Use rm -rf only if you''re sure'
Get-ChildItem -Path $Path `
  -Recurse | Remove-Item -Force
"""


@pytest.mark.parametrize(
    "content",
    [
        _HELP_SCRIPT,
        "# Don't run rm -rf `here\nWrite-Host 'ok'\n",
        "<#\n rm -rf won't run\n#>\nWrite-Host 'ok'\n",
        "Write-Host 'Never run rm -rf $(whoami) here'\n",
        "Write-Host 'Tip: rm -rf won''t help'\n",
        'Write-Host "rm -rf $($x.Name) won\'t"\n',
        "rm -rf `\n  $Path\n",
        "if ($help) {\n    Write-Host 'Use rm -rf won''t help'\n}\n",
        "Get-Item x | Write-Host 'Never run rm -rf $(whoami) here'\n",
        "Write-Output 'Use rm -rf won''t help'\n",
        "Write-Host 'a',\n  'Never run rm -rf $(whoami) here'\n",
    ],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_powershell_comments_and_messages_have_complete_analysis(
    content: str, newline: str
) -> None:
    content = content.replace("\n", newline)
    # Each case exercises a host-syntax false positive present before #711.
    assert tm.has_bounded_parse_exhaustion(content, lambda: None, file_type="shell") is True

    result = _scan(content)

    assert _limited(content) is False
    assert all(row["outcome"] is LedgerOutcome.COMPLETED for row in result["inspection_ledger"])


def test_referenced_powershell_help_is_complete(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "SKILL.md").write_text(
        "---\nname: ps-help\ndescription: Clean a build cache.\n---\n\nRun `scripts/helper.ps1`.\n"
    )
    (tmp_path / "scripts/helper.ps1").write_text(_HELP_SCRIPT)

    result = graph.invoke({"input_path": str(tmp_path), "use_llm": False, "output_format": "json"})

    assert result["analysis_completeness"]["status"] == "complete"
    assert not any(finding.rule_id == "AE1" for finding in result["findings"])


@pytest.mark.parametrize(
    "content",
    [
        "bash -c 'rm -rf /'\n",
        "& bash -c 'rm -rf /'\n",
        "Invoke-Expression 'rm -rf /'\n",
        "[scriptblock]::Create('rm -rf /').Invoke()\n",
        "Write-Host \"Result: $(bash -c 'rm -rf /')\"\n",
        "# cleanup\nrm -rf /\n",
    ],
)
def test_executable_powershell_payloads_remain_visible(content: str) -> None:
    result = _scan(content)

    assert any(finding.rule_id == "TM1" for finding in result["findings"])


@pytest.mark.parametrize(
    "content",
    [
        # Unclosed strings, here-strings and payloads.
        'bash -c "rm -rf $(\n',
        'Invoke-Expression "rm -rf `$(foo"\n',
        "$s = @'\nrm -rf $(\n",
        "& { rm -rf $(\n",
        "1..3 | ForEach-Object -Parallel { rm -rf $( }\n",
        # A message whose value reaches code is not a message.
        "Write-Output 'rm -rf $(' | bash\n",
        "Write-Output 'rm -rf $('\n| bash\n",
        "$cmd = Write-Output 'rm -rf $('\n",
        "$x = Get-Item . && Write-Output 'rm -rf $('\n",
        "bash -c \"$(Write-Output 'rm -rf $(')\"\n",
        "Write-Host 'rm -rf $(' 6>&1 | bash\n",
        "Write-Host 'rm -rf $(' -InformationVariable cmd\n",
        "Write-Host 'rm -rf $(' -iv cmd\n",
        "Write-Host @params 'rm -rf $('\n",
        "Write-Host 'rm -rf $('.ForEach({ bash -c $_ })\n",
        "& Write-Host 'rm -rf $('\n",
        "@{ Write-Host = 'rm -rf $(' }\n",
        # The file can rebind the message command or run computed code.
        "Set-Alias Write-Host bash\nWrite-Host 'rm -rf $('\n",
        "function Write-Host { bash -c $args[0] }\nWrite-Host 'rm -rf $('\n",
        "Import-Module ./helpers.psm1\nWrite-Host 'rm -rf $('\n",
        ". ./helpers.ps1\nWrite-Host 'rm -rf $('\n",
        "$PSDefaultParameterValues['Write-Host:InformationVariable'] = 'c'\n"
        "Write-Host 'rm -rf $('\n",
        # Tokens whose meaning depends on parser mode keep the raw source.
        "Write-Host --% 'rm -rf $(' # it's\n",
        "Write-Host x#it's\nWrite-Host 'rm -rf $('\n",
        "Write-Host ` 'rm -rf $('[0]\n",
        "Write-Host `, # c\n'rm -rf $('\n",
    ],
)
def test_unproven_powershell_ownership_keeps_partial_status(content: str) -> None:
    assert _limited(content) is True
    result = _scan(content)
    assert any(
        row["outcome"] is LedgerOutcome.PARTIAL
        and row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
        for row in result["inspection_ledger"]
    )


@pytest.mark.parametrize(
    "content",
    [
        "Write-Warning \u2018rm -rf won\u2019\u2019t $(help\u2019\n",
        "Write-Host @'\nrm -rf won't $(x\n'@\n",
        'Write-Host @"\nrm -rf won`"t $(Get-Date)\n"@\n',
    ],
)
def test_typographic_quotes_and_here_string_messages_are_owned(content: str) -> None:
    projected = tm._powershell_shell_text(content, lambda: None)

    assert "rm -rf" not in projected
    assert ("$(Get-Date)" in projected) == ("$(Get-Date)" in content)
    assert _limited(content) is False


def test_projection_preserves_offsets_and_keeps_message_subexpressions() -> None:
    content = "Write-Host \"it's $(bash -c 'id') done\" # won't\r\nGet-Item `\r\n  x\r\n"

    projected = tm._powershell_shell_text(content, lambda: None)

    assert len(projected) == len(content)
    assert [i for i, c in enumerate(projected) if c in "\r\n"] == [
        i for i, c in enumerate(content) if c in "\r\n"
    ]
    assert "$(bash -c 'id')" in projected
    assert "it's" not in projected and "won't" not in projected
    assert "Get-Item \\\r\n" in projected


@pytest.mark.parametrize(
    "content",
    [
        "Write-Host 'unclosed\n",
        "<# unclosed\nWrite-Host 'x'\n",
        "Write-Host @'\nno closer\n",
        "Write-Host (1\n",
        "Write-Host 1)\n",
        "Write-Host 'a'b # it's\n",
    ],
)
def test_unsupported_powershell_source_is_returned_unchanged(content: str) -> None:
    assert tm._powershell_shell_text(content, lambda: None) == content


def test_only_complete_powershell_source_can_grant_ownership() -> None:
    content = "# Don't run rm -rf `here\nWrite-Host 'ok'\n"

    assert (
        tm.has_bounded_parse_exhaustion(
            content, lambda: None, file_type="powershell", complete_context=False
        )
        is True
    )


def test_powershell_projection_honors_runtime_checks() -> None:
    content = "Write-Host '" + "x" * 100_000 + "'\n"
    calls = 0

    def check_runtime() -> None:
        nonlocal calls
        calls += 1
        if calls > 3:
            raise TimeoutError

    with pytest.raises(TimeoutError):
        tm._powershell_shell_text(content, check_runtime)
