# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PowerShell source must not exempt embedded shell commands from completeness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.graph import graph
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.mcp_server import run_scan
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner

_EMBEDDED_SHELL_COMMANDS = [
    "bash -c '$CMD -rf /'",
    'bash -c "$CMD -rf /"',
    "& bash -lc '$CMD -rf /'",
    "/bin/bash --noprofile -c '$CMD -rf /'",
    "Write-Output 'safe'; bash -c '$CMD -rf /'",
    "Write-Output \"$($text -replace '%s', 'safe')\"\n" + "bash -c '$CMD -rf /'",
]
_EMBEDDED_SHELL_IDS = [
    "bash",
    "dynamic-command-string",
    "call-operator",
    "absolute-shell",
    "after-separator",
    "after-benign-expression",
]
_BENIGN_REPLACE_VALUES = [
    "Write-Output \"$($text -replace '%TEMP%', $env:TEMP)\"",
    "Write-Host -NoNewline \"$($text -replace '%s', 'r m')\"",
    "Write-Output (\"$($text -replace 'old', 'printf')\")",
    "\"$($text -replace '%s', 'r m')\" | Write-Output",
    "$result = \"$($text -replace '%s', 'r m')\"",
]


def _write_powershell_bundle(root: Path, content: str) -> None:
    # These are inert scanner fixtures: no PowerShell or shell process is run.
    (root / "SKILL.md").write_text("# Example\n\nA bundled script example.\n", encoding="utf-8")
    (root / "example.ps1").write_text(content + "\n", encoding="utf-8")


def _assert_shell_parse_limit(completeness: dict) -> None:
    assert completeness["execution_successful"] is True
    assert completeness["status"] == "partial"
    assert completeness["is_complete"] is False
    assert any(
        row["path"] == "example.ps1" and row["reason_code"] == "static_parse_limit"
        for row in completeness["ledger_exceptions"]
    )
    assert not any(row["fatal"] for row in completeness["ledger_exceptions"])


@pytest.mark.parametrize("content", _EMBEDDED_SHELL_COMMANDS, ids=_EMBEDDED_SHELL_IDS)
def test_powershell_embedded_runtime_shell_records_parse_limit(content: str) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["example.ps1"], "file_cache": {"example.ps1": content}},
        [tm_module],
    )

    event = result["inspection_ledger"][0]
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.asyncio
@pytest.mark.parametrize("content", _EMBEDDED_SHELL_COMMANDS, ids=_EMBEDDED_SHELL_IDS)
async def test_powershell_embedded_runtime_shell_fails_closed_publicly(
    tmp_path: Path, content: str
) -> None:
    _write_powershell_bundle(tmp_path, content)

    result = graph.invoke({"input_path": str(tmp_path), "output_format": "json", "use_llm": False})
    _assert_shell_parse_limit(result["analysis_completeness"])
    assert result["risk_recommendation"] == "CAUTION"

    arguments = ["scan", str(tmp_path), "--format", "json", "--no-llm"]
    for strict in (False, True):
        cli_result = CliRunner().invoke(
            app, arguments + (["--fail-on-incomplete"] if strict else [])
        )
        assert cli_result.exit_code == (1 if strict else 0), cli_result.output
        report = json.loads(cli_result.output)
        _assert_shell_parse_limit(report["analysis_completeness"])
        assert report["risk_assessment"]["recommendation"] == "CAUTION"

    verdict = await run_scan(str(tmp_path), use_llm=False, output_format="json")
    _assert_shell_parse_limit(verdict["analysis_completeness"])
    assert verdict["recommendation"] == "CAUTION"
    assert verdict["safe_to_install"] is False


@pytest.mark.parametrize("content", _BENIGN_REPLACE_VALUES)
def test_powershell_replace_values_remain_complete_in_ledger(content: str) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["example.ps1"], "file_cache": {"example.ps1": content}},
        [tm_module],
    )

    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "content",
    [
        "Write-Output \"bash -c '$CMD -rf /'\"",
        "Write-Output 'bash -c ''$CMD -rf /'''",
    ],
)
def test_powershell_quoted_shell_text_remains_complete(content: str) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["example.ps1"], "file_cache": {"example.ps1": content}},
        [tm_module],
    )

    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.asyncio
async def test_powershell_replace_values_remain_complete_publicly(tmp_path: Path) -> None:
    _write_powershell_bundle(tmp_path, "\n".join(_BENIGN_REPLACE_VALUES))

    result = graph.invoke({"input_path": str(tmp_path), "output_format": "json", "use_llm": False})
    assert result["analysis_completeness"]["status"] == "complete"
    assert result["analysis_completeness"]["ledger_exceptions"] == []
    assert result["risk_recommendation"] == "SAFE"

    cli_result = CliRunner().invoke(
        app,
        ["scan", str(tmp_path), "--format", "json", "--no-llm", "--fail-on-incomplete"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    report = json.loads(cli_result.output)
    assert report["analysis_completeness"]["status"] == "complete"
    assert report["risk_assessment"]["recommendation"] == "SAFE"

    verdict = await run_scan(str(tmp_path), use_llm=False, output_format="json")
    assert verdict["analysis_completeness"]["status"] == "complete"
    assert verdict["recommendation"] == "SAFE"
    assert verdict["safe_to_install"] is True
