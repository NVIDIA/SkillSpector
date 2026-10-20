# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsupported brace targets preserve uncertainty for runtime-selected commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.mcp_server import run_scan
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner
from tests.nodes.analyzers.test_documentation_reconstruction import _assert_llm_mode
from tests.nodes.analyzers.test_documentation_reconstruction import (
    successful_llm_transport as successful_llm_transport,
)

_OPERANDS = [
    pytest.param("-rf {/,a,b,c,d,e,f,g}", False, id="eight-alternatives-with-root"),
    pytest.param("-rf {/,a,b,c,d,e,f,g,h}", False, id="nine-alternatives-with-root"),
    pytest.param("-rf {a,b,c,d,e,f,g,h}", True, id="eight-relative-alternatives"),
    pytest.param("-rf {a,b,c,d,e,f,g,h,i}", True, id="nine-relative-alternatives"),
    pytest.param("-rf {a,{b,{c,{d,{/,e}}}}}", False, id="nesting-exceeds-four-rounds"),
    pytest.param("-rf {/,{1..2}}", False, id="unsupported-range-in-alternatives"),
    pytest.param("-{r,f,a,b,c,d,e,g} /", False, id="eight-option-alternatives"),
    pytest.param("-{r,f,a,b,c,d,e,g,h} /", False, id="nine-option-alternatives"),
    pytest.param("-rf '{/,a,b,c,d,e,f,g,h}'", True, id="quoted-brace-target"),
    pytest.param("'-{r,f,a,b,c,d,e,g,h}' /", True, id="quoted-brace-options"),
]


@pytest.mark.parametrize("operands,complete", _OPERANDS)
@pytest.mark.parametrize("command", ["$CMD", "$($CMD)", "$(env $CMD)", "`$CMD`"])
@pytest.mark.parametrize("file_type", ["shell", "markdown"])
def test_dynamic_command_brace_limits_reach_the_static_ledger(
    operands: str, complete: bool, command: str, file_type: str
) -> None:
    # These are inert scanner inputs. Relative-only operands remain a control:
    # exceeding the brace bound alone does not establish a root-delete context.
    source = f"{command} {operands}\n"
    path = "scripts/command.sh"
    if file_type == "markdown":
        source = f"```bash\n{source}```\n"
        path = "SKILL.md"
    assert (
        tm_module.has_bounded_parse_exhaustion(source, lambda: None, file_type=file_type)
        is not complete
    )
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: source}}, [tm_module]
    )
    row = result["inspection_ledger"][0]
    assert row["path"] == path
    assert row["analyzer_id"] == "static_patterns_tool_misuse"
    assert row["outcome"] is (LedgerOutcome.COMPLETED if complete else LedgerOutcome.PARTIAL)
    if not complete:
        assert row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
    assert result["findings"] == []


@pytest.mark.parametrize("operands,complete", _OPERANDS)
@pytest.mark.parametrize("command", ["$CMD", "$($CMD)"])
@pytest.mark.parametrize("use_llm", [False, True], ids=["no-llm", "llm"])
def test_dynamic_brace_limits_preserve_cli_and_mcp_gates(
    tmp_path: Path,
    operands: str,
    complete: bool,
    command: str,
    use_llm: bool,
    successful_llm_transport: list[str],
) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: command-example\n"
        "description: Inspect a documented command example.\n---\n\n"
        f"```bash\n{command} {operands}\n```\n",
        encoding="utf-8",
    )
    args = ["scan", str(tmp_path), "--format", "json", "--fail-on-incomplete"]
    if not use_llm:
        args.append("--no-llm")
    cli = CliRunner().invoke(app, args)
    cli_calls = list(successful_llm_transport)
    successful_llm_transport.clear()
    mcp = asyncio.run(run_scan(str(tmp_path), use_llm=use_llm, output_format="json"))

    assert cli.exit_code == (0 if complete else 1), cli.output
    assert mcp["safe_to_install"] is complete
    assert mcp["llm_used"] is use_llm
    for report, calls in [
        (json.loads(cli.output), cli_calls),
        (json.loads(mcp["report"]), successful_llm_transport),
    ]:
        _assert_llm_mode(report, use_llm, calls)
        assert report["issues"] == []
        assert report["risk_assessment"]["score"] == 0
        coverage = report["analysis_completeness"]
        assert coverage["execution_successful"] is True
        assert coverage["is_complete"] is complete
        assert report["risk_assessment"]["recommendation"] == ("SAFE" if complete else "CAUTION")
        if complete:
            assert coverage["ledger_exceptions"] == []
        else:
            assert any(
                event["reason_code"] == LedgerReason.STATIC_PARSE_LIMIT
                and event["path"] == "SKILL.md"
                and "static_patterns_tool_misuse" in event["analyzers"]
                for event in coverage["ledger_exceptions"]
            )
