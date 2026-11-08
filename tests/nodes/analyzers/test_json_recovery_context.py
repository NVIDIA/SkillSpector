# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover JSON commands without changing their original Markdown ownership."""

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

_RUNTIME_COMMAND = "`$(resolve_tool).example` -rf /"


def _json_fence(value: str, padded: bool, marker: str = "~~~", container: str = "plain") -> str:
    # These are inert scanner inputs. The surrounding entries make an earlier
    # shell parse consume the command, requiring recovery of its JSON string.
    values = ["`ordinary ", value, "`"] if padded else [value]
    opener, prefix = {
        "plain": ("", ""),
        "quote": ("> ", "> "),
        "list": ("- ", "  "),
        "tab-list": ("-\t", "\t"),
    }[container]
    return f"{opener}{marker}json\n{prefix}{json.dumps(values)}\n{prefix}{marker}\n"


@pytest.mark.parametrize("container", ["plain", "quote", "list", "tab-list"])
@pytest.mark.parametrize("marker", ["~~~", "```"], ids=["tildes", "backticks"])
@pytest.mark.parametrize("padded", [False, True], ids=["single-entry", "multiple-entries"])
def test_json_recovery_preserves_literal_fence_context(
    container: str, marker: str, padded: bool
) -> None:
    content = _json_fence(_RUNTIME_COMMAND, padded, marker, container)
    assert len(tm_module.validated_json_string_spans(content, lambda: None)) == (3 if padded else 1)
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}, [tm_module]
    )
    assert any(
        row["outcome"] is LedgerOutcome.PARTIAL
        and row["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
        and row["analyzer_id"] == "static_patterns_tool_misuse"
        and row["path"] == "SKILL.md"
        for row in result["inspection_ledger"]
    )
    assert result["findings"] == []


@pytest.mark.parametrize("padded", [False, True], ids=["single-entry", "multiple-entries"])
@pytest.mark.parametrize(
    "value,complete",
    [
        (_RUNTIME_COMMAND, False),
        ("Use `$(hostname).example` for the host name.", False),
        ("Ordinary prose `example`", True),
    ],
    ids=["unresolved-command", "literal-runtime-in-prose", "benign-prose"],
)
@pytest.mark.parametrize("use_llm", [False, True], ids=["no-llm", "llm"])
def test_json_recovery_preserves_cli_and_mcp_completeness_gates(
    tmp_path: Path,
    padded: bool,
    value: str,
    complete: bool,
    use_llm: bool,
    successful_llm_transport: list[str],
) -> None:
    content = (
        "---\nname: json-command-example\n"
        "description: Inspect the command stored in the JSON array.\n---\n\n"
        "Inspect the command stored in the JSON array:\n\n" + _json_fence(value, padded)
    )
    (tmp_path / "SKILL.md").write_text(content, encoding="utf-8")
    args = ["scan", str(tmp_path), "--format", "json", "--fail-on-incomplete"]
    if not use_llm:
        args.append("--no-llm")
    cli = CliRunner().invoke(app, args)
    cli_calls = list(successful_llm_transport)
    successful_llm_transport.clear()
    mcp = asyncio.run(run_scan(str(tmp_path), use_llm=use_llm, output_format="json"))
    mcp_calls = list(successful_llm_transport)

    assert cli.exception is None or isinstance(cli.exception, SystemExit)
    assert cli.exit_code == (0 if complete else 1), cli.output
    assert mcp["safe_to_install"] is complete
    assert mcp["llm_used"] is use_llm
    assert mcp["analysis_completeness"]["is_complete"] is complete
    for report, calls in [
        (json.loads(cli.output), cli_calls),
        (json.loads(mcp["report"]), mcp_calls),
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
