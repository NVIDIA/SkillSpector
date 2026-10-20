# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent composition contracts for the two PR516 ownership repairs.

The full corpus, not its source text, is scanned; all shell-looking inputs are
inert. A JSON false-positive must not conceal missing table uncertainty.
"""

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

_JSON = json.dumps(["<omit on first request; reuse the returned identifier later>", "x" * 8292])
_TABLES = {
    "same-cell": "| Example |\n| --- |\n| `$(hostname).example` |",
    "different-rows": "| Example |\n| --- |\n| `$(resolve_tool).example |\n| ` -rf / |",
    "different-cells": "| Left | Right |\n| --- | --- |\n| `$(resolve_tool).example | ` -rf / |",
}


def _document(indent: str, table: str, order: str) -> str:
    # Both prefixes define valid list-relative JSON fences at Markdown columns.
    prefix = "\t" if indent == "tabs" else "  "
    marker_padding = "\t" if indent == "tabs" else " "
    fence = f"-{marker_padding}```json\n{prefix}{_JSON}\n{prefix}```"
    blocks = [fence, _TABLES[table]]
    if order == "table-first":
        blocks.reverse()
    # A heading explicitly closes the preceding list/table before the next block.
    return "\n\n# Next example\n\n".join(blocks) + "\n"


_CASES = [
    pytest.param(
        _document(indent, table, order), table == "same-cell", id=f"{indent}-{table}-{order}"
    )
    for indent in ("spaces", "tabs")
    for table in _TABLES
    for order in ("json-first", "table-first")
]


@pytest.mark.parametrize("content,complete", _CASES)
def test_table_and_json_composition_preserves_independent_completeness(
    content: str, complete: bool
) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {
            "components": ["references/combined.md"],
            "file_cache": {"references/combined.md": content},
        },
        [tm_module],
    )
    event = result["inspection_ledger"][0]
    assert event["outcome"] is (LedgerOutcome.COMPLETED if complete else LedgerOutcome.PARTIAL)
    if not complete:
        assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT
    assert result["findings"] == []


@pytest.mark.parametrize("content,complete", _CASES)
@pytest.mark.parametrize("use_llm", [False, True], ids=["no-llm", "llm"])
def test_table_and_json_composition_reaches_both_public_gates(
    tmp_path: Path,
    content: str,
    complete: bool,
    use_llm: bool,
    successful_llm_transport: list[str],
) -> None:
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
    assert cli.exit_code in (0, 1), cli.output
    reports = [(json.loads(cli.output), cli_calls), (json.loads(mcp["report"]), mcp_calls)]
    # Both interfaces execute before correctness assertions, even in the red phase.
    for report, calls in reports:
        _assert_llm_mode(report, use_llm, calls)
        if use_llm:
            assert len(calls) >= report["metadata"]["llm_calls_attempted"] >= 3
        else:
            assert calls == []
            assert report["metadata"].get("llm_calls_attempted", 0) == 0
            assert report["metadata"].get("llm_calls_succeeded", 0) == 0
        coverage = report["analysis_completeness"]
        assert coverage["execution_successful"] is True
        statuses = {
            status["analyzer_id"]: status
            for status in coverage["analyzer_statuses"]
            if status["analyzer_id"].startswith("semantic_")
        }
        assert set(statuses) == {
            "semantic_developer_intent",
            "semantic_quality_policy",
            "semantic_security_discovery",
        }
        for status in statuses.values():
            assert status["status"] == ("completed" if use_llm else "disabled")
            assert status["partial"] == status["failed"] == status["unaccounted"] == 0
            if use_llm:
                assert status["completed"] == status["planned_work"] > 0
            else:
                assert status["completed"] == status["planned_work"] == 0
    assert mcp["llm_used"] is use_llm
    assert cli.exit_code == (0 if complete else 1), cli.output
    assert mcp["safe_to_install"] is complete
    for report, _ in reports:
        coverage = report["analysis_completeness"]
        assert coverage["is_complete"] is complete
        assert not any(issue["id"] == "TM1" for issue in report["issues"])
        if complete:
            assert coverage["coverage_percent"] == 100.0
            assert not any(issue["id"] == "AE1" for issue in report["issues"])
        else:
            assert report["risk_assessment"]["recommendation"] != "SAFE"
            assert any(
                event.get("reason_code") == "static_parse_limit"
                and event.get("path") == "SKILL.md"
                and event.get("outcome") == "partial"
                and "static_patterns_tool_misuse" in event.get("analyzers", [])
                for event in coverage.get("ledger_exceptions", [])
            )
