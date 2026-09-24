# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capacity diagnostics survive public completeness and installation gates."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.inspection_ledger import (
    LedgerOutcome,
    LedgerReason,
    _exception_from_event,
    ledger_event,
)
from skillspector.mcp_server import run_scan
from tests.nodes.analyzers.test_documentation_reconstruction import _assert_llm_mode
from tests.nodes.analyzers.test_documentation_reconstruction import (
    successful_llm_transport as successful_llm_transport,
)


@pytest.mark.parametrize("use_llm", [False, True], ids=["static", "semantic"])
@pytest.mark.parametrize("size", [65535, 65536, 65537, 74978, 131071, 131072, 131073])
def test_capacity_preserves_cli_mcp_gates(
    tmp_path: Path, use_llm: bool, size: int, successful_llm_transport: list[str]
) -> None:
    # A compact synthetic object with unique records avoids context-stuffing
    # findings while exercising the JSON closing quote after a placeholder.
    def encode(records: int) -> str:
        return json.dumps(
            {
                "batch": "<omit on first request; reuse the returned identifier later>",
                "records": [
                    {"index": i, "label": f"Example record {i:05d} for review."}
                    for i in range(records)
                ],
            }
        )

    # Size the distinct short records, then use less than one record of padding.
    records = size // 64
    body = encode(records)
    while len(body) > size:
        records -= 1
        body = encode(records)
    while len(candidate := encode(records + 1)) <= size:
        records += 1
        body = candidate
    assert 0 <= size - len(body) < 80
    body += " " * (size - len(body))
    prefix = "---\nname: json-capacity\ndescription: Review a JSON example.\n---\n"
    (tmp_path / "SKILL.md").write_text(prefix + body, encoding="utf-8")
    args = ["scan", str(tmp_path), "--format", "json", "--fail-on-incomplete"]
    if not use_llm:
        args.append("--no-llm")
    cli = CliRunner().invoke(app, args)
    cli_calls = list(successful_llm_transport)
    successful_llm_transport.clear()
    mcp = asyncio.run(run_scan(str(tmp_path), use_llm=use_llm, output_format="json"))
    complete = size <= 131072
    assert cli.exit_code == (0 if complete else 1), cli.output
    assert mcp["safe_to_install"] is complete
    for report, calls in [
        (json.loads(cli.output), cli_calls),
        (json.loads(mcp["report"]), successful_llm_transport),
    ]:
        _assert_llm_mode(report, use_llm, calls)
        assert report["execution_successful"] is True
        assert report["issues"] == []
        coverage = report["analysis_completeness"]
        assert coverage["is_complete"] is complete
        assert coverage["coverage_percent"] == (100.0 if complete else 0.0)
        assert report["risk_assessment"]["recommendation"] == ("SAFE" if complete else "CAUTION")
        if complete:
            assert coverage["ledger_exceptions"] == []
        else:
            assert len(coverage["ledger_exceptions"]) == 1
            event = coverage["ledger_exceptions"][0]
            assert event["reason_code"] == LedgerReason.JSON_QUOTE_OWNERSHIP_LIMIT
            assert event["path"] == "SKILL.md"
            assert event["source_start_offset"] == len(prefix)
            assert event["source_end_offset"] == len(prefix) + size
            assert event["observed_characters"] == size
            assert event["limit_characters"] == 131072
            assert f"[{len(prefix)}, {len(prefix) + size})" in event["message"]
            assert "Split" in event["message"]
            assert "validity remains unverified" in event["message"]


@pytest.mark.parametrize("value", ["private source text", -1, True, 3.5, None])
def test_public_capacity_projection_rejects_malformed_metrics(value: object) -> None:
    event = ledger_event(
        outcome=LedgerOutcome.PARTIAL,
        phase="static",
        path="SKILL.md",
        reason=LedgerReason.JSON_QUOTE_OWNERSHIP_LIMIT,
    )
    event.update(
        source_start_offset=value,
        source_end_offset=70000,
        observed_characters=70000,
        limit_characters=65536,
    )
    event["message"] = "private source text"
    projected = _exception_from_event(event, fatal=False)
    assert "source_start_offset" not in projected
    assert "private source text" not in projected["message"]
