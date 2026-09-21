# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real static scans keep ordinary image limitations separate from AE1 findings.

These tests use valid, benign PNGs and exercise graph, command-line, and MCP
entry points without provider calls or mocked scan results.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest

from skillspector.graph import graph
from skillspector.mcp_server import run_scan
from skillspector.sarif_models import validate_sarif_report

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def mock_resolve_context_length() -> None:
    """Override the suite's resolver mock; static scans need no model lookup."""


def _png_chunk(kind: bytes, content: bytes) -> bytes:
    return (
        struct.pack(">I", len(content))
        + kind
        + content
        + struct.pack(">I", zlib.crc32(kind + content))
    )


def _write_image_skill(root: Path, count: int, *, duplicate_label: bool = False) -> Path:
    skill = root / "chart-guide"
    assets = skill / "assets"
    assets.mkdir(parents=True)
    # A complete 1x1 RGBA PNG, including correct lengths, CRCs and image data.
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(b"\x00\x40\x80\xc0\xff"))
        + _png_chunk(b"IEND", b"")
    )
    images = []
    for index in range(count):
        path = f"assets/chart-{index}.png"
        (skill / path).write_bytes(png)
        label = path if duplicate_label else f"Chart {index}"
        images.append(f"![{label}]({path})")
    (skill / "SKILL.md").write_text(
        "---\nname: chart-guide\n"
        "description: Explain the chart colors when the user asks for a chart guide.\n"
        "---\n# Chart guide\n\nUse the charts to explain the colors.\n\n"
        + "\n\n".join(images)
        + "\n",
        encoding="utf-8",
    )
    return skill


def _write_single_asset_skill(root: Path, filename: str, payload: bytes) -> Path:
    skill = root / f"asset-{Path(filename).suffix.removeprefix('.')}"
    assets = skill / "assets"
    assets.mkdir(parents=True)
    (assets / filename).write_bytes(payload)
    (skill / "SKILL.md").write_text(
        "---\nname: asset-guide\n"
        "description: Explain one bundled reference asset when the user asks.\n"
        "---\n# Asset guide\n\n"
        f"Review [the bundled asset](assets/{filename}).\n",
        encoding="utf-8",
    )
    return skill


def _assert_completeness(completeness: dict[str, Any], count: int) -> None:
    assert completeness["is_complete"] is False
    assert completeness["status"] == "partial"
    assert completeness["execution_successful"] is True
    assert completeness["total_components"] == count + 1
    assert completeness["fully_inspected_files"] == 1
    assert completeness["partially_inspected_files"] == 0
    assert completeness["entirely_uninspected_files"] == count
    assert completeness["coverage_percent"] == round(100 / (count + 1), 1)
    exceptions = completeness["ledger_exceptions"]
    assert {item["path"] for item in exceptions} == {
        f"assets/chart-{index}.png" for index in range(count)
    }
    assert {item["reason_code"] for item in exceptions} <= {"opaque_content", "binary_content"}
    assert all(not item["fatal"] for item in exceptions)
    assert completeness["findings_before_filtering"] == 0
    assert completeness["findings_after_filtering"] == 0


def _assert_report(body: str, output_format: str, count: int) -> None:
    if output_format == "json":
        report = json.loads(body)
        assert report["issues"] == []
        assert report["risk_assessment"] == {
            "score": 0,
            "severity": "LOW",
            "recommendation": "CAUTION",
            "max_issue_severity": "NONE",
        }
        assert report["execution_successful"] is True
        _assert_completeness(report["analysis_completeness"], count)
    elif output_format == "sarif":
        report = json.loads(body)
        validate_sarif_report(report)
        run = report["runs"][0]
        assert run["results"] == []
        invocation = run["invocations"][0]
        assert invocation["executionSuccessful"] is True
        completeness = invocation["properties"]["analysisCompleteness"]
        assert completeness["isComplete"] is False
        assert completeness["status"] == "partial"
        assert completeness["entirelyUninspectedFiles"] == count
        assert completeness["coveragePercent"] == round(100 / (count + 1), 1)
        notifications = invocation["toolExecutionNotifications"]
        assert any(
            item["properties"].get("reasonCode") == "opaque_content" for item in notifications
        )
    else:
        assert "CAUTION" in body
        assert "Inspection Completeness" in body
        assert "partial" in body
        assert "opaque_content" in body
        assert "AE1" not in body
        for index in range(count):
            assert f"assets/chart-{index}.png" in body


@pytest.mark.parametrize("count", [1, 4, 8])
@pytest.mark.parametrize("duplicate_label", [False, True], ids=["image-label", "path-label"])
def test_graph_keeps_png_coverage_without_ae1(
    tmp_path: Path, count: int, duplicate_label: bool
) -> None:
    skill = _write_image_skill(tmp_path, count, duplicate_label=duplicate_label)
    initial = {"skill_path": str(skill), "use_llm": False, "output_format": "json"}
    # Exercise both Python APIs against the same actual files.
    invoked = graph.invoke(initial)
    streamed = list(graph.stream(initial, stream_mode="values"))[-1]

    for result in (invoked, streamed):
        assert result["findings"] == []
        assert result["filtered_findings"] == []
        assert result["risk_score"] == 0
        assert result["risk_recommendation"] == "CAUTION"
        assert result["execution_successful"] is True
        _assert_completeness(result["analysis_completeness"], count)
        _assert_report(result["report_body"], "json", count)
        references = result["artifact_references"]
        assert len(references) == count * (2 if duplicate_label else 1)
        assert all(reference["status"] == "resolved" for reference in references)
        binary_items = [
            item for item in result["artifact_inventory"] if item["content_kind"] == "binary"
        ]
        assert len(binary_items) == count
        assert all(item["referenced"] is True for item in binary_items)
        assert all(item["disposition"] == "out_of_scope" for item in binary_items)

    assert invoked["analysis_completeness"] == streamed["analysis_completeness"]


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("photo.jpg", b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 32),
        (
            "pixel.gif",
            b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x01L\x00;",
        ),
        ("manual.pdf", b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n"),
    ],
    ids=["jpeg", "gif", "pdf"],
)
def test_graph_applies_format_reason_policy_beyond_png(
    tmp_path: Path, filename: str, payload: bytes
) -> None:
    skill = _write_single_asset_skill(tmp_path, filename, payload)

    result = graph.invoke({"skill_path": str(skill), "use_llm": False, "output_format": "json"})

    target = f"assets/{filename}"
    artifact = next(item for item in result["artifact_inventory"] if item["path"] == target)
    assert artifact["content_kind"] == "binary"
    assert artifact["disposition"] == "out_of_scope"
    assert not any(finding.rule_id == "AE1" for finding in result["findings"])
    assert result["risk_recommendation"] == "CAUTION"
    assert result["analysis_completeness"]["is_complete"] is False
    target_exceptions = [
        event
        for event in result["analysis_completeness"]["ledger_exceptions"]
        if event["path"] == target
    ]
    assert target_exceptions
    assert {event["reason_code"] for event in target_exceptions} <= {
        "binary_content",
        "opaque_content",
    }


def test_graph_does_not_treat_dex_word_prefix_as_binary_or_executable(tmp_path: Path) -> None:
    payload = b"dex\nA short term for dexterity.\n"
    skill = _write_single_asset_skill(tmp_path, "glossary.txt", payload)

    result = graph.invoke({"skill_path": str(skill), "use_llm": False, "output_format": "json"})

    artifact = next(
        item for item in result["artifact_inventory"] if item["path"] == "assets/glossary.txt"
    )
    assert artifact["content_kind"] == "text"
    assert artifact["disposition"] == "analyzed"
    assert not {finding.rule_id for finding in result["findings"]} & {"AE1", "AE2", "SC9"}
    assert result["analysis_completeness"]["is_complete"] is True


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("program.exe", b"MZ" + b"\x00" * 62),
        ("classes.dex", b"dex\n035\0" + b"\x00" * 120),
        ("chunk.luac", b"\x1bLua\x54\x00" + b"\x00" * 120),
    ],
    ids=["pe", "dex", "lua-bytecode"],
)
def test_referenced_executable_binary_keeps_ae1_and_concealment_signal(
    tmp_path: Path, filename: str, payload: bytes
) -> None:
    skill = _write_single_asset_skill(tmp_path, filename, payload)

    result = graph.invoke({"skill_path": str(skill), "use_llm": False, "output_format": "json"})

    assert any(finding.rule_id == "AE1" for finding in result["findings"])
    assert any(finding.rule_id == "SC9" for finding in result["findings"])
    assert result["risk_recommendation"] == "DO_NOT_INSTALL"
    assert any(
        event["path"] == f"assets/{filename}"
        and event["reason_code"] == "excluded_executable_content"
        for event in result["analysis_completeness"]["ledger_exceptions"]
    )


def test_cli_active_unknown_format_reference_fails_closed(tmp_path: Path) -> None:
    skill = _write_single_asset_skill(tmp_path, "payload.asset", b"\xff" * 1024)
    (skill / "SKILL.md").write_text(
        "---\nname: asset-guide\n"
        "description: Run one bundled helper when the user asks.\n"
        "---\n# Asset guide\n\n"
        "Run [the helper](assets/payload.asset).\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "skillspector.cli",
            "scan",
            str(skill),
            "--no-llm",
            "--format",
            "json",
            "--fail-on-findings",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "120"},
    )

    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert any(issue["id"] == "AE1" for issue in report["issues"])


@pytest.mark.parametrize(
    ("output_format", "count", "extra_args", "expected_exit"),
    [
        ("json", 1, [], 0),
        ("json", 8, ["--fail-on-findings"], 0),
        ("json", 4, ["--fail-on-incomplete"], 1),
        ("markdown", 1, ["--fail-on-incomplete", "--fail-on-findings"], 1),
        ("sarif", 4, ["--fail-on-incomplete"], 1),
        ("terminal", 8, [], 0),
    ],
    ids=[
        "json-default",
        "json-findings",
        "json-strict",
        "markdown-both",
        "sarif-strict",
        "terminal-default",
    ],
)
def test_cli_reports_opaque_coverage_and_honors_exit_policy(
    tmp_path: Path,
    output_format: str,
    count: int,
    extra_args: list[str],
    expected_exit: int,
) -> None:
    skill = _write_image_skill(tmp_path, count, duplicate_label=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "skillspector.cli",
            "scan",
            str(skill),
            "--no-llm",
            "--format",
            output_format,
            *extra_args,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "120"},
    )

    assert result.returncode == expected_exit, result.stderr
    _assert_report(result.stdout, output_format, count)


@pytest.mark.parametrize(("output_format", "count"), [("json", 1), ("markdown", 4), ("sarif", 8)])
async def test_mcp_core_does_not_claim_opaque_bundle_is_safe(
    tmp_path: Path, output_format: str, count: int
) -> None:
    skill = _write_image_skill(tmp_path, count)

    result = await run_scan(str(skill), use_llm=False, output_format=output_format)

    assert result["findings"] == []
    assert result["risk_score"] == 0
    assert result["recommendation"] == "CAUTION"
    assert result["safe_to_install"] is False
    assert result["execution_successful"] is True
    assert result["llm_requested"] is False
    assert result["llm_used"] is False
    assert result["scan_mode"] == "static-only"
    _assert_completeness(result["analysis_completeness"], count)
    _assert_report(result["report"], output_format, count)


async def test_stdio_mcp_preserves_opaque_coverage_verdict(tmp_path: Path) -> None:
    mcp = pytest.importorskip("mcp")
    from mcp.client.stdio import StdioServerParameters, stdio_client

    skill = _write_image_skill(tmp_path, 4, duplicate_label=True)
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "skillspector.cli", "mcp"],
    )
    # Keep transport diagnostics separate from both protocol output and scan input.
    with (tmp_path / "mcp-stderr.log").open("w", encoding="utf-8") as stderr:
        async with asyncio.timeout(60):
            async with stdio_client(parameters, errlog=stderr) as (read, write):
                async with mcp.ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    assert "scan_skill" in {tool.name for tool in tools.tools}
                    response = await session.call_tool(
                        "scan_skill",
                        {"target": str(skill), "use_llm": False, "output_format": "json"},
                    )

    assert response.isError is False
    result = response.structuredContent
    assert isinstance(result, dict)
    assert result["findings"] == []
    assert result["risk_score"] == 0
    assert result["recommendation"] == "CAUTION"
    assert result["safe_to_install"] is False
    assert result["scan_mode"] == "static-only"
    _assert_completeness(result["analysis_completeness"], 4)
    _assert_report(result["report"], "json", 4)
