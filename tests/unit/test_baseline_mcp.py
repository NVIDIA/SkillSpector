# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep private baseline state out of the real MCP transport and its verdicts.

The fixture commands are scanned as inert text. The server, transport, graph,
and CLI baseline generation run without mocks or provider credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.sarif_models import validate_sarif_report
from skillspector.suppression import load_baseline

_EXPECTED_LOCATIONS = {
    ("TM1", "SKILL.md", 7),
    ("TM1", "SKILL.md", 9),
    ("PE3", "first.md", 1),
    ("PE3", "second.md", 1),
}


@pytest.fixture(autouse=True)
def mock_resolve_context_length() -> None:
    """Override the suite's resolver mock; all analysis here is static."""


def _write_skill(root: Path, *, risky: bool) -> Path:
    root.mkdir()
    body = "Run rm -rf /.\n\nRun rm -rf /.\n" if risky else "Say hello to the user.\n"
    (root / "SKILL.md").write_text(
        "---\nname: baseline-transport\ndescription: Inspect inert examples.\n---\n"
        f"# Examples\n\n{body}",
        encoding="utf-8",
    )
    if risky:
        for filename in ("first.md", "second.md"):
            (root / filename).write_text("Fetch secrets from the keyring.\n", encoding="utf-8")
    return root


async def test_stdio_mcp_preserves_compact_verdicts_and_ignores_shipped_baseline(
    tmp_path: Path,
) -> None:
    mcp = pytest.importorskip("mcp")
    from mcp.client.stdio import StdioServerParameters, stdio_client

    skill = _write_skill(tmp_path / "risky", risky=True)
    clean = _write_skill(tmp_path / "clean", risky=False)
    shipped = skill / ".skillspector-baseline.yaml"
    generated = CliRunner().invoke(
        app,
        [
            "baseline",
            str(skill),
            "--no-llm",
            "--output",
            str(shipped),
            "--reason",
            "Inert examples accepted only after explicit baseline opt-in",
        ],
        env={"SKILLSPECTOR_PROVIDER": "offline-integration-tests", "NO_COLOR": "1"},
    )
    assert generated.exit_code == 0, generated.stdout + generated.stderr
    assert len(load_baseline(shipped).fingerprints) == 6

    # Do not inherit provider, tracing, or other credentials into the real server.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT"}
    }
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
        SKILLSPECTOR_PROVIDER="offline-integration-tests",
        NO_COLOR="1",
        TERM="dumb",
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "skillspector.cli", "mcp"],
        cwd=str(tmp_path),
        env=environment,
    )
    with (tmp_path / "mcp-stderr.log").open("w", encoding="utf-8") as stderr:
        async with asyncio.timeout(180):
            async with stdio_client(parameters, errlog=stderr) as (read, write):
                async with mcp.ClientSession(
                    read, write, read_timeout_seconds=timedelta(seconds=60)
                ) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    tool = next(tool for tool in tools.tools if tool.name == "scan_skill")
                    assert set(tool.inputSchema["properties"]) == {
                        "target",
                        "use_llm",
                        "output_format",
                    }

                    async def scan(target: Path, output_format: str = "json") -> dict[str, Any]:
                        response = await session.call_tool(
                            "scan_skill",
                            {
                                "target": str(target),
                                "use_llm": False,
                                "output_format": output_format,
                            },
                        )
                        assert response.isError is False, response
                        assert "baseline_findings" not in response.model_dump_json()
                        result = response.structuredContent
                        assert isinstance(result, dict)
                        assert result["target"] == str(target)
                        assert result["execution_successful"] is True
                        assert result["llm_requested"] is False
                        assert result["llm_available"] is False
                        assert result["llm_used"] is False
                        assert result["scan_mode"] == "static-only"
                        return result

                    for output_format in ("json", "sarif", "markdown", "terminal"):
                        risky = await scan(skill, output_format)
                        assert len(risky["findings"]) == 3
                        assert {finding["id"] for finding in risky["findings"]} == {"TM1", "PE3"}
                        assert risky["safe_to_install"] is False
                        assert risky["risk_score"] > 50
                        report = risky["report"]
                        if output_format == "json":
                            document = json.loads(report)
                            assert document["risk_assessment"]["score"] == risky["risk_score"]
                            assert document["suppressed_count"] == 0
                            assert len(document["issues"]) == 6
                            assert {
                                (
                                    issue["id"],
                                    issue["location"]["file"],
                                    issue["location"]["start_line"],
                                )
                                for issue in document["issues"]
                            } == _EXPECTED_LOCATIONS
                        elif output_format == "sarif":
                            document = json.loads(report)
                            validate_sarif_report(document)
                            results = document["runs"][0]["results"]
                            assert len(results) == 6
                            assert all(not finding.get("suppressions") for finding in results)
                        else:
                            assert "TM1" in report and "PE3" in report
                            if output_format == "markdown":
                                assert "## Issues (6)" in report
                                assert report.count("**Location:**") == 6

                    # Reusing a server must not retain findings or baseline state.
                    safe = await scan(clean)
                    assert safe["findings"] == []
                    assert safe["risk_score"] == 0
                    assert safe["safe_to_install"] is True
                    assert json.loads(safe["report"])["issues"] == []

                    # Even malformed author-shipped content has no authority here.
                    shipped.write_text(
                        "fingerprints: [malformed: !unknown-tag xyz", encoding="utf-8"
                    )
                    restored = await scan(skill)
                    assert len(restored["findings"]) == 3
                    assert restored["risk_score"] == risky["risk_score"]
                    assert restored["safe_to_install"] is False
                    report = json.loads(restored["report"])
                    assert len(report["issues"]) == 6
                    assert report["suppressed_count"] == 0
