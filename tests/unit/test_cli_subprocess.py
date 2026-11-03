# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI regressions exercised in fresh processes without graph or provider mocks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_python(
    code: str, *args: str, provider: str = "nv_build"
) -> subprocess.CompletedProcess[str]:
    # Keep platform essentials, but exclude inherited credentials and scanner options.
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP")
        if key in os.environ
    }
    env.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
        SKILLSPECTOR_PROVIDER=provider,
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
        NO_COLOR="1",
    )
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env=env,
        check=False,
    )


def _run_cli(*args: str, provider: str = "nv_build") -> subprocess.CompletedProcess[str]:
    return _run_python("from skillspector.cli_entry import app; app()", *args, provider=provider)


def _write_skill(path: Path, body: str = "# Helper\n\nFormat a table.\n") -> Path:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "flags",
    [
        ["--use-shipped-baseline"],
        ["--transitive"],
        ["--transitive-depth", "1"],
        ["--transitive-depth", "2"],
        ["--transitive-allow-prefix", "https://example.com/"],
        ["--transitive-deny-prefix", "https://example.com/"],
    ],
)
def test_registry_unsupported_options_stop_before_input_or_output_access(
    tmp_path: Path, flags: list[str]
) -> None:
    output = tmp_path / "report.json"
    output.write_text("previous report", encoding="utf-8")

    result = _run_cli(
        "scan",
        str(tmp_path / "missing-registry.json"),
        "--mcp-registry",
        "--format",
        "json",
        "--output",
        str(output),
        *flags,
    )

    assert result.returncode == 2
    assert "cannot be combined" in result.stderr
    assert flags[0] in result.stderr
    assert result.stdout == ""
    assert output.read_text(encoding="utf-8") == "previous report"


def test_registry_default_options_still_scan(tmp_path: Path) -> None:
    source = tmp_path / "registry.json"
    source.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "server": {
                            "name": "example/notes",
                            "version": "1.0.0",
                            "repository": {
                                "url": "https://github.com/example/notes",
                                "source": "github",
                            },
                            "packages": [
                                {
                                    "registryType": "npm",
                                    "identifier": "example-notes",
                                    "version": "1.0.0",
                                    "fileSha256": "a" * 64,
                                    "transport": {"type": "stdio"},
                                }
                            ],
                        },
                        "_meta": {
                            "io.modelcontextprotocol.registry/official": {"status": "active"}
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    saved = _run_cli(
        "scan", str(source), "--mcp-registry", "--format", "json", "--output", str(output)
    )
    assert saved.returncode == 0, saved.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["server_count"] == 1

    scanned = _run_cli("scan", str(source), "--mcp-registry", "--format", "json")

    assert scanned.returncode == 0, scanned.stderr
    payload = json.loads(scanned.stdout)
    assert payload["mcp_registry"] is True
    assert payload["server_count"] == 1
    assert payload["risk_score"] == 0


@pytest.mark.parametrize("output_format", ["json", "sarif", "markdown", "terminal"])
def test_recursive_medium_recommendation_matches_direct_scan(
    tmp_path: Path, output_format: str
) -> None:
    """A complete MEDIUM scan stays CAUTION when reported as a collection."""
    collection = tmp_path / "collection"
    marked = _write_skill(collection / "marked", "# Notes\n\nordinary project notes\n")
    _write_skill(collection / "clean")
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "marker.yar").write_text(
        "rule ordinary_notes_marker {\n"
        " meta:\n"
        '  severity = "HIGH"\n'
        '  confidence = "1.0"\n'
        '  description = "Harmless marker for report parity"\n'
        " strings:\n"
        '  $marker = "ordinary project notes"\n'
        " condition:\n"
        "  $marker\n"
        "}\n",
        encoding="utf-8",
    )
    options = ["--no-llm", "--yara-rules-dir", str(rules)]
    direct = _run_cli("scan", str(marked), *options, "--format", "json")
    assert direct.returncode == 0, direct.stderr
    direct_risk = json.loads(direct.stdout)["risk_assessment"]
    assert direct_risk["score"] == 25
    assert direct_risk["severity"] == "MEDIUM"
    assert direct_risk["recommendation"] == "CAUTION"

    result = _run_cli("scan", str(collection), "--recursive", *options, "--format", output_format)
    assert result.returncode == 0, result.stderr
    if output_format == "json":
        payload = json.loads(result.stdout)
        assert payload["analysis_completeness"]["is_complete"] is True
        assert payload["execution_successful"] is True
        assert payload["max_risk_score"] == 25
        assert payload["risk_severity"] == "MEDIUM"
        assert payload["risk_recommendation"] == "CAUTION"
        children = {child["path"]: child for child in payload["skills"]}
        assert children["marked"]["risk_assessment"]["recommendation"] == "CAUTION"
        assert children["clean"]["risk_assessment"]["recommendation"] == "SAFE"
    elif output_format == "sarif":
        payload = json.loads(result.stdout)
        aggregate = payload["runs"][-1]["invocations"][0]
        assert aggregate["executionSuccessful"] is True
        assert aggregate["properties"]["analysisCompleteness"]["is_complete"] is True
        assert aggregate["properties"]["riskAssessment"] == {
            "maxRiskScore": 25,
            "severity": "MEDIUM",
            "recommendation": "CAUTION",
        }
    else:
        # Complete human-readable collections preserve their child reports.
        assert "MEDIUM" in result.stdout
        assert "CAUTION" in result.stdout
