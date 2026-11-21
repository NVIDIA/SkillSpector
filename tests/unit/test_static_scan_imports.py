# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process coverage for static scans' deferred provider imports.

These tests deliberately reject the optional chat-client modules in the child
process.  A static ``--no-llm`` scan must still produce its ordinary report and
retain the disabled-LLM analysis ledger when those clients are unavailable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_BLOCKED_CLIENTS = (
    "langchain_openai",
    "langchain_anthropic",
    "openai",
    "anthropic",
)


def _run_static_scan(
    skill: Path,
    *,
    temperature: str | None = None,
    reasoning_effort: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the real console app in a fresh process with chat clients blocked."""
    source_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import sys

class RejectChatClients(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = __BLOCKED_CLIENTS__
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("static scan imported chat client: " + fullname)

sys.meta_path.insert(0, RejectChatClients())
from skillspector.cli_entry import app
app(prog_name="skillspector")
""".replace("__BLOCKED_CLIENTS__", repr(_BLOCKED_CLIENTS))
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP")
        if key in os.environ
    }
    env.update(
        PYTHONPATH=str(source_root / "src"),
        LANG="C.UTF-8",
        NO_COLOR="1",
        PYTHON_DOTENV_DISABLED="1",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
        AWS_EC2_METADATA_DISABLED="true",
        SKILLSPECTOR_PROVIDER="invalid-provider",
    )
    if temperature is not None:
        env["SKILLSPECTOR_TEMPERATURE"] = temperature
    if reasoning_effort is not None:
        env["SKILLSPECTOR_REASONING_EFFORT"] = reasoning_effort
    return subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "scan",
            str(skill),
            "--no-llm",
            "--format",
            "json",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    ("temperature", "reasoning_effort"),
    [
        pytest.param(None, None, id="defaults"),
        pytest.param("0.2", "low", id="valid-controls"),
        pytest.param("warm", "provider specific value", id="invalid-controls"),
    ],
)
def test_static_scan_keeps_report_coverage_without_chat_clients(
    temperature: str | None, reasoning_effort: str | None
) -> None:
    """A clean static scan retains complete coverage and disabled LLM status."""
    skill = Path(__file__).resolve().parents[1] / "fixtures" / "safe_skill"
    result = _run_static_scan(skill, temperature=temperature, reasoning_effort=reasoning_effort)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    metadata = report["metadata"]
    completeness = report["analysis_completeness"]
    assert metadata["llm_requested"] is False
    assert metadata["llm_available"] is False
    assert metadata["meta_analysis_applied"] is False
    assert metadata["inference_usage"] == []
    assert completeness["is_complete"] is True
    assert completeness["coverage_percent"] == 100.0
    assert completeness["execution_successful"] is True
    statuses = {
        analyzer["analyzer_id"]: analyzer["status"]
        for analyzer in completeness["analyzer_statuses"]
    }
    assert {
        "semantic_developer_intent",
        "semantic_quality_policy",
        "semantic_security_discovery",
    } <= statuses.keys()
    assert all(
        statuses[analyzer_id] == "disabled"
        for analyzer_id in (
            "semantic_developer_intent",
            "semantic_quality_policy",
            "semantic_security_discovery",
        )
    )
    sampling = metadata["llm_provenance"]["sampling"]
    if temperature is None:
        assert sampling["temperature"]["requested"] is None
    elif temperature == "warm":
        assert sampling["temperature"]["source"] == "invalid_environment"
        assert sampling["temperature"]["forwarded_to_client"] is None
    else:
        assert sampling["temperature"]["requested"] == float(temperature)
        assert sampling["temperature"]["forwarded_to_client"] is None
    if reasoning_effort is None:
        assert sampling["reasoning_effort"]["requested"] is None
    elif reasoning_effort == "provider specific value":
        assert sampling["reasoning_effort"]["source"] == "invalid_environment"
        assert sampling["reasoning_effort"]["forwarded_to_client"] is None
    else:
        assert sampling["reasoning_effort"]["requested"] == reasoning_effort
        assert sampling["reasoning_effort"]["forwarded_to_client"] is None


def test_static_scan_still_reports_findings_without_chat_clients() -> None:
    """Static findings remain available when optional chat clients are absent."""
    skill = Path(__file__).resolve().parents[1] / "fixtures" / "malicious_skill"
    result = _run_static_scan(skill)

    # Findings are a successful scan even though the CLI uses status 1 to
    # signal that the report contains issues.
    assert result.returncode in {0, 1}, result.stderr
    report = json.loads(result.stdout)
    assert report["metadata"]["llm_requested"] is False
    assert report["analysis_completeness"]["is_complete"] is True
    assert report["issues"], "static analyzers must still report findings"
