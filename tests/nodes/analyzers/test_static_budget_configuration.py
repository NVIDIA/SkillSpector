# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configurable static allowances remain bounded by the parent workflow."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from skillspector.artifacts import classify_artifact
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason, finalize_ledger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.nodes import report as report_module
from skillspector.nodes.analyzers import static_runner, static_yara


@pytest.mark.parametrize("value", ["", "invalid", "0", "-1", "nan", "inf", "-inf"])
def test_invalid_static_allowance_warns_and_retains_default(value: str, caplog) -> None:
    assert static_runner._static_max_seconds_from_environment(value) == 300.0
    assert "SKILLSPECTOR_MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT" in caplog.text


@pytest.mark.parametrize("value,expected", [(None, 300.0), ("45.5", 45.5), ("900", 900.0)])
def test_fresh_process_shares_configured_allowance_with_yara(
    value: str | None, expected: float
) -> None:
    env = os.environ.copy()
    name = "SKILLSPECTOR_MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT"
    env.pop(name, None)
    if value is not None:
        env[name] = value
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from skillspector.nodes.analyzers import static_runner, static_yara; "
            "print(json.dumps([static_runner.MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT, "
            "static_yara.MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT]))",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == [expected, expected]


@pytest.mark.parametrize(
    "configured,parent,limited", [(300.0, 600.0, False), (20.0, 600.0, True), (300.0, 20.0, True)]
)
def test_static_work_beyond_thirty_seconds_respects_effective_allowance(
    monkeypatch: pytest.MonkeyPatch, configured: float, parent: float, limited: bool
) -> None:
    now = 0.0

    class SlowModule:
        ANALYZER_ID = "static_tool_misuse"

        @staticmethod
        def analyze(**_kwargs):
            nonlocal now
            now = 31.0
            return []

    monkeypatch.setattr(static_runner, "MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT", configured)
    monkeypatch.setattr(static_runner.time, "monotonic", lambda: now)
    findings, reason, metrics = static_runner._scan_all_views_detailed(
        "example.txt", "ordinary text", [SlowModule], None, timeout_seconds=parent
    )
    assert findings == []
    assert reason == (LedgerReason.RUNTIME_LIMIT if limited else None)
    if limited:
        assert metrics == {"observed_seconds": 31.0, "limit_seconds": 20.0}


def test_runtime_aware_static_module_retains_prefix_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0

    class RuntimeAwareModule:
        ANALYZER_ID = "runtime_aware_static"
        USES_RUNTIME_CHECK = True

        @staticmethod
        def analyze(*, content, file_path, file_type, check_runtime):
            nonlocal now
            del content, file_type
            AnalyzerFinding(
                rule_id="P9",
                message="Bounded prefix evidence",
                severity=Severity.LOW,
                confidence=0.1,
                location=Location(file=file_path, start_line=1),
            )
            now = 31.0
            check_runtime()
            raise AssertionError("expired callback must stop the analyzer")

    monkeypatch.setattr(static_runner, "MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT", 30.0)
    monkeypatch.setattr(static_runner.time, "monotonic", lambda: now)
    monkeypatch.setattr(report_module, "is_llm_available", lambda **_: (False, "disabled"))
    content = "ordinary text"
    state = {
        "components": ["SKILL.md"],
        "file_cache": {"SKILL.md": content},
        "artifact_inventory": [classify_artifact("SKILL.md", content.encode())],
        "component_metadata": [
            {"path": "SKILL.md", "type": "markdown", "lines": 1, "executable": False}
        ],
        "output_format": "json",
        "use_llm": False,
    }

    response = static_runner.run_static_patterns_with_ledger(state, [RuntimeAwareModule])

    assert len(response["findings"]) == 1
    assert response["findings"][0].rule_id == "P9"
    event = response["inspection_ledger"][0]
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.RUNTIME_LIMIT
    assert event["emitted_finding_ids"] == [response["findings"][0].finding_id]

    merged_state = {**state, **response}
    completeness, effective_ids = finalize_ledger(merged_state)
    rendered = report_module.report(
        {
            **merged_state,
            "analysis_completeness": completeness,
            "effective_finding_ids": effective_ids,
        }
    )

    assert completeness["is_complete"] is False
    assert any(
        row["reason_code"] is LedgerReason.RUNTIME_LIMIT
        for row in completeness["ledger_exceptions"]
    )
    assert rendered["risk_recommendation"] == "CAUTION"


@pytest.mark.parametrize(
    "configured,parent,expected", [(300.0, 600.0, 300), (45.5, 600.0, 45), (300.0, 42.5, 42)]
)
def test_yara_engine_receives_effective_allowance(
    monkeypatch: pytest.MonkeyPatch, configured: float, parent: float, expected: int
) -> None:
    calls = []

    class RecordingRules:
        def match(self, **kwargs):
            calls.append(kwargs)
            return []

    monkeypatch.setattr(static_yara, "MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT", configured)
    result = static_yara._match_file(
        RecordingRules(), "ordinary text", "example.txt", timeout_seconds=parent, clock=lambda: 0.0
    )
    assert result.reason is None
    assert calls[0]["timeout"] == expected
