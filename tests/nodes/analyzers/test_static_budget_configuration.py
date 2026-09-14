# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configurable static allowances remain bounded by the parent workflow."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from skillspector.inspection_ledger import LedgerReason
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
