# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configured dependency-source ceilings must affect real scans (#696)."""

from types import SimpleNamespace

import pytest

from skillspector import dependency_sources
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.models import Finding
from skillspector.nodes.analyzers import static_patterns_supply_chain as supply_chain


@pytest.fixture
def slow_findings(monkeypatch):
    """Advance a private clock during analysis without sleeping or fetching URLs."""
    clock = SimpleNamespace(now=0.0, cost=6.0)
    monkeypatch.setattr(dependency_sources, "time", SimpleNamespace(monotonic=lambda: clock.now))
    original = dependency_sources._finding

    def finding(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.now += clock.cost
        return result

    monkeypatch.setattr(dependency_sources, "_finding", finding)
    return clock


@pytest.mark.parametrize(
    "ceiling,caller_timeout,cost,expected_count,expired_limit",
    [
        pytest.param(5.0, None, 6.0, 1, 5.0, id="default-still-stops-at-five"),
        pytest.param(30.0, None, 6.0, 2, None, id="raised-ceiling-without-caller-timeout"),
        pytest.param(30.0, 30.0, 6.0, 2, None, id="raised-ceiling-with-long-workflow"),
        pytest.param(30.0, 2.0, 3.0, 1, 2.0, id="shorter-caller-timeout-wins"),
        pytest.param(2.0, 30.0, 3.0, 1, 2.0, id="shorter-configured-ceiling-wins"),
        pytest.param(30.0, 0.0, 1.0, 0, 0.0, id="exhausted-caller-budget"),
        pytest.param(30.0, -1.0, 1.0, 0, 0.0, id="negative-caller-budget"),
    ],
)
def test_configured_ceiling_controls_scan_completion(
    monkeypatch, slow_findings, ceiling, caller_timeout, cost, expected_count, expired_limit
):
    monkeypatch.setattr(dependency_sources, "MAX_ANALYSIS_SECONDS", ceiling)
    slow_findings.cost = cost
    files = {f"{name}/.npmrc": "registry=https://packages.example.invalid\n" for name in ("a", "b")}

    result = dependency_sources.analyze_dependency_sources_detailed(
        list(files), files, timeout_seconds=caller_timeout
    )

    assert [f.file for f in result.findings] == list(files)[:expected_count]
    assert all(f.rule_id == "SC10" and f.severity == "HIGH" for f in result.findings)
    if expired_limit is None:
        assert result.limitations == []
        assert slow_findings.now == 12.0
    else:
        assert len(result.limitations) == 1
        limitation = result.limitations[0]
        assert limitation.reason is LedgerReason.RUNTIME_LIMIT
        assert limitation.path == "a/.npmrc"
        assert limitation.limit_seconds == expired_limit
        assert limitation.observed_seconds == slow_findings.now


@pytest.mark.parametrize("ceiling,complete", [(5.0, False), (30.0, True)])
def test_slow_parse_without_findings_still_accounts_for_elapsed_time(
    monkeypatch, ceiling, complete
):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(dependency_sources, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(dependency_sources, "MAX_ANALYSIS_SECONDS", ceiling)
    original = dependency_sources._changes_for_file

    def slow_parse(*args, **kwargs):
        original(*args, **kwargs)
        clock.now += 6.0

    monkeypatch.setattr(dependency_sources, "_changes_for_file", slow_parse)
    files = {".npmrc": "registry=https://registry.npmjs.org/\n"}
    result = dependency_sources.analyze_dependency_sources_detailed(list(files), files)

    assert result.findings == []
    if complete:
        assert result.limitations == []
    else:
        assert len(result.limitations) == 1
        assert result.limitations[0].reason is LedgerReason.RUNTIME_LIMIT
        assert result.limitations[0].limit_seconds == 5.0
        assert result.limitations[0].observed_seconds == 6.0


def test_node_preserves_partial_status_and_findings_under_short_workflow(
    monkeypatch, slow_findings
):
    monkeypatch.setattr(dependency_sources, "MAX_ANALYSIS_SECONDS", 30.0)
    slow_findings.cost = 3.0
    earlier = Finding(rule_id="SC2", severity="HIGH", file="setup.sh", message="Existing risk")
    monkeypatch.setattr(
        supply_chain.static_runner,
        "run_static_patterns_with_ledger",
        lambda *_args: {
            "findings": [earlier],
            "inspection_ledger": [],
            "analyzer_status_events": [],
        },
    )
    files = {".npmrc": "registry=https://packages.example.invalid\n"}
    response = supply_chain.node(
        {
            "components": list(files),
            "file_cache": files,
            "workflow_resource_budget": SimpleNamespace(
                remaining_seconds=lambda: max(0.0, 2.0 - slow_findings.now)
            ),
        }
    )

    assert response["findings"][0] is earlier
    assert [f.rule_id for f in response["findings"]] == ["SC2", "SC10"]
    partial = [
        event
        for event in response["inspection_ledger"]
        if "dependency_source" in event["analyzer_id"] and event["outcome"] is LedgerOutcome.PARTIAL
    ]
    assert len(partial) == 1
    assert partial[0]["reason_code"] is LedgerReason.RUNTIME_LIMIT
    assert partial[0]["limit_seconds"] == 2.0
    assert response["analyzer_status_events"][0]["status"] == "degraded"
