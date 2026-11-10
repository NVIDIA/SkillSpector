# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoded bundle syntax is independent of the carrier filename."""

import json

import pytest

from skillspector.artifacts import SecurityTextView
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.analyzers import static_runner
from skillspector.oms import OMSProjection


def _scan(monkeypatch, path, projected, *, complete=True):
    carrier = '{"dsseEnvelope": "synthetic carrier"}'
    monkeypatch.setattr(
        static_runner,
        "project_oms_content",
        lambda _content: OMSProjection(SecurityTextView("oms-decoded", projected), complete),
    )
    return static_runner._scan_all_views_detailed(path, carrier, [tm], None)


@pytest.mark.parametrize("path", ["skill.oms.sig", "bundle.json", "notes.md", "bundle.py"])
def test_projected_json_owns_quotes_regardless_of_carrier_name(monkeypatch, path):
    projected = json.dumps({"readable": ["label`with'apostrophe", "ordinary reference"]})
    findings, reason, metrics = _scan(monkeypatch, path, projected)
    assert findings == []
    assert reason is None
    assert getattr(metrics, "parser_diagnostic", None) is None


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("path", ["skill.oms.sig", "bundle.json", "bundle.py"])
def test_projected_commands_keep_findings_and_original_carrier(monkeypatch, path, complete):
    projected = json.dumps({"instruction": "rm -rf /"})
    findings, _, _ = _scan(monkeypatch, path, projected, complete=complete)
    destructive = [finding for finding in findings if finding.rule_id == "TM1"]
    assert destructive
    assert all(
        finding.file == path and finding.start_line == 1 and finding.evidence.get("oms_projection")
        for finding in destructive
    )


def test_unrecognized_raw_bundle_keeps_runtime_diagnostic(monkeypatch):
    monkeypatch.setattr(
        static_runner,
        "project_oms_content",
        lambda content: OMSProjection(SecurityTextView("raw", content), False),
    )
    _, _, metrics = static_runner._scan_all_views_detailed(
        "skill.oms.sig", 'sh -c "$CMD -rf /"', [tm], None
    )
    assert metrics.parser_diagnostic is not None
