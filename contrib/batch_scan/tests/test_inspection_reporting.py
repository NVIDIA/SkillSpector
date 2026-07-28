# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch propagation tests for canonical inspection completeness."""

from __future__ import annotations

import json
from pathlib import Path

from contrib.batch_scan.reports import _format_json, _format_markdown, _format_terminal
from contrib.batch_scan.runner import entry_from_result


def test_entry_from_result_preserves_analysis_completeness(tmp_path: Path) -> None:
    completeness = {
        "execution_successful": False,
        "ledger_exceptions": [{"reason_code": "read_error", "path": "x.py"}],
        "scope_exclusions": [],
        "analyzer_statuses": [],
    }
    entry = entry_from_result(
        {
            "analysis_completeness": completeness,
            "execution_successful": False,
            "risk_score": 0,
            "risk_severity": "LOW",
            "risk_recommendation": "CAUTION",
            "component_metadata": [],
            "manifest": {"name": "broken"},
            "filtered_findings": [],
        },
        tmp_path,
        tmp_path,
    )

    assert entry["analysis_completeness"] == completeness
    assert entry["execution_successful"] is False


def test_batch_formats_preserve_every_child_ledger_exception() -> None:
    entry = {
        "skill": {"name": "ledger-skill", "language": "en"},
        "risk_assessment": {"score": 0, "severity": "LOW", "recommendation": "CAUTION"},
        "components": [],
        "issues": [{"id": "P1", "finding_id": "finding-batch-1"}],
        "analysis_completeness": {
            "execution_successful": False,
            "ledger_exceptions": [
                {"reason_code": "read_error", "path": "a.py", "message": "could not read"},
                {"reason_code": "syntax_error", "path": "b.py", "message": "could not parse"},
            ],
            "scope_exclusions": [],
            "analyzer_statuses": [],
        },
        "execution_successful": False,
    }

    payload = json.loads(_format_json([entry]))
    exceptions = payload["skills"][0]["analysis_completeness"]["ledger_exceptions"]
    assert [item["reason_code"] for item in exceptions] == ["read_error", "syntax_error"]
    assert payload["skills"][0]["issues"][0]["finding_id"] == "finding-batch-1"

    for rendered in (_format_terminal([entry]), _format_markdown([entry])):
        assert "read_error" in rendered
        assert "syntax_error" in rendered
