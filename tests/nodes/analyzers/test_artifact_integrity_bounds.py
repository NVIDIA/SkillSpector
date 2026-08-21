# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource-bound regressions for the artifact-integrity analyzer."""

from __future__ import annotations

from skillspector.nodes.analyzers import artifact_integrity


class _ExpiringWorkflowBudget:
    def __init__(self, positive_calls: int) -> None:
        self._positive_calls = positive_calls
        self.calls = 0

    def remaining_seconds(self) -> float:
        self.calls += 1
        return 0.001 if self.calls <= self._positive_calls else 0.0


def test_deadline_during_content_marks_current_and_remaining_partial() -> None:
    workflow_budget = _ExpiringWorkflowBudget(5)
    result = artifact_integrity.node(
        {
            "components": ["first.png", "second.md"],
            "local_file_cache": {
                "first.png": "plain text",
                "second.md": "ordinary text",
            },
            "artifact_inventory": [
                {"path": "first.png", "misleading_extension": True},
                {"path": "second.md", "misleading_extension": False},
            ],
            "workflow_resource_budget": workflow_budget,
        }
    )

    assert [finding.rule_id for finding in result["findings"]] == ["AE2"]
    assert [event["outcome"] for event in result["inspection_ledger"]] == [
        "partial",
        "partial",
    ]
    assert all(event["reason_code"] == "runtime_limit" for event in result["inspection_ledger"])
    assert result["analyzer_status_events"][0]["status"] == "degraded"


def test_finding_cap_stops_construction_and_marks_affected_suffix_partial(
    monkeypatch,
) -> None:
    monkeypatch.setattr(artifact_integrity, "MAX_FINDINGS_PER_ARTIFACT", 2)
    monkeypatch.setattr(artifact_integrity, "MAX_FINDINGS_PER_ANALYZER", 2)
    result = artifact_integrity.node(
        {
            "components": ["first.png", "second.png"],
            "local_file_cache": {"first.png": "\x00", "second.png": "plain text"},
            "artifact_inventory": [
                {
                    "path": "first.png",
                    "misleading_extension": True,
                    "contains_nul": True,
                },
                {"path": "second.png", "misleading_extension": True},
            ],
        }
    )

    assert len(result["findings"]) == 2
    assert [finding.rule_id for finding in result["findings"]] == ["AE2", "AE3"]
    assert [event["outcome"] for event in result["inspection_ledger"]] == [
        "partial",
        "partial",
    ]
    assert result["inspection_ledger"][0]["observed_findings"] == 3
    assert result["inspection_ledger"][0]["limit_findings"] == 2
    assert result["analyzer_status_events"][0]["status"] == "degraded"
