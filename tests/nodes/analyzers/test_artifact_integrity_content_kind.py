# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary-artifact gating regressions for the artifact-integrity analyzer."""

from __future__ import annotations

from skillspector.artifacts import ContentKind
from skillspector.nodes.analyzers import artifact_integrity


def test_binary_kind_artifact_skips_text_signal_findings() -> None:
    result = artifact_integrity.node(
        {
            "components": ["assets/diagram.png"],
            "local_file_cache": {"assets/diagram.png": "\x00\x89PNG\ufffd noise"},
            "artifact_inventory": [
                {
                    "path": "assets/diagram.png",
                    "content_kind": ContentKind.BINARY,
                    "contains_nul": True,
                }
            ],
        }
    )

    assert result["findings"] == []
    assert [event["outcome"] for event in result["inspection_ledger"]] == ["completed"]
    assert result["analyzer_status_events"][0]["status"] == "completed"


def test_text_kind_artifact_still_reports_nul_and_unicode_signals() -> None:
    result = artifact_integrity.node(
        {
            "components": ["payload.md"],
            "local_file_cache": {"payload.md": "safe\x00text"},
            "artifact_inventory": [
                {
                    "path": "payload.md",
                    "content_kind": ContentKind.TEXT,
                    "contains_nul": True,
                }
            ],
        }
    )

    assert [finding.rule_id for finding in result["findings"]] == ["AE3", "AE4"]


def test_missing_classification_keeps_prior_behavior() -> None:
    result = artifact_integrity.node(
        {
            "components": ["legacy.bin"],
            "local_file_cache": {"legacy.bin": "safe\x00text"},
            "artifact_inventory": [
                {"path": "legacy.bin", "contains_nul": True},
            ],
        }
    )

    assert [finding.rule_id for finding in result["findings"]] == ["AE3", "AE4"]
