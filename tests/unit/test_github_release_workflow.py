# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract coverage for the label-gated GitHub release workflow."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


def test_release_workflow_publishes_only_labeled_merged_main_prs() -> None:
    """The PR label is the sole release-qualification signal."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert 'branches: ["main"]' in workflow
    assert "types: [closed]" in workflow
    assert "github.event.pull_request.merged == true" in workflow
    assert "github.event.pull_request.labels.*.name, 'release:publish'" in workflow
    assert "release/oss-*" not in workflow
    assert "skillspector-release.json" not in workflow


def test_release_workflow_tags_the_merged_pr_commit() -> None:
    """The helper reads the version and tags the merge that caused the event."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "ref: ${{ github.event.pull_request.merge_commit_sha }}" in workflow
    assert '--target "${{ github.event.pull_request.merge_commit_sha }}"' in workflow
