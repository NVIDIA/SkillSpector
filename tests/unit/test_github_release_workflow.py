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


def test_release_workflow_builds_and_attaches_supported_distributions() -> None:
    """The GitHub Release includes both supported distribution formats."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "uv run --no-sync python -m build --no-isolation" in workflow
    assert "uv run --no-sync twine check dist/*" in workflow
    assert "--asset dist/*.whl" in workflow
    assert "--asset dist/*.tar.gz" in workflow


def test_release_workflow_uses_skillspectors_locked_uv_environment() -> None:
    """Release artifacts are built with the repository's pinned tooling."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86 # v5" in workflow
    assert 'UV_VERSION: "0.10.10"' in workflow
    assert 'version: "${{ env.UV_VERSION }}"' in workflow
    assert "cache-dependency-glob: uv.lock" in workflow
    assert 'python-version: "${{ env.PYTHON_VERSION }}"' in workflow
    assert "uv sync --locked --extra dev --no-install-project" in workflow
    assert "python -m pip install" not in workflow

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lockfile = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert '"hatchling>=1.31.0"' in pyproject
    assert 'name = "hatchling"' in lockfile
