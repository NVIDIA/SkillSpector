# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract coverage for the label-gated GitHub release workflow."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


def test_release_workflow_pins_actions_to_immutable_commits() -> None:
    """Actions with release write permissions must not follow mutable tags."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    action_refs = re.findall(r"\buses:\s*([^\s#]+)", workflow)

    assert action_refs
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", ref) for ref in action_refs)


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
    assert "persist-credentials: false" in workflow
    assert '--target "${{ github.event.pull_request.merge_commit_sha }}"' in workflow


def test_release_workflow_builds_and_attaches_supported_distributions() -> None:
    """The GitHub Release includes both supported distribution formats."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "uv run --no-sync python -m build --no-isolation" in workflow
    assert "uv run --no-sync twine check dist/*" in workflow
    assert "check_distribution_assets.py dist/*.whl dist/*.tar.gz" in workflow
    assert "--asset dist/*.whl" in workflow
    assert "--asset dist/*.tar.gz" in workflow


def test_release_workflow_uses_skillspectors_locked_uv_environment() -> None:
    """Release artifacts are built with the repository's pinned tooling."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86 # v5.4.2" in workflow
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


def test_index_publish_uses_only_artifacts_from_the_qualified_release_job() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert workflow["on"] == {"pull_request": {"branches": ["main"], "types": ["closed"]}}

    release_job = workflow["jobs"]["publish"]
    index_job = workflow["jobs"]["publish-index"]
    assert index_job["needs"] == "publish"
    assert "github.repository == 'NVIDIA/SkillSpector'" in index_job["if"]

    upload = next(
        step for step in release_job["steps"] if "actions/upload-artifact@" in step.get("uses", "")
    )
    download = next(
        step for step in index_job["steps"] if "actions/download-artifact@" in step.get("uses", "")
    )
    assert "github.event.pull_request.head.repo.full_name == github.repository" in index_job["if"]
    assert upload["with"]["name"] == release_job["outputs"]["distribution-artifact"]
    assert "github.run_attempt" in upload["with"]["name"]
    assert download["with"]["name"] == "${{ needs.publish.outputs.distribution-artifact }}"
    assert upload["with"]["path"].splitlines() == ["dist/*.whl", "dist/*.tar.gz"]
    assert download["with"]["path"] == "dist"


def test_only_the_index_job_can_request_oidc_for_digest_checked_uploads() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert "id-token" not in workflow["permissions"]
    assert "id-token" not in workflow["jobs"]["publish"]["permissions"]

    index_job = workflow["jobs"]["publish-index"]
    assert index_job["permissions"] == {"id-token": "write"}
    assert "pypi-production" in index_job["environment"]["name"]
    assert "pypi-test" in index_job["environment"]["name"]
    assert all("actions/checkout@" not in step.get("uses", "") for step in index_job["steps"])
    command = index_job["steps"][-1]["run"]
    assert "--trusted-publishing always" in command
    assert "--check-url" in command


def test_index_retry_keeps_the_index_selected_for_the_release_artifacts() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    release_job = workflow["jobs"]["publish"]
    index_job = workflow["jobs"]["publish-index"]

    selected_index = "needs.publish.outputs.distribution-index"
    index_setting = "vars.SKILLSPECTOR_PUBLISH_INDEX"
    assert release_job["outputs"].get("distribution-index") == "${{ " + index_setting + " }}"
    assert f"{selected_index} == 'pypi'" in index_job["if"]
    assert f"{selected_index} == 'testpypi'" in index_job["if"]
    assert f"{index_setting} == {selected_index}" in index_job["if"]

    endpoints = index_job["steps"][-1]["env"]
    for selection in (*index_job["environment"].values(), *endpoints.values()):
        assert selected_index in selection
        assert index_setting not in selection
