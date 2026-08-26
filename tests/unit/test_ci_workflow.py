# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract coverage for pull-request and manual CI checkout selection."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CHECKOUT_ACTION = "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2"
CHECKOUT_REF = (
    "ref: ${{ github.event_name == 'pull_request' "
    "&& github.event.pull_request.head.sha || github.sha }}"
)
BASE_SHA_EXPRESSION = (
    "BASE_SHA: ${{ github.event_name == 'pull_request' "
    "&& github.event.pull_request.base.sha || github.event_name == 'workflow_dispatch' "
    "&& inputs.change_base_sha || github.event.before }}"
)
HEAD_SHA_EXPRESSION = (
    "HEAD_SHA: ${{ github.event_name == 'pull_request' "
    "&& github.event.pull_request.head.sha || github.sha }}"
)


def test_ci_accepts_pull_requests_to_any_base_and_requires_a_manual_change_base() -> None:
    """A stacked PR starts CI without naming its temporary base branch."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    triggers = workflow.split("# Least privilege:", maxsplit=1)[0]

    assert "  pull_request:\n" in triggers
    assert "  pull_request:\n    branches:" not in triggers
    assert '  push:\n    branches: ["main"]' in triggers
    assert "  workflow_dispatch:\n" in triggers
    assert "      change_base_sha:\n" in triggers
    assert '        description: "Full base commit SHA for change detection"' in triggers
    assert "        required: true" in triggers
    assert "        type: string" in triggers
    assert "pull_request_target" not in workflow


def test_every_ci_checkout_uses_the_immutable_event_head_without_credentials() -> None:
    """No job executes PR content from a merge ref or with persisted Git credentials."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    checkout_count = workflow.count(CHECKOUT_ACTION)

    assert checkout_count == 6
    assert workflow.count(CHECKOUT_REF) == checkout_count
    assert workflow.count("persist-credentials: false") == checkout_count
    assert workflow.count("fetch-depth: 0") == 2


def test_change_filter_selects_and_validates_each_event_base() -> None:
    """Manual CI rejects an absent or non-commit base instead of silently running Docker."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert BASE_SHA_EXPRESSION in workflow
    assert HEAD_SHA_EXPRESSION in workflow
    assert '[[ ! "$BASE_SHA" =~ ^[0-9a-f]{40}$ ]]' in workflow
    assert 'git cat-file -e "${BASE_SHA}^{commit}"' in workflow
    assert 'echo "$BASE_SHA"' not in workflow


def test_parser_wheel_smoke_installs_runtime_before_importing_production_loader() -> None:
    """The clean wheel environment can import the production parser module."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    wheel_install = workflow.index("- name: Install exact parser wheels only")
    runtime_install = workflow.index("- name: Install SkillSpector runtime")
    loader_smoke = workflow.index("- name: Smoke production parser loader")

    assert wheel_install < runtime_install < loader_smoke
    assert "uv pip install --python .parser-smoke-venv ." in workflow
    assert "from skillspector.shell_frontend import" in workflow
    assert "runpy.run_path('src/skillspector/shell_frontend.py')" not in workflow
