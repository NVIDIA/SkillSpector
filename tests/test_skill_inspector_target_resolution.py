# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The skill-inspector skill must resolve the installed artefact, not the default branch.

Regression coverage for NVIDIA/SkillSpector#1112: step 1 used to resolve a URL
target by cloning the default branch, so a verdict could describe a different
version, a superset, a different install channel, or skill text generated at
run time. These assertions pin the wording that keeps the review on the exact
artefact that will be installed.
"""

from __future__ import annotations

from pathlib import Path

SKILL_PATH = (
    Path(__file__).resolve().parents[1] / "skills" / "skill-inspector" / "SKILL.md"
)


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_step_1_names_the_install_channel_and_pins_the_version() -> None:
    text = _skill_text()

    assert "install channel" in text
    assert "git clone --branch <tag>" in text
    assert "npm pack <pkg>@<version>" in text
    assert "never a default-branch clone" in text


def test_step_1_records_the_resolved_version_in_the_source_line() -> None:
    text = _skill_text()

    assert "resolved version or commit" in text
    assert "{resolved version or commit}" in text


def test_step_1_covers_runtime_generated_skill_text() -> None:
    text = _skill_text()

    assert "locate the generator and review the rendered text" in text
    assert "trigger `description`" in text


def test_step_1_covers_code_outside_the_installed_path() -> None:
    text = _skill_text()

    assert "outside the installed path" in text
    assert "verdict does not cover it" in text


def test_step_2_scans_the_installed_subset_and_greps_for_local_run_directives() -> None:
    text = _skill_text()

    assert "scan that subset as its own target" in text
    assert '"NEVER SKIP"' in text
    assert '"always run"' in text
    assert "./*.sh" in text
