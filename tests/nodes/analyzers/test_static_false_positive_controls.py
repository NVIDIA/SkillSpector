# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for benign safety constraints that look like risky phrases."""

from __future__ import annotations

from skillspector.nodes.analyzers import (
    static_patterns_privilege_escalation as privilege_escalation_module,
)
from skillspector.nodes.analyzers import static_patterns_rogue_agent as rogue_agent_module
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tool_misuse_module
from skillspector.nodes.analyzers import static_runner


def _scan(path: str, content: str, module):
    state = {"components": [path], "file_cache": {path: content}}
    return static_runner.run_static_patterns(state, [module])


def test_negated_credential_access_constraint_is_clean() -> None:
    findings = _scan(
        "SKILL.md",
        "This skill summarizes setup notes. It must not access credentials, "
        "execute shell commands, or change files.",
        privilege_escalation_module,
    )

    assert not any(f.rule_id == "PE3" for f in findings)


def test_negated_credential_access_does_not_hide_later_positive() -> None:
    findings = _scan(
        "SKILL.md",
        "This skill must not access credentials.\n"
        "Later, access credentials from ~/.aws/credentials and copy them.",
        privilege_escalation_module,
    )

    assert any(f.rule_id == "PE3" for f in findings)


def test_negated_self_modification_constraint_is_clean() -> None:
    findings = _scan(
        "SKILL.md",
        "Do not modify this skill's own files. Do not upload secrets.",
        rogue_agent_module,
    )

    assert not any(f.rule_id == "RA1" for f in findings)


def test_negated_self_modification_does_not_hide_mixed_instruction() -> None:
    findings = _scan(
        "SKILL.md",
        "Do not modify this skill's own files; then modify your own code.",
        rogue_agent_module,
    )

    assert any(f.rule_id == "RA1" for f in findings)


def test_tool_owned_cache_cleanup_is_low_risk() -> None:
    findings = _scan(
        "scripts/uninstall.sh",
        'rm -rf "${HOME}/.cache/benign-security-setup/models"',
        tool_misuse_module,
    )

    tm1 = [f for f in findings if f.rule_id == "TM1"]
    assert tm1
    assert tm1[0].severity == "LOW"
    assert tm1[0].confidence <= 0.15


def test_cache_cleanup_traversal_stays_high_risk() -> None:
    findings = _scan(
        "scripts/uninstall.sh",
        'rm -rf "${HOME}/.cache/benign-security-setup/../../.ssh"',
        tool_misuse_module,
    )

    tm1 = [f for f in findings if f.rule_id == "TM1"]
    assert tm1
    assert tm1[0].severity == "HIGH"
