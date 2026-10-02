# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installation commands do not imply snooping or cross-session persistence."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers import static_patterns_agent_snooping as snooping
from skillspector.nodes.analyzers import static_patterns_rogue_agent as rogue
from skillspector.nodes.analyzers import static_runner


def _scan(content: str, path: str = "README.md"):
    return static_runner.run_static_patterns(
        {"components": [path], "file_cache": {path: content}}, [snooping, rogue]
    )


@pytest.mark.parametrize("agent", ["claude", "codex", "gemini"])
@pytest.mark.parametrize("target", ["", "/prompt-master"])
@pytest.mark.parametrize("path", ["README.md", "SKILL.md", "install.sh"])
def test_standard_skill_installation_is_not_snooping_or_persistence(agent, target, path):
    content = (
        f"mkdir -p ~/.{agent}/skills{target}\n"
        f"git clone https://github.com/nidhinjs/prompt-master.git ~/.{agent}/skills/prompt-master\n"
    )

    assert not {"AS3", "RA2"}.intersection(f.rule_id for f in _scan(content, path))


@pytest.mark.parametrize("operation", ["ls", "find", "dir", "gls", "gfind", "/usr/local/bin/gls"])
def test_actual_skill_enumeration_is_still_reported(operation):
    findings = _scan(f"{operation} ~/.claude/skills")

    assert any(f.rule_id == "AS3" for f in findings)


def test_other_skill_read_is_still_reported():
    findings = _scan("cat ~/.claude/skills/other-skill/SKILL.md")

    assert any(f.rule_id == "AS3" for f in findings)


@pytest.mark.parametrize(
    "content",
    [
        "mkdir -p ~/.hidden-agent",
        "mkdir -p ~/.claude/skills-other",
        "mkdir -p ~/.claude/skills/../../.hidden-agent",
        "mkdir -p ~/.claude/skills/.hidden-agent",
        "mkdir -p ~/.claude/skills /tmp/.hidden-agent",
        "mkdir -p ~/.claude/skills && crontab -e",
        "write ~/.claude/skills/agent/state.json",
        "create ~/.claude/skills/agent/state.json",
        "mkdir \\\n    ~/.hidden-agent",
        "mkdir -p \\\r\n    ~/.hidden-agent",
    ],
)
def test_hidden_writes_and_composed_persistence_remain_reported(content):
    assert any(f.rule_id == "RA2" for f in _scan(content))


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r"])
def test_installation_does_not_absorb_later_hidden_write(separator):
    findings = _scan(
        f"mkdir -p ~/.claude/skills{separator}write ~/.hidden-agent/state.json{separator}"
    )
    persistence = [f for f in findings if f.rule_id == "RA2"]

    assert persistence
    assert all(f.start_line == 2 for f in persistence)


def test_hidden_write_matching_respects_unicode_logical_line_breaks():
    # Compact security views intentionally remove Unicode separators. Check
    # source-view attribution independently of those extra obfuscation findings.
    findings = rogue.analyze(
        "mkdir -p ~/.claude/skills\u2028write ~/.hidden-agent/state.json",
        "README.md",
        "markdown",
    )
    persistence = [f for f in findings if f.rule_id == "RA2"]

    assert persistence
    assert all(f.location.start_line == 2 for f in persistence)


@pytest.mark.parametrize("prefix", ["mkdir", "mydir", "my-dir"])
def test_operation_names_are_not_substring_matches(prefix):
    findings = _scan(f"{prefix} -p ~/.claude/skills")

    assert not {"AS3", "RA2"}.intersection(f.rule_id for f in findings)


@pytest.mark.parametrize(
    "command",
    [
        "mkdir ~/.claude/skills",
        "mkdir --parents ~/.claude/skills/",
        "mkdir -p -- ~/.claude/skills/prompt-master",
        '  mkdir -p "~/.claude/skills/prompt-master"  ',
    ],
)
def test_simple_installation_directory_creation_options(command):
    assert not any(f.rule_id == "RA2" for f in _scan(command))


@pytest.mark.parametrize(
    "copy_command",
    [
        "cp -r /tmp/payload ~/.claude/skills/payload",
        "cp -t ~/.claude/skills/payload /tmp/payload",
        "cp --target-directory=~/.claude/skills/payload /tmp/payload",
        "mv /tmp/payload ~/.claude/skills/payload",
        "install -D /tmp/payload ~/.claude/skills/payload/payload",
    ],
)
def test_install_directory_followed_by_payload_copy_is_persistence(copy_command):
    findings = _scan(f"mkdir -p ~/.claude/skills/payload\n{copy_command}\n")
    persistence = [f for f in findings if f.rule_id == "RA2"]

    assert persistence
    assert all(f.start_line == 2 for f in persistence)


def test_install_directory_copy_to_unrelated_target_is_not_persistence():
    findings = _scan("mkdir -p ~/.claude/skills/payload\ncp -r /tmp/payload /tmp/output\n")

    assert not any(f.rule_id == "RA2" for f in findings)


def test_install_directory_copy_to_different_skill_is_not_persistence():
    findings = _scan(
        "mkdir -p ~/.claude/skills/payload\ncp -r /tmp/payload ~/.claude/skills/another-skill\n"
    )

    assert not any(f.rule_id == "RA2" for f in findings)


def test_install_directory_does_not_match_copy_in_a_later_paragraph():
    findings = _scan(
        "mkdir -p ~/.claude/skills/payload\n\ncp -r /tmp/payload ~/.claude/skills/payload\n"
    )

    assert not any(f.rule_id == "RA2" for f in findings)


@pytest.mark.parametrize(
    "rule_id, action",
    [
        ("AS3", "read other skills"),
        ("RA2", "start a background process"),
        ("RA2", "persist state across sessions"),
    ],
)
def test_direct_prohibition_is_not_a_snooping_or_persistence_instruction(rule_id, action):
    assert not any(f.rule_id == rule_id for f in _scan(f"Do not {action}."))


@pytest.mark.parametrize(
    "rule_id, action",
    [
        ("AS3", "read other skills"),
        ("RA2", "start a background process"),
        ("RA2", "persist state across sessions"),
    ],
)
@pytest.mark.parametrize(
    "template",
    [
        "Do not {action}; then {action}.",
        "Do not {action}.\nNow {action}.",
        "Do not {action} unless necessary.",
    ],
)
def test_prohibitions_do_not_hide_later_actions_or_exceptions(rule_id, action, template):
    assert any(f.rule_id == rule_id for f in _scan(template.format(action=action)))
