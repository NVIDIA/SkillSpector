# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MCP rug-pull analyzer (RP1–RP3)."""

from __future__ import annotations

import json

import pytest

from skillspector.nodes.analyzers.mcp_rug_pull import _strip_yaml_comment, node
from skillspector.nodes.build_context import build_context
from skillspector.nodes.deduplicate import deduplicate
from skillspector.nodes.report import report
from skillspector.state import SkillspectorState


def _state(
    manifest: dict | None = None, file_cache: dict[str, str] | None = None
) -> SkillspectorState:
    state: SkillspectorState = {}
    if manifest is not None:
        state["manifest"] = manifest
    if file_cache is not None:
        state["file_cache"] = file_cache
    return state


def test_rp1_npx_unpinned():
    """RP1 detects npx without @version suffix."""
    result = node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={"setup.sh": "npx @scope/mcp-server\n"},
        )
    )
    rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
    assert len(rp1) == 1
    assert "npx @scope/mcp-server" in rp1[0].matched_text
    issue = json.loads(report({"filtered_findings": rp1, "output_format": "json"})["report_body"])[
        "issues"
    ][0]
    assert issue["pattern"] == rp1[0].message
    assert issue["finding"] == "npx @scope/mcp-server"


def test_rp1_npx_match_does_not_cross_lines():
    """A trailing ``npx`` must not combine with the next line as a command."""
    for content in (
        "---\nname: npx\ndescription: repro\n---\n",
        "Install it with npx\nthe package manager.\n",
    ):
        result = node(_state(file_cache={"SKILL.md": content}))
        assert not [finding for finding in result["findings"] if finding.rule_id == "RP1"]


def test_rp1_pnpx_unpinned():
    """RP1 also detects pnpm's npx-style runner without a version pin."""
    result = node(_state(file_cache={"setup.sh": "pnpx @scope/mcp-server\n"}))
    rp1 = [finding for finding in result["findings"] if finding.rule_id == "RP1"]

    assert len(rp1) == 1
    assert rp1[0].matched_text == "pnpx @scope/mcp-server"


def test_rp1_npx_requires_a_word_boundary():
    """An unrelated identifier ending in ``npx`` is not a command."""
    result = node(_state(file_cache={"setup.sh": "foonpx @scope/mcp-server\n"}))

    assert not [finding for finding in result["findings"] if finding.rule_id == "RP1"]


def test_rp1_yaml_mcp_config_unpinned():
    """RP1 detects unpinned npx-style commands in YAML MCP config args."""
    configs = (
        """mcpServers:\n  fs:\n    command: npx\n    args: ["-y", "@scope/mcp-server"]\n""",
        """servers:\n  goose:\n    cmd: pnpx\n    args:\n      - "-y"\n      - "@scope/mcp-server"\n""",
    )

    for config in configs:
        result = node(_state(file_cache={"mcp.yaml": config}))
        rp1 = [finding for finding in result["findings"] if finding.rule_id == "RP1"]

        assert len(rp1) == 1
        assert "@scope/mcp-server" in rp1[0].matched_text


def test_rp1_yaml_mcp_config_pinned_no_finding():
    """RP1 skips YAML MCP args whose package token pins a version."""
    configs = (
        """mcpServers:\n  fs:\n    command: npx\n    args: ["-y", "@scope/mcp-server@1.2.3"]\n""",
        """servers:\n  goose:\n    cmd: pnpx\n    args:\n      - "-y"\n      - "@scope/mcp-server@1.2.3"\n""",
    )

    for config in configs:
        result = node(_state(file_cache={"mcp.yaml": config}))

        assert not [finding for finding in result["findings"] if finding.rule_id == "RP1"]


@pytest.mark.parametrize("style", ["flow", "block"])
@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["-y", "@scope/server", ""], 1),
        (["-y", "@scope/server@1.2.3", ""], 0),
        (["", "@scope/server"], 0),
        (["-y", "", "@scope/server"], 0),
        (["-y", "", "@scope/server@1.2.3"], 0),
        ([""], 0),
        (["-y", ""], 0),
    ],
)
def test_rp1_yaml_empty_arguments_do_not_crash_or_shift_package(style, quote, arguments, expected):
    quoted = [quote + argument + quote for argument in arguments]
    args = (
        "    args: [" + ", ".join(quoted) + "]\n"
        if style == "flow"
        else "    args:\n" + "".join("      - " + argument + "\n" for argument in quoted)
    )
    result = node(_state(file_cache={"mcp.yaml": "mcpServers:\n  fs:\n    command: npx\n" + args}))
    rp1 = [finding for finding in result["findings"] if finding.rule_id == "RP1"]
    assert len(rp1) == expected
    if rp1:
        assert rp1[0].start_line == 3
        assert "@scope/server" in rp1[0].matched_text


def test_rp1_yaml_empty_package_does_not_abort_other_configs_or_files():
    content = (
        'mcpServers:\n  empty:\n    command: npx\n    args: ["-y", ""]\n'
        '  real:\n    command: pnpx\n    args: ["@scope/server"]\n'
    )
    result = node(_state(file_cache={"mcp.yaml": content, "setup.sh": "npx another-server\n"}))
    rp1 = [finding for finding in result["findings"] if finding.rule_id == "RP1"]
    assert [(finding.file, finding.start_line) for finding in rp1] == [
        ("mcp.yaml", 6),
        ("setup.sh", 1),
    ]
    assert all(event["outcome"] == "completed" for event in result["inspection_ledger"])


@pytest.mark.parametrize(
    "layout",
    [
        'mcpServers:\n  fs:\n    command: npx\n    env:\n      FOO: bar\n    args: ["-y", "PACKAGE"]\n',
        'mcpServers:\n  fs:\n    command: npx\n    type: stdio\n    cwd: /tmp\n    description: server\n    args: ["-y", "PACKAGE"]\n',
        'servers:\n  - command: npx\n    args: ["-y", "PACKAGE"]\n',
        'servers:\n  - command: pnpx\n    type: stdio\n    args:\n      - "-y"\n      - "PACKAGE"\n',
        'mcpServers:\n  fs:\n    args: ["-y", "PACKAGE"]\n    env: {}\n    command: npx\n',
        'mcpServers:\n  fs:\n    args:\n      - "-y"\n      - "PACKAGE"\n    command: npx\n',
        'servers:\n  - args: ["-y", "PACKAGE"]\n    command: npx\n',
        'servers:\n  - name: fs\n    args:\n      - "-y"\n      - "PACKAGE"\n    command: npx\n',
        'mcpServers:\n  fs:\n    command: /usr/local/bin/npx\n    args: ["-y", "PACKAGE"]\n',
        'mcpServers:\n  fs:\n    args: ["-y", "PACKAGE"]\n    command: "./node_modules/.bin/pnpx"\n',
    ],
)
@pytest.mark.parametrize("pinned", [False, True])
def test_rp1_yaml_sibling_layouts_preserve_pin_behavior(layout, pinned):
    package = "@scope/server@1.2.3" if pinned else "@scope/server"
    content = layout.replace("PACKAGE", package)
    rp1 = [
        f for f in node(_state(file_cache={"mcp.yaml": content}))["findings"] if f.rule_id == "RP1"
    ]
    assert len(rp1) == (0 if pinned else 1)
    if rp1:
        assert "@scope/server" in rp1[0].matched_text
        assert rp1[0].start_line == next(
            index for index, line in enumerate(content.splitlines(), 1) if "command:" in line
        )


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    ("header", "command", "middle", "args"),
    [
        (
            "mcpServers:\n  fs:\n",
            "    command: npx\n",
            "",
            '    args:\n    - -y\n    - "PACKAGE"\n',
        ),
        ("servers:\n- name: fs\n", "  command: pnpx\n", "", '  args:\n  - -y\n  - "PACKAGE"\n'),
        (
            "mcpServers:\n  fs:\n",
            "    command: npx\n",
            "    autoApprove:\n    - read_file\n",
            '    args: ["PACKAGE"]\n',
        ),
        (
            "mcpServers:\n  fs:\n",
            "    command: npx\n",
            "    env:\n"
            + "      KEY{}: value\n".format(0)
            + "".join(f"      KEY{i}: value\n" for i in range(1, 12)),
            '    args: ["PACKAGE"]\n',
        ),
        (
            "mcpServers:\n  fs:\n",
            "    command: npx\n",
            "    description: |\n" + "      description text\n" * 12,
            '    args: ["PACKAGE"]\n',
        ),
        (
            "mcpServers:\n  fs:\n",
            "    command: npx\n",
            "",
            "    args:\n" + "      - -y\n" * 12 + '      - "PACKAGE"\n',
        ),
    ],
)
def test_rp1_yaml_indentless_and_long_values(
    header, command, middle, args, pinned, reverse, newline
):
    package = "@scope/server@1.2.3" if pinned else "@scope/server"
    pair = args + middle + command if reverse else command + middle + args
    content = (header + pair).replace("PACKAGE", package).replace("\n", newline)
    rp1 = [
        f for f in node(_state(file_cache={"mcp.yaml": content}))["findings"] if f.rule_id == "RP1"
    ]
    assert len(rp1) == (0 if pinned else 1)
    if rp1:
        assert rp1[0].start_line == next(
            i for i, line in enumerate(content.splitlines(), 1) if "command:" in line
        )


@pytest.mark.parametrize(
    "args",
    [
        'args: # "@scope/server@1.2.3"\n      - "@scope/server"',
        'args:\n      - "-y" # "decoy@1.2.3"\n      - "@scope/server"',
        'args: ["@scope/server"] # "decoy@1.2.3"',
    ],
)
def test_rp1_yaml_comments_cannot_supply_a_fake_package_pin(args):
    content = "mcpServers:\n  fs:\n    command: npx\n    " + args + "\n"
    rp1 = [
        f for f in node(_state(file_cache={"mcp.yaml": content}))["findings"] if f.rule_id == "RP1"
    ]
    assert len(rp1) == 1


@pytest.mark.parametrize(
    "content",
    [
        'servers:\n  - command: npx\n  - args: ["@scope/server"]\n',
        'servers:\n  - args: ["@scope/server"]\n  - command: npx\n',
        'mcpServers:\n  first:\n    command: npx\n  second:\n    args: ["@scope/server"]\n',
        'mcpServers:\n  first:\n    args: ["@scope/server"]\n  second:\n    command: npx\n',
        'mcpServers:\n  fs:\n    command: npx\n    env:\n      args: ["@scope/server"]\n',
    ],
)
def test_rp1_yaml_does_not_bind_args_from_another_mapping(content):
    assert not [
        f for f in node(_state(file_cache={"mcp.yaml": content}))["findings"] if f.rule_id == "RP1"
    ]


@pytest.mark.parametrize("direction", ["before", "after"])
@pytest.mark.parametrize("distance", [8, 9])
def test_rp1_yaml_sibling_search_remains_bounded(direction, distance):
    command = "    command: npx\n"
    args = '    args: ["@scope/server"]\n'
    intervening = "".join(f"    field{i}: value\n" for i in range(distance - 1))
    pair = args + intervening + command if direction == "before" else command + intervening + args
    content = "mcpServers:\n  fs:\n" + pair
    rp1 = [
        f for f in node(_state(file_cache={"mcp.yaml": content}))["findings"] if f.rule_id == "RP1"
    ]
    assert len(rp1) == (1 if distance == 8 else 0)


def test_rp1_yaml_nested_pinned_args_do_not_hide_sibling_package():
    content = (
        "servers:\r\n  - command: npx\r\n    env:\r\n"
        '      args: ["decoy@1.2.3"]\r\n    args: ["@scope/server"]\r\n'
    )
    rp1 = [
        f for f in node(_state(file_cache={"mcp.yaml": content}))["findings"] if f.rule_id == "RP1"
    ]
    assert len(rp1) == 1
    assert rp1[0].start_line == 2


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('"pkg#fragment" # "decoy@1.2.3"', '"pkg#fragment" '),
        ("'it''s # quoted' # tail", "'it''s # quoted' "),
        ('"escaped\\" # quoted" # tail', '"escaped\\" # quoted" '),
        ("pkg#fragment", "pkg#fragment"),
    ],
)
def test_yaml_comment_stripping_preserves_quoted_content(line, expected):
    assert _strip_yaml_comment(line) == expected


def test_rp1_npx_still_matches_flags_on_the_same_line():
    """Common npx flags remain supported after restricting whitespace."""
    result = node(_state(file_cache={"setup.sh": "npx -y @scope/mcp-server\n"}))
    rp1 = [finding for finding in result["findings"] if finding.rule_id == "RP1"]

    assert len(rp1) == 1
    assert rp1[0].matched_text == "npx -y @scope/mcp-server"


def test_rp1_scans_cached_files_without_a_manifest():
    """Cache-based RP1 checks remain applicable when manifest parsing failed."""
    result = node(_state(file_cache={"setup.sh": "npx @scope/mcp-server\n"}))

    assert [finding.rule_id for finding in result["findings"]] == ["RP1"]


def test_cache_only_scan_skips_manifest_comparison_checks():
    """A prior manifest cannot be diffed against an absent current manifest."""
    state = _state(file_cache={"setup.sh": "npx @scope/mcp-server\n"})
    state["previous_manifest"] = {
        "triggers": ["legacy"],
        "parameters": [{"name": "token", "type": "string"}],
    }

    result = node(state)

    assert [finding.rule_id for finding in result["findings"]] == ["RP1"]


def test_rp1_npx_pinned_no_finding():
    """RP1 does not fire when npx has @version."""
    result = node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={"setup.sh": "npx @scope/mcp-server@1.2.3\n"},
        )
    )
    rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
    assert len(rp1) == 0


def test_rp1_unrelated_version_pin_does_not_suppress():
    """A pin on another argument or command on the same line does not pin the package."""
    for content, expected in (
        ("npx evil-package --label helper@1.2.3\n", "npx evil-package"),
        ("npx @scope/mcp-server http://localhost:3000/sse\n", "npx @scope/mcp-server"),
        ("npx @scope/server-a && npx @scope/server-b@1.2.3\n", "npx @scope/server-a"),
        ("uvx my-mcp-server --with helper==1.2.3\n", "uvx my-mcp-server"),
        ("pip install my-mcp-server other-package==1.2.3\n", "pip install my-mcp-server"),
    ):
        result = node(_state(file_cache={"setup.sh": content}))
        rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
        assert [f.matched_text for f in rp1] == [expected], content


def test_rp1_version_pin_attached_to_package_no_finding():
    """A pin attached to the package operand still counts when other arguments follow."""
    for content in (
        "npx -y @scope/mcp-server@1.2.3 --label helper\n",
        "npx -p @scope/mcp-server@1.2.3 mcp-server\n",
        "uvx my-mcp-server==1.2.3 --host 127.0.0.1:8000\n",
        "pip install my-mcp-server[cli]==1.2.3\n",
    ):
        result = node(_state(file_cache={"setup.sh": content}))
        assert not [f for f in result["findings"] if f.rule_id == "RP1"], content


def test_rp1_uvx_unpinned():
    """RP1 detects uvx without ==version."""
    result = node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={"install.sh": "uvx my-mcp-server\n"},
        )
    )
    rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
    assert len(rp1) >= 1
    assert any("uvx" in f.matched_text for f in rp1)


def test_rp1_docker_unpinned():
    """RP1 detects docker run without tag."""
    node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={"Dockerfile": "FROM org/mcp-server\n"},
        )
    )
    # RP1 docker pattern matches "docker pull|run|create"
    # FROM in Dockerfile isn't matched by our regex, so update test
    result2 = node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={"setup.sh": "docker run org/mcp-server\n"},
        )
    )
    rp1 = [f for f in result2["findings"] if f.rule_id == "RP1"]
    assert len(rp1) >= 1


def test_rp1_docker_credentials_are_redacted_in_reports():
    result = node(
        _state(
            file_cache={
                "setup.sh": "docker pull https://deploy:s3cret@registry.example.com/team/image"
            }
        )
    )
    rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
    assert len(rp1) == 1

    json_body = report({"filtered_findings": rp1, "output_format": "json"})["report_body"]
    issue = json.loads(json_body)["issues"][0]
    assert "https://***@registry.example.com" in issue["pattern"]
    assert "https://***@registry.example.com" in issue["finding"]
    sarif_body = report({"filtered_findings": rp1, "output_format": "sarif"})["report_body"]
    for body in (json_body, sarif_body):
        assert "deploy:s3cret" not in body
        assert "s3cret" not in body


def test_rp1_multiple_patterns():
    """Multiple unpinned references produce multiple RP1 findings."""
    result = node(
        _state(
            manifest={"name": "test-skill"},
            file_cache={
                "setup.sh": "npx @scope/server-a\nnpx @org/server-b\n",
            },
        )
    )
    rp1 = [f for f in result["findings"] if f.rule_id == "RP1"]
    assert len(rp1) == 2


def test_rp3_version_wildcard():
    """RP3 detects wildcard version."""
    result = node(
        _state(
            manifest={"version": "*", "name": "test"},
        )
    )
    rp3 = [f for f in result["findings"] if f.rule_id == "RP3"]
    assert len(rp3) >= 1


def test_rp3_version_wildcard_from_skill_frontmatter(tmp_path):
    """RP3 receives the version projected from real skill frontmatter."""
    (tmp_path / "SKILL.md").write_text(
        '---\nname: test-skill\ndescription: For tests\nversion: "*"\n---\n',
        encoding="utf-8",
    )

    result = node(build_context({"skill_path": str(tmp_path)}))

    rp3 = [finding for finding in result["findings"] if finding.rule_id == "RP3"]
    assert len(rp3) == 1
    assert rp3[0].matched_text == "*"


def test_rp3_broad_version_preview_preserves_full_value_identity() -> None:
    prefix = "^" + "1" * 200
    complete_values = (prefix + "first", prefix + "second")
    findings = [
        next(
            finding
            for finding in node(_state(manifest={"version": value}))["findings"]
            if finding.rule_id == "RP3"
        )
        for value in complete_values
    ]

    assert findings[0].matched_text == findings[1].matched_text
    assert len({finding.fingerprint() for finding in findings}) == 2
    assert len(deduplicate(findings)) == 2
    for finding, complete_value in zip(findings, complete_values, strict=True):
        assert complete_value not in json.dumps(finding.to_dict(), sort_keys=True)


def test_rp3_version_ok_no_finding():
    """RP3 does not fire on pinned version."""
    result = node(
        _state(
            manifest={"version": "1.2.3", "name": "test"},
        )
    )
    rp3 = [f for f in result["findings"] if f.rule_id == "RP3"]
    assert len(rp3) == 0


def test_empty_state_returns_no_findings():
    """Empty state produces no findings."""
    result = node({})
    assert result["findings"] == []
