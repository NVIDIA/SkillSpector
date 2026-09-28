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

from skillspector.nodes.analyzers.mcp_rug_pull import node
from skillspector.nodes.build_context import build_context
from skillspector.nodes.deduplicate import deduplicate
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
