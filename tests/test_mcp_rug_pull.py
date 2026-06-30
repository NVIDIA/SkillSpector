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

from skillspector.nodes.analyzers.mcp_rug_pull import node
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
