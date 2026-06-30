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

"""Unit tests for the mcp_rug_pull analyzer node (B.3.3 RP1-RP3)."""

from __future__ import annotations

from skillspector.models import Finding
from skillspector.nodes.analyzers.mcp_rug_pull import node


class TestMcpRugPullNode:
    """Tests for the MCP rug-pull comparison logic."""

    def test_no_previous_manifest_skips(self) -> None:
        """Returns empty findings when previous_manifest is absent."""
        state = {
            "manifest": {
                "name": "my-skill",
                "permissions": ["read"],
            },
            "previous_manifest": None,
        }
        result = node(state)
        assert result == {"findings": []}

    def test_missing_previous_manifest_key_skips(self) -> None:
        """Returns empty findings when previous_manifest key is missing in state."""
        state = {
            "manifest": {
                "name": "my-skill",
                "permissions": ["read"],
            },
        }
        result = node(state)
        assert result == {"findings": []}

    def test_identical_manifests_returns_empty(self) -> None:
        """Returns empty findings when current and previous manifests are identical."""
        manifest = {
            "name": "my-skill",
            "description": "benign skill",
            "permissions": ["read", "write"],
            "triggers": ["run"],
            "parameters": [
                {
                    "name": "path",
                    "type": "string",
                    "description": "file path",
                    "default": "/tmp",
                }
            ],
        }
        state = {
            "manifest": manifest,
            "previous_manifest": manifest,
        }
        result = node(state)
        assert result == {"findings": []}

    def test_rp1_permission_expansion(self) -> None:
        """RP1 is triggered when a new permission is added in the current manifest."""
        state = {
            "manifest": {
                "permissions": ["read", "write", "execute"],
            },
            "previous_manifest": {
                "permissions": ["read", "write"],
            },
        }
        result = node(state)
        findings = result["findings"]
        rp1_findings = [f for f in findings if f.rule_id == "RP1"]
        assert len(rp1_findings) == 1
        f = rp1_findings[0]
        assert isinstance(f, Finding)
        assert f.severity == "HIGH"
        assert f.confidence == 0.90
        assert f.file == "SKILL.md"
        assert f.category == "MCP Rug Pull"
        assert "ASI02" in f.tags
        assert "execute" in f.message
        assert f.explanation is not None
        assert f.remediation is not None

    def test_rp1_normalization_avoids_false_positives(self) -> None:
        """RP1 does not trigger when permission case or whitespace differs slightly."""
        state = {
            "manifest": {
                "permissions": ["Read ", "WRITE"],
            },
            "previous_manifest": {
                "permissions": ["read", "write"],
            },
        }
        result = node(state)
        assert result == {"findings": []}

    def test_rp2_trigger_added(self) -> None:
        """RP2 is triggered when a trigger phrase is added."""
        state = {
            "manifest": {
                "triggers": ["run", "execute_task"],
            },
            "previous_manifest": {
                "triggers": ["run"],
            },
        }
        result = node(state)
        findings = result["findings"]
        rp2_findings = [f for f in findings if f.rule_id == "RP2"]
        assert len(rp2_findings) == 1
        f = rp2_findings[0]
        assert f.severity == "MEDIUM"
        assert f.confidence == 0.85
        assert "added: execute_task" in f.message

    def test_rp2_trigger_removed(self) -> None:
        """RP2 is triggered when a trigger phrase is removed."""
        state = {
            "manifest": {
                "triggers": ["run"],
            },
            "previous_manifest": {
                "triggers": ["run", "helper"],
            },
        }
        result = node(state)
        findings = result["findings"]
        rp2_findings = [f for f in findings if f.rule_id == "RP2"]
        assert len(rp2_findings) == 1
        f = rp2_findings[0]
        assert "removed: helper" in f.message

    def test_rp3_parameter_added(self) -> None:
        """RP3 is triggered when a parameter is added."""
        state = {
            "manifest": {
                "parameters": [
                    {"name": "path", "type": "string"},
                    {"name": "force", "type": "boolean"},
                ]
            },
            "previous_manifest": {
                "parameters": [
                    {"name": "path", "type": "string"},
                ]
            },
        }
        result = node(state)
        findings = result["findings"]
        rp3_findings = [f for f in findings if f.rule_id == "RP3"]
        assert len(rp3_findings) == 1
        f = rp3_findings[0]
        assert f.severity == "MEDIUM"
        assert f.confidence == 0.80
        assert "added: force" in f.message

    def test_rp3_parameter_removed(self) -> None:
        """RP3 is triggered when a parameter is removed."""
        state = {
            "manifest": {
                "parameters": [
                    {"name": "path", "type": "string"},
                ]
            },
            "previous_manifest": {
                "parameters": [
                    {"name": "path", "type": "string"},
                    {"name": "force", "type": "boolean"},
                ]
            },
        }
        result = node(state)
        findings = result["findings"]
        rp3_findings = [f for f in findings if f.rule_id == "RP3"]
        assert len(rp3_findings) == 1
        f = rp3_findings[0]
        assert "removed: force" in f.message

    def test_rp3_parameter_type_changed(self) -> None:
        """RP3 is triggered when a parameter type is modified."""
        state = {
            "manifest": {"parameters": [{"name": "path", "type": "integer"}]},
            "previous_manifest": {"parameters": [{"name": "path", "type": "string"}]},
        }
        result = node(state)
        findings = result["findings"]
        rp3_findings = [f for f in findings if f.rule_id == "RP3"]
        assert len(rp3_findings) == 1
        f = rp3_findings[0]
        assert "modified: path (type changed from string to integer)" in f.message

    def test_rp3_parameter_default_changed(self) -> None:
        """RP3 is triggered when a parameter default value is modified."""
        state = {
            "manifest": {"parameters": [{"name": "path", "default": "/var/log"}]},
            "previous_manifest": {"parameters": [{"name": "path", "default": "/tmp"}]},
        }
        result = node(state)
        findings = result["findings"]
        rp3_findings = [f for f in findings if f.rule_id == "RP3"]
        assert len(rp3_findings) == 1
        f = rp3_findings[0]
        assert "modified: path (default changed from /tmp to /var/log)" in f.message

    def test_rp3_parameter_description_changed(self) -> None:
        """RP3 is triggered when a parameter description is modified."""
        state = {
            "manifest": {"parameters": [{"name": "path", "description": "new description"}]},
            "previous_manifest": {
                "parameters": [{"name": "path", "description": "old description"}]
            },
        }
        result = node(state)
        findings = result["findings"]
        rp3_findings = [f for f in findings if f.rule_id == "RP3"]
        assert len(rp3_findings) == 1
        f = rp3_findings[0]
        assert "modified: path (description changed)" in f.message

    def test_complex_manifest_change_triggers_multiple_findings(self) -> None:
        """Multiple rules are triggered when permissions, triggers, and parameters all change."""
        state = {
            "manifest": {
                "permissions": ["read", "write", "net"],
                "triggers": ["do_something"],
                "parameters": [{"name": "url", "type": "string"}],
            },
            "previous_manifest": {
                "permissions": ["read", "write"],
                "triggers": ["run"],
                "parameters": [{"name": "path", "type": "string"}],
            },
        }
        result = node(state)
        findings = result["findings"]
        rule_ids = {f.rule_id for f in findings}
        assert rule_ids == {"RP1", "RP2", "RP3"}
        assert len(findings) == 3
