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

"""Tests for analyzer node registry alignment with SADD spec."""

from __future__ import annotations

from skillspector.nodes.analyzers import ANALYZER_NODE_IDS, ANALYZER_NODES

# Expected analyzer node IDs per SADD spec workflow reference table.
# Order: static (14), behavioral (2), mcp (3), semantic (3).
EXPECTED_ANALYZER_NODE_IDS: list[str] = [
    "static_patterns_prompt_injection",
    "static_patterns_data_exfiltration",
    "static_patterns_privilege_escalation",
    "static_patterns_supply_chain",
    "static_patterns_harmful_content",
    "static_patterns_excessive_agency",
    "static_patterns_output_handling",
    "static_patterns_system_prompt_leakage",
    "static_patterns_memory_poisoning",
    "static_patterns_tool_misuse",
    "static_patterns_rogue_agent",
    "static_patterns_agent_snooping",
    "static_patterns_anti_refusal",
    "static_patterns_ssrf",
    "static_yara",
    "behavioral_ast",
    "behavioral_taint_tracking",
    "mcp_least_privilege",
    "mcp_tool_poisoning",
    "mcp_rug_pull",
    "semantic_security_discovery",
    "semantic_developer_intent",
    "semantic_quality_policy",
]


class TestAnalyzerRegistry:
    """Registry matches SADD spec node set and order."""

    def test_analyzer_node_ids_match_sadd_spec(self):
        """ANALYZER_NODE_IDS equals expected list from SADD spec."""
        assert ANALYZER_NODE_IDS == EXPECTED_ANALYZER_NODE_IDS

    def test_analyzer_nodes_has_entry_for_every_id(self):
        """Every ANALYZER_NODE_IDS entry has a corresponding ANALYZER_NODES entry."""
        for node_id in ANALYZER_NODE_IDS:
            assert node_id in ANALYZER_NODES, f"Missing ANALYZER_NODES entry for {node_id}"

    def test_analyzer_nodes_has_no_extra_entries(self):
        """ANALYZER_NODES has no entries beyond ANALYZER_NODE_IDS."""
        for node_id in ANALYZER_NODES:
            assert node_id in ANALYZER_NODE_IDS, f"Extra ANALYZER_NODES entry: {node_id}"
