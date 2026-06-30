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

"""Analyzer node registry for the Skillspector workflow."""

from __future__ import annotations

from skillspector.nodes.analyzers.behavioral_ast import node as behavioral_ast_node
from skillspector.nodes.analyzers.behavioral_taint_tracking import (
    node as behavioral_taint_tracking_node,
)
from skillspector.nodes.analyzers.mcp_least_privilege import node as mcp_least_privilege_node
from skillspector.nodes.analyzers.mcp_rug_pull import node as mcp_rug_pull_node
from skillspector.nodes.analyzers.mcp_tool_poisoning import node as mcp_tool_poisoning_node
from skillspector.nodes.analyzers.semantic_developer_intent import (
    node as semantic_developer_intent_node,
)
from skillspector.nodes.analyzers.semantic_quality_policy import (
    node as semantic_quality_policy_node,
)
from skillspector.nodes.analyzers.semantic_security_discovery import (
    node as semantic_security_discovery_node,
)
from skillspector.nodes.analyzers.static_patterns_agent_snooping import (
    node as static_patterns_agent_snooping_node,
)
from skillspector.nodes.analyzers.static_patterns_anti_refusal import (
    node as static_patterns_anti_refusal_node,
)
from skillspector.nodes.analyzers.static_patterns_data_exfiltration import (
    node as static_patterns_data_exfiltration_node,
)
from skillspector.nodes.analyzers.static_patterns_excessive_agency import (
    node as static_patterns_excessive_agency_node,
)
from skillspector.nodes.analyzers.static_patterns_harmful_content import (
    node as static_patterns_harmful_content_node,
)
from skillspector.nodes.analyzers.static_patterns_memory_poisoning import (
    node as static_patterns_memory_poisoning_node,
)
from skillspector.nodes.analyzers.static_patterns_output_handling import (
    node as static_patterns_output_handling_node,
)
from skillspector.nodes.analyzers.static_patterns_privilege_escalation import (
    node as static_patterns_privilege_escalation_node,
)
from skillspector.nodes.analyzers.static_patterns_prompt_injection import (
    node as static_patterns_prompt_injection_node,
)
from skillspector.nodes.analyzers.static_patterns_rogue_agent import (
    node as static_patterns_rogue_agent_node,
)
from skillspector.nodes.analyzers.static_patterns_ssrf import (
    node as static_patterns_ssrf_node,
)
from skillspector.nodes.analyzers.static_patterns_supply_chain import (
    node as static_patterns_supply_chain_node,
)
from skillspector.nodes.analyzers.static_patterns_system_prompt_leakage import (
    node as static_patterns_system_prompt_leakage_node,
)
from skillspector.nodes.analyzers.static_patterns_tool_misuse import (
    node as static_patterns_tool_misuse_node,
)
from skillspector.nodes.analyzers.static_yara import node as static_yara_node

ANALYZER_NODE_IDS: list[str] = [
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

ANALYZER_NODES = {
    "static_patterns_prompt_injection": static_patterns_prompt_injection_node,
    "static_patterns_data_exfiltration": static_patterns_data_exfiltration_node,
    "static_patterns_privilege_escalation": static_patterns_privilege_escalation_node,
    "static_patterns_supply_chain": static_patterns_supply_chain_node,
    "static_patterns_harmful_content": static_patterns_harmful_content_node,
    "static_patterns_excessive_agency": static_patterns_excessive_agency_node,
    "static_patterns_output_handling": static_patterns_output_handling_node,
    "static_patterns_system_prompt_leakage": static_patterns_system_prompt_leakage_node,
    "static_patterns_memory_poisoning": static_patterns_memory_poisoning_node,
    "static_patterns_tool_misuse": static_patterns_tool_misuse_node,
    "static_patterns_rogue_agent": static_patterns_rogue_agent_node,
    "static_patterns_agent_snooping": static_patterns_agent_snooping_node,
    "static_patterns_anti_refusal": static_patterns_anti_refusal_node,
    "static_patterns_ssrf": static_patterns_ssrf_node,
    "static_yara": static_yara_node,
    "behavioral_ast": behavioral_ast_node,
    "behavioral_taint_tracking": behavioral_taint_tracking_node,
    "mcp_least_privilege": mcp_least_privilege_node,
    "mcp_tool_poisoning": mcp_tool_poisoning_node,
    "mcp_rug_pull": mcp_rug_pull_node,
    "semantic_security_discovery": semantic_security_discovery_node,
    "semantic_developer_intent": semantic_developer_intent_node,
    "semantic_quality_policy": semantic_quality_policy_node,
}

__all__ = ["ANALYZER_NODE_IDS", "ANALYZER_NODES"]
