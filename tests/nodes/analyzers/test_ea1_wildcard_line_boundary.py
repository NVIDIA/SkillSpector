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

"""Tests for EA1's wildcard-tool-access pattern: line-boundary and bold-markdown fixes.

Confirmed on the official anthropics/skills repo, mcp-builder/SKILL.md:
the original EA1 regex used bare ``\\s*`` between the colon and the expected
wildcard value. Because Python's ``\\s`` matches newlines, the gap could span
a blank line and bridge two unrelated headings (SKILL.md:96, "For each
tool:" + blank line + "**Input Schema:**"). The pattern also had no check
that the matched ``*`` was a standalone token, so the first ``*`` of a
closing ``**`` bold-markdown span satisfied it (SKILL.md:25, "**API Coverage
vs. Workflow Tools:**"). Both false positives fired as EA1/MEDIUM and
survived to the final report.
"""

from __future__ import annotations

from skillspector.nodes.analyzers import (
    static_patterns_excessive_agency as ea_module,
)


class TestEA1LineBoundary:
    """The wildcard value must be on the same line as the colon, not bridged
    across a blank line to an unrelated heading."""

    def test_heading_intro_followed_by_blank_line_and_next_heading_not_flagged(self) -> None:
        findings = ea_module.analyze(
            "For each tool:\n\n**Input Schema:**\n",
            "SKILL.md",
            "markdown",
        )
        assert not any(f.rule_id == "EA1" for f in findings)

    def test_wildcard_on_next_line_without_blank_gap_still_not_flagged(self) -> None:
        """The value must be on the *same* line as the colon; even a single
        newline before it is not a same-line wildcard grant."""
        findings = ea_module.analyze(
            "tools:\n*\n",
            "SKILL.md",
            "markdown",
        )
        assert not any(f.rule_id == "EA1" for f in findings)


class TestEA1BoldMarkdownCollision:
    """The matched ``*`` must be a standalone token, not the first asterisk
    of a ``**bold**`` span."""

    def test_bold_heading_ending_in_tools_colon_not_flagged(self) -> None:
        findings = ea_module.analyze(
            "**API Coverage vs. Workflow Tools:**\n",
            "SKILL.md",
            "markdown",
        )
        assert not any(f.rule_id == "EA1" for f in findings)

    def test_bolded_list_of_named_tools_not_flagged(self) -> None:
        """A bolded list of specific tools is the opposite of an unrestricted
        grant and must not match."""
        findings = ea_module.analyze(
            "Tools: **Read**, **Write**\n",
            "SKILL.md",
            "markdown",
        )
        assert not any(f.rule_id == "EA1" for f in findings)


class TestEA1GenuineWildcardStillFlagged:
    """Real single-line wildcard grants must still fire — the fix narrows
    the pattern, it must not blind it."""

    def test_tools_quoted_wildcard_flagged(self) -> None:
        findings = ea_module.analyze(
            'tools: "*"\n',
            "SKILL.md",
            "markdown",
        )
        assert any(f.rule_id == "EA1" for f in findings)

    def test_tools_bracketed_wildcard_flagged(self) -> None:
        findings = ea_module.analyze(
            "tools: [*]\n",
            "SKILL.md",
            "markdown",
        )
        assert any(f.rule_id == "EA1" for f in findings)

    def test_permissions_single_quoted_wildcard_flagged(self) -> None:
        findings = ea_module.analyze(
            "permissions: '*'\n",
            "SKILL.md",
            "markdown",
        )
        assert any(f.rule_id == "EA1" for f in findings)

    def test_unquoted_wildcard_flagged(self) -> None:
        findings = ea_module.analyze(
            "tools: *\n",
            "SKILL.md",
            "markdown",
        )
        assert any(f.rule_id == "EA1" for f in findings)


class TestEA1FootnoteAndPreColonNotFlagged:
    """Remaining false-positive paths from #444: a footnote legend after a
    'Tools:' label, and a blank-line gap before the colon."""

    def test_footnote_legend_after_tools_label_not_flagged(self) -> None:
        findings = ea_module.analyze(
            "Tools: * = requires authentication\n",
            "SKILL.md",
            "markdown",
        )
        assert not any(f.rule_id == "EA1" for f in findings)

    def test_footnote_prose_after_tools_label_not_flagged(self) -> None:
        findings = ea_module.analyze(
            "Tools: * marks optional parameters\n",
            "SKILL.md",
            "markdown",
        )
        assert not any(f.rule_id == "EA1" for f in findings)

    def test_blank_line_before_definition_list_colon_not_flagged(self) -> None:
        """The gap before the colon must stay on one line too — a markdown
        definition list two paragraphs later is not a grant."""
        findings = ea_module.analyze(
            "several tools\n\n: * item\n",
            "SKILL.md",
            "markdown",
        )
        assert not any(f.rule_id == "EA1" for f in findings)

    def test_bare_wildcard_with_trailing_comment_still_flagged(self) -> None:
        """A '#' comment after the bare wildcard is still a grant, not prose."""
        findings = ea_module.analyze(
            "tools: * # allow everything\n",
            "SKILL.md",
            "markdown",
        )
        assert any(f.rule_id == "EA1" for f in findings)


class TestEA1BlockListAndJsonFormsFlagged:
    """Detection gaps from #445: the idiomatic YAML block-list and JSON
    quoted-key encodings of a wildcard grant."""

    def test_yaml_block_list_quoted_wildcard_flagged(self) -> None:
        findings = ea_module.analyze(
            'tools:\n  - "*"\n',
            "SKILL.md",
            "markdown",
        )
        assert any(f.rule_id == "EA1" for f in findings)

    def test_yaml_block_list_bare_wildcard_flagged(self) -> None:
        findings = ea_module.analyze(
            "tools:\n  - *\n",
            "SKILL.md",
            "markdown",
        )
        assert any(f.rule_id == "EA1" for f in findings)

    def test_yaml_block_list_zero_indent_flagged(self) -> None:
        findings = ea_module.analyze(
            'permissions:\n- "*"\n',
            "SKILL.md",
            "markdown",
        )
        assert any(f.rule_id == "EA1" for f in findings)

    def test_json_quoted_key_list_wildcard_flagged(self) -> None:
        findings = ea_module.analyze(
            '"tools": ["*"]\n',
            "config.json",
            "json",
        )
        assert any(f.rule_id == "EA1" for f in findings)

    def test_json_quoted_key_scalar_wildcard_flagged(self) -> None:
        findings = ea_module.analyze(
            '"permissions": "*"\n',
            "config.json",
            "json",
        )
        assert any(f.rule_id == "EA1" for f in findings)

    def test_inline_list_wildcard_not_first_flagged(self) -> None:
        findings = ea_module.analyze(
            'tools: ["search", "*"]\n',
            "SKILL.md",
            "markdown",
        )
        assert any(f.rule_id == "EA1" for f in findings)

    def test_markdown_dash_list_of_bold_tools_not_flagged(self) -> None:
        """The block-list branch must not collide with a markdown list of
        specific bolded tool names."""
        findings = ea_module.analyze(
            "Tools:\n- **Read**\n- **Write**\n",
            "SKILL.md",
            "markdown",
        )
        assert not any(f.rule_id == "EA1" for f in findings)

    def test_blank_line_before_dash_item_not_flagged(self) -> None:
        """A blank line between the key and a dash item breaks the block-list
        association — the cross-paragraph bridge from #405 must not return."""
        findings = ea_module.analyze(
            'tools:\n\n  - "*"\n',
            "SKILL.md",
            "markdown",
        )
        assert not any(f.rule_id == "EA1" for f in findings)


class TestEA1NestedObjectArgumentNotFlagged:
    """A '*' nested inside an explicit tool object is an argument value for a
    specifically named tool, not a wildcard grant — the wildcard must be a
    top-level list element."""

    def test_json_tool_object_with_wildcard_argument_not_flagged(self) -> None:
        findings = ea_module.analyze(
            'tools: [{"name": "grep", "pattern": "*"}]\n',
            "config.json",
            "json",
        )
        assert not any(f.rule_id == "EA1" for f in findings)

    def test_yaml_flow_tool_object_with_wildcard_argument_not_flagged(self) -> None:
        findings = ea_module.analyze(
            'tools: [{name: search, glob: "*"}]\n',
            "SKILL.md",
            "markdown",
        )
        assert not any(f.rule_id == "EA1" for f in findings)

    def test_top_level_wildcard_after_named_element_still_flagged(self) -> None:
        """Narrowing to top-level elements must not lose the plain mixed list."""
        findings = ea_module.analyze(
            'tools: ["search", "*"]\n',
            "SKILL.md",
            "markdown",
        )
        assert any(f.rule_id == "EA1" for f in findings)

    def test_nested_array_wildcard_not_flagged(self) -> None:
        """A '*' inside a nested array is not a top-level tool-list element."""
        findings = ea_module.analyze(
            'tools: ["search", ["grep", "*"]]\n',
            "SKILL.md",
            "markdown",
        )
        assert not any(f.rule_id == "EA1" for f in findings)

    def test_json_tool_object_with_nested_arguments_not_flagged(self) -> None:
        """A '*' buried in a named tool's argument object is not a grant."""
        findings = ea_module.analyze(
            '"tools": [{"name": "search", "arguments": {"glob": "*"}}]\n',
            "config.json",
            "json",
        )
        assert not any(f.rule_id == "EA1" for f in findings)
