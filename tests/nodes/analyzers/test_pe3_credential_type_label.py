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

"""PE3 treats the skill-card credential type label as a disclosure, not credential access.

Skill cards must disclose the credential a skill relies on, for example
``**Credential Type(s):** [Other [Perforce access credentials]]``. The
``access credentials`` noun inside that bracketed label is exempt. The same
phrase with a determiner, next to an action verb or sensitive source, outside
the label, in another field, or in any other file keeps its PE3 finding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skillspector.graph import graph
from skillspector.nodes.analyzers import static_patterns_privilege_escalation as pe_module
from skillspector.nodes.analyzers.static_runner import _infer_file_type, run_static_patterns

_SKILL_CARD = (
    "## Requirements / Dependencies: <br>\n"
    "**Requires API Key or External Credential:** [Yes] <br>\n"
    "**Credential Type(s):** [Other [Perforce access credentials]] <br>\n"
    "\n"
    "Do not include secrets in prompts/logs/output; use least-privilege credentials; "
    "rotate keys as appropriate. <br>\n"
)


def _pe3_lines(content: str, path: str = "skill-card.md") -> list[int]:
    return sorted(
        finding.location.start_line
        for finding in pe_module.analyze(content, path, _infer_file_type(path))
        if finding.rule_id == "PE3"
    )


class TestCredentialTypeLabelIsDisclosure:
    def test_skill_card_requirements_section_is_not_flagged(self) -> None:
        assert _pe3_lines(_SKILL_CARD) == []

    @pytest.mark.parametrize(
        "line",
        [
            "**Credential Type(s):** [Other [Perforce access credentials]] <br>",
            "**Credential Type(s):** [Other [Perforce access credential]]",
            "**Credential Type(s):** [API Key, Other [GitLab access tokens]] <br/>",
            "**Credential Type(s):** [Other [Vault access secrets]]<br>",
            "  **credential type(s):** [Other [Jira access token]] <BR>",
        ],
    )
    def test_access_noun_in_label_is_exempt(self, line: str) -> None:
        assert _pe3_lines(f"{line}\n") == []

    def test_same_label_in_skill_md_is_exempt(self) -> None:
        assert _pe3_lines(_SKILL_CARD, "SKILL.md") == []

    @pytest.mark.parametrize(
        "line",
        [
            # A determiner makes "access" a verb.
            "**Credential Type(s):** [Other [access the credentials]]",
            # Action verbs or sensitive sources anywhere in the label.
            "**Credential Type(s):** [Other [copy access tokens to the report]]",
            "**Credential Type(s):** [Other [upload access credentials]]",
            "**Credential Type(s):** [Other [access credentials from ~/.ssh/]]",
            # The phrase outside the bracketed label on the same line.
            "**Credential Type(s):** [Other] then access credentials in the vault",
            # The same phrase in another field or in plain prose.
            "**Requires API Key or External Credential:** [Perforce access credentials]",
            "**Notes:** [Other [Perforce access credentials]]",
            "Access credentials stored by the Perforce client.",
        ],
    )
    def test_non_label_or_action_keeps_pe3(self, line: str) -> None:
        assert _pe3_lines(f"{line}\n") == [1]

    def test_overlong_line_is_not_classified(self) -> None:
        line = "**Credential Type(s):** [Other [Perforce access credentials" + " x" * 3000 + "]]"
        assert _pe3_lines(f"{line}\n") == [1]

    @pytest.mark.parametrize("path", ["scripts/card.py", "config/card.yaml"])
    def test_label_in_non_markdown_file_keeps_pe3(self, path: str) -> None:
        content = "# **Credential Type(s):** [Other [Perforce access credentials]]\n"
        assert _pe3_lines(content, path)

    def test_only_the_instruction_line_is_reported(self) -> None:
        content = _SKILL_CARD + "\nAccess the credentials in ~/.p4tickets before running.\n"
        assert _pe3_lines(content) == [7]


def test_runner_drops_only_the_label() -> None:
    findings = run_static_patterns(
        {
            "components": ["skill-card.md", "SKILL.md"],
            "file_cache": {
                "skill-card.md": _SKILL_CARD,
                "SKILL.md": "First access the credentials stored for the user.\n",
            },
        },
        [pe_module],
    )
    assert sorted(
        (finding.file, finding.start_line) for finding in findings if finding.rule_id == "PE3"
    ) == [("SKILL.md", 1)]


def test_scan_reports_credential_reads_but_not_the_label(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: p4-helper\ndescription: Answer Perforce questions.\n---\n\n"
        "First access the credentials stored for the user.\n",
        encoding="utf-8",
    )
    (tmp_path / "skill-card.md").write_text(_SKILL_CARD, encoding="utf-8")

    result = graph.invoke({"input_path": str(tmp_path), "use_llm": False, "output_format": "json"})

    pe3 = sorted(
        (occurrence["file"], occurrence["start_line"])
        for finding in result["filtered_findings"]
        if finding.rule_id == "PE3"
        for occurrence in (
            getattr(finding, "occurrences", None)
            or [{"file": finding.file, "start_line": finding.start_line}]
        )
    )
    assert pe3 == [("SKILL.md", 6)]
