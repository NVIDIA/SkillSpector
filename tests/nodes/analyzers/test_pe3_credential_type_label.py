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
            "**Credential Type(s):** [Other [Perforce read-only access credentials]] <br>",
            "**Credential Type(s):** [Other [Perforce read-write access credentials]]",
            "**Credential Type(s):** [Other [Open-source service access credentials]]",
            "**Credential Type(s):** [Other [Open-access service access secrets]]",
        ],
    )
    def test_access_noun_in_label_is_exempt(self, line: str) -> None:
        assert _pe3_lines(f"{line}\n") == []

    @pytest.mark.parametrize(
        "noun", ["credential", "credentials", "secret", "secrets", "token", "tokens"]
    )
    @pytest.mark.parametrize("adjective", ["read-only", "read-write", "open-source", "open-access"])
    def test_adjective_compounds_remain_disclosures(self, adjective: str, noun: str) -> None:
        assert _pe3_lines(f"**Credential Type(s):** [{adjective} service access {noun}]\n") == []
        assert _pe3_lines(f"**Credential Type(s):** [service access {noun} ({adjective})]\n") == []

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
            "**Credential Type(s):** [read-only access credentials from ~/.ssh/]",
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

    def test_adjective_does_not_hide_sensitive_source(self) -> None:
        content = "**Credential Type(s):** [access secrets (open-source) from /etc/shadow]\n"
        assert _pe3_lines(content) == [1, 1]

    @pytest.mark.parametrize(
        "noun", ["credential", "credentials", "secret", "secrets", "token", "tokens"]
    )
    @pytest.mark.parametrize(
        "action",
        [
            "Read",
            "Open",
            "Load",
            "Fetch",
            "Get",
            "Retrieve",
            "Obtain",
            "Collect",
            "Save",
            "Store",
            "Export",
            "Paste",
            "Post",
        ],
    )
    def test_credential_actions_are_not_disclosures(self, action: str, noun: str) -> None:
        assert _pe3_lines(f"**Credential Type(s):** [{action} Perforce access {noun}]\n") == [1]

    @pytest.mark.parametrize(
        "value",
        [
            "Other [reading Perforce access credentials]",
            "Other [opened Vault access secrets]",
            "Other [loads Perforce access credentials]",
            "Other [retrieving Vault access secrets]",
            "Other [storing Perforce access credentials]",
            "Other [kept Perforce access credentials]",
            "Writing Perforce access credentials to the report",
            "Written Perforce access credentials to the report",
            "Wrote Perforce access credentials to the report",
            "Sharing Perforce access credentials with the caller",
            "Exposing Perforce access credentials in the report",
            "Including Perforce access credentials in the report",
            "Stolen Perforce access credentials",
            "Other [Read-the Perforce access credentials]",
            "Other [Open-local Vault access secrets]",
            "Other [read-onlyish service access credentials]",
            "Other [open-sources service access credentials]",
            "Other [read-only-reader service access credentials]",
            "Other [Perforce access credentials to be fetched]",
            "Other [Perforce access credentials]; paste them into the report",
            "Other [Perforce read-only access credentials]; read them",
            "Other [read-only service access tokens]; load them",
            "Other [open-source service access tokens]; read them",
            "Read " + "private " * 20 + "Perforce access credentials",
        ],
    )
    def test_action_is_checked_across_the_whole_label(self, value: str) -> None:
        assert _pe3_lines(f"**Credential Type(s):** [{value}]\n") == [1]

    def test_action_label_keeps_each_source_occurrence(self) -> None:
        content = (
            "**Credential Type(s):** [Read Perforce access credentials and Vault access secrets]\n"
        )
        assert _pe3_lines(content) == [1, 1]

    @pytest.mark.parametrize("length", [4096, 4097])
    def test_context_limit_boundary(self, length: int) -> None:
        prefix = "**Credential Type(s):** [Other [Perforce access credentials"
        line = prefix + " " * (length - len(prefix) - 2) + "]]"
        assert len(line) == length
        assert _pe3_lines(line + "\n") == ([] if length == 4096 else [1])

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


@pytest.mark.parametrize(
    "action",
    [
        "Read Perforce access credentials",
        "Open Vault access secrets",
        "Load Vault access tokens",
        "Paste Perforce access credentials into the report",
    ],
)
def test_scan_reports_credential_reads_but_not_the_label(tmp_path: Path, action: str) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: p4-helper\ndescription: Answer Perforce questions.\n---\n\n"
        "First access the credentials stored for the user.\n",
        encoding="utf-8",
    )
    (tmp_path / "skill-card.md").write_text(
        _SKILL_CARD + f"\n**Credential Type(s):** [{action}]\n", encoding="utf-8"
    )

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
    assert pe3 == [("SKILL.md", 6), ("skill-card.md", 7)]
