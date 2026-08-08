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

"""Tests for cross_skill_dependency analyzer (CS1-CS3)."""

from __future__ import annotations

from skillspector.nodes.analyzers import cross_skill_dependency as csd


def _run_node(components: dict[str, str]) -> list:
    state = {"components": list(components), "file_cache": components}
    return csd.node(state)["findings"]


class TestSkillNameFromPath:
    def test_returns_parent_dir_not_file_stem(self):
        assert csd._skill_name_from_path("skill-a/SKILL.md") == "skill-a"

    def test_prefers_known_skill_name(self):
        assert (
            csd._ancestor_skill_names("skills/alpha/scripts/helper.py", {"alpha", "beta"})
            == {"alpha"}
        )


class TestCS1CrossSkillReference:
    def test_cross_skill_reference_flagged(self):
        components = {
            "alpha/SKILL.md": "Depends on 'beta' for formatting.\n",
            "beta/SKILL.md": "# Beta skill\n",
        }
        findings = _run_node(components)
        cs1 = [f for f in findings if f.rule_id == "CS1"]
        assert len(cs1) == 1
        assert cs1[0].file == "alpha/SKILL.md"

    def test_self_reference_not_flagged(self):
        components = {
            "alpha/SKILL.md": "This skill uses alpha internally to keep things simple.\n",
            "beta/SKILL.md": "# Beta skill\n",
        }
        findings = _run_node(components)
        assert not any(f.rule_id == "CS1" for f in findings)


class TestMultiSkillGate:
    def test_single_skill_with_scripts_subdir_is_skipped(self):
        """'tool scripts' prose must not fire when only one real skill exists."""
        components = {
            "my-skill/SKILL.md": "tool scripts that help you do work\n",
            "my-skill/scripts/helper.py": "print('hi')\n",
        }
        findings = _run_node(components)
        assert findings == []

    def test_two_skills_triggers_analysis(self):
        components = {
            "alpha/SKILL.md": "# Alpha\n",
            "beta/SKILL.md": "# Beta\n",
            "alpha/scripts/helper.py": "print('hi')\n",
        }
        findings = _run_node(components)
        assert isinstance(findings, list)


class TestCS2PrivilegeEscalation:
    def test_grants_permission_to_other_skill(self):
        components = {
            "alpha/SKILL.md": "grant permission to 'beta' for all file access\n",
            "beta/SKILL.md": "# Beta skill\n",
        }
        findings = _run_node(components)
        cs2 = [f for f in findings if f.rule_id == "CS2"]
        assert len(cs2) == 1

    def test_shares_credentials_with_other_skill(self):
        components = {
            "alpha/SKILL.md": "share credentials with 'beta'\n",
            "beta/SKILL.md": "# Beta skill\n",
        }
        findings = _run_node(components)
        assert any(f.rule_id == "CS2" for f in findings)


class TestCS3SharedState:
    def test_shared_state_only_with_cross_skill_ref(self):
        components = {
            "alpha/SKILL.md": "Use skill 'beta'. We keep shared state in the common cache.\n",
            "beta/SKILL.md": "# Beta skill\n",
        }
        findings = _run_node(components)
        assert any(f.rule_id == "CS3" for f in findings)

    def test_bare_mutex_mention_is_benign(self):
        components = {
            "alpha/SKILL.md": "Acquire the mutex before writing. The lockfile lives in /tmp.\n",
            "beta/SKILL.md": "# Beta skill\n",
        }
        findings = _run_node(components)
        assert not any(f.rule_id == "CS3" for f in findings)


class TestCircularReferences:
    def test_detects_simple_cycle(self):
        refs = {"a": {"b"}, "b": {"a"}}
        cycles = csd._detect_circular_references(refs)
        assert len(cycles) >= 1

    def test_no_cycle_for_dag(self):
        refs = {"a": {"b"}, "b": set()}
        assert csd._detect_circular_references(refs) == []
