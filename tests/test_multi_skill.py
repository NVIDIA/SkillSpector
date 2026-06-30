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

"""Tests for multi-skill directory detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillspector.multi_skill import detect_skills


@pytest.fixture
def multi_skill_dir(tmp_path: Path) -> Path:
    """Create a directory with 3 sub-skills, no root SKILL.md."""
    for name in ("weather-lookup", "email-sender", "file-manager"):
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: A {name} skill\n---\n# {name}\n",
            encoding="utf-8",
        )
        (skill_dir / "tool.py").write_text(f"# {name} implementation\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def single_skill_dir(tmp_path: Path) -> Path:
    """Create a single-skill directory with root SKILL.md."""
    (tmp_path / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: A test skill\n---\n# My Skill\n",
        encoding="utf-8",
    )
    (tmp_path / "tool.py").write_text("# implementation\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def nested_with_root(tmp_path: Path) -> Path:
    """A directory with root SKILL.md AND sub-skill SKILL.md files."""
    (tmp_path / "SKILL.md").write_text(
        "---\nname: parent-skill\n---\n# Parent\n",
        encoding="utf-8",
    )
    sub = tmp_path / "sub-skill"
    sub.mkdir()
    (sub / "SKILL.md").write_text(
        "---\nname: sub-skill\n---\n# Sub\n",
        encoding="utf-8",
    )
    return tmp_path


class TestDetectSkills:
    """Tests for detect_skills()."""

    def test_multi_skill_directory_detected(self, multi_skill_dir: Path) -> None:
        """Directory with no root SKILL.md and multiple sub-skills is detected."""
        result = detect_skills(multi_skill_dir)
        assert result.is_multi_skill is True
        assert len(result.skills) == 3
        assert result.has_root_skill is False

    def test_skill_names_extracted_from_frontmatter(self, multi_skill_dir: Path) -> None:
        """Skill names come from SKILL.md frontmatter."""
        result = detect_skills(multi_skill_dir)
        names = {s.name for s in result.skills}
        assert names == {"weather-lookup", "email-sender", "file-manager"}

    def test_single_skill_not_multi(self, single_skill_dir: Path) -> None:
        """Directory with root SKILL.md is not multi-skill."""
        result = detect_skills(single_skill_dir)
        assert result.is_multi_skill is False
        assert result.has_root_skill is True
        assert len(result.skills) == 0

    def test_root_skill_overrides_nested(self, nested_with_root: Path) -> None:
        """Root SKILL.md means it's a single skill even with nested SKILL.md."""
        result = detect_skills(nested_with_root)
        assert result.is_multi_skill is False
        assert result.has_root_skill is True

    def test_empty_directory_not_multi(self, tmp_path: Path) -> None:
        """Empty directory is not multi-skill."""
        result = detect_skills(tmp_path)
        assert result.is_multi_skill is False
        assert len(result.skills) == 0

    def test_single_sub_skill_not_multi(self, tmp_path: Path) -> None:
        """Only one sub-skill is not considered multi-skill (need >= 2)."""
        sub = tmp_path / "only-skill"
        sub.mkdir()
        (sub / "SKILL.md").write_text("---\nname: only\n---\n# Only\n", encoding="utf-8")
        result = detect_skills(tmp_path)
        assert result.is_multi_skill is False
        assert len(result.skills) == 1

    def test_hidden_directories_skipped(self, tmp_path: Path) -> None:
        """Directories starting with '.' are not scanned for skills."""
        for name in ("skill-a", "skill-b"):
            sub = tmp_path / name
            sub.mkdir()
            (sub / "SKILL.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")
        hidden = tmp_path / ".hidden-skill"
        hidden.mkdir()
        (hidden / "SKILL.md").write_text("---\nname: hidden\n---\n", encoding="utf-8")
        result = detect_skills(tmp_path)
        assert result.is_multi_skill is True
        assert len(result.skills) == 2
        names = {s.name for s in result.skills}
        assert "hidden" not in names

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        """Non-existent path returns not multi-skill."""
        result = detect_skills(tmp_path / "does-not-exist")
        assert result.is_multi_skill is False

    def test_skill_directory_paths_are_absolute(self, multi_skill_dir: Path) -> None:
        """SkillDirectory.path should be an absolute path."""
        result = detect_skills(multi_skill_dir)
        for skill in result.skills:
            assert skill.path.is_absolute()

    def test_relative_path_is_dirname(self, multi_skill_dir: Path) -> None:
        """SkillDirectory.relative_path is just the directory name."""
        result = detect_skills(multi_skill_dir)
        for skill in result.skills:
            assert "/" not in skill.relative_path
            assert skill.relative_path == skill.path.name

    def test_fallback_name_from_dirname(self, tmp_path: Path) -> None:
        """If SKILL.md has no name in frontmatter, use directory name."""
        for name in ("skill-a", "skill-b"):
            sub = tmp_path / name
            sub.mkdir()
            (sub / "SKILL.md").write_text("---\ndescription: no name\n---\n", encoding="utf-8")
        result = detect_skills(tmp_path)
        assert result.is_multi_skill is True
        names = {s.name for s in result.skills}
        assert names == {"skill-a", "skill-b"}

    def test_lowercase_skill_md_detected(self, tmp_path: Path) -> None:
        """skill.md (lowercase) is also recognized."""
        for name in ("alpha", "beta"):
            sub = tmp_path / name
            sub.mkdir()
            (sub / "skill.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")
        result = detect_skills(tmp_path)
        assert result.is_multi_skill is True
        assert len(result.skills) == 2
