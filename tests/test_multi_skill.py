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

import json
from pathlib import Path

import pytest

import skillspector.multi_skill as multi_skill_module
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


def _write_aisop_bundle(path: Path) -> None:
    """Write a valid minimal AISOP/AISP bundle file."""
    bundle = [
        {
            "role": "system",
            "content": {
                "protocol": "AISOP V1",
                "format": "AIModal",
            },
        },
        {
            "role": "user",
            "content": {
                "aisop": {"main": "graph TD"},
                "functions": {
                    "lookup": {"constraints": ["query"]},
                    "schedule": {"constraints": ["time"]},
                },
                "declared_tools": ["search", "calendar"],
                "aisp_contract": {
                    "resources": {
                        "calendar": {"path": "resources/calendar.json"},
                        "memory": {"path": "resources/memory.md"},
                    }
                },
            },
        },
    ]
    path.write_text(json.dumps(bundle), encoding="utf-8")


def _make_nested_resources(depth: int) -> dict[str, object]:
    """Build a deeply nested resources tree for recursion-guard tests."""
    current: dict[str, object] = {"path": "resources/final.json"}
    for idx in range(depth, -1, -1):
        current = {f"resource_{idx}": {"resources": current}}
    return current


class TestDetectSkills:
    """Tests for detect_skills()."""

    def test_multi_skill_directory_detected(self, multi_skill_dir: Path) -> None:
        """Directory with no root SKILL.md and multiple sub-skills is detected."""
        result = detect_skills(multi_skill_dir)
        assert result.is_multi_skill is True
        assert len(result.skills) == 3
        assert result.has_root_skill is False
        assert result.complete is True
        assert [skill.relative_path for skill in result.skills] == [
            "email-sender",
            "file-manager",
            "weather-lookup",
        ]

    def test_skill_names_extracted_from_frontmatter(self, multi_skill_dir: Path) -> None:
        """Skill names come from SKILL.md frontmatter."""
        result = detect_skills(multi_skill_dir)
        names = {s.name for s in result.skills}
        assert names == {"weather-lookup", "email-sender", "file-manager"}

    def test_structured_skill_subdir_detected(self, tmp_path: Path) -> None:
        """An immediate subdirectory with a valid AISOP/AISP bundle is detected."""
        sub = tmp_path / "workflow-bundle"
        sub.mkdir()
        _write_aisop_bundle(sub / "workflow.aisop.json")
        result = detect_skills(tmp_path)
        assert result.is_multi_skill is False
        assert result.has_root_skill is False
        assert len(result.skills) == 1
        assert result.skills[0].name == "workflow-bundle"
        assert result.skills[0].path == sub

    def test_structured_bundles_under_aisop_directories_are_detected(self, tmp_path: Path) -> None:
        """The supported .aisop container is not treated as a generic hidden tree."""
        for name in ("workflow-a", "workflow-b"):
            sub = tmp_path / name
            bundle_dir = sub / ".aisop"
            bundle_dir.mkdir(parents=True)
            _write_aisop_bundle(bundle_dir / "workflow.aisop.json")

        result = detect_skills(tmp_path)

        assert result.complete is True
        assert result.is_multi_skill is True
        assert {skill.name for skill in result.skills} == {"workflow-a", "workflow-b"}

    def test_single_structured_child_not_multi(self, tmp_path: Path) -> None:
        """One structured subdirectory should not force multi-skill mode."""
        sub = tmp_path / "only-structured"
        sub.mkdir()
        _write_aisop_bundle(sub / "workflow.aisop.json")
        result = detect_skills(tmp_path)
        assert result.is_multi_skill is False
        assert len(result.skills) == 1

    @pytest.mark.parametrize("ancestor", [".claude", "venv"])
    def test_structured_child_under_ancestor_detected(self, tmp_path: Path, ancestor: str) -> None:
        """Structured children remain discoverable under external ancestors."""
        skills_dir = tmp_path / ancestor / "skills"
        sub = skills_dir / "workflow-bundle"
        sub.mkdir(parents=True)
        _write_aisop_bundle(sub / "workflow.aisop.json")

        result = detect_skills(skills_dir)

        assert len(result.skills) == 1
        assert result.skills[0].path == sub

    @pytest.mark.parametrize("skip_dir", ["venv", "node_modules", "__pycache__"])
    def test_skip_dir_children_with_bundles_ignored(self, tmp_path: Path, skip_dir: str) -> None:
        """Skip-dir children do not become phantom skills from vendored bundles."""
        real_skill = tmp_path / "real-skill"
        real_skill.mkdir()
        (real_skill / "SKILL.md").write_text("---\nname: real\n---\n", encoding="utf-8")

        bundled_child = tmp_path / skip_dir / "pkg"
        bundled_child.mkdir(parents=True)
        _write_aisop_bundle(bundled_child / "workflow.aisop.json")

        result = detect_skills(tmp_path)

        assert result.is_multi_skill is False
        assert len(result.skills) == 1
        assert result.skills[0].path == real_skill

    def test_structured_bundle_ignored_when_partial(self, tmp_path: Path) -> None:
        """Malformed AISOP/AISP JSON does not count as a structured child skill."""
        malformed = tmp_path / "bad-bundle"
        malformed.mkdir()
        (malformed / "workflow.aisop.json").write_text(
            json.dumps(
                [
                    {
                        "role": "system",
                        "content": {"protocol": "AISOP V1"},
                    },
                    {"content": {"functions": []}},
                ]
            ),
            encoding="utf-8",
        )
        result = detect_skills(tmp_path)
        assert result.is_multi_skill is False
        assert len(result.skills) == 0

    def test_structured_bundle_ignored_when_over_nested(self, tmp_path: Path) -> None:
        """Over-nested structured bundles fail closed instead of crashing detection."""
        nested = tmp_path / "deep-bundle"
        nested.mkdir()
        bundle = [
            {
                "role": "system",
                "content": {
                    "protocol": "AISP V1",
                    "format": "contract",
                },
            },
            {
                "role": "user",
                "content": {
                    "functions": {"lookup": {"constraints": ["query"]}},
                    "aisp_contract": {"resources": _make_nested_resources(140)},
                },
            },
        ]
        (nested / "workflow.aisop.json").write_text(json.dumps(bundle), encoding="utf-8")

        result = detect_skills(tmp_path)

        assert result.is_multi_skill is False
        assert len(result.skills) == 0

    def test_root_skill_still_overrides_structured_nested(self, tmp_path: Path) -> None:
        """A root SKILL.md still forces single-skill mode with nested structured bundles."""
        (tmp_path / "SKILL.md").write_text("---\nname: root-skill\n---\n# Root\n", encoding="utf-8")
        nested = tmp_path / "nested-structured"
        nested.mkdir()
        _write_aisop_bundle(nested / "workflow.aisop.json")
        result = detect_skills(tmp_path)
        assert result.is_multi_skill is False
        assert result.has_root_skill is True
        assert len(result.skills) == 0

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

    def test_dot_prefixed_child_skill_is_discovered_with_explicit_skips(
        self, tmp_path: Path
    ) -> None:
        """A dot-prefixed child skill is scanned without traversing explicit skips or links."""
        for name in ("skill-a", "skill-b"):
            sub = tmp_path / name
            sub.mkdir()
            (sub / "SKILL.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")
        dot_prefixed = tmp_path / ".review-helper"
        dot_prefixed.mkdir()
        (dot_prefixed / "SKILL.md").write_text("---\nname: review-helper\n---\n", encoding="utf-8")
        skipped = tmp_path / ".git"
        skipped.mkdir()
        (skipped / "SKILL.md").write_text("---\nname: skipped\n---\n", encoding="utf-8")
        linked_target = tmp_path.parent / f"{tmp_path.name}-linked-target"
        linked_target.mkdir()
        (linked_target / "SKILL.md").write_text("---\nname: linked\n---\n", encoding="utf-8")
        try:
            (tmp_path / "linked-skill").symlink_to(linked_target, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are not supported on this filesystem")

        result = detect_skills(tmp_path)

        assert result.is_multi_skill is True
        assert [skill.relative_path for skill in result.skills] == [
            ".review-helper",
            "skill-a",
            "skill-b",
        ]
        assert {skill.name for skill in result.skills} == {
            "review-helper",
            "skill-a",
            "skill-b",
        }

    def test_symlinked_skill_directory_is_skipped(self, tmp_path: Path) -> None:
        """Detection must not read a skill manifest through a directory symlink."""
        for name in ("skill-a", "skill-b"):
            sub = tmp_path / name
            sub.mkdir()
            (sub / "SKILL.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")
        external = tmp_path.parent / "external-skill"
        external.mkdir()
        (external / "SKILL.md").write_text("---\nname: private\n---\n", encoding="utf-8")
        try:
            (tmp_path / "linked-skill").symlink_to(external, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are not supported on this filesystem")

        result = detect_skills(tmp_path)

        assert result.is_multi_skill is True
        assert {skill.name for skill in result.skills} == {"skill-a", "skill-b"}

    def test_symlinked_root_is_not_detected(self, tmp_path: Path) -> None:
        """Direct callers cannot use detection to inspect a symlinked root."""
        external = tmp_path / "external"
        external.mkdir()
        (external / "SKILL.md").write_text("---\nname: private\n---\n", encoding="utf-8")
        symlink = tmp_path / "linked-root"
        try:
            symlink.symlink_to(external, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are not supported on this filesystem")

        result = detect_skills(symlink)

        assert result.is_multi_skill is False
        assert result.has_root_skill is False

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

    def test_relative_root_produces_absolute_skill_paths(
        self, multi_skill_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Relative caller input is normalized before child paths are returned."""
        monkeypatch.chdir(multi_skill_dir.parent)

        result = detect_skills(Path(multi_skill_dir.name))

        assert result.complete is True
        assert all(skill.path.is_absolute() for skill in result.skills)

    def test_real_high_entry_count_fails_closed_without_partial_skills(
        self, tmp_path: Path
    ) -> None:
        """A directory beyond the real retained-entry bound is explicitly partial."""
        for index in range(multi_skill_module.MAX_MULTI_SKILL_DIRECTORY_ENTRIES + 1):
            (tmp_path / f"entry-{index:05d}").touch()

        result = detect_skills(tmp_path)

        assert result.complete is False
        assert result.is_multi_skill is False
        assert result.skills == []
        assert result.entries_examined == (multi_skill_module.MAX_MULTI_SKILL_DIRECTORY_ENTRIES + 1)
        limitation = result.limitations[0]
        assert limitation.reason_code == "artifact_count_limit"
        assert limitation.resource == "multi_skill_directory_entries"
        assert limitation.observed_artifacts == result.entries_examined
        assert limitation.limit_artifacts == (multi_skill_module.MAX_MULTI_SKILL_DIRECTORY_ENTRIES)
        assert limitation.as_ledger_metadata()["path"] == "."

    def test_shared_runtime_limit_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deadline covers enumeration and every child classifier."""
        (tmp_path / "ordinary-entry").touch()
        ticks = iter((0.0, 0.0, multi_skill_module.MAX_MULTI_SKILL_RUNTIME_SECONDS))
        monkeypatch.setattr(
            multi_skill_module.time,
            "monotonic",
            lambda: next(ticks, multi_skill_module.MAX_MULTI_SKILL_RUNTIME_SECONDS),
        )

        result = detect_skills(tmp_path)

        assert result.complete is False
        assert result.skills == []
        limitation = result.limitations[0]
        assert limitation.reason_code == "runtime_limit"
        assert limitation.resource == "multi_skill_runtime"
        assert limitation.limit_seconds == multi_skill_module.MAX_MULTI_SKILL_RUNTIME_SECONDS

    def test_real_oversized_manifest_frontmatter_fails_closed(self, tmp_path: Path) -> None:
        """Name extraction reads at most the documented frontmatter prefix."""
        oversized = tmp_path / "a-oversized"
        oversized.mkdir()
        payload = b"---\nname: " + b"x" * (
            multi_skill_module.MAX_MULTI_SKILL_MANIFEST_FRONTMATTER_BYTES + 1
        )
        (oversized / "SKILL.md").write_bytes(payload)
        normal = tmp_path / "b-normal"
        normal.mkdir()
        (normal / "SKILL.md").write_text("---\nname: normal\n---\n", encoding="utf-8")

        result = detect_skills(tmp_path)

        assert len(payload) > multi_skill_module.MAX_MULTI_SKILL_MANIFEST_FRONTMATTER_BYTES
        assert result.complete is False
        assert result.is_multi_skill is False
        assert result.skills == []
        limitation = result.limitations[0]
        assert limitation.reason_code == "manifest_parse_limit"
        assert limitation.resource == "multi_skill_manifest_bytes"
        assert limitation.observed_bytes == (
            multi_skill_module.MAX_MULTI_SKILL_MANIFEST_FRONTMATTER_BYTES + 1
        )
        assert limitation.limit_bytes == (
            multi_skill_module.MAX_MULTI_SKILL_MANIFEST_FRONTMATTER_BYTES
        )

    def test_large_body_after_bounded_frontmatter_is_complete(self, tmp_path: Path) -> None:
        """Only frontmatter, rather than an unrelated large body, is bounded."""
        for name in ("alpha", "beta"):
            child = tmp_path / name
            child.mkdir()
            (child / "SKILL.md").write_bytes(
                f"---\nname: {name}\n---\n".encode()
                + b"x" * multi_skill_module.MAX_MULTI_SKILL_MANIFEST_FRONTMATTER_BYTES
            )

        result = detect_skills(tmp_path)

        assert result.complete is True
        assert result.is_multi_skill is True
        assert {skill.name for skill in result.skills} == {"alpha", "beta"}

    def test_structured_candidates_share_one_aggregate_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Many child probes cannot multiply the structured candidate ceiling."""
        monkeypatch.setattr(multi_skill_module, "MAX_MULTI_SKILL_STRUCTURED_CANDIDATES", 1)
        candidate_dir = tmp_path / "a-candidates"
        candidate_dir.mkdir()
        (candidate_dir / "one.aisop.json").write_text("{}", encoding="utf-8")
        (candidate_dir / "two.aisop.json").write_text("{}", encoding="utf-8")
        normal = tmp_path / "b-normal"
        normal.mkdir()
        (normal / "SKILL.md").write_text("---\nname: normal\n---\n", encoding="utf-8")

        result = detect_skills(tmp_path)

        assert result.complete is False
        assert result.skills == []
        assert result.structured_candidates_examined == 2
        limitation = result.limitations[0]
        assert limitation.reason_code == "artifact_count_limit"
        assert limitation.resource == "multi_skill_structured_candidates"

    def test_symlinked_ancestor_is_explicitly_incomplete(self, tmp_path: Path) -> None:
        """Direct callers cannot make discovery traverse a symlinked ancestor."""
        actual_parent = tmp_path / "actual-parent"
        skill_root = actual_parent / "skills"
        skill_root.mkdir(parents=True)
        link = tmp_path / "linked-parent"
        try:
            link.symlink_to(actual_parent, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are not supported on this filesystem")

        result = detect_skills(link / "skills")

        assert result.complete is False
        assert result.skills == []
        assert result.limitations[0].resource == "multi_skill_input_path"

    def test_console_facing_names_are_sanitized(self, tmp_path: Path) -> None:
        """Manifest and directory names cannot inject Rich markup or controls."""
        for directory_name, manifest_name in (
            ("alpha[bold]", "[red]alpha[/red]"),
            ("beta", "beta\x1b[31m"),
        ):
            child = tmp_path / directory_name
            child.mkdir()
            (child / "SKILL.md").write_text(
                f"---\nname: '{manifest_name}'\n---\n",
                encoding="utf-8",
            )

        result = detect_skills(tmp_path)

        assert result.complete is True
        assert result.is_multi_skill is True
        for skill in result.skills:
            assert "[" not in skill.name
            assert "]" not in skill.name
            assert "\x1b" not in skill.name
            assert "[" not in skill.relative_path
            assert "]" not in skill.relative_path

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
