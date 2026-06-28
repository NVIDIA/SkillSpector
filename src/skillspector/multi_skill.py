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

"""Multi-skill directory detection and per-skill scanning.

Detects when a scanned directory contains multiple independent skills
(each with their own SKILL.md) and supports scanning each independently
to produce per-skill reports instead of one inflated monolithic result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from skillspector.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SkillDirectory:
    """A detected skill within a multi-skill directory."""

    path: Path
    name: str
    relative_path: str


@dataclass
class MultiSkillDetectionResult:
    """Result of scanning a directory for multiple skills."""

    is_multi_skill: bool
    skills: list[SkillDirectory] = field(default_factory=list)
    has_root_skill: bool = False


def detect_skills(directory: Path) -> MultiSkillDetectionResult:
    """Detect whether a directory contains multiple independent skills.

    A directory is considered multi-skill when:
    - It has NO root-level SKILL.md (or skill.md)
    - At least 2 immediate subdirectories contain SKILL.md (or skill.md)

    If a root SKILL.md exists, the directory is treated as a single skill
    (the standard behavior) regardless of nested SKILL.md files.

    Returns a MultiSkillDetectionResult with detected skills.
    """
    if not directory.is_dir():
        return MultiSkillDetectionResult(is_multi_skill=False)

    has_root = _has_skill_md(directory)
    if has_root:
        return MultiSkillDetectionResult(is_multi_skill=False, has_root_skill=True)

    skills: list[SkillDirectory] = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if _has_skill_md(child):
            name = _extract_skill_name(child)
            skills.append(
                SkillDirectory(
                    path=child,
                    name=name,
                    relative_path=child.name,
                )
            )

    is_multi = len(skills) >= 2
    return MultiSkillDetectionResult(
        is_multi_skill=is_multi,
        skills=skills,
        has_root_skill=False,
    )


def _has_skill_md(directory: Path) -> bool:
    """Check if directory contains a SKILL.md or skill.md at root level."""
    return (directory / "SKILL.md").is_file() or (directory / "skill.md").is_file()


def _extract_skill_name(skill_dir: Path) -> str:
    """Extract skill name from SKILL.md frontmatter, falling back to directory name."""
    import re

    import yaml

    for name in ("SKILL.md", "skill.md"):
        path = skill_dir / name
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not content.startswith("---"):
            break
        end_match = re.search(r"\n---\s*\n", content[3:])
        if not end_match:
            break
        frontmatter = content[3 : end_match.start() + 3]
        try:
            # WARNING: Do not change this to yaml.load() without an explicit Loader.
            # yaml.safe_load() is used intentionally to avoid arbitrary code execution.
            data = yaml.safe_load(frontmatter)
        except yaml.YAMLError:
            break
        if isinstance(data, dict) and "name" in data:
            return str(data["name"])
        break

    return skill_dir.name
