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

"""Skill discovery — recursively find skill directories under a root path.

A directory is a skill if it directly contains a ``SKILL.md`` file.
The root directory itself is never treated as a skill.
"""

from __future__ import annotations

from pathlib import Path


def discover_skills(root: Path) -> list[Path]:
    """Recursively find all skill directories under *root*.

    Returns a list of ``Path`` objects sorted alphabetically by path.
    Each path points to a directory that contains a ``SKILL.md`` file.
    """
    skills: list[Path] = []
    for skill_md in sorted(root.rglob("SKILL.md")):
        skill_dir = skill_md.parent
        if skill_dir == root:
            continue
        skills.append(skill_dir)
    return skills
