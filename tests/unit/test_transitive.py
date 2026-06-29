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

"""Tests for transitive source extraction and traversal planning."""

from skillspector import transitive


def test_plan_blocks_circular_reference() -> None:
    """Visited identities block repeated canonical targets before second resolution."""
    refs = [
        "https://github.com/org/dup.git",
        "git@github.com:org/dup.git",
        "https://github.com/org/dup",
    ]
    visited: set[str] = set()
    first = transitive.plan_transitive_targets(
        refs, visited=visited, current_depth=1, max_depth=3, allow_prefixes=(), deny_prefixes=()
    )
    second = transitive.plan_transitive_targets(
        refs, visited=visited, current_depth=1, max_depth=3, allow_prefixes=(), deny_prefixes=()
    )

    assert first == ["https://github.com/org/dup"]
    assert second == []
    assert visited == {"https://github.com/org/dup"}


def test_extract_excludes_badges_docs_and_issue_urls() -> None:
    """Non-scan URLs should be filtered out, even when they look URL-like."""
    file_cache = {
        "SKILL.md": (
            "badge https://img.shields.io/github/stars/user/repo?style=flat-square, "
            "issue https://github.com/NVIDIA/SkillSpector/issues/12, "
            "docs https://github.com/NVIDIA/SkillSpector/wiki, "
            "ci https://github.com/NVIDIA/SkillSpector/actions, "
            "src https://raw.githubusercontent.com/NVIDIA/SkillSpector/main/tool.py, "
            "zip https://huggingface.co/abc/archive/main.zip"
        ),
    }

    refs = transitive.extract_external_refs(file_cache)
    assert refs == [
        "https://raw.githubusercontent.com/NVIDIA/SkillSpector/main/tool.py",
        "https://huggingface.co/abc/archive/main.zip",
    ]


def test_extract_keeps_repos_with_reserved_word_names() -> None:
    """Reserved UI words in org or repo names should not block valid repository targets."""
    file_cache = {
        "SKILL.md": (
            "https://github.com/wiki-tools/skill.git "
            "https://github.com/org/actions.git "
            "https://github.com/badger/skill.git"
        ),
    }

    refs = transitive.extract_external_refs(file_cache)
    assert refs == [
        "https://github.com/wiki-tools/skill",
        "https://github.com/org/actions",
        "https://github.com/badger/skill",
    ]


def test_plan_depth_limit_prevents_next_wave() -> None:
    """When current depth exceeds max depth, no targets are returned."""
    refs = ["https://github.com/org/repo.git"]
    visited: set[str] = set()
    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=visited,
        current_depth=4,
        max_depth=3,
        allow_prefixes=(),
        deny_prefixes=(),
    )

    assert result == []
    assert visited == set()


def test_plan_applies_allow_prefix() -> None:
    """Only identities matching allow prefixes are returned."""
    refs = [
        "https://github.com/ok/repo.git",
        "https://github.com/skip/repo.git",
    ]
    visited: set[str] = set()
    allowed = ("https://github.com/ok/",)

    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=visited,
        current_depth=1,
        max_depth=2,
        allow_prefixes=allowed,
        deny_prefixes=(),
    )

    assert result == ["https://github.com/ok/repo"]


def test_plan_allow_prefix_respects_path_boundaries() -> None:
    """Allow prefixes should not match sibling org names sharing a string prefix."""
    refs = [
        "https://github.com/trusted/repo.git",
        "https://github.com/trusted-malicious/repo.git",
    ]
    visited: set[str] = set()

    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=visited,
        current_depth=1,
        max_depth=2,
        allow_prefixes=("https://github.com/trusted/",),
        deny_prefixes=(),
    )

    assert result == ["https://github.com/trusted/repo"]


def test_plan_applies_deny_prefix() -> None:
    """Deny prefixes skip matching identities even if they are otherwise valid."""
    refs = [
        "https://github.com/ok/repo.git",
        "https://github.com/skip/repo.git",
    ]
    visited: set[str] = set()
    denied = ("https://github.com/skip/",)

    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=visited,
        current_depth=1,
        max_depth=2,
        allow_prefixes=(),
        deny_prefixes=denied,
    )

    assert result == ["https://github.com/ok/repo"]


def test_plan_deny_prefix_respects_path_boundaries() -> None:
    """Deny prefixes should not block sibling org names that only share a string prefix."""
    refs = [
        "https://github.com/trusted/repo.git",
        "https://github.com/trusted-malicious/repo.git",
    ]
    visited: set[str] = set()

    result = transitive.plan_transitive_targets(
        refs=refs,
        visited=visited,
        current_depth=1,
        max_depth=2,
        allow_prefixes=(),
        deny_prefixes=("https://github.com/trusted/",),
    )

    assert result == ["https://github.com/trusted-malicious/repo"]
