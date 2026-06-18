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

"""Unit tests for build_context node.

Uses skill spec layout: SKILL.md, references/, scripts/, assets/
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skillspector.constants import MODEL_CONFIG
from skillspector.nodes.build_context import build_context
from skillspector.state import SkillspectorState


def _make_skill_spec_dir(root: Path, *, skill_md_name: str = "SKILL.md") -> None:
    """Populate root with skill spec: SKILL.md, references/, scripts/, assets/."""
    if skill_md_name == "SKILL.md":
        (root / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: For tests\ntriggers: [a, b]\npermissions: [read]\n---\n\n# Skill\n",
            encoding="utf-8",
        )
    (root / "references").mkdir(exist_ok=True)
    (root / "references" / "guide.md").write_text("# Reference guide\n", encoding="utf-8")
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "run.py").write_text("print(1)\n", encoding="utf-8")
    (root / "assets").mkdir(exist_ok=True)
    (root / "assets" / "icon.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    if skill_md_name == "skill.md":
        (root / "skill.md").write_text(
            "---\nname: lower\ndescription: d\n---\n",
            encoding="utf-8",
        )


def test_build_context_real_directory_with_skill_md(tmp_path: Path) -> None:
    """skill_path with skill spec (SKILL.md, references/, scripts/, assets/) yields components, file_cache, manifest."""
    _make_skill_spec_dir(tmp_path)

    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)

    assert "components" in result
    components = result["components"]
    assert isinstance(components, list)
    assert "SKILL.md" in components
    assert "references/guide.md" in components
    assert "scripts/run.py" in components
    assert "assets/icon.png" in components
    assert result["file_cache"]
    assert result["file_cache"].get("SKILL.md", "").startswith("---")
    assert result["file_cache"].get("references/guide.md") == "# Reference guide\n"
    assert result["file_cache"].get("scripts/run.py") == "print(1)\n"
    assert result["manifest"] == {
        "name": "test-skill",
        "description": "For tests",
        "triggers": ["a", "b"],
        "permissions": ["read"],
        "parameters": [],
    }
    assert result["ast_cache"] == {}
    assert result["previous_manifest"] is None
    assert "component_metadata" in result
    assert isinstance(result["component_metadata"], list)
    assert len(result["component_metadata"]) == len(result["components"])
    run_py_meta = next(
        (m for m in result["component_metadata"] if m.get("path") == "scripts/run.py"), None
    )
    assert run_py_meta is not None
    assert run_py_meta.get("type") == "python"
    assert run_py_meta.get("executable") is True
    assert run_py_meta.get("lines") == 1
    assert "has_executable_scripts" in result
    assert result["has_executable_scripts"] is True


def test_build_context_missing_skill_path() -> None:
    """Missing skill_path raises instead of producing a clean empty scan."""
    state: SkillspectorState = {}
    with pytest.raises(ValueError, match="skill_path is required"):
        build_context(state)


def test_build_context_empty_skill_path() -> None:
    """Empty skill_path raises instead of producing a clean empty scan."""
    state: SkillspectorState = {"skill_path": ""}
    with pytest.raises(ValueError, match="skill_path is required"):
        build_context(state)


def test_build_context_nonexistent_path() -> None:
    """Non-existent path raises instead of producing a clean empty scan."""
    state: SkillspectorState = {"skill_path": "/nonexistent/path/xyz"}
    with pytest.raises(ValueError, match="not an existing directory"):
        build_context(state)


def test_build_context_path_is_file_not_dir(tmp_path: Path) -> None:
    """Path that is a file raises instead of producing a clean empty scan."""
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    state: SkillspectorState = {"skill_path": str(f)}
    with pytest.raises(ValueError, match="not an existing directory"):
        build_context(state)


def test_build_context_empty_directory_is_valid_empty_scan(tmp_path: Path) -> None:
    """An existing empty directory is a valid scan target with no components."""
    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)
    assert result["components"] == []
    assert result["file_cache"] == {}
    assert result["manifest"] == {}
    assert result["model_config"] == MODEL_CONFIG


def test_build_context_skips_skip_dirs(tmp_path: Path) -> None:
    """Skip dirs like __pycache__ and node_modules are not included in components."""
    _make_skill_spec_dir(tmp_path)
    (tmp_path / "__pycache__" / "x.pyc").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "__pycache__" / "x.pyc").write_text("", encoding="utf-8")
    (tmp_path / "node_modules" / "pkg" / "index.js").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("", encoding="utf-8")

    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)

    components = result["components"]
    assert "SKILL.md" in components
    assert "references/guide.md" in components
    assert "scripts/run.py" in components
    assert not any("__pycache__" in p for p in components)
    assert not any("node_modules" in p for p in components)


def test_build_context_no_skill_md_returns_empty_manifest(tmp_path: Path) -> None:
    """Skill spec dir without SKILL.md or skill.md yields empty manifest."""
    (tmp_path / "references").mkdir(exist_ok=True)
    (tmp_path / "references" / "doc.md").write_text("x", encoding="utf-8")
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "assets").mkdir(exist_ok=True)
    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)
    assert result["manifest"] == {}
    assert "references/doc.md" in result["components"]
    assert result["file_cache"].get("references/doc.md") == "x"


def test_build_context_no_executable_scripts_when_only_markdown(tmp_path: Path) -> None:
    """Directory with only .md files has has_executable_scripts False."""
    (tmp_path / "SKILL.md").write_text("---\nname: docs-only\n---\n# Doc", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# Readme", encoding="utf-8")
    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)
    assert result["has_executable_scripts"] is False
    assert len(result["component_metadata"]) == 2
    for meta in result["component_metadata"]:
        assert meta.get("executable") is False


def test_build_context_skill_md_lowercase(tmp_path: Path) -> None:
    """skill.md (lowercase) is used when SKILL.md absent; skill spec layout."""
    _make_skill_spec_dir(tmp_path, skill_md_name="skill.md")
    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)
    assert result["manifest"]["name"] == "lower"
    assert result["manifest"]["description"] == "d"
    assert "skill.md" in result["components"]
    assert "references/guide.md" in result["components"]


def test_build_context_parses_parameters_from_frontmatter(tmp_path: Path) -> None:
    """`parameters` frontmatter is preserved as dicts so MCP TP checks can reach it.

    Regression guard: without this, the mcp_tool_poisoning parameter checks
    (TP3 and parameter-scoped TP1/TP2) never fire on real scans because the
    manifest carried no `parameters` key.
    """
    (tmp_path / "SKILL.md").write_text(
        "---\n"
        "name: reader\n"
        "description: reads data\n"
        "parameters:\n"
        "  - name: path\n"
        "    description: file path to read\n"
        "  - not-a-dict\n"  # non-dict entries are dropped
        "---\n",
        encoding="utf-8",
    )
    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)
    assert result["manifest"]["parameters"] == [
        {"name": "path", "description": "file path to read"}
    ]
