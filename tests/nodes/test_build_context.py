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

import json
from pathlib import Path

import pytest

from skillspector.constants import MODEL_CONFIG
from skillspector.nodes.build_context import build_context
from skillspector.state import SkillspectorState

_OMS_FIXTURE = Path(__file__).parents[1] / "fixtures" / "oms" / "mcore-split-pr.skill.oms.sig"
# Pinned from NVIDIA/skills at commit 1f01acfe1aece58ba95d124eafdfb5bb93523db6:
# skills/mcore-split-pr/skill.oms.sig


def _write_real_oms_signature(root: Path, relative_path: str = "skill.oms.sig") -> Path:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_OMS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    return target


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
        "allowed-tools": [],
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


def test_build_context_inventories_but_excludes_valid_root_oms_signature(
    tmp_path: Path,
) -> None:
    """A real OMS signature is reported as metadata but withheld from analyzers."""
    (tmp_path / "SKILL.md").write_text("---\nname: signed\n---\n# Signed\n", encoding="utf-8")
    signature_path = _write_real_oms_signature(tmp_path)

    result = build_context({"skill_path": str(tmp_path)})

    assert "skill.oms.sig" in result["components"]
    assert "skill.oms.sig" not in result["file_cache"]
    assert result["analysis_excluded_components"] == ["skill.oms.sig"]
    signature_meta = next(
        item for item in result["component_metadata"] if item["path"] == "skill.oms.sig"
    )
    assert signature_meta == {
        "path": "skill.oms.sig",
        "type": "oms_signature",
        "lines": 1,
        "executable": False,
        "size_bytes": signature_path.stat().st_size,
    }


@pytest.mark.parametrize(
    "invalid_case", ["malformed_json", "wrong_media_type", "message_signature"]
)
def test_build_context_scans_unrecognized_root_oms_signature(
    tmp_path: Path,
    invalid_case: str,
) -> None:
    """Malformed and non-OMS Sigstore files retain normal scanner behavior."""
    content = _OMS_FIXTURE.read_text(encoding="utf-8")
    if invalid_case == "malformed_json":
        content = "{not-json"
    else:
        bundle = json.loads(content)
        if invalid_case == "wrong_media_type":
            bundle["mediaType"] = "application/vnd.dev.sigstore.bundle.v0.2+json"
        else:
            bundle["messageSignature"] = {"signature": "YWJj"}
            del bundle["dsseEnvelope"]
        content = json.dumps(bundle)
    (tmp_path / "skill.oms.sig").write_text(content, encoding="utf-8")

    result = build_context({"skill_path": str(tmp_path)})

    assert result["file_cache"]["skill.oms.sig"] == content
    assert result["analysis_excluded_components"] == []
    signature_meta = next(
        item for item in result["component_metadata"] if item["path"] == "skill.oms.sig"
    )
    assert signature_meta["type"] == "other"


def test_build_context_scans_nested_oms_signature(tmp_path: Path) -> None:
    """Only the signature at the skill root is eligible for recognition."""
    nested = _write_real_oms_signature(tmp_path, "nested/skill.oms.sig")

    result = build_context({"skill_path": str(tmp_path)})

    assert result["file_cache"]["nested/skill.oms.sig"] == nested.read_text(encoding="utf-8")
    assert result["analysis_excluded_components"] == []
    signature_meta = next(
        item for item in result["component_metadata"] if item["path"] == "nested/skill.oms.sig"
    )
    assert signature_meta["type"] == "other"


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


def test_build_context_parses_allowed_tools_list(tmp_path: Path) -> None:
    """`allowed-tools` list form is preserved so LP3 treats it as a declaration."""
    (tmp_path / "SKILL.md").write_text(
        "---\nname: deployer\ndescription: deploys services\nallowed-tools: [Bash, Read]\n---\n",
        encoding="utf-8",
    )
    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)
    assert result["manifest"]["allowed-tools"] == ["Bash", "Read"]


def test_build_context_allowed_tools_malformed_value(tmp_path: Path) -> None:
    """A non-list, non-string `allowed-tools` value normalizes to an empty list."""
    (tmp_path / "SKILL.md").write_text(
        "---\nname: deployer\ndescription: deploys services\nallowed-tools: 42\n---\n",
        encoding="utf-8",
    )
    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)
    assert result["manifest"]["allowed-tools"] == []


def test_build_context_parses_allowed_tools_comma_string(tmp_path: Path) -> None:
    """`allowed-tools` comma-separated string form is normalized to a list."""
    (tmp_path / "SKILL.md").write_text(
        "---\nname: deployer\ndescription: deploys services\nallowed-tools: Bash, Read\n---\n",
        encoding="utf-8",
    )
    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)
    assert result["manifest"]["allowed-tools"] == ["Bash", "Read"]
