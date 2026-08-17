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

import base64
import json
import os
from pathlib import Path
from typing import BinaryIO

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from skillspector.constants import MODEL_CONFIG
from skillspector.nodes.build_context import build_context
from skillspector.providers import reset_provider, use_provider
from skillspector.python_ast import ParsedPythonFile, get_python_ast
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
    python_ast_cache_key = result["python_ast_cache_key"]
    assert isinstance(python_ast_cache_key, str)
    parsed_python = get_python_ast(
        python_ast_cache_key,
        result["file_cache"]["scripts/run.py"],
        "scripts/run.py",
    )
    assert isinstance(parsed_python, ParsedPythonFile)
    assert parsed_python.is_parseable
    assert parsed_python.tree is not None
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


def test_build_context_ast_cache_skips_oversized_python(tmp_path: Path) -> None:
    """Prewarming respects the same source-size limit as AST analyzers."""
    from skillspector.python_ast import MAX_PYTHON_AST_SOURCE_CHARS

    (tmp_path / "oversized.py").write_text("x = 1\n" + "#" * MAX_PYTHON_AST_SOURCE_CHARS)

    result = build_context({"skill_path": str(tmp_path)})

    assert result["python_ast_cache_key"] is None


def test_build_context_ast_cache_handle_is_checkpoint_serializable(tmp_path: Path) -> None:
    """Raw AST objects remain in runtime storage, not checkpointed graph state."""
    (tmp_path / "script.py").write_text("import os\n", encoding="utf-8")

    result = build_context({"skill_path": str(tmp_path)})

    serializer = JsonPlusSerializer()
    assert serializer.dumps_typed(result)


def test_build_context_reads_directory_with_windows_secure_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows' handle-based fallback keeps normal directory scans usable."""
    _make_skill_spec_dir(tmp_path)

    def open_with_windows_handle(path: Path) -> BinaryIO:
        return path.open("rb")

    monkeypatch.setattr("skillspector.input_handler._HAS_SECURE_DIR_FD", False)
    monkeypatch.setattr("skillspector.input_handler._IS_WINDOWS", True)
    monkeypatch.setattr(
        "skillspector.input_handler._open_regular_file_from_windows_handle",
        open_with_windows_handle,
    )

    result = build_context({"skill_path": str(tmp_path)})

    assert result["file_cache"]["SKILL.md"].startswith("---")
    assert result["file_cache"]["scripts/run.py"] == "print(1)\n"
    assert result["manifest"]["name"] == "test-skill"


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


def test_build_context_model_config_uses_bound_provider(tmp_path: Path) -> None:
    class _BoundProvider:
        DEFAULT_MODEL = "bound-default"
        SLOT_DEFAULTS = {"meta_analyzer": "bound-meta"}

        def get_context_length(self, model: str) -> int | None:
            return 4096

        def get_max_output_tokens(self, model: str) -> int | None:
            return 128

        def resolve_model(self, slot: str = "default") -> str:
            return self.SLOT_DEFAULTS.get(slot, self.DEFAULT_MODEL)

        def resolve_credentials(self) -> tuple[str, str | None] | None:
            return None

        def create_chat_model(self, model: str, *, max_tokens: int, timeout: float | None = 120):
            return object()

    token = use_provider(_BoundProvider())
    try:
        result = build_context({"skill_path": str(tmp_path)})
    finally:
        reset_provider(token)

    assert result["model_config"]["default"] == "bound-default"
    assert result["model_config"]["meta_analyzer"] == "bound-meta"


def test_build_context_inventories_but_excludes_valid_root_oms_signature(
    tmp_path: Path,
) -> None:
    """A real OMS signature is reported as metadata but withheld from analyzers."""
    (tmp_path / "SKILL.md").write_text("---\nname: signed\n---\n# Signed\n", encoding="utf-8")
    signature_path = _write_real_oms_signature(tmp_path)

    result = build_context({"skill_path": str(tmp_path)})

    assert "skill.oms.sig" not in result["components"]
    assert "skill.oms.sig" not in result["file_cache"]
    assert any(
        event["path"] == "skill.oms.sig" and event["reason_code"] == "oms_signature"
        for event in result["inspection_ledger"]
    )
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


def test_build_context_excludes_future_oms_predicate_version(tmp_path: Path) -> None:
    """OMS predicate revisions remain excluded without relaxing the namespace check."""
    bundle = json.loads(_OMS_FIXTURE.read_text(encoding="utf-8"))
    payload = json.loads(base64.b64decode(bundle["dsseEnvelope"]["payload"]))
    payload["predicateType"] = "https://model_signing/signature/v1.1"
    bundle["dsseEnvelope"]["payload"] = base64.b64encode(
        json.dumps(payload).encode("utf-8")
    ).decode("ascii")
    (tmp_path / "skill.oms.sig").write_text(json.dumps(bundle), encoding="utf-8")

    result = build_context({"skill_path": str(tmp_path)})

    assert "skill.oms.sig" not in result["components"]
    assert any(
        event["path"] == "skill.oms.sig" and event["reason_code"] == "oms_signature"
        for event in result["inspection_ledger"]
    )


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
    signature_meta = next(
        item for item in result["component_metadata"] if item["path"] == "skill.oms.sig"
    )
    assert signature_meta["type"] == "other"


def test_build_context_scans_nested_oms_signature(tmp_path: Path) -> None:
    """Only the signature at the skill root is eligible for recognition."""
    nested = _write_real_oms_signature(tmp_path, "nested/skill.oms.sig")

    result = build_context({"skill_path": str(tmp_path)})

    assert result["file_cache"]["nested/skill.oms.sig"] == nested.read_text(encoding="utf-8")
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
        "  - not-a-dict\n"
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


def test_build_context_reports_exclusion_boundary_without_descendants(tmp_path: Path) -> None:
    """Excluded directory trees produce one boundary record, not child records."""
    (tmp_path / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    excluded = tmp_path / "node_modules" / "pkg"
    excluded.mkdir(parents=True)
    (excluded / "index.js").write_text("alert(1)\n", encoding="utf-8")

    result = build_context({"skill_path": str(tmp_path)})
    exclusions = [
        event for event in result["inspection_ledger"] if event["outcome"] == "out_of_scope"
    ]

    assert [event["path"] for event in exclusions] == ["node_modules/"]
    assert "node_modules/pkg/index.js" not in result["components"]


def test_build_context_inventories_hidden_file_for_local_analysis(tmp_path: Path) -> None:
    """Hidden regular files stay local and never enter the LLM-visible cache."""
    (tmp_path / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=not-reported\n", encoding="utf-8")

    result = build_context({"skill_path": str(tmp_path)})
    assert ".env" in result["components"]
    assert ".env" in result["local_file_cache"]
    assert ".env" not in result["file_cache"]
    assert ".env" not in result["llm_components"]
    assert not any(
        event.get("reason_code") == "hidden_file" for event in result["inspection_ledger"]
    )


def test_build_context_reports_read_error_without_fake_empty_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unreadable files remain inventoried but are absent from the content cache."""
    target = tmp_path / "broken.py"
    target.write_text("print(1)\n", encoding="utf-8")

    def deny_open(*args: object, **kwargs: object) -> int:
        raise PermissionError("sensitive operating-system detail")

    monkeypatch.setattr("skillspector.input_handler.os.open", deny_open)
    result = build_context({"skill_path": str(tmp_path)})

    assert "broken.py" in result["components"]
    assert "broken.py" not in result["file_cache"]
    event = next(entry for entry in result["inspection_ledger"] if entry["path"] == "broken.py")
    assert event["reason_code"] == "read_error"
    assert event["error_class"] == "PermissionError"
    assert "sensitive" not in event["message"]


def test_build_context_records_non_regular_files_in_the_ledger(tmp_path: Path) -> None:
    """Named pipes are inventoried so the cache phase can report their failure."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable on this platform")
    pipe = tmp_path / "events.pipe"
    os.mkfifo(pipe)

    result = build_context({"skill_path": str(tmp_path)})

    assert "events.pipe" in result["components"]
    assert "events.pipe" not in result["file_cache"]
    event = next(entry for entry in result["inspection_ledger"] if entry["path"] == "events.pipe")
    assert event["reason_code"] == "not_regular_file"


def test_build_context_excludes_dangling_symlink_from_scan_scope(tmp_path: Path) -> None:
    """Symlinks are excluded rather than read as files from an unknown target."""
    dangling = tmp_path / "missing.py"
    try:
        dangling.symlink_to("no-longer-present.py")
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    result = build_context({"skill_path": str(tmp_path)})

    assert "missing.py" not in result["components"]
    assert "missing.py" not in result["file_cache"]
    event = next(entry for entry in result["inspection_ledger"] if entry["path"] == "missing.py")
    assert event["reason_code"] == "not_regular_file"


def test_build_context_records_stat_errors_in_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unstatable discovered entry produces structured STAT_ERROR evidence."""
    target = tmp_path / "protected.py"
    target.write_text("print(1)\n", encoding="utf-8")
    original = Path.stat

    def fail_target(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path == target:
            raise PermissionError("sensitive operating-system detail")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_target)
    result = build_context({"skill_path": str(tmp_path)})

    assert "protected.py" in result["components"]
    assert "protected.py" not in result["file_cache"]
    event = next(entry for entry in result["inspection_ledger"] if entry["path"] == "protected.py")
    assert event["reason_code"] == "stat_error"
    assert event["error_class"] == "PermissionError"


def test_build_context_records_non_regular_entries_in_the_ledger(tmp_path: Path) -> None:
    """A discovered FIFO is retained as failed ledger evidence, never silently skipped."""
    fifo = tmp_path / "inspection.pipe"
    os.mkfifo(fifo)

    result = build_context({"skill_path": str(tmp_path)})

    assert "inspection.pipe" in result["components"]
    assert "inspection.pipe" not in result["file_cache"]
    event = next(
        entry for entry in result["inspection_ledger"] if entry["path"] == "inspection.pipe"
    )
    assert event["reason_code"] == "not_regular_file"


def test_build_context_rejects_symlink_to_external_file(tmp_path: Path) -> None:
    """A symlinked file outside skill_dir must not enter the component cache."""
    secret = tmp_path.parent / "external_secret.txt"
    secret.write_text("AWS_SECRET=hunter2", encoding="utf-8")

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n", encoding="utf-8")
    (skill_dir / "creds.md").symlink_to(secret)

    result = build_context({"skill_path": str(skill_dir)})

    assert "creds.md" not in result["components"]
    assert "creds.md" not in result["file_cache"]
    assert all("hunter2" not in content for content in result["file_cache"].values())


def test_build_context_rejects_symlinked_directory(tmp_path: Path) -> None:
    """A symlinked subdirectory outside skill_dir must not be traversed."""
    external = tmp_path.parent / "external_dir"
    external.mkdir(exist_ok=True)
    (external / "leak.md").write_text("PRIVATE_KEY=xyz", encoding="utf-8")

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n", encoding="utf-8")
    (skill_dir / "linked").symlink_to(external, target_is_directory=True)

    result = build_context({"skill_path": str(skill_dir)})

    assert not any(path.startswith("linked/") for path in result["components"])
    assert all("PRIVATE_KEY" not in content for content in result["file_cache"].values())


def test_build_context_rejects_junctioned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows junctions must be excluded before os.walk can traverse them."""
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "leak.md").write_text("PRIVATE_KEY=xyz", encoding="utf-8")
    original_is_junction = Path.is_junction

    def is_junction(path: Path) -> bool:
        return path == linked or original_is_junction(path)

    monkeypatch.setattr(Path, "is_junction", is_junction)
    result = build_context({"skill_path": str(tmp_path)})

    assert not any(path.startswith("linked/") for path in result["components"])
    assert all("PRIVATE_KEY" not in content for content in result["file_cache"].values())
    event = next(entry for entry in result["inspection_ledger"] if entry["path"] == "linked/")
    assert event["reason_code"] == "not_regular_file"


def test_build_context_rejects_in_tree_symlink(tmp_path: Path) -> None:
    """Even an in-tree symlink is skipped rather than read through."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "real.md").write_text("real content", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n", encoding="utf-8")
    (skill_dir / "alias.md").symlink_to(skill_dir / "real.md")

    result = build_context({"skill_path": str(skill_dir)})

    assert "real.md" in result["components"]
    assert "alias.md" not in result["components"]


def test_build_context_rejects_file_swapped_to_symlink_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path replaced after stat must not leak its new symlink target."""
    from skillspector.nodes.build_context import _open_regular_file_no_follow

    secret = tmp_path.parent / "external_secret.txt"
    secret.write_text("AWS_SECRET=hunter2", encoding="utf-8")
    target = tmp_path / "payload.md"
    target.write_text("safe", encoding="utf-8")

    def replace_target(path: Path) -> BinaryIO:
        if path.name == target.name:
            path.unlink()
            path.symlink_to(secret)
        return _open_regular_file_no_follow(path)

    monkeypatch.setattr(
        "skillspector.nodes.build_context._open_regular_file_no_follow", replace_target
    )
    result = build_context({"skill_path": str(tmp_path)})

    assert "payload.md" in result["components"]
    assert "payload.md" not in result["file_cache"]
    assert all("hunter2" not in content for content in result["file_cache"].values())
    event = next(entry for entry in result["inspection_ledger"] if entry["path"] == "payload.md")
    assert event["reason_code"] == "not_regular_file"


def test_build_context_rejects_symlinked_manifest(tmp_path: Path) -> None:
    """Manifest parsing cannot bypass symlink rejection applied to the cache."""
    external = tmp_path.parent / "external_manifest.md"
    external.write_text(
        "---\nname: private-name\ndescription: private-description\n---\n", encoding="utf-8"
    )
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").symlink_to(external)

    result = build_context({"skill_path": str(skill_dir)})

    assert result["manifest"] == {}
    assert "SKILL.md" not in result["components"]
    assert "SKILL.md" not in result["file_cache"]


def _write_aisop_bundle(path: Path) -> None:
    """Write a valid minimal AISOP/AISP bundle file."""
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
                "functions": {
                    "inbox": {"constraints": ["Read-only inspection must not modify files."]}
                },
                "aisp_contract": {
                    "resources": {
                        "state": {"path": "resources/state.json"},
                    },
                    "declared_tools": ["mail", "search"],
                },
            },
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle), encoding="utf-8")


def _make_nested_functions(depth: int) -> dict[str, object]:
    """Build a deeply nested functions tree for recursion-guard tests."""
    current: dict[str, object] = {"constraints": ["depth.guard"]}
    for idx in range(depth, -1, -1):
        current = {f"node_{idx}": {"constraints": [f"depth_{idx}"], "functions": current}}
    return current


def test_build_context_populates_structured_skill_context(tmp_path: Path) -> None:
    """Valid AISOP/AISP bundle yields structured_skill_context metadata in scan context."""
    _write_aisop_bundle(tmp_path / "workflow.aisop.json")
    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)

    assert "structured_skill_context" in result
    context = result["structured_skill_context"]
    assert isinstance(context, dict)
    assert context["protocol"] == "AISP V1"
    assert context["layout_kind"] == "AISP"
    assert context["format"] == "contract"
    assert context["bundle_path"] == str((tmp_path / "workflow.aisop.json").resolve())
    assert context["workflow_nodes"] == ["inbox"]
    assert context["constraint_anchors"] == ["Read-only inspection must not modify files."]
    assert context["resource_anchors"] == ["resources/state.json"]
    assert context["declared_tools"] == ["mail", "search"]


@pytest.mark.parametrize("ancestor", [".claude", "venv"])
def test_build_context_structured_bundle_under_ancestor(tmp_path: Path, ancestor: str) -> None:
    """Scan-root-relative filters keep bundles under external ancestors."""
    skill_dir = tmp_path / ancestor / "skills" / "foo"
    skill_dir.mkdir(parents=True)
    _write_aisop_bundle(skill_dir / "workflow.aisop.json")

    result = build_context({"skill_path": str(skill_dir)})

    assert "workflow.aisop.json" in result["components"]
    assert "structured_skill_context" in result


def test_build_context_manifest_may_be_empty_when_only_structured(tmp_path: Path) -> None:
    """A structured bundle can populate context while manifest stays empty."""
    _write_aisop_bundle(tmp_path / "workflow.aisop.json")
    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)
    assert result["manifest"] == {}
    assert "structured_skill_context" in result


def test_build_context_structured_context_absent_for_malformed_bundle(tmp_path: Path) -> None:
    """Malformed AISOP/AISP JSON leaves structured_skill_context unset."""
    (tmp_path / "bad.aisop.json").write_text(
        json.dumps([{"role": "system", "content": {"protocol": "AISOP V1"}}, {}]),
        encoding="utf-8",
    )
    state: SkillspectorState = {"skill_path": str(tmp_path)}
    result = build_context(state)
    assert "structured_skill_context" not in result


def test_build_context_deduplicates_nested_workflow_names(tmp_path: Path) -> None:
    """Nested function names stay unique in structured_skill_context."""
    bundle = [
        {
            "role": "system",
            "content": {
                "protocol": "AISOP V1",
                "format": "workflow",
            },
        },
        {
            "role": "user",
            "content": {
                "aisop": {"main": "graph TD"},
                "functions": {
                    "lookup": {
                        "functions": {
                            "lookup": {
                                "constraints": ["nested.query"],
                            }
                        }
                    }
                },
            },
        },
    ]
    (tmp_path / "nested.aisop.json").write_text(json.dumps(bundle), encoding="utf-8")
    result = build_context({"skill_path": str(tmp_path)})
    context = result["structured_skill_context"]
    assert context["workflow_nodes"] == ["lookup"]


def test_build_context_ignores_over_nested_structured_bundle(tmp_path: Path) -> None:
    """Over-nested structured bundles fail closed instead of crashing build_context."""
    bundle = [
        {
            "role": "system",
            "content": {
                "protocol": "AISOP V1",
                "format": "workflow",
            },
        },
        {
            "role": "user",
            "content": {
                "aisop": {"main": "graph TD"},
                "functions": _make_nested_functions(140),
            },
        },
    ]
    (tmp_path / "deep.aisop.json").write_text(json.dumps(bundle), encoding="utf-8")

    result = build_context({"skill_path": str(tmp_path)})

    assert "structured_skill_context" not in result
