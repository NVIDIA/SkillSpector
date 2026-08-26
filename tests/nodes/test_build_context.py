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
import logging
import os
from pathlib import Path
from time import monotonic
from typing import BinaryIO

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from skillspector.artifacts import ArtifactDisposition
from skillspector.constants import MAX_ANALYZABLE_FILE_BYTES, MODEL_CONFIG
from skillspector.inspection_ledger import LedgerReason
from skillspector.nodes.build_context import build_context
from skillspector.providers import reset_provider, use_provider
from skillspector.python_ast import ParsedPythonFile, get_python_ast
from skillspector.state import (
    MAX_WORKFLOW_ARTIFACTS,
    MAX_WORKFLOW_BYTES,
    MAX_WORKFLOW_SECONDS,
    SkillspectorState,
    WorkflowResourceBudget,
)
from skillspector.url_redaction import (
    REDACTED_REMAINDER,
    TextRedactionIncompleteReason,
    TextRedactionResult,
)

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


def test_build_context_starts_and_returns_default_graph_wide_budget(tmp_path: Path) -> None:
    payload = b"# bounded workflow\n"
    (tmp_path / "SKILL.md").write_bytes(payload)

    result = build_context({"skill_path": str(tmp_path)})

    budget = result["workflow_resource_budget"]
    assert isinstance(budget, WorkflowResourceBudget)
    assert budget.max_seconds == MAX_WORKFLOW_SECONDS == 60.0
    assert budget.max_bytes == MAX_WORKFLOW_BYTES == 64 * 1024 * 1024
    assert budget.max_artifacts == MAX_WORKFLOW_ARTIFACTS == 10_000
    assert budget.started_at is not None
    assert budget.scanned_bytes == len(payload)
    assert budget.scanned_artifacts == 1


def test_build_context_reuses_supplied_stricter_transitive_budget(tmp_path: Path) -> None:
    from skillspector.cli import _TransitiveBudget, _TransitiveTraversalState

    (tmp_path / "SKILL.md").write_text("# child\n", encoding="utf-8")
    traversal = _TransitiveTraversalState(
        budget=_TransitiveBudget(max_bytes=16, max_seconds=3.0, max_artifacts=2)
    )

    result = build_context(
        {
            "skill_path": str(tmp_path),
            "transitive_traversal_state": traversal,
        }
    )

    assert result["workflow_resource_budget"] is traversal
    assert traversal.started_at is not None
    assert traversal.budget.max_bytes == 16
    assert traversal.budget.max_seconds == 3.0
    assert traversal.budget.max_artifacts == 2


def test_expired_graph_wide_budget_marks_discovery_partial(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("# unstarted\n", encoding="utf-8")
    budget = WorkflowResourceBudget(max_seconds=0.0)

    result = build_context(
        {
            "skill_path": str(tmp_path),
            "workflow_resource_budget": budget,
        }
    )

    assert result["workflow_resource_budget"] is budget
    assert result["components"] == []
    event = result["inspection_ledger"][0]
    assert event["outcome"] == "partial"
    assert event["reason_code"] == LedgerReason.RUNTIME_LIMIT


def test_scandir_checks_shared_deadline_for_each_directory_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lazy directory enumeration cannot run past the graph-wide clock."""
    import skillspector.nodes.build_context as build_context_module

    (tmp_path / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("second\n", encoding="utf-8")

    clock_values = iter((0.0, 0.1, 0.6))
    monkeypatch.setattr(build_context_module, "monotonic", lambda: next(clock_values))

    class FakeSharedBudget:
        def __init__(self) -> None:
            self.remaining_values = iter((0.5, 0.4, 0.0))
            self.reasons: list[str] = []

        def remaining_seconds(self) -> float:
            return next(self.remaining_values)

        def note_truncation(self, reason: str) -> None:
            self.reasons.append(reason)

    budget = FakeSharedBudget()
    paths, events = build_context_module._walk_skill_files(
        tmp_path,
        {"workflow_resource_budget": budget},
    )

    assert paths == []
    assert len(events) == 1
    assert events[0]["outcome"] == "partial"
    assert events[0]["reason_code"] == LedgerReason.RUNTIME_LIMIT
    assert events[0]["phase"] == "discovery"
    assert events[0]["observed_seconds"] == pytest.approx(0.6)
    assert events[0]["limit_seconds"] == pytest.approx(0.5)
    assert budget.reasons == ["time budget exhausted during discovery at SKILL.md"]


def test_discovery_marks_single_entry_partial_when_path_check_crosses_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last entry cannot be accepted merely because its pre-check was timely."""
    import skillspector.nodes.build_context as build_context_module

    (tmp_path / "SKILL.md").write_text("# skill\n", encoding="utf-8")

    class FakeClock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()
    original_resolves_outside = build_context_module._resolves_outside

    def slow_path_check(path: Path, root: Path) -> bool:
        result = original_resolves_outside(path, root)
        clock.now = 2.0
        return result

    monkeypatch.setattr(build_context_module, "monotonic", clock)
    monkeypatch.setattr(build_context_module, "MAX_BUNDLE_DISCOVERY_SECONDS", 1.0)
    monkeypatch.setattr(build_context_module, "_resolves_outside", slow_path_check)

    paths, events = build_context_module._walk_skill_files(tmp_path)

    assert paths == []
    assert len(events) == 1
    assert events[0]["phase"] == "discovery"
    assert events[0]["path"] == "SKILL.md"
    assert events[0]["reason_code"] == LedgerReason.RUNTIME_LIMIT
    assert events[0]["observed_seconds"] == pytest.approx(2.0)
    assert events[0]["limit_seconds"] == pytest.approx(1.0)


def test_file_cache_stops_at_progressing_shared_deadline_with_affected_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deadline reached between files marks the deterministic unread suffix partial."""
    import skillspector.nodes.build_context as build_context_module

    (tmp_path / "first.txt").write_text("first\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("second\n", encoding="utf-8")
    clock_values = iter((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6))
    monkeypatch.setattr(build_context_module, "monotonic", lambda: next(clock_values))

    class FakeSharedBudget:
        def __init__(self) -> None:
            self.remaining_values = iter((0.5, 0.4, 0.35, 0.3, 0.2, 0.1, 0.05, 0.0))
            self.reasons: list[str] = []
            self.scanned_bytes = 0

        def remaining_seconds(self) -> float:
            return next(self.remaining_values)

        def remaining_bytes(self) -> int:
            return 1_024 - self.scanned_bytes

        def record_bytes(self, count: int) -> None:
            self.scanned_bytes += count

        def note_truncation(self, reason: str) -> None:
            self.reasons.append(reason)

    budget = FakeSharedBudget()
    state: SkillspectorState = {"workflow_resource_budget": budget}
    file_cache, raw_cache, _llm_cache, inventory, events = build_context_module._read_file_cache(
        tmp_path,
        ["first.txt", "second.txt"],
        started_at=0.0,
        state=state,
    )

    assert file_cache == {"first.txt": "first\n"}
    assert raw_cache == {"first.txt": b"first\n"}
    assert [(item["path"], item["disposition"]) for item in inventory] == [
        ("first.txt", ArtifactDisposition.ANALYZED),
        ("second.txt", ArtifactDisposition.PARTIAL),
    ]
    assert len(events) == 1
    assert events[0]["phase"] == "cache"
    assert events[0]["path"] == "second.txt"
    assert events[0]["reason_code"] == LedgerReason.RUNTIME_LIMIT
    assert events[0]["observed_seconds"] == pytest.approx(0.6)
    assert events[0]["limit_seconds"] == pytest.approx(0.5)
    assert budget.reasons == ["time budget exhausted before reading second.txt"]


def test_file_cache_marks_single_file_partial_when_read_crosses_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow final read is discarded and accounted as partial, not analyzed."""
    import skillspector.nodes.build_context as build_context_module

    (tmp_path / "SKILL.md").write_text("# skill\n", encoding="utf-8")

    class FakeClock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()
    original_read = build_context_module._read_bytes_no_follow

    def slow_read(path: Path, *, max_bytes: int) -> bytes:
        result = original_read(path, max_bytes=max_bytes)
        clock.now = 2.0
        return result

    monkeypatch.setattr(build_context_module, "monotonic", clock)
    monkeypatch.setattr(build_context_module, "MAX_BUNDLE_CACHE_SECONDS", 1.0)
    monkeypatch.setattr(build_context_module, "_read_bytes_no_follow", slow_read)

    file_cache, raw_cache, llm_cache, inventory, events = build_context_module._read_file_cache(
        tmp_path,
        ["SKILL.md"],
        started_at=0.0,
    )

    assert file_cache == raw_cache == llm_cache == {}
    assert [(row["path"], row["disposition"]) for row in inventory] == [
        ("SKILL.md", ArtifactDisposition.PARTIAL)
    ]
    assert inventory[0]["reason"] == LedgerReason.RUNTIME_LIMIT.value
    assert len(events) == 1
    assert events[0]["phase"] == "cache"
    assert events[0]["path"] == "SKILL.md"
    assert events[0]["reason_code"] == LedgerReason.RUNTIME_LIMIT
    assert events[0]["observed_seconds"] == pytest.approx(2.0)
    assert events[0]["limit_seconds"] == pytest.approx(1.0)


def test_file_cache_deadline_overrides_slow_unsafe_path_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A safety check crossing the deadline remains partial, not merely excluded."""
    import skillspector.nodes.build_context as build_context_module

    (tmp_path / "SKILL.md").write_text("# skill\n", encoding="utf-8")

    class FakeClock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()

    def slow_unsafe_check(_path: Path, _root: Path) -> bool:
        clock.now = 2.0
        return True

    monkeypatch.setattr(build_context_module, "monotonic", clock)
    monkeypatch.setattr(build_context_module, "MAX_BUNDLE_CACHE_SECONDS", 1.0)
    monkeypatch.setattr(build_context_module, "_resolves_outside", slow_unsafe_check)

    file_cache, raw_cache, llm_cache, inventory, events = build_context_module._read_file_cache(
        tmp_path,
        ["SKILL.md"],
        started_at=0.0,
    )

    assert file_cache == raw_cache == llm_cache == {}
    assert inventory[0]["disposition"] == ArtifactDisposition.PARTIAL
    assert inventory[0]["reason"] == LedgerReason.RUNTIME_LIMIT.value
    assert [event["reason_code"] for event in events] == [LedgerReason.RUNTIME_LIMIT]
    assert events[0]["observed_seconds"] == pytest.approx(2.0)
    assert events[0]["limit_seconds"] == pytest.approx(1.0)


def test_dense_directory_discovery_and_cache_complete_with_modest_real_elapsed_time(
    tmp_path: Path,
) -> None:
    """A normal dense bundle stays comfortably below its documented local ceilings."""
    import skillspector.nodes.build_context as build_context_module

    for index in range(256):
        (tmp_path / f"file-{index:03d}.txt").write_text("x", encoding="utf-8")

    started = monotonic()
    paths, discovery_events = build_context_module._walk_skill_files(tmp_path)
    _text, raw, _llm, inventory, cache_events = build_context_module._read_file_cache(
        tmp_path,
        paths,
    )
    elapsed = monotonic() - started

    assert len(paths) == len(raw) == len(inventory) == 256
    assert not discovery_events
    assert not cache_events
    assert elapsed < 5.0


def test_workflow_budget_exact_limits_are_allowed_without_false_truncation() -> None:
    budget = WorkflowResourceBudget(max_bytes=4, max_artifacts=2)

    budget.record_bytes(4)
    budget.record_artifacts(2)

    assert budget.remaining_bytes() == 0
    assert budget.remaining_artifacts() == 0
    assert budget.truncation_reasons == []
    assert budget.budget_exhausted is False


def test_build_context_accepts_bundle_exactly_at_shared_limits(tmp_path: Path) -> None:
    payload = b"# exact\n"
    (tmp_path / "SKILL.md").write_bytes(payload)
    budget = WorkflowResourceBudget(max_bytes=len(payload), max_artifacts=1)

    result = build_context(
        {
            "skill_path": str(tmp_path),
            "workflow_resource_budget": budget,
        }
    )

    assert result["components"] == ["SKILL.md"]
    assert result["raw_file_cache"]["SKILL.md"] == payload
    assert budget.remaining_bytes() == 0
    assert budget.remaining_artifacts() == 0
    assert budget.truncation_reasons == []
    assert not any(
        event.get("reason_code")
        in {LedgerReason.ARTIFACT_COUNT_LIMIT, LedgerReason.TOTAL_BYTES_LIMIT}
        for event in result["inspection_ledger"]
    )


def test_workflow_budget_records_only_actual_over_limit_work() -> None:
    budget = WorkflowResourceBudget(max_bytes=4, max_artifacts=2)
    budget.record_bytes(5)
    budget.record_artifacts(3)

    assert budget.remaining_bytes() == 0
    assert budget.remaining_artifacts() == 0
    assert budget.truncation_reasons == [
        "byte budget 4 exceeded",
        "artifact budget 2 exceeded",
    ]
    assert budget.budget_exhausted is True


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


def test_build_context_parses_manifest_from_cached_snapshot_after_file_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-cache filesystem mutation cannot suppress canonical frontmatter."""
    import skillspector.nodes.build_context as build_context_module

    manifest_path = tmp_path / "SKILL.md"
    manifest_path.write_text(
        "---\nname: cached-snapshot\ndescription: bounded\n---\n# Skill\n",
        encoding="utf-8",
    )
    original_read_cache = build_context_module._read_file_cache

    def deleting_read_cache(*args: object, **kwargs: object) -> object:
        result = original_read_cache(*args, **kwargs)  # type: ignore[arg-type]
        manifest_path.unlink()
        return result

    monkeypatch.setattr(build_context_module, "_read_file_cache", deleting_read_cache)

    result = build_context_module.build_context({"skill_path": str(tmp_path)})

    assert result["manifest"]["name"] == "cached-snapshot"
    assert result["manifest"]["description"] == "bounded"
    primary = next(row for row in result["artifact_inventory"] if row["path"] == "SKILL.md")
    assert primary["disposition"] == ArtifactDisposition.ANALYZED
    assert not any(event["phase"] == "manifest" for event in result["inspection_ledger"])


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


@pytest.mark.parametrize(
    ("field", "error_class"),
    [
        (f"value: {'9' * 5_000}", "ValueError"),
        ("unknown: !!bool maybe", "KeyError"),
        ("unknown: !!timestamp abc", "AttributeError"),
        ("unknown: !!timestamp 999999-01-01", "AttributeError"),
        ("unknown: !!int ''", "IndexError"),
        ("unknown: !!float ''", "IndexError"),
    ],
)
def test_manifest_scalar_conversion_error_marks_primary_partial(
    tmp_path: Path,
    field: str,
    error_class: str,
) -> None:
    """A bounded but unconvertible YAML scalar is malformed input, not a crash."""
    (tmp_path / "SKILL.md").write_text(
        f"---\nname: bounded\n{field}\n---\n# Skill\n",
        encoding="utf-8",
    )

    result = build_context({"skill_path": str(tmp_path)})

    assert result["manifest"] == {}
    artifact = next(item for item in result["artifact_inventory"] if item["path"] == "SKILL.md")
    assert artifact["disposition"] == ArtifactDisposition.PARTIAL
    assert artifact["reason"] == LedgerReason.MANIFEST_PARSE_ERROR.value
    event = next(
        row
        for row in result["inspection_ledger"]
        if row.get("reason_code") == LedgerReason.MANIFEST_PARSE_ERROR
    )
    assert event["path"] == "SKILL.md"
    assert event["error_class"] == error_class


def test_manifest_unrelated_loader_error_remains_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The conversion guard must not hide unrelated implementation defects."""
    import skillspector.nodes.build_context as build_context_module

    (tmp_path / "SKILL.md").write_text(
        "---\nname: bounded\n---\n# Skill\n",
        encoding="utf-8",
    )

    def unexpected_failure(_loader: object) -> object:
        raise RuntimeError("unexpected loader defect")

    monkeypatch.setattr(
        build_context_module._BoundedManifestLoader,
        "get_single_data",
        unexpected_failure,
    )

    with pytest.raises(RuntimeError, match="unexpected loader defect"):
        build_context({"skill_path": str(tmp_path)})


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
    assert result["local_file_cache"][".env"] == "TOKEN=not-reported\n"
    assert result["raw_file_cache"][".env"] == b"TOKEN=not-reported\n"
    assert ".env" not in result["file_cache"]
    assert ".env" not in result["llm_file_cache"]
    assert ".env" not in result["llm_components"]
    assert not any(
        event.get("reason_code") == "hidden_file" for event in result["inspection_ledger"]
    )


def test_build_context_redacts_visible_config_urls_before_provider_cache(tmp_path: Path) -> None:
    """Visible authored configs cross the URL redactor; hidden configs remain local-only."""
    sentinel = "task7-visible-credential"
    (tmp_path / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    (tmp_path / "pip.conf").write_text(
        "[global]\n"
        f"index-url = https://user:{sentinel}@packages.example.invalid/private?token={sentinel}\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        f'[tool.uv]\nindex-url = "https://user:{sentinel}@python.example.invalid/simple"\n',
        encoding="utf-8",
    )
    (tmp_path / ".npmrc").write_text(
        f"registry=https://user:{sentinel}@npm.example.invalid/private\n",
        encoding="utf-8",
    )

    result = build_context({"skill_path": str(tmp_path)})

    provider_projection = json.dumps(result["llm_file_cache"], sort_keys=True)
    assert sentinel not in provider_projection
    assert "packages.example.invalid" in provider_projection
    assert "python.example.invalid" in provider_projection
    assert ".npmrc" not in result["llm_file_cache"]
    assert sentinel in result["local_file_cache"][".npmrc"]
    assert result["llm_redaction_incomplete_paths"] == []


def test_build_context_redacts_embedded_scheme_relative_url_before_provider_cache(
    tmp_path: Path,
) -> None:
    sentinel = "task9-build-context-scheme-relative-secret"
    (tmp_path / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    (tmp_path / "pip.conf").write_text(
        "[global]\n"
        f"index-url=//user:{sentinel}@packages.example.invalid/private?token={sentinel}\n",
        encoding="utf-8",
    )

    result = build_context({"skill_path": str(tmp_path)})

    provider_projection = json.dumps(result["llm_file_cache"], sort_keys=True)
    assert sentinel not in provider_projection
    assert "[REDACTED_URL]" in result["llm_file_cache"]["pip.conf"]
    assert sentinel in result["local_file_cache"]["pip.conf"]


@pytest.mark.parametrize(
    "template",
    [
        "<url>//user:{sentinel}@packages.example.invalid/{private_path}</url>",
        "url(//user:{sentinel}@packages.example.invalid/{private_path})",
    ],
    ids=("element-markup", "functional-markup"),
)
def test_build_context_redacts_markup_embedded_scheme_relative_url_before_provider_cache(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    template: str,
) -> None:
    sentinel = "round2-build-context-scheme-relative-secret"
    private_path = "round2-build-context-private-path"
    raw = template.format(sentinel=sentinel, private_path=private_path)
    (tmp_path / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    (tmp_path / "pip.conf").write_text(f"index-url={raw}\n", encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger="skillspector"):
        result = build_context({"skill_path": str(tmp_path)})

    provider_projection = json.dumps(result["llm_file_cache"], sort_keys=True)
    assert sentinel not in provider_projection
    assert private_path not in provider_projection
    assert "[REDACTED_URL]" in result["llm_file_cache"]["pip.conf"]
    assert raw in result["local_file_cache"]["pip.conf"]
    assert result["llm_redaction_incomplete_paths"] == []
    assert sentinel not in caplog.text
    assert private_path not in caplog.text


def test_build_context_omits_visible_artifact_when_url_redaction_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incomplete visible-artifact redaction is omitted and projected by bounded path."""
    import skillspector.nodes.build_context as build_context_module

    sentinel = "task7-incomplete-visible-artifact"
    (tmp_path / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    (tmp_path / "pip.conf").write_text(
        f"index-url = https://user:{sentinel}@packages.example.invalid/private\n",
        encoding="utf-8",
    )
    real_redactor = build_context_module.redact_text_result

    def bounded_redactor(value: str) -> TextRedactionResult:
        if sentinel in value:
            return TextRedactionResult(
                REDACTED_REMAINDER,
                False,
                0,
                TextRedactionIncompleteReason.CANDIDATE_LIMIT,
            )
        return real_redactor(value)

    monkeypatch.setattr(build_context_module, "redact_text_result", bounded_redactor)

    result = build_context({"skill_path": str(tmp_path)})

    assert "pip.conf" not in result["llm_file_cache"]
    assert "pip.conf" not in result["llm_components"]
    assert result["llm_redaction_incomplete_paths"] == ["pip.conf"]
    assert sentinel in result["local_file_cache"]["pip.conf"]


def test_component_metadata_stat_failure_redacts_credential_shaped_path_in_debug_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import skillspector.nodes.build_context as build_context_module

    sentinel = "task9-stat-log-secret"
    path = f"registry=//user:{sentinel}@host.invalid/private"
    target = tmp_path / path
    real_stat = Path.stat

    def fail_target_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == target:
            raise OSError("stat failed")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_target_stat)

    with caplog.at_level(logging.DEBUG, logger="skillspector"):
        build_context_module._build_component_metadata(
            tmp_path,
            [path],
            {path: "safe content"},
        )

    assert sentinel not in caplog.text
    assert path not in caplog.text


@pytest.mark.parametrize("failure_kind", ["open", "os"], ids=("open-error", "os-error"))
def test_cache_read_failures_redact_credential_shaped_path_in_debug_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure_kind: str,
) -> None:
    import skillspector.nodes.build_context as build_context_module

    sentinel = "task9-read-log-secret"
    path = f"registry=//user:{sentinel}@host.invalid/private"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text("safe content\n", encoding="utf-8")

    def fail_read(file_path: Path, *, max_bytes: int | None = None) -> bytes:
        del max_bytes
        if failure_kind == "open":
            raise build_context_module._FileOpenError(file_path, PermissionError("denied"))
        raise OSError("read failed")

    monkeypatch.setattr(build_context_module, "_read_bytes_no_follow", fail_read)

    with caplog.at_level(logging.DEBUG, logger="skillspector"):
        build_context_module._read_file_cache(tmp_path, [path])

    assert sentinel not in caplog.text
    assert path not in caplog.text


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


def test_structured_limit_marks_candidate_inventory_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import skillspector.structured_skill as structured_skill

    monkeypatch.setattr(structured_skill, "MAX_STRUCTURED_DOCUMENT_BYTES", 32)
    (tmp_path / "large.aisop.json").write_text("x" * 33, encoding="utf-8")

    result = build_context({"skill_path": str(tmp_path)})

    artifact = next(
        item for item in result["artifact_inventory"] if item["path"] == "large.aisop.json"
    )
    assert artifact["disposition"] == ArtifactDisposition.PARTIAL
    assert artifact["reason"] == LedgerReason.SIZE_LIMIT.value
    assert any(
        event.get("phase") == "structured_skill"
        and event.get("path") == "large.aisop.json"
        and event.get("reason_code") == LedgerReason.SIZE_LIMIT
        for event in result["inspection_ledger"]
    )


def test_build_context_reports_files_beyond_supported_envelope_as_partial(
    tmp_path: Path,
) -> None:
    (tmp_path / "SKILL.md").write_text("# Large data helper", encoding="utf-8")
    (tmp_path / "huge.dat").write_bytes(b"x" * (MAX_ANALYZABLE_FILE_BYTES + 1))

    result = build_context({"skill_path": str(tmp_path)})

    assert len(result["raw_file_cache"]["huge.dat"]) == MAX_ANALYZABLE_FILE_BYTES
    artifact = next(item for item in result["artifact_inventory"] if item["path"] == "huge.dat")
    assert artifact["size_bytes"] == MAX_ANALYZABLE_FILE_BYTES + 1
    assert artifact["disposition"] == "partial"
    assert any(
        event["path"] == "huge.dat"
        and event["outcome"] == "partial"
        and event["reason_code"] == "size_limit"
        for event in result["inspection_ledger"]
    )


def test_build_context_shares_artifact_budget_across_child_bundles(tmp_path: Path) -> None:
    """A second child sees the artifact allowance already consumed by its sibling."""
    from skillspector.cli import _TransitiveBudget, _TransitiveTraversalState

    children = [tmp_path / "first", tmp_path / "second"]
    for child in children:
        child.mkdir()
        (child / "SKILL.md").write_text("# child\n", encoding="utf-8")
        (child / "run.py").write_text("print('child')\n", encoding="utf-8")
    traversal = _TransitiveTraversalState(
        budget=_TransitiveBudget(
            max_targets=2,
            max_bytes=1_000_000,
            max_seconds=60.0,
            max_artifacts=3,
        )
    )

    first = build_context(
        {
            "skill_path": str(children[0]),
            "transitive_traversal_state": traversal,
        }
    )
    second = build_context(
        {
            "skill_path": str(children[1]),
            "transitive_traversal_state": traversal,
        }
    )

    assert len(first["artifact_inventory"]) == 2
    assert traversal.scanned_artifacts <= traversal.budget.max_artifacts
    assert traversal.remaining_artifacts() == 1
    assert not second["components"]
    assert any(
        event.get("reason_code") == LedgerReason.ARTIFACT_COUNT_LIMIT
        for event in second["inspection_ledger"]
    )
    assert any("artifact budget exhausted" in reason for reason in traversal.truncation_reasons)


def test_build_context_shares_byte_budget_across_child_bundles(tmp_path: Path) -> None:
    """Raw bytes retained by one child reduce the next child's exact read allowance."""
    from skillspector.cli import _TransitiveBudget, _TransitiveTraversalState

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_bytes = b"# first child payload\n"
    second_bytes = b"# second child payload is deliberately longer\n"
    (first_dir / "SKILL.md").write_bytes(first_bytes)
    (second_dir / "SKILL.md").write_bytes(second_bytes)
    byte_limit = len(first_bytes) + 5
    traversal = _TransitiveTraversalState(
        budget=_TransitiveBudget(
            max_targets=2,
            max_bytes=byte_limit,
            max_seconds=60.0,
            max_artifacts=100,
        )
    )

    first = build_context(
        {
            "skill_path": str(first_dir),
            "transitive_traversal_state": traversal,
        }
    )
    second = build_context(
        {
            "skill_path": str(second_dir),
            "transitive_traversal_state": traversal,
        }
    )

    assert first["raw_file_cache"]["SKILL.md"] == first_bytes
    assert second["raw_file_cache"]["SKILL.md"] == second_bytes[:5]
    assert traversal.scanned_bytes == byte_limit
    assert traversal.remaining_bytes() == 0
    second_primary = next(
        item for item in second["artifact_inventory"] if item["path"] == "SKILL.md"
    )
    assert second_primary["disposition"] == ArtifactDisposition.PARTIAL
    assert second_primary["reason"] == LedgerReason.TOTAL_BYTES_LIMIT.value
    assert any("byte budget exhausted" in reason for reason in traversal.truncation_reasons)


def test_build_context_shares_deadline_across_child_bundles(tmp_path: Path) -> None:
    """An expired traversal clock prevents the next child from restarting its own timer."""
    from skillspector.cli import _TransitiveBudget, _TransitiveTraversalState

    children = [tmp_path / "first", tmp_path / "second"]
    for child in children:
        child.mkdir()
        (child / "SKILL.md").write_text("# child\n", encoding="utf-8")
    traversal = _TransitiveTraversalState(
        budget=_TransitiveBudget(
            max_targets=2,
            max_bytes=1_000_000,
            max_seconds=1.0,
            max_artifacts=100,
        )
    )

    first = build_context(
        {
            "skill_path": str(children[0]),
            "transitive_traversal_state": traversal,
        }
    )
    assert first["components"] == ["SKILL.md"]
    assert traversal.started_at is not None
    traversal.started_at -= 2.0
    second = build_context(
        {
            "skill_path": str(children[1]),
            "transitive_traversal_state": traversal,
        }
    )

    assert not second["components"]
    assert any(
        event.get("reason_code") == LedgerReason.RUNTIME_LIMIT
        for event in second["inspection_ledger"]
    )
    assert any("time budget exhausted" in reason for reason in traversal.truncation_reasons)
