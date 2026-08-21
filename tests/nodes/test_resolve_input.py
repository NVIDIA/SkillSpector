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

"""Tests for resolve_input node."""

from pathlib import Path

import pytest

from skillspector.input_handler import TransitiveIngestTruncatedError
from skillspector.nodes.resolve_input import resolve_input


def test_resolve_input_with_input_path_directory(tmp_path: Path) -> None:
    """When input_path is a local directory, skill_path is set; temp_dir_for_cleanup is None."""
    (tmp_path / "SKILL.md").write_text("# Test", encoding="utf-8")
    state = {"input_path": str(tmp_path)}
    update = resolve_input(state)
    assert update["skill_path"] == str(tmp_path.resolve())
    assert update.get("temp_dir_for_cleanup") is None


def test_resolve_input_with_skill_path_only(tmp_path: Path) -> None:
    """When only skill_path is set, it is normalized; temp_dir_for_cleanup is None."""
    (tmp_path / "SKILL.md").write_text("# Test", encoding="utf-8")
    state = {"skill_path": str(tmp_path)}
    update = resolve_input(state)
    assert update["skill_path"] == str(tmp_path.resolve())
    assert update.get("temp_dir_for_cleanup") is None


def test_resolve_input_rejects_skill_path_with_symlinked_parent(tmp_path: Path) -> None:
    """The skill_path-only route must enforce the same symlink policy as input_path."""
    external_skill = tmp_path / "external" / "skill"
    external_skill.mkdir(parents=True)
    (external_skill / "SKILL.md").write_text("# External skill", encoding="utf-8")
    symlinked_parent = tmp_path / "linked"
    try:
        symlinked_parent.symlink_to(external_skill.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not supported on this filesystem")

    with pytest.raises(ValueError, match="symlinked parent"):
        resolve_input({"skill_path": str(symlinked_parent / external_skill.name)})


def test_resolve_input_prefers_input_path_over_skill_path(tmp_path: Path) -> None:
    """When both are set, input_path wins."""
    (tmp_path / "SKILL.md").write_text("# Test", encoding="utf-8")
    state = {"input_path": str(tmp_path), "skill_path": "/other/path"}
    update = resolve_input(state)
    assert update["skill_path"] == str(tmp_path.resolve())


def test_resolve_input_empty_input_returns_none_skill_path() -> None:
    """When neither input_path nor skill_path is set (or empty), skill_path becomes None."""
    update = resolve_input({})
    assert update["skill_path"] is None
    assert update.get("temp_dir_for_cleanup") is None

    update2 = resolve_input({"input_path": "  ", "skill_path": ""})
    assert update2["skill_path"] is None


def test_workflow_budget_starts_before_input_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolve, context construction, and analyzers reuse one already-started clock."""
    captured: list[object] = []

    class CapturingHandler:
        def __init__(self, transitive_budget: object | None = None) -> None:
            assert transitive_budget is not None
            assert getattr(transitive_budget, "started_at", None) is not None
            captured.append(transitive_budget)

        def resolve(self, _input_path: str) -> tuple[Path, str]:
            return tmp_path, "directory"

        def temp_dir_for_cleanup(self) -> None:
            return None

    monkeypatch.setattr("skillspector.nodes.resolve_input.InputHandler", CapturingHandler)

    update = resolve_input({"input_path": str(tmp_path)})

    assert update["workflow_resource_budget"] is captured[0]


def test_transitive_truncation_is_typed_sanitized_and_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node never converts a truncated remote child into a clean empty input."""
    cleaned: list[bool] = []

    class TruncatedHandler:
        def __init__(self, transitive_budget: object | None = None) -> None:
            assert transitive_budget is not None

        def resolve(self, _input_path: str) -> tuple[Path, str]:
            raise TransitiveIngestTruncatedError("byte_budget_exhausted", "download")

        def cleanup(self) -> None:
            cleaned.append(True)

    monkeypatch.setattr("skillspector.nodes.resolve_input.InputHandler", TruncatedHandler)
    with pytest.raises(TransitiveIngestTruncatedError) as raised:
        resolve_input(
            {
                "input_path": "https://raw.githubusercontent.com/private/source",
                "transitive_traversal_state": object(),
            }
        )

    assert raised.value.truncation.as_dict() == {
        "code": "byte_budget_exhausted",
        "source_type": "download",
        "message": "Transitive download ingest truncated (byte_budget_exhausted)",
    }
    assert cleaned == [True]
    assert "private/source" not in str(raised.value)
