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

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import httpx
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
    assert update.get("selected_source_identity") == tmp_path.name


def test_resolve_input_with_skill_path_only(tmp_path: Path) -> None:
    """When only skill_path is set, it is normalized; temp_dir_for_cleanup is None."""
    (tmp_path / "SKILL.md").write_text("# Test", encoding="utf-8")
    state = {"skill_path": str(tmp_path)}
    update = resolve_input(state)
    assert update["skill_path"] == str(tmp_path.resolve())
    assert update.get("temp_dir_for_cleanup") is None
    assert update.get("selected_source_identity") == tmp_path.name


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
        primary_file_path = None

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


def test_selected_source_identity_from_resolved_git_and_archive_inputs() -> None:
    """Identity follows successful materialization, including preserved archive roots."""
    from skillspector.input_handler import selected_source_identity_for_input

    temp_dir = Path("/tmp/skillspector_abc")
    clone_root = temp_dir / "repo"
    assert (
        selected_source_identity_for_input(
            "https://github.com/acme/example-skill.git",
            source_type="git",
            resolved_path=clone_root,
            temp_dir=temp_dir,
        )
        == "example-skill"
    )
    assert (
        selected_source_identity_for_input(
            "git@github.com:acme/example-skill.git",
            source_type="git",
            resolved_path=clone_root,
            temp_dir=temp_dir,
        )
        == "example-skill"
    )
    assert (
        selected_source_identity_for_input(
            "/tmp/packs/example-skill.zip",
            source_type="zip",
            resolved_path=temp_dir / "extracted",
            temp_dir=temp_dir,
        )
        == "example-skill"
    )
    assert (
        selected_source_identity_for_input(
            "/tmp/packs/peer-skill.zip",
            source_type="zip",
            resolved_path=temp_dir / "extracted" / "example-skill",
            temp_dir=temp_dir,
        )
        == "example-skill"
    )
    assert (
        selected_source_identity_for_input(
            "/tmp/skillspector_abc/repo",
            source_type="directory",
            resolved_path=clone_root,
            temp_dir=None,
        )
        == "repo"
    )


@pytest.mark.parametrize("source", ["local-zip", "downloaded-zip"])
def test_failed_materialization_removes_the_handler_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str
) -> None:
    """An input that fails to materialize leaves no temp directory or download behind."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("SKILL.md", "# Skill\n")
        bundle.writestr("../outside.md", "escape\n")
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def mkdtemp(*args: str, **kwargs: str) -> str:
        path = real_mkdtemp(*args, dir=tmp_path, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr("skillspector.input_handler.tempfile.mkdtemp", mkdtemp)
    if source == "local-zip":
        local_zip = tmp_path / "slip.zip"
        local_zip.write_bytes(archive.getvalue())
        input_path = str(local_zip)
    else:
        real_client = httpx.Client
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, content=archive.getvalue())
        )
        monkeypatch.setattr(
            "skillspector.input_handler.httpx.Client",
            lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
        )
        monkeypatch.setattr("skillspector.input_handler._is_private_ip", lambda _host: False)
        input_path = "https://raw.githubusercontent.com/org/repo/main/skill.zip"

    with pytest.raises(ValueError, match="zip-slip"):
        resolve_input({"input_path": input_path})

    assert created
    assert [path for path in created if path.exists()] == []


def test_interrupted_extraction_removes_the_handler_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An interrupt after the download landed in the temp directory leaves nothing behind."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("SKILL.md", "# Skill\n")
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def mkdtemp(*args: str, **kwargs: str) -> str:
        path = real_mkdtemp(*args, dir=tmp_path, **kwargs)
        created.append(Path(path))
        return path

    def interrupted_extract(_self: object, zip_path: Path) -> Path:
        assert zip_path.exists()
        raise KeyboardInterrupt

    real_client = httpx.Client
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, content=archive.getvalue())
    )
    monkeypatch.setattr("skillspector.input_handler.tempfile.mkdtemp", mkdtemp)
    monkeypatch.setattr(
        "skillspector.input_handler.httpx.Client",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )
    monkeypatch.setattr("skillspector.input_handler._is_private_ip", lambda _host: False)
    monkeypatch.setattr("skillspector.input_handler.InputHandler._extract_zip", interrupted_extract)

    with pytest.raises(KeyboardInterrupt):
        resolve_input({"input_path": "https://raw.githubusercontent.com/org/repo/main/skill.zip"})

    assert created
    assert [path for path in created if path.exists()] == []


def test_interrupted_clone_stops_git_and_removes_the_handler_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An interrupt while Git is cloning terminates the child before the tree is removed."""
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def mkdtemp(*args: str, **kwargs: str) -> str:
        path = real_mkdtemp(*args, dir=tmp_path, **kwargs)
        created.append(Path(path))
        return path

    class InterruptedClone:
        instances: list[InterruptedClone] = []

        def __init__(self, command: list[str], **_kwargs: object) -> None:
            self.clone_dir = Path(command[-1])
            self.polls = 0
            self.terminated = False
            InterruptedClone.instances.append(self)

        def poll(self) -> int | None:
            self.polls += 1
            if self.polls == 1:
                # Git has started writing when the interrupt arrives.
                (self.clone_dir / ".git").mkdir(parents=True)
                raise KeyboardInterrupt
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr("skillspector.input_handler.tempfile.mkdtemp", mkdtemp)
    monkeypatch.setattr("skillspector.input_handler.subprocess.Popen", InterruptedClone)
    monkeypatch.setattr("skillspector.input_handler._is_private_ip", lambda _host: False)

    with pytest.raises(KeyboardInterrupt):
        resolve_input({"input_path": "https://github.com/org/repo.git"})

    assert [clone.terminated for clone in InterruptedClone.instances] == [True]
    assert created
    assert [path for path in created if path.exists()] == []
