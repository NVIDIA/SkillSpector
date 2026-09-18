# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scan temp-directory cleanup."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from skillspector.cleanup import cleanup_result
from skillspector.input_handler import InputHandler


def _refuse_read_only_unlink(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply Windows semantics everywhere: a read-only file cannot be unlinked."""
    real_unlink = os.unlink

    def unlink(path: str, *args: object, dir_fd: int | None = None) -> None:
        mode = os.stat(path, dir_fd=dir_fd, follow_symlinks=False).st_mode
        if not mode & stat.S_IWRITE:
            raise PermissionError(13, "Access is denied", path)
        real_unlink(path, *args, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", unlink)


def _clone_with_read_only_pack(root: Path) -> Path:
    """Lay out the read-only pack files ``git clone`` leaves in a temp checkout."""
    pack_dir = root / "repo" / ".git" / "objects" / "pack"
    pack_dir.mkdir(parents=True)
    for name in ("pack-1.idx", "pack-1.pack"):
        pack = pack_dir / name
        pack.write_bytes(b"PACK")
        pack.chmod(stat.S_IREAD)
    (root / "repo" / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    return root


def test_cleanup_result_removes_read_only_git_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Git URL scan's temp clone is removed even though Git marks packs read-only."""
    temp_dir = _clone_with_read_only_pack(tmp_path / "skillspector_scan")
    _refuse_read_only_unlink(monkeypatch)

    cleanup_result({"temp_dir_for_cleanup": str(temp_dir)})

    assert not temp_dir.exists()


def test_cleanup_result_stays_best_effort_when_a_file_cannot_be_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file that is still locked is left behind; cleanup never fails the scan."""
    temp_dir = tmp_path / "skillspector_locked"
    temp_dir.mkdir()
    locked = temp_dir / "locked.pack"
    locked.write_bytes(b"PACK")
    (temp_dir / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    real_unlink = os.unlink

    def unlink(path: str, *args: object, dir_fd: int | None = None) -> None:
        if os.path.basename(path) == locked.name:
            raise PermissionError(32, "The file is in use by another process", path)
        real_unlink(path, *args, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", unlink)

    cleanup_result({"temp_dir_for_cleanup": str(temp_dir)})

    assert locked.exists()
    assert not (temp_dir / "SKILL.md").exists()


def test_input_handler_cleanup_removes_read_only_git_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The handler's own cleanup path removes the same read-only clone."""
    handler = InputHandler()
    handler._temp_dir = _clone_with_read_only_pack(tmp_path / "skillspector_handler")
    temp_dir = handler._temp_dir
    _refuse_read_only_unlink(monkeypatch)

    handler.cleanup()

    assert not temp_dir.exists()
    assert handler.temp_dir_for_cleanup() is None
