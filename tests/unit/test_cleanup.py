# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scan temp-directory cleanup."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from skillspector.cleanup import _retry_writable, cleanup_result
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
    """A file that is still locked is left behind; cleanup never fails the scan.

    The directory that still holds it fails ``rmdir`` with a non-permission
    error, which must not be retried: its mode stays as it was, so the tree
    remains searchable on POSIX.
    """
    temp_dir = tmp_path / "skillspector_locked"
    temp_dir.mkdir()
    locked = temp_dir / "locked.pack"
    locked.write_bytes(b"PACK")
    (temp_dir / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    real_unlink = os.unlink
    chmod_calls: list[str] = []
    real_chmod = os.chmod

    def unlink(path: str, *args: object, dir_fd: int | None = None) -> None:
        if os.path.basename(path) == locked.name:
            raise PermissionError(32, "The file is in use by another process", path)
        real_unlink(path, *args, dir_fd=dir_fd)

    def chmod(path: str, mode: int, *args: object, **kwargs: object) -> None:
        chmod_calls.append(os.path.basename(path))
        real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "chmod", chmod)
    mode_before = stat.S_IMODE(temp_dir.stat().st_mode)

    cleanup_result({"temp_dir_for_cleanup": str(temp_dir)})

    assert locked.exists()
    assert not (temp_dir / "SKILL.md").exists()
    assert stat.S_IMODE(temp_dir.stat().st_mode) == mode_before
    assert chmod_calls == [locked.name]


def test_non_permission_failures_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only a permission error is worth a chmod; anything else is left to rmtree."""
    calls: list[str] = []
    monkeypatch.setattr(os, "chmod", lambda path, mode, **kwargs: calls.append(path))
    (tmp_path / "child").write_bytes(b"")

    _retry_writable(os.rmdir, str(tmp_path), OSError(39, "Directory not empty", str(tmp_path)))
    _retry_writable(os.unlink, str(tmp_path / "gone"), FileNotFoundError(2, "No such file"))

    assert calls == []
    assert (tmp_path / "child").exists()


def test_incompatible_callbacks_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An fd-based rmtree reports os.open and os.scandir too; those are never retried."""
    calls: list[str] = []
    monkeypatch.setattr(os, "chmod", lambda path, mode, **kwargs: calls.append(path))
    target = tmp_path / "file"
    target.write_bytes(b"")

    _retry_writable(os.open, str(target), PermissionError(13, "Access is denied", str(target)))
    _retry_writable(os.scandir, str(tmp_path), PermissionError(13, "Access is denied"))

    assert calls == []
    assert target.exists()


def test_retry_failures_never_escape_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A retry that raises something other than OSError still leaves cleanup non-fatal."""
    temp_dir = tmp_path / "skillspector_retry"
    temp_dir.mkdir()
    stubborn = temp_dir / "stubborn.pack"
    stubborn.write_bytes(b"PACK")
    stubborn.chmod(stat.S_IREAD)
    real_unlink = os.unlink
    attempts: list[str] = []

    def unlink(path: str, *args: object, dir_fd: int | None = None) -> None:
        if os.path.basename(path) == stubborn.name:
            attempts.append(path)
            if len(attempts) == 1:
                raise PermissionError(13, "Access is denied", path)
            raise TypeError("retried with an argument this callback cannot take")
        real_unlink(path, *args, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", unlink)

    cleanup_result({"temp_dir_for_cleanup": str(temp_dir)})

    assert len(attempts) == 2
    assert stubborn.exists()


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
