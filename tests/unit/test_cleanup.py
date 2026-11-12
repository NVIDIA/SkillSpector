# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scan temp-directory cleanup."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from skillspector.cleanup import TempDirTracker, _retry_writable, cleanup_result
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


def _scan_temp_dir(root: Path) -> Path:
    """Create a directory named like the ones ``InputHandler`` materializes."""
    path = root / "skillspector_abc123"
    path.mkdir()
    (path / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    return path


def test_tracker_records_a_scan_temp_dir(tmp_path: Path) -> None:
    """A node output naming an existing skillspector_ directory is recorded."""
    temp_dir = _scan_temp_dir(tmp_path)
    tracker = TempDirTracker()

    tracker.on_chain_end({"temp_dir_for_cleanup": str(temp_dir)})

    assert tracker.temp_dir == str(temp_dir)


@pytest.mark.parametrize("later", [None, "", 42, ["x"]])
def test_tracker_keeps_recorded_path_when_a_later_output_has_none(
    tmp_path: Path, later: object
) -> None:
    """A later output without a usable path does not erase the recorded one."""
    temp_dir = _scan_temp_dir(tmp_path)
    tracker = TempDirTracker()
    tracker.on_chain_end({"temp_dir_for_cleanup": str(temp_dir)})

    tracker.on_chain_end({"temp_dir_for_cleanup": later})
    tracker.on_chain_end({"other": "value"})
    tracker.on_chain_end("not a mapping")

    assert tracker.temp_dir == str(temp_dir)


def test_tracker_ignores_paths_that_are_not_scan_temp_dirs(tmp_path: Path) -> None:
    """Only an existing, non-symlink skillspector_ directory can become a deletion target."""
    user_dir = tmp_path / "my-skill"
    user_dir.mkdir()
    missing = tmp_path / "skillspector_missing"
    a_file = tmp_path / "skillspector_file"
    a_file.write_text("x", encoding="utf-8")
    tracker = TempDirTracker()

    for candidate in (user_dir, missing, a_file):
        tracker.on_chain_end({"temp_dir_for_cleanup": str(candidate)})

    assert tracker.temp_dir is None
    assert user_dir.exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_tracker_ignores_a_symlink_named_like_a_scan_temp_dir(tmp_path: Path) -> None:
    """A symlink is never recorded, even with the scan prefix and a directory target."""
    target = tmp_path / "keep"
    target.mkdir()
    link = tmp_path / "skillspector_link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("cannot create symlinks here")
    tracker = TempDirTracker()

    tracker.on_chain_end({"temp_dir_for_cleanup": str(link)})

    assert tracker.temp_dir is None


def test_tracker_remove_twice_is_a_no_op(tmp_path: Path) -> None:
    """Removing an already-removed directory does not raise."""
    temp_dir = _scan_temp_dir(tmp_path)
    tracker = TempDirTracker()
    tracker.on_chain_end({"temp_dir_for_cleanup": str(temp_dir)})

    tracker.remove()
    tracker.remove()

    assert not temp_dir.exists()


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_removing_on_error_reraises_the_original_exception(
    tmp_path: Path, error: type[BaseException]
) -> None:
    """The wrapped run's exception propagates unchanged after the directory is removed."""
    temp_dir = _scan_temp_dir(tmp_path)
    tracker = TempDirTracker()
    original = error("stopped")

    with pytest.raises(error) as raised:
        with tracker.removing_on_error():
            tracker.on_chain_end({"temp_dir_for_cleanup": str(temp_dir)})
            raise original

    assert raised.value is original
    assert not temp_dir.exists()


def test_removing_on_error_leaves_the_directory_on_success(tmp_path: Path) -> None:
    """A run that completes leaves removal to the caller's cleanup_result."""
    temp_dir = _scan_temp_dir(tmp_path)
    tracker = TempDirTracker()

    with tracker.removing_on_error():
        tracker.on_chain_end({"temp_dir_for_cleanup": str(temp_dir)})

    assert temp_dir.exists()
