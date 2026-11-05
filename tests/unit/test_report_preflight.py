# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-destructive, descriptor-relative report destination preflight."""

import errno
import os
import socket
from stat import S_IFIFO

import pytest

from skillspector import file_output
from skillspector.file_output import preflight_output, write_text_no_follow

pytestmark = pytest.mark.skipif(
    not file_output._SECURE_OUTPUT_SUPPORTED,
    reason="descriptor-relative secure output is unavailable",
)


def snapshot(root):
    """Capture identities, permissions, contents and modification timestamps."""
    result = {}
    for path in [root, *sorted(root.rglob("*"))]:
        info = path.lstat()
        content = (
            os.readlink(path)
            if path.is_symlink()
            else path.read_bytes() if path.is_file() else None
        )
        result[str(path.relative_to(root))] = (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            content,
        )
    return result


@pytest.fixture
def non_destructive(monkeypatch):
    """Forbid mutating operations, including truncating or creating opens."""
    original_open = os.open

    def checked_open(path, flags, *args, **kwargs):
        assert not flags & (os.O_CREAT | os.O_TRUNC | os.O_EXCL)
        if not flags & os.O_DIRECTORY:
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_NONBLOCK
            assert kwargs["dir_fd"] is not None
        return original_open(path, flags, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("preflight attempted a filesystem modification")

    monkeypatch.setattr(file_output.os, "open", checked_open)
    for name in ("unlink", "rename", "replace", "write", "fdopen"):
        monkeypatch.setattr(file_output.os, name, forbidden)


@pytest.mark.parametrize("existing", [False, True])
def test_preflight_returns_path_and_identity_without_changes(
    tmp_path, monkeypatch, non_destructive, existing
):
    output = tmp_path / "report=v1.json"
    if existing:
        output.write_bytes(b"previous report\n")
    before = snapshot(tmp_path)
    monkeypatch.chdir(tmp_path)

    normalized, info = preflight_output("./report=v1.json")

    assert normalized == tmp_path / output.name
    if existing:
        assert info == output.stat()
    else:
        assert info is None
    assert snapshot(tmp_path) == before


def test_preflight_returns_hard_link_identity(tmp_path, non_destructive):
    source = tmp_path / "source"
    source.write_bytes(b"protected")
    output = tmp_path / "report"
    os.link(source, output)
    before = snapshot(tmp_path)
    _, info = preflight_output(output)
    assert (info.st_dev, info.st_ino) == (source.stat().st_dev, source.stat().st_ino)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("path", [".", "/"])
def test_preflight_rejects_directory_only_paths(
    tmp_path, monkeypatch, non_destructive, path
):
    monkeypatch.chdir(tmp_path)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="non-regular|not writable or searchable"):
        preflight_output(path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ["missing", "file", "symlink", "nested-symlink"])
def test_preflight_rejects_invalid_parent(tmp_path, non_destructive, kind):
    parent = tmp_path / "reports"
    if kind == "file":
        parent.write_bytes(b"not a directory")
    elif "symlink" in kind:
        real = tmp_path / "real"
        real.mkdir()
        (real / "nested").mkdir()
        parent.symlink_to(real, target_is_directory=True)
        if kind == "nested-symlink":
            parent /= "nested"
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="output directory"):
        preflight_output(parent / "report.json")
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ["directory", "symlink", "dangling", "fifo", "socket"])
def test_preflight_rejects_non_regular_leaf(tmp_path, monkeypatch, non_destructive, kind):
    output = tmp_path / "report"
    with socket.socket(socket.AF_UNIX) as sock:
        if kind == "directory":
            output.mkdir()
        elif kind in {"symlink", "dangling"}:
            target = tmp_path / "protected"
            if kind == "symlink":
                target.write_bytes(b"protected")
            output.symlink_to(target)
        elif kind == "fifo":
            os.mkfifo(output)
        else:
            # macOS limits AF_UNIX paths to 104 bytes; pytest roots can exceed it.
            monkeypatch.chdir(tmp_path)
            sock.bind(output.name)
        before = snapshot(tmp_path)
        with pytest.raises(ValueError, match="non-regular"):
            preflight_output(output)
        assert snapshot(tmp_path) == before


@pytest.mark.parametrize("existing", [False, True])
def test_preflight_rejects_parent_without_write_or_search_access(
    tmp_path, monkeypatch, non_destructive, existing
):
    output = tmp_path / "report"
    if existing:
        output.write_bytes(b"previous")
    before = snapshot(tmp_path)

    def deny_access(path, mode, **kwargs):
        assert path == "."
        assert mode == os.W_OK | os.X_OK
        assert kwargs["effective_ids"] is True
        assert os.fstat(kwargs["dir_fd"]).st_ino == tmp_path.stat().st_ino
        return False

    monkeypatch.setattr(file_output.os, "access", deny_access)
    with pytest.raises(ValueError, match="not writable or searchable"):
        preflight_output(output)
    assert snapshot(tmp_path) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write read-only files/directories")
@pytest.mark.parametrize("kind", ["output", "parent-write", "parent-search"])
def test_preflight_checks_actual_permissions(tmp_path, kind):
    parent = tmp_path / "reports"
    parent.mkdir()
    output = parent / "report"
    output.write_bytes(b"previous")
    if kind == "output":
        output.chmod(0o400)
    else:
        parent.chmod(0o500 if kind == "parent-write" else 0o600)
    try:
        with pytest.raises(ValueError, match="Permission denied|not writable or searchable"):
            preflight_output(output)
    finally:
        parent.chmod(0o700)
        output.chmod(0o600)
    assert output.read_bytes() == b"previous"
    assert list(parent.iterdir()) == [output]


@pytest.mark.parametrize("stage", ["access", "stat", "open", "fstat"])
def test_preflight_wraps_errors_and_closes_descriptors(
    tmp_path, monkeypatch, non_destructive, stage
):
    output = tmp_path / "report"
    output.write_bytes(b"previous")
    before = snapshot(tmp_path)
    original = getattr(os, stage)
    opened = []
    original_open = os.open

    def track_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(file_output.os, "open", track_open)

    def fail(*args, **kwargs):
        if stage == "open" and args[0] != output.name:
            return track_open(*args, **kwargs)
        if stage == "stat" and not kwargs.get("dir_fd"):
            return original(*args, **kwargs)
        raise PermissionError(errno.EACCES, "synthetic permission failure")

    with monkeypatch.context() as patch:
        patch.setattr(file_output.os, stage, fail)
        with pytest.raises(ValueError, match="synthetic permission failure"):
            preflight_output(output)
    for fd in opened:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(fd)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("capability", ["secure-output", "effective-access"])
def test_preflight_fails_closed_on_unsupported_platform(
    tmp_path, monkeypatch, non_destructive, capability
):
    before = snapshot(tmp_path)
    if capability == "secure-output":
        monkeypatch.setattr(file_output, "_SECURE_OUTPUT_SUPPORTED", False)
    else:

        def unsupported(*args, **kwargs):
            raise NotImplementedError

        monkeypatch.setattr(file_output.os, "access", unsupported)
    with pytest.raises(ValueError, match="unsupported on this platform"):
        preflight_output(tmp_path / "report")
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("replacement", ["symlink", "regular", "fifo"])
def test_preflight_rejects_leaf_swap_before_open(tmp_path, monkeypatch, replacement):
    output = tmp_path / "report"
    output.write_bytes(b"previous")
    protected = tmp_path / "protected"
    protected.write_bytes(b"protected")
    moved = tmp_path / "original"
    original_open = os.open

    def swap_before_open(path, flags, *args, **kwargs):
        if path == output.name:
            output.rename(moved)
            if replacement == "symlink":
                output.symlink_to(protected)
            elif replacement == "regular":
                os.link(protected, output)
            else:
                os.mkfifo(output)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(file_output.os, "open", swap_before_open)
    with pytest.raises(ValueError):
        preflight_output(output)
    assert protected.read_bytes() == b"protected"
    assert moved.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".skillspector-output-*"))


def test_preflight_rechecks_opened_file_type(tmp_path, monkeypatch, non_destructive):
    output = tmp_path / "report"
    output.write_bytes(b"previous")
    before = snapshot(tmp_path)
    real_fstat = os.fstat

    def non_regular_stat(fd):
        values = list(real_fstat(fd))
        values[0] = S_IFIFO | 0o600
        return os.stat_result(values)

    with monkeypatch.context() as patch:
        patch.setattr(file_output.os, "fstat", non_regular_stat)
        with pytest.raises(ValueError, match="non-regular"):
            preflight_output(output)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("timing", ["before-open", "after-open"])
def test_preflight_parent_swap_uses_no_follow_anchor(tmp_path, monkeypatch, timing):
    parent = tmp_path / "reports"
    parent.mkdir()
    output = parent / "report"
    output.write_bytes(b"previous")
    expected = output.stat()
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / output.name
    protected.write_bytes(b"protected")
    outside_before = snapshot(outside)
    moved = tmp_path / "original-reports"
    original_open = os.open

    def swap_parent(path, flags, *args, **kwargs):
        if path == parent.name:
            if timing == "after-open":
                fd = original_open(path, flags, *args, **kwargs)
            parent.rename(moved)
            parent.symlink_to(outside, target_is_directory=True)
            if timing == "after-open":
                return fd
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(file_output.os, "open", swap_parent)
    if timing == "before-open":
        with pytest.raises(ValueError, match="symlink or is not a directory"):
            preflight_output(output)
    else:
        normalized, info = preflight_output(output)
        assert normalized == output
        assert (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino)
    assert (moved / output.name).read_bytes() == b"previous"
    assert snapshot(outside) == outside_before
    assert not list(tmp_path.rglob(".skillspector-output-*"))


def test_writer_still_rejects_symlink_created_after_preflight(tmp_path):
    output = tmp_path / "report"
    assert preflight_output(output)[1] is None
    protected = tmp_path / "protected"
    protected.write_bytes(b"protected")
    output.symlink_to(protected)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="non-regular"):
        write_text_no_follow(output, "new report")
    assert snapshot(tmp_path) == before
