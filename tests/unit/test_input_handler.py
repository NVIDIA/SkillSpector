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

"""Tests for skillspector input_handler (resolve directory, zip, single file)."""

import ctypes
import os
import sys
from errno import ENOENT
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from skillspector.input_handler import (
    ALLOWED_GIT_HOSTS,
    InputHandler,
    _open_regular_file_from_windows_handle,
    _open_regular_file_no_follow,
)


def _mock_windows_secure_open(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    *,
    handle: int = 1,
    attributes: int = 0,
    final_path: str | None = None,
) -> None:
    """Install a handle-level Windows open simulation on any platform."""

    def get_file_information(_handle: int, information: object) -> bool:
        information._obj.dwFileAttributes = attributes  # type: ignore[attr-defined]
        return True

    def get_final_path(_handle: int, buffer: object, _size: int, _flags: int) -> int:
        opened_path = final_path or str(source)
        buffer.value = opened_path  # type: ignore[attr-defined]
        return len(opened_path)

    kernel32 = SimpleNamespace(
        CreateFileW=lambda *_args: handle,
        GetFileInformationByHandle=get_file_information,
        GetFinalPathNameByHandleW=get_final_path,
        CloseHandle=lambda _handle: True,
    )
    msvcrt = SimpleNamespace(open_osfhandle=lambda _handle, _flags: os.open(source, os.O_RDONLY))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: kernel32, raising=False)
    monkeypatch.setattr(os, "O_BINARY", 0, raising=False)
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt)


def test_resolve_directory(tmp_path: Path) -> None:
    """Resolving a local directory returns path and source_type directory."""
    (tmp_path / "SKILL.md").write_text("# Skill", encoding="utf-8")
    handler = InputHandler()
    try:
        resolved, source_type = handler.resolve(str(tmp_path))
        assert resolved.is_dir()
        assert (resolved / "SKILL.md").exists()
        assert source_type == "directory"
    finally:
        handler.cleanup()


def test_resolve_single_md_file(tmp_path: Path) -> None:
    """Resolving a single .md file wraps it in a temp dir."""
    f = tmp_path / "doc.md"
    f.write_text("# Doc", encoding="utf-8")
    handler = InputHandler()
    try:
        resolved, source_type = handler.resolve(str(f))
        assert resolved.is_dir()
        assert (resolved / "doc.md").exists()
        assert source_type == "file"
    finally:
        handler.cleanup()


def test_resolve_single_symlinked_file_raises(tmp_path: Path) -> None:
    """Standalone file inputs must not dereference symlinks before scanning."""
    secret = tmp_path / "external_secret.md"
    secret.write_text("AWS_SECRET=hunter2", encoding="utf-8")
    symlink = tmp_path / "SKILL.md"
    try:
        symlink.symlink_to(secret)
    except OSError:
        pytest.skip("symlinks are not supported on this filesystem")

    handler = InputHandler()
    try:
        with pytest.raises(ValueError, match="symlinked input"):
            handler.resolve(str(symlink))
        assert handler.temp_dir_for_cleanup() is None
    finally:
        handler.cleanup()


def test_resolve_file_with_symlinked_parent_raises(tmp_path: Path) -> None:
    """Standalone file inputs must not traverse a symlinked parent directory."""
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    (external_dir / "secret.md").write_text("AWS_SECRET=hunter2", encoding="utf-8")
    symlinked_parent = tmp_path / "linked"
    try:
        symlinked_parent.symlink_to(external_dir, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not supported on this filesystem")

    handler = InputHandler()
    try:
        with pytest.raises(ValueError, match="symlinked parent"):
            handler.resolve(str(symlinked_parent / "secret.md"))
        assert handler.temp_dir_for_cleanup() is None
    finally:
        handler.cleanup()


def test_resolve_directory_with_symlinked_parent_raises(tmp_path: Path) -> None:
    """Directory inputs must not escape through a symlinked ancestor."""
    external_skill = tmp_path / "external" / "skill"
    external_skill.mkdir(parents=True)
    (external_skill / "SKILL.md").write_text("# External skill", encoding="utf-8")
    symlinked_parent = tmp_path / "linked"
    try:
        symlinked_parent.symlink_to(external_skill.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not supported on this filesystem")

    handler = InputHandler()
    try:
        with pytest.raises(ValueError, match="symlinked parent"):
            handler.resolve(str(symlinked_parent / external_skill.name))
        assert handler.temp_dir_for_cleanup() is None
    finally:
        handler.cleanup()


def test_resolve_junctioned_directory_raises(tmp_path: Path) -> None:
    """Directory inputs must reject terminal Windows junctions too."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    handler = InputHandler()
    try:
        with patch.object(Path, "is_junction", autospec=True) as is_junction:
            is_junction.side_effect = lambda path: path == skill_dir
            with pytest.raises(ValueError, match="junctioned input"):
                handler.resolve(str(skill_dir))
    finally:
        handler.cleanup()


def test_resolve_file_with_junction_parent_raises(tmp_path: Path) -> None:
    """Standalone file inputs must not traverse Windows junctions."""
    source = tmp_path / "linked" / "SKILL.md"
    source.parent.mkdir()
    source.write_text("# Skill", encoding="utf-8")
    handler = InputHandler()
    try:
        with patch.object(Path, "is_junction", autospec=True) as is_junction:
            is_junction.side_effect = lambda path: path == source.parent
            with pytest.raises(ValueError, match="symlinked parent"):
                handler.resolve(str(source))
        assert handler.temp_dir_for_cleanup() is None
    finally:
        handler.cleanup()


def test_resolve_file_through_root_owned_system_alias(tmp_path: Path) -> None:
    """Root-owned system aliases do not prevent scanning ordinary local files."""
    root_alias = Path("/var")
    try:
        relative_path = tmp_path.relative_to("/private/var")
    except ValueError:
        pytest.skip("temporary directory is not below the macOS /var alias")
    if not root_alias.is_symlink():
        pytest.skip("/var is not a system alias on this platform")

    source = tmp_path / "SKILL.md"
    source.write_text("# Skill", encoding="utf-8")
    handler = InputHandler()
    try:
        resolved, source_type = handler.resolve(str(root_alias / relative_path / source.name))
        assert (resolved / source.name).read_text(encoding="utf-8") == "# Skill"
        assert source_type == "file"
    finally:
        handler.cleanup()


def test_resolve_symlinked_zip_raises(tmp_path: Path) -> None:
    """Local archives must be rejected before their symlink target is opened."""
    archive = tmp_path / "external_archive.zip"
    archive.write_bytes(b"not opened")
    symlink = tmp_path / "skill.zip"
    try:
        symlink.symlink_to(archive)
    except OSError:
        pytest.skip("symlinks are not supported on this filesystem")

    handler = InputHandler()
    try:
        with pytest.raises(ValueError, match="symlinked input"):
            handler.resolve(str(symlink))
        assert handler.temp_dir_for_cleanup() is None
    finally:
        handler.cleanup()


def test_resolve_file_open_failure_does_not_create_temp_dir(tmp_path: Path) -> None:
    """Failed secure opens leave no handler-owned temporary directory behind."""
    source = tmp_path / "SKILL.md"
    source.write_text("# Skill", encoding="utf-8")
    handler = InputHandler()
    try:
        with patch("skillspector.input_handler.os.open", side_effect=OSError("denied")):
            with pytest.raises(ValueError, match="Could not safely open"):
                handler.resolve(str(source))
        assert handler.temp_dir_for_cleanup() is None
    finally:
        handler.cleanup()


@pytest.mark.skipif(not hasattr(os, "O_PATH"), reason="requires O_PATH (Linux)")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses directory permissions"
)
def test_secure_open_traverses_search_only_ancestors(tmp_path: Path) -> None:
    """Traversal needs search access on ancestors, not read access."""
    parent = tmp_path / "search_only"
    parent.mkdir()
    source = parent / "SKILL.md"
    source.write_text("# Skill", encoding="utf-8")
    os.chmod(parent, 0o111)
    try:
        with _open_regular_file_no_follow(source) as opened:
            assert opened.read() == b"# Skill"
    finally:
        os.chmod(parent, 0o755)


def test_resolve_file_rejects_platform_without_safe_open_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scanning must fail closed when neither secure-open implementation is available."""
    source = tmp_path / "SKILL.md"
    source.write_text("# Skill", encoding="utf-8")
    handler = InputHandler()
    try:
        monkeypatch.setattr("skillspector.input_handler._HAS_SECURE_DIR_FD", False)
        monkeypatch.setattr("skillspector.input_handler._IS_WINDOWS", False)
        with pytest.raises(ValueError, match="Secure no-follow file opens are unavailable"):
            handler.resolve(str(source))
        assert handler.temp_dir_for_cleanup() is None
    finally:
        handler.cleanup()


def test_windows_no_follow_open_reads_verified_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows secure-open accepts a verified regular file handle."""
    source = tmp_path / "SKILL.md"
    source.write_text("# Skill", encoding="utf-8")
    _mock_windows_secure_open(monkeypatch, source)

    with _open_regular_file_from_windows_handle(source) as opened:
        assert opened.read() == b"# Skill"


def test_windows_no_follow_open_rejects_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows secure-open reports a missing file without exposing a handle."""
    source = tmp_path / "missing.md"
    _mock_windows_secure_open(monkeypatch, source, handle=ctypes.c_void_p(-1).value)
    monkeypatch.setattr(
        "skillspector.input_handler._windows_last_error", lambda: OSError(ENOENT, "missing")
    )

    with pytest.raises(FileNotFoundError, match="File not found"):
        _open_regular_file_from_windows_handle(source)


def test_windows_no_follow_open_rejects_reparse_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows secure-open rejects a reparse-point handle before reading it."""
    source = tmp_path / "SKILL.md"
    source.write_text("# Skill", encoding="utf-8")
    _mock_windows_secure_open(monkeypatch, source, attributes=0x00000400)

    with pytest.raises(ValueError, match="Could not safely open"):
        _open_regular_file_from_windows_handle(source)


def test_windows_no_follow_open_rejects_canonical_path_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows secure-open rejects a handle whose resolved path changed."""
    source = tmp_path / "SKILL.md"
    source.write_text("# Skill", encoding="utf-8")
    _mock_windows_secure_open(monkeypatch, source, final_path=str(tmp_path / "outside.md"))

    with pytest.raises(ValueError, match="Could not safely open"):
        _open_regular_file_from_windows_handle(source)


def test_resolve_zip_file(tmp_path: Path) -> None:
    """Resolving a .zip file extracts and returns the extract dir."""
    import zipfile

    (tmp_path / "SKILL.md").write_text("# Skill", encoding="utf-8")
    zip_path = tmp_path / "skill.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(tmp_path / "SKILL.md", "SKILL.md")
    handler = InputHandler()
    try:
        resolved, source_type = handler.resolve(str(zip_path))
        assert resolved.is_dir()
        assert source_type == "zip"
    finally:
        handler.cleanup()


def test_resolve_nonexistent_raises() -> None:
    """Resolving a nonexistent path raises FileNotFoundError or ValueError."""
    handler = InputHandler()
    with pytest.raises((FileNotFoundError, ValueError)):
        handler.resolve("/nonexistent/path/xyz")


def test_resolve_single_non_md_file(tmp_path: Path) -> None:
    """Resolving a single non-.md file (e.g. .txt) wraps it in a temp dir."""
    f = tmp_path / "readme.txt"
    f.write_text("Read me", encoding="utf-8")
    handler = InputHandler()
    try:
        resolved, source_type = handler.resolve(str(f))
        assert resolved.is_dir()
        assert (resolved / "readme.txt").exists()
        assert source_type == "file"
    finally:
        handler.cleanup()


def test_cleanup_idempotent(tmp_path: Path) -> None:
    """cleanup() can be called after resolve and does not raise."""
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    handler = InputHandler()
    handler.resolve(str(tmp_path / "a.md"))
    handler.cleanup()
    handler.cleanup()


def test_clone_git_disables_symlinks() -> None:
    """git clone must prevent symlinked entries from materializing as links."""
    handler = InputHandler()
    try:

        def fake_clone(command: list[str], **_kwargs: object) -> MagicMock:
            Path(command[-1]).mkdir(parents=True)
            process = MagicMock()
            process.poll.return_value = 0
            process.wait.return_value = 0
            return process

        with (
            patch.object(handler, "_validate_url_host", return_value="github.com"),
            patch(
                "skillspector.input_handler.subprocess.Popen",
                side_effect=fake_clone,
            ) as mock_run,
        ):
            handler._clone_git("https://github.com/example/repo.git")
            cmd = mock_run.call_args.args[0]

        assert cmd[:2] == ["git", "-c"]
        assert "core.symlinks=false" in cmd
        assert cmd.index("-c") < cmd.index("clone")
    finally:
        handler.cleanup()


def test_scp_url_is_git_url() -> None:
    """scp-style SSH URL is recognised as a Git URL."""
    assert InputHandler()._is_git_url("git@github.com:org/repo.git") is True


def test_http_urls_are_not_accepted_as_remote_inputs() -> None:
    """Network inputs require HTTPS unless they use SSH's scp-style syntax."""
    handler = InputHandler()
    assert handler._is_git_url("http://github.com/org/repo.git") is False
    assert handler._is_file_url("http://raw.githubusercontent.com/org/repo/SKILL.md") is False


def test_validate_url_host_scp_extracts_github() -> None:
    """_validate_url_host extracts 'github.com' from an scp-style URL."""
    with patch("skillspector.input_handler._is_private_ip", return_value=False):
        host = InputHandler()._validate_url_host("git@github.com:org/repo.git", ALLOWED_GIT_HOSTS)
    assert host == "github.com"


def test_scp_valid_host_clones() -> None:
    """resolve() calls git clone with the scp URL when the host is allowed."""
    handler = InputHandler()
    try:

        def fake_clone(command: list[str], **_kwargs: object) -> MagicMock:
            Path(command[-1]).mkdir(parents=True)
            process = MagicMock()
            process.poll.return_value = 0
            process.wait.return_value = 0
            return process

        with (
            patch("skillspector.input_handler._is_private_ip", return_value=False),
            patch(
                "skillspector.input_handler.subprocess.Popen", side_effect=fake_clone
            ) as mock_run,
        ):
            handler.resolve("git@github.com:org/repo.git")
        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "git@github.com:org/repo.git" in call_args
    finally:
        handler.cleanup()


def test_scp_disallowed_host_raises() -> None:
    """_validate_url_host rejects an scp URL whose host is not in the allowlist."""
    with pytest.raises(ValueError, match="not in the allowed hosts"):
        InputHandler()._validate_url_host("git@malicious.internal:org/repo.git", ALLOWED_GIT_HOSTS)


def test_scp_private_ip_raises() -> None:
    """_validate_url_host rejects an scp URL whose extracted host is not in the allowlist."""
    with pytest.raises(ValueError):
        InputHandler()._validate_url_host("git@169.254.169.254:org/repo.git", ALLOWED_GIT_HOSTS)


def test_https_url_unchanged() -> None:
    """https URLs continue to extract the host via urlparse without hitting the scp fallback."""
    with patch("skillspector.input_handler._is_private_ip", return_value=False):
        host = InputHandler()._validate_url_host(
            "https://github.com/org/repo.git", ALLOWED_GIT_HOSTS
        )
    assert host == "github.com"


def test_scp_ssrf_gate_fires() -> None:
    """SSRF gate raises ValueError for an scp URL whose host resolves to a private IP."""
    with patch("skillspector.input_handler._is_private_ip", return_value=True):
        with pytest.raises(ValueError, match="private/internal IP"):
            InputHandler()._validate_url_host("git@github.com:org/repo.git", ALLOWED_GIT_HOSTS)
