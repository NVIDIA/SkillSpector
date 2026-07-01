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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skillspector.input_handler import ALLOWED_GIT_HOSTS, InputHandler


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


def test_scp_url_is_git_url() -> None:
    """scp-style SSH URL is recognised as a Git URL."""
    assert InputHandler()._is_git_url("git@github.com:org/repo.git") is True


def test_validate_url_host_scp_extracts_github() -> None:
    """_validate_url_host extracts 'github.com' from an scp-style URL."""
    host = InputHandler()._validate_url_host("git@github.com:org/repo.git", ALLOWED_GIT_HOSTS)
    assert host == "github.com"


def test_scp_valid_host_clones() -> None:
    """resolve() calls git clone with the scp URL when the host is allowed."""
    handler = InputHandler()
    try:
        with patch(
            "skillspector.input_handler.subprocess.run", return_value=MagicMock()
        ) as mock_run:
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
    host = InputHandler()._validate_url_host("https://github.com/org/repo.git", ALLOWED_GIT_HOSTS)
    assert host == "github.com"


def test_scp_ssrf_gate_fires() -> None:
    """SSRF gate raises ValueError for an scp URL whose host resolves to a private IP."""
    with patch("skillspector.input_handler._is_private_ip", return_value=True):
        with pytest.raises(ValueError, match="private/internal IP"):
            InputHandler()._validate_url_host("git@github.com:org/repo.git", ALLOWED_GIT_HOSTS)
