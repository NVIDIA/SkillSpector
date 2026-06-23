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

"""Tests for SSRF protection and zip-slip prevention in input_handler."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from skillspector.input_handler import (
    ALLOWED_DOWNLOAD_HOSTS,
    ALLOWED_GIT_HOSTS,
    InputHandler,
    _is_private_ip,
)


class TestPrivateIPDetection:
    """_is_private_ip blocks internal network addresses."""

    def test_localhost_blocked(self) -> None:
        assert _is_private_ip("127.0.0.1") is True

    def test_ipv6_loopback_blocked(self) -> None:
        assert _is_private_ip("::1") is True

    def test_private_10_range_blocked(self) -> None:
        assert _is_private_ip("10.0.0.1") is True

    def test_private_172_range_blocked(self) -> None:
        assert _is_private_ip("172.16.0.1") is True

    def test_private_192_range_blocked(self) -> None:
        assert _is_private_ip("192.168.1.1") is True

    def test_link_local_169_blocked(self) -> None:
        assert _is_private_ip("169.254.169.254") is True

    def test_public_ip_allowed(self) -> None:
        assert _is_private_ip("140.82.121.3") is False

    def test_unresolvable_host_blocked(self) -> None:
        assert _is_private_ip("definitely-not-a-real-host-xyz123.invalid") is True


class TestGitCloneSSRF:
    """Git clone validates URLs against allowlist and SSRF."""

    def test_internal_git_url_blocked(self) -> None:
        handler = InputHandler()
        with pytest.raises(ValueError, match="not in the allowed hosts"):
            handler._clone_git("https://internal-gitlab.corp.local/repo.git")
        handler.cleanup()

    def test_localhost_git_url_blocked(self) -> None:
        handler = InputHandler()
        with pytest.raises(ValueError, match="not in the allowed hosts"):
            handler._clone_git("http://127.0.0.1:8080/repo.git")
        handler.cleanup()

    def test_metadata_endpoint_blocked(self) -> None:
        handler = InputHandler()
        with pytest.raises(ValueError, match="not in the allowed hosts"):
            handler._clone_git("http://169.254.169.254/latest/meta-data/")
        handler.cleanup()

    @patch("skillspector.input_handler.subprocess.run")
    def test_github_url_allowed(self, mock_run) -> None:
        mock_run.return_value = None
        handler = InputHandler()
        handler._clone_git("https://github.com/NVIDIA/SkillSpector.git")
        mock_run.assert_called_once()
        handler.cleanup()

    @patch("skillspector.input_handler.subprocess.run")
    def test_gitlab_url_allowed(self, mock_run) -> None:
        mock_run.return_value = None
        handler = InputHandler()
        handler._clone_git("https://gitlab.com/user/repo.git")
        mock_run.assert_called_once()
        handler.cleanup()


class TestDownloadSSRF:
    """File download validates URLs against allowlist and SSRF."""

    def test_internal_url_blocked(self) -> None:
        handler = InputHandler()
        with pytest.raises(ValueError, match="not in the allowed hosts"):
            handler._download_file("http://192.168.1.100/secrets.txt")
        handler.cleanup()

    def test_cloud_metadata_blocked(self) -> None:
        handler = InputHandler()
        with pytest.raises(ValueError, match="not in the allowed hosts"):
            handler._download_file("http://169.254.169.254/latest/meta-data/iam/")
        handler.cleanup()

    def test_arbitrary_host_blocked(self) -> None:
        handler = InputHandler()
        with pytest.raises(ValueError, match="not in the allowed hosts"):
            handler._download_file("https://evil-attacker.com/payload.md")
        handler.cleanup()

    @patch("skillspector.input_handler.httpx.Client")
    def test_raw_githubusercontent_allowed(self, mock_client_cls) -> None:
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_response = mock_client.get.return_value
        mock_response.content = b"# SKILL.md content"
        mock_response.headers = {}
        handler = InputHandler()
        result = handler._download_file(
            "https://raw.githubusercontent.com/NVIDIA/SkillSpector/main/SKILL.md"
        )
        assert result.is_dir()
        handler.cleanup()

    @patch("skillspector.input_handler.httpx.Client")
    def test_download_does_not_follow_redirects(self, mock_client_cls) -> None:
        """Redirects are disabled to prevent SSRF via open-redirect on allowed hosts."""
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_client.get.return_value.content = b"# content"
        mock_client.get.return_value.headers = {}
        handler = InputHandler()
        try:
            handler._download_file(
                "https://raw.githubusercontent.com/NVIDIA/SkillSpector/main/SKILL.md"
            )
        except Exception:
            pass
        mock_client_cls.assert_called_once_with(follow_redirects=False, timeout=30)
        handler.cleanup()


class TestZipSlipPrevention:
    """Zip extraction blocks path traversal attacks."""

    def test_zip_slip_blocked(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "malicious.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../etc/passwd", "root:x:0:0:")
        handler = InputHandler()
        handler._temp_dir = tmp_path / "work"
        handler._temp_dir.mkdir()
        with pytest.raises(ValueError, match="zip-slip"):
            handler._extract_zip(zip_path)
        handler.cleanup()

    def test_normal_zip_extracts_fine(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "normal.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("skill/SKILL.md", "# Normal skill")
            zf.writestr("skill/tool.py", "print('hello')")
        handler = InputHandler()
        handler._temp_dir = tmp_path / "work"
        handler._temp_dir.mkdir()
        result = handler._extract_zip(zip_path)
        assert result.is_dir()
        assert (result / "SKILL.md").exists()
        handler.cleanup()

    def test_deeply_nested_path_allowed(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "deep.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a/b/c/d/file.txt", "deep content")
        handler = InputHandler()
        handler._temp_dir = tmp_path / "work"
        handler._temp_dir.mkdir()
        result = handler._extract_zip(zip_path)
        assert result.is_dir()
        handler.cleanup()


class TestAllowlistConfiguration:
    """Allowlists contain expected hosts."""

    def test_git_hosts_include_major_forges(self) -> None:
        assert "github.com" in ALLOWED_GIT_HOSTS
        assert "gitlab.com" in ALLOWED_GIT_HOSTS
        assert "bitbucket.org" in ALLOWED_GIT_HOSTS

    def test_download_hosts_include_raw_github(self) -> None:
        assert "raw.githubusercontent.com" in ALLOWED_DOWNLOAD_HOSTS

    def test_download_hosts_include_huggingface(self) -> None:
        assert "huggingface.co" in ALLOWED_DOWNLOAD_HOSTS
