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

import pytest

from skillspector.input_handler import InputHandler


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


def test_extract_zip_slip_traversal(tmp_path: Path) -> None:
    """Verify that extracting a zip file containing directory traversal members raises a ValueError."""
    import zipfile
    from skillspector.input_handler import InputHandler

    zip_file = tmp_path / "malicious.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        zf.writestr("../../../evil.txt", "malicious payload")

    handler = InputHandler()
    # Force handler's temp dir to local tmp_path
    handler._temp_dir = tmp_path

    import pytest
    with pytest.raises(ValueError, match="Zip Slip detected"):
        handler._extract_zip(zip_file)

